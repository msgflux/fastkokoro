from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper

from fastkokoro.config import Settings

logger = logging.getLogger("uvicorn.error")


def resolve_adain_fused_model_path(model_path: Path, settings: Settings) -> Path:
    if not settings.onnx_adain_fusion:
        return model_path

    if settings.onnx_adain_custom_op_library is None:
        raise ValueError(
            "FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY is required when "
            "FASTKOKORO_ONNX_ADAIN_FUSION is enabled"
        )
    if not settings.onnx_adain_custom_op_library.exists():
        raise FileNotFoundError(settings.onnx_adain_custom_op_library)

    if settings.onnx_adain_model_path is not None:
        if not settings.onnx_adain_model_path.exists():
            raise FileNotFoundError(settings.onnx_adain_model_path)
        return settings.onnx_adain_model_path

    cache_path = _default_adain_cache_path(model_path, settings)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fused = fuse_generator_adain(model_path, cache_path)
    logger.info("Generated AdaIN-fused ONNX model: path=%s fused=%s", cache_path, fused)
    return cache_path


def _default_adain_cache_path(model_path: Path, settings: Settings) -> Path:
    digest = hashlib.sha256(str(model_path.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = model_path.name.removesuffix(".onnx")
    return settings.cache_dir / "onnx" / f"{stem}.adain.{digest}.onnx"


def fuse_generator_adain(input_path: Path, output_path: Path) -> int:
    model = onnx.load(input_path)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    fused = 0
    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            continue
        if "/decoder/decoder/generator/" not in node.name:
            continue
        match = _match_adain(node, consumers)
        if match is None:
            continue
        remove_nodes, source, scale_input, shift_input, output = match
        replacements[node.name] = helper.make_node(
            "AdaIn",
            inputs=[source, scale_input, shift_input],
            outputs=[output],
            name=node.name + "_AdaIn",
            domain="fastkokoro",
        )
        remove.update(remove_nodes)
        fused += 1

    if fused == 0:
        raise ValueError(
            f"No generator AdaIN patterns found in ONNX model: {input_path}"
        )

    new_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(node.name)
        if replacement is not None:
            new_nodes.append(replacement)
        elif node.name not in remove:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def fuse_cuda_instance_adain(input_path: Path, output_path: Path) -> int:
    model = onnx.load(input_path)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    fused = 0
    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            continue
        if "/decoder/generator/" not in node.name:
            continue
        match = _find_affine_after_instance_norm(node, consumers)
        if match is None:
            continue
        remove_nodes, scale_input, shift_input, output_name = match
        replacements[node.name] = helper.make_node(
            "AdaInCudaFp16",
            inputs=[
                node.input[0],
                node.input[1],
                node.input[2],
                scale_input,
                shift_input,
            ],
            outputs=[output_name],
            name=node.name + "_AdaInCudaFp16",
            domain="fastkokoro",
        )
        remove.update(remove_nodes)
        fused += 1

    if fused == 0:
        raise ValueError(
            "No generator InstanceNorm+AdaIN patterns found in ONNX model: "
            f"{input_path}"
        )

    new_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(node.name)
        if replacement is not None:
            new_nodes.append(replacement)
        elif node.name not in remove:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def fuse_cuda_adain_snake(input_path: Path, output_path: Path) -> int:
    model = onnx.load(input_path)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    fused = 0
    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            continue
        if "/decoder/generator/" not in node.name:
            continue
        match = _find_snake_after_instance_norm(node, consumers)
        if match is None:
            continue
        remove_nodes, scale_input, shift_input, alpha_input, output_name = match
        replacements[node.name] = helper.make_node(
            "AdaInSnakeCudaFp16",
            inputs=[
                node.input[0],
                node.input[1],
                node.input[2],
                scale_input,
                shift_input,
                alpha_input,
            ],
            outputs=[output_name],
            name=node.name + "_AdaInSnakeCudaFp16",
            domain="fastkokoro",
        )
        remove.update(remove_nodes)
        fused += 1

    if fused == 0:
        raise ValueError(
            "No generator InstanceNorm+AdaIN+Snake patterns found in ONNX model: "
            f"{input_path}"
        )

    new_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(node.name)
        if replacement is not None:
            new_nodes.append(replacement)
        elif node.name not in remove:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def fuse_cuda_atan2(input_path: Path, output_path: Path) -> int:
    model = onnx.load(input_path)
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            consumers[name].append(node)

    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    fused = 0
    for node in model.graph.node:
        if node.op_type != "Atan":
            continue
        match = _find_atan2_phase(node, producers, consumers)
        if match is None:
            continue
        remove_nodes, imag_input, real_input, output_name = match
        replacements[node.name] = helper.make_node(
            "Atan2CudaFp16",
            inputs=[imag_input, real_input],
            outputs=[output_name],
            name=node.name + "_Atan2CudaFp16",
            domain="fastkokoro",
        )
        remove.update(remove_nodes)
        fused += 1

    if fused == 0:
        raise ValueError(f"No atan2 phase pattern found in ONNX model: {input_path}")

    new_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(node.name)
        if replacement is not None:
            new_nodes.append(replacement)
        elif node.name not in remove:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def _match_adain(
    reduce_mean: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[set[str], str, str, str, str] | None:
    source = reduce_mean.input[0]
    source_consumers = consumers.get(source, [])
    sub = next((node for node in source_consumers if node.op_type == "Sub"), None)
    if sub is None or sub.input[0] != source or sub.input[1] != reduce_mean.output[0]:
        return None

    square = next(
        (
            node
            for node in consumers.get(sub.output[0], [])
            if node.op_type == "Mul"
            and node.input[0] == sub.output[0]
            and node.input[1] == sub.output[0]
        ),
        None,
    )
    if square is None:
        return None

    reduce_var = _only_consumer(consumers, square.output[0], "ReduceMean")
    add_eps = (
        _only_consumer(consumers, reduce_var.output[0], "Add") if reduce_var else None
    )
    sqrt = _only_consumer(consumers, add_eps.output[0], "Sqrt") if add_eps else None
    div = next(
        (
            node
            for node in consumers.get(sub.output[0], [])
            if node.op_type == "Div"
            and sqrt is not None
            and node.input[1] == sqrt.output[0]
        ),
        None,
    )
    if div is None or div.input[0] != sub.output[0]:
        return None

    mul_scale = _only_consumer(consumers, div.output[0], "Mul")
    add_shift = (
        _only_consumer(consumers, mul_scale.output[0], "Add") if mul_scale else None
    )
    if add_shift is None:
        return None
    scale_input = (
        mul_scale.input[0]
        if mul_scale.input[1] == div.output[0]
        else mul_scale.input[1]
    )
    shift_input = (
        add_shift.input[0]
        if add_shift.input[1] == mul_scale.output[0]
        else add_shift.input[1]
    )
    remove = {
        reduce_mean.name,
        sub.name,
        square.name,
        reduce_var.name,
        add_eps.name,
        sqrt.name,
        div.name,
        mul_scale.name,
        add_shift.name,
    }
    return remove, source, scale_input, shift_input, add_shift.output[0]


def _find_atan2_phase(
    atan: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[set[str], str, str, str] | None:
    div = producers.get(atan.input[0])
    if div is None or div.op_type != "Div" or len(div.input) != 2:
        return None
    imag_input, real_input = div.input

    add_pi = _only_consumer_with_input(consumers, atan.output[0], "Add")
    sub_pi = _only_consumer_with_input(consumers, atan.output[0], "Sub")
    where_quadrant = None
    if add_pi is not None and sub_pi is not None:
        for candidate in consumers.get(add_pi.output[0], []):
            if candidate.op_type == "Where" and sub_pi.output[0] in candidate.input:
                where_quadrant = candidate
                break
    if where_quadrant is None:
        return None

    where_atan = None
    for candidate in consumers.get(where_quadrant.output[0], []):
        if candidate.op_type == "Where" and atan.output[0] in candidate.input:
            where_atan = candidate
            break
    if where_atan is None:
        return None

    where_zero = _only_consumer(consumers, where_atan.output[0], "Where")
    if where_zero is None:
        return None

    remove = {
        div.name,
        atan.name,
        add_pi.name,
        sub_pi.name,
        where_quadrant.name,
        where_atan.name,
        where_zero.name,
    }
    return remove, imag_input, real_input, where_zero.output[0]


def _find_affine_after_instance_norm(
    instance_norm: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[set[str], str, str, str] | None:
    mul_scale = _only_consumer(consumers, instance_norm.output[0], "Mul")
    add_shift = (
        _only_consumer(consumers, mul_scale.output[0], "Add") if mul_scale else None
    )
    if mul_scale is None or add_shift is None:
        return None
    scale_input = (
        mul_scale.input[0]
        if mul_scale.input[1] == instance_norm.output[0]
        else mul_scale.input[1]
    )
    shift_input = (
        add_shift.input[0]
        if add_shift.input[1] == mul_scale.output[0]
        else add_shift.input[1]
    )
    return (
        {instance_norm.name, mul_scale.name, add_shift.name},
        scale_input,
        shift_input,
        add_shift.output[0],
    )


def _find_snake_after_instance_norm(
    instance_norm: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[set[str], str, str, str, str] | None:
    affine = _find_affine_after_instance_norm(instance_norm, consumers)
    if affine is None:
        return None
    affine_nodes, scale_input, shift_input, adain_output = affine

    alpha_mul = _find_binary_consumer(consumers, adain_output, "Mul")
    snake_add = _find_binary_consumer(consumers, adain_output, "Add")
    if alpha_mul is None or snake_add is None:
        return None
    alpha_input = (
        alpha_mul.input[0] if alpha_mul.input[1] == adain_output else alpha_mul.input[1]
    )
    sin = _only_consumer(consumers, alpha_mul.output[0], "Sin")
    pow_node = _only_consumer(consumers, sin.output[0], "Pow") if sin else None
    scale_mul = (
        _find_binary_consumer(consumers, pow_node.output[0], "Mul")
        if pow_node
        else None
    )
    if scale_mul is None:
        return None

    reciprocal = None
    for candidate in consumers.get(alpha_input, []):
        if candidate.op_type == "Reciprocal":
            reciprocal = candidate
            break
    if reciprocal is None or reciprocal.output[0] not in scale_mul.input:
        return None
    if reciprocal.input[0] != alpha_input:
        return None
    if scale_mul.output[0] not in snake_add.input:
        return None

    remove = set(affine_nodes)
    remove.update(
        {
            reciprocal.name,
            alpha_mul.name,
            sin.name,
            pow_node.name,
            scale_mul.name,
            snake_add.name,
        }
    )
    return remove, scale_input, shift_input, alpha_input, snake_add.output[0]


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = [node for node in consumers.get(output, []) if node.op_type == op_type]
    if len(nodes) != 1:
        return None
    return nodes[0]


def _find_binary_consumer(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = [
        node
        for node in consumers.get(output, [])
        if node.op_type == op_type and output in node.input and len(node.input) == 2
    ]
    if len(nodes) != 1:
        return None
    return nodes[0]


def _only_consumer_with_input(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = [
        node
        for node in consumers.get(output, [])
        if node.op_type == op_type and output in node.input
    ]
    if len(nodes) != 1:
        return None
    return nodes[0]
