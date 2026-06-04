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


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = [node for node in consumers.get(output, []) if node.op_type == op_type]
    if len(nodes) != 1:
        return None
    return nodes[0]
