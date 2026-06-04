from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper


def _attrs(node: onnx.NodeProto) -> dict[str, list[int] | int]:
    values: dict[str, list[int] | int] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INTS:
            values[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.INT:
            values[attr.name] = attr.i
    return values


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = consumers.get(output, [])
    if len(nodes) != 1 or nodes[0].op_type != op_type:
        return None
    return nodes[0]


def _find_adain_after_conv(
    conv: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[set[str], str, str, str] | None:
    conv_output = conv.output[0]
    first = consumers.get(conv_output, [])
    if {n.op_type for n in first} != {"ReduceMean", "Sub"} or len(first) != 2:
        return None
    reduce_mean = next((n for n in first if n.op_type == "ReduceMean"), None)
    sub = next((n for n in first if n.op_type == "Sub"), None)
    if reduce_mean is None or sub is None:
        return None
    if sub.input[0] != conv_output or sub.input[1] != reduce_mean.output[0]:
        return None
    square = next(
        (
            n
            for n in consumers.get(sub.output[0], [])
            if n.op_type == "Mul"
            and n.input[0] == sub.output[0]
            and n.input[1] == sub.output[0]
        ),
        None,
    )
    if (
        square is None
        or square.input[0] != sub.output[0]
        or square.input[1] != sub.output[0]
    ):
        return None
    reduce_var = _only_consumer(consumers, square.output[0], "ReduceMean")
    add_eps = (
        _only_consumer(consumers, reduce_var.output[0], "Add") if reduce_var else None
    )
    sqrt = _only_consumer(consumers, add_eps.output[0], "Sqrt") if add_eps else None
    div = next(
        (
            n
            for n in consumers.get(sub.output[0], [])
            if n.op_type == "Div"
            and len(n.input) == 2
            and sqrt is not None
            and n.input[1] == sqrt.output[0]
        ),
        None,
    )
    if div is None or div.input[0] != sub.output[0] or div.input[1] != sqrt.output[0]:
        return None
    mul_scale = _only_consumer(consumers, div.output[0], "Mul")
    add_shift = (
        _only_consumer(consumers, mul_scale.output[0], "Add") if mul_scale else None
    )
    if mul_scale is None or add_shift is None:
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
    return remove, add_shift.output[0], scale_input, shift_input


def fuse(input_path: Path, output_path: Path, name_contains: list[str]) -> int:
    model = onnx.load(input_path)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    fused = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if "/decoder/decoder/generator/" not in node.name:
            continue
        if name_contains and not any(value in node.name for value in name_contains):
            continue
        match = _find_adain_after_conv(node, consumers)
        if match is None:
            continue
        attrs = _attrs(node)
        pads = attrs.get("pads", [0, 0])
        dilations = attrs.get("dilations", [1])
        strides = attrs.get("strides", [1])
        groups = attrs.get("group", 1)
        if (
            strides != [1]
            or groups != 1
            or not isinstance(pads, list)
            or not isinstance(dilations, list)
        ):
            continue
        remove_nodes, output_name, scale_input, shift_input = match
        fused_node = helper.make_node(
            "Conv1dAdaIn",
            inputs=[
                node.input[0],
                node.input[1],
                node.input[2],
                scale_input,
                shift_input,
            ],
            outputs=[output_name],
            name=node.name + "_Conv1dAdaIn",
            domain="fastkokoro",
            pad_left=int(pads[0]),
            dilation=int(dilations[0]),
        )
        replacements[node.name] = fused_node
        remove.update(remove_nodes)
        fused += 1

    new_nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
        elif node.name not in remove:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name-contains", action="append", default=[])
    args = parser.parse_args()
    fused = fuse(args.input, args.output, args.name_contains)
    print(f"fused={fused} output={args.output}")


if __name__ == "__main__":
    main()
