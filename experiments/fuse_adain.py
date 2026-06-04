from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], output: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = [node for node in consumers.get(output, []) if node.op_type == op_type]
    if len(nodes) != 1:
        return None
    return nodes[0]


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
    return remove, source, scale_input, shift_input, add_shift.output[0]


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
        if node.op_type != "ReduceMean":
            continue
        if "/decoder/decoder/generator/" not in node.name:
            continue
        if name_contains and not any(value in node.name for value in name_contains):
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
