from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
from collections import Counter
from pathlib import Path

import onnx
from onnx.external_data_helper import convert_model_from_external_data

from fastkokoro.graph_fusion import fuse_portable_atan2
from fastkokoro.onnx_simplification import simplify_onnx_model


def remove_unused_initializers(model: onnx.ModelProto) -> int:
    used_values = {output.name for output in model.graph.output}

    def collect_graph_inputs(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            used_values.update(name for name in node.input if name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect_graph_inputs(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        collect_graph_inputs(subgraph)

    collect_graph_inputs(model.graph)
    kept = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_values
    ]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply ORT 1.18-compatible Kokoro ONNX graph optimizations."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxsim before graph fusion and clean the fused graph afterward.",
    )
    parser.add_argument(
        "--atan2",
        choices=("none", "portable"),
        default="none",
        help="Replace the vocoder atan2 graph with standard portable ONNX ops.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not (args.simplify or args.atan2 != "none"):
        raise ValueError("select at least one graph optimization")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "input": str(args.input),
        "output": str(args.output),
    }
    with tempfile.TemporaryDirectory(
        prefix=".fastkokoro-opt-",
        dir=args.output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        current = args.input.resolve()
        stage = 0

        def next_path(label: str) -> Path:
            nonlocal stage
            stage += 1
            return temporary / f"{stage:02d}-{label}.onnx"

        if args.simplify:
            destination = next_path("simplified")
            current = simplify_onnx_model(current, destination)
            report["simplified"] = True

        if args.atan2 == "portable":
            destination = next_path("atan2-portable")
            report["atan2_fusions"] = fuse_portable_atan2(current, destination)
            current = destination

        model = onnx.load(current)
        metadata = {item.key: item.value for item in model.metadata_props}
        if args.simplify:
            metadata["fastkokoro.graph_simplifier"] = "onnxsim"
            metadata["fastkokoro.onnxsim_version"] = importlib.metadata.version(
                "onnxsim"
            )
        if args.atan2 == "portable":
            metadata["fastkokoro.atan2"] = "portable-polynomial-fp32-v1"
        onnx.helper.set_model_props(model, metadata)
        report["unused_initializers_removed"] = remove_unused_initializers(model)
        convert_model_from_external_data(model)
        onnx.save_model(model, args.output, save_as_external_data=False)

    onnx.checker.check_model(str(args.output), full_check=False)
    final_model = onnx.load(args.output, load_external_data=False)
    operators = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in final_model.graph.node
    )
    report.update(
        {
            "bytes": args.output.stat().st_size,
            "nodes": len(final_model.graph.node),
            "opsets": {
                opset.domain or "ai.onnx": opset.version
                for opset in final_model.opset_import
            },
            "standard_atan_nodes": operators["ai.onnx::Atan"],
        }
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
