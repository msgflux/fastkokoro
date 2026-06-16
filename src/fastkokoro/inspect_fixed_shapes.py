from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastkokoro.fixed_shape_analysis import (
    inspect_fixed_shape_readiness,
    render_fixed_shape_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an ONNX graph for fixed-shape readiness and report where "
            "dynamic shapes remain reachable from token inputs."
        )
    )
    parser.add_argument("model", type=Path, help="Path to the ONNX model")
    parser.add_argument(
        "--seed-input",
        dest="seed_inputs",
        action="append",
        default=[],
        help="Input tensor name to treat as the fixed-shape entry point",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Render the report as JSON instead of plain text",
    )
    parser.add_argument(
        "--no-shape-inference",
        action="store_true",
        help="Skip ONNX shape inference and inspect only existing graph metadata",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = inspect_fixed_shape_readiness(
        args.model,
        seed_inputs=tuple(args.seed_inputs) or ("tokens", "input_ids"),
        infer_shapes=not args.no_shape_inference,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_fixed_shape_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
