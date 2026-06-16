from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastkokoro.fixed_shape_analysis import inspect_fixed_shape_readiness
from fastkokoro.fixed_shape_experiments import (
    EXPERIMENTAL_VARIANTS,
    FixedShapeVariantSpec,
    write_fixed_shape_variant,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-shape experiment variants and compare how much "
            "dynamic shape remains reachable from token inputs."
        )
    )
    parser.add_argument("model", type=Path, help="Path to the base ONNX model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo-output/fixed-shape-variants"),
        help="Directory where generated ONNX variants and reports will be written",
    )
    parser.add_argument(
        "--input-bucket",
        type=int,
        default=64,
        help="Bucket to use for input-fixing variants",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=120000,
        help="Output length to use for padding variants",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        _resolve_variant(spec, args.input_bucket, args.output_length)
        for spec in EXPERIMENTAL_VARIANTS
    ]
    summary = []
    for variant in variants:
        if variant.name == "base":
            model_path = args.model
        else:
            model_path = args.output_dir / f"{args.model.stem}.{variant.name}.onnx"
            write_fixed_shape_variant(
                args.model,
                model_path,
                fixed_input_bucket=variant.fixed_input_bucket,
                fixed_output_length=variant.fixed_output_length,
                bert_attention_mask=variant.bert_attention_mask,
                bert_fixed_embedding_indices=variant.bert_fixed_embedding_indices,
                bert_fixed_sequence_length=variant.bert_fixed_sequence_length,
                bert_fixed_attention_reshapes=variant.bert_fixed_attention_reshapes,
                predictor_text_encoder_shapes=variant.predictor_text_encoder_shapes,
            )

        report = inspect_fixed_shape_readiness(model_path)
        record = {
            "name": variant.name,
            "model_path": str(model_path),
            "fixed_input_bucket": variant.fixed_input_bucket,
            "fixed_output_length": variant.fixed_output_length,
            "bert_attention_mask": variant.bert_attention_mask,
            "bert_fixed_embedding_indices": variant.bert_fixed_embedding_indices,
            "bert_fixed_sequence_length": variant.bert_fixed_sequence_length,
            "bert_fixed_attention_reshapes": variant.bert_fixed_attention_reshapes,
            "predictor_text_encoder_shapes": variant.predictor_text_encoder_shapes,
            "dynamic_tensors": len(report.dynamic_tensors),
            "reachable_dynamic_tensors": len(report.reachable_dynamic_tensors),
            "reachable_dynamic_nodes": len(report.reachable_dynamic_nodes),
            "input_slice_barriers": len(report.input_slice_barriers),
            "shape_driver_counts": report.shape_driver_counts,
            "notes": list(report.notes),
        }
        summary.append(record)
        print(
            f"{variant.name:12} dynamic_tensors={record['dynamic_tensors']:4d} "
            f"reachable_dynamic_nodes={record['reachable_dynamic_nodes']:4d} "
            f"input_slice_barriers={record['input_slice_barriers']}"
        )

    summary_path = args.output_dir / f"{args.model.stem}.comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nsummary_json={summary_path}")
    return 0


def _resolve_variant(
    spec: FixedShapeVariantSpec,
    input_bucket: int,
    output_length: int,
) -> FixedShapeVariantSpec:
    return FixedShapeVariantSpec(
        name=spec.name,
        fixed_input_bucket=(
            input_bucket if spec.fixed_input_bucket is not None else None
        ),
        fixed_output_length=(
            output_length if spec.fixed_output_length is not None else None
        ),
        bert_attention_mask=spec.bert_attention_mask,
        bert_fixed_embedding_indices=spec.bert_fixed_embedding_indices,
        bert_fixed_sequence_length=spec.bert_fixed_sequence_length,
        bert_fixed_attention_reshapes=spec.bert_fixed_attention_reshapes,
        predictor_text_encoder_shapes=spec.predictor_text_encoder_shapes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
