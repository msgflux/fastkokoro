from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort

from fastkokoro.fixed_shape_experiments import (
    EXPERIMENTAL_VARIANTS,
    FixedShapeVariantSpec,
    write_fixed_shape_variant,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-shape variants and validate ORT loadability."
    )
    parser.add_argument("model", type=Path, help="Path to the base ONNX model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo-output/fixed-shape-variants"),
        help="Directory where generated variants and summary JSON are written",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider used for load validation",
    )
    parser.add_argument("--input-bucket", type=int, default=64)
    parser.add_argument("--output-length", type=int, default=120000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in EXPERIMENTAL_VARIANTS:
        variant = _resolve_variant(spec, args.input_bucket, args.output_length)
        model_path = args.model
        if variant.name != "base":
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
                encoder_static_batch_annotations=(
                    variant.encoder_static_batch_annotations
                ),
                graph_static_batch_annotations=(
                    variant.graph_static_batch_annotations
                ),
                encoder_core_lstm_states=variant.encoder_core_lstm_states,
                decoder_entry_annotations=variant.decoder_entry_annotations,
                decoder_entry_value_annotations=(
                    variant.decoder_entry_value_annotations
                ),
                decoder_generator_prestft_annotations=(
                    variant.decoder_generator_prestft_annotations
                ),
                decoder_generator_istft_annotations=(
                    variant.decoder_generator_istft_annotations
                ),
                decoder_output_annotations=variant.decoder_output_annotations,
                text_encoder_lstm_reshapes=variant.text_encoder_lstm_reshapes,
            )

        record = _load_record(variant.name, model_path, args.provider)
        records.append(record)
        status = record["status"]
        message = record.get("error", "")
        print(f"{variant.name:64s} {status:4s} {message}")

    summary_path = args.output_dir / f"{args.model.stem}.ort-load-summary.json"
    summary_path.write_text(json.dumps(records, indent=2, sort_keys=True))
    print(f"\nsummary_json={summary_path}")
    return 0


def _load_record(name: str, model_path: Path, provider: str) -> dict[str, object]:
    try:
        session = ort.InferenceSession(str(model_path), providers=[provider])
    except Exception as exc:
        return {
            "name": name,
            "model_path": str(model_path),
            "status": "fail",
            "error": str(exc).splitlines()[0],
        }

    return {
        "name": name,
        "model_path": str(model_path),
        "status": "ok",
        "inputs": [
            {"name": item.name, "shape": list(item.shape), "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": list(item.shape), "type": item.type}
            for item in session.get_outputs()
        ],
    }


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
        encoder_static_batch_annotations=spec.encoder_static_batch_annotations,
        graph_static_batch_annotations=spec.graph_static_batch_annotations,
        encoder_core_lstm_states=spec.encoder_core_lstm_states,
        decoder_entry_annotations=spec.decoder_entry_annotations,
        decoder_entry_value_annotations=spec.decoder_entry_value_annotations,
        decoder_generator_prestft_annotations=(
            spec.decoder_generator_prestft_annotations
        ),
        decoder_generator_istft_annotations=(
            spec.decoder_generator_istft_annotations
        ),
        decoder_output_annotations=spec.decoder_output_annotations,
        text_encoder_lstm_reshapes=spec.text_encoder_lstm_reshapes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
