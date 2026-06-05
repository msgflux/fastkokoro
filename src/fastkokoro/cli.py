from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from fastkokoro.build_adain_op import build_adain_custom_op
from fastkokoro.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-custom-op",
        action="store_true",
        help="Build and enable the optional CPU AdaIN custom op before startup.",
    )
    parser.add_argument(
        "--custom-op-output",
        type=Path,
        default=None,
        help="Output path for --build-custom-op.",
    )
    parser.add_argument(
        "--custom-op-cc",
        default=os.getenv("CC", "gcc"),
        help="C compiler for --build-custom-op. Defaults to CC or gcc.",
    )
    parser.add_argument(
        "--custom-op-no-openmp",
        action="store_true",
        help="Build --build-custom-op without OpenMP.",
    )
    parser.add_argument(
        "--warmup-multi-shape",
        action="store_true",
        help="Enable experimental multi-shape warmup for lower first chunk latency.",
    )
    parser.add_argument(
        "--warmup-multi-shape-buckets",
        default=None,
        help="Comma-separated shape buckets to use with --warmup-multi-shape.",
    )
    args = parser.parse_args()

    if args.build_custom_op:
        custom_op_library = build_adain_custom_op(
            output=args.custom_op_output,
            cc=args.custom_op_cc,
            openmp=not args.custom_op_no_openmp,
        )
        os.environ["FASTKOKORO_ONNX_ADAIN_FUSION"] = "true"
        os.environ["FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY"] = str(custom_op_library)
        os.environ.setdefault("FASTKOKORO_ONNX_PROVIDERS", "CPUExecutionProvider")

    if args.warmup_multi_shape or args.warmup_multi_shape_buckets:
        os.environ["FASTKOKORO_WARMUP_MULTI_SHAPE"] = "true"
        if args.warmup_multi_shape_buckets:
            os.environ["FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS"] = (
                args.warmup_multi_shape_buckets
            )

    settings = Settings.from_env()
    uvicorn.run(
        "fastkokoro.server:app",
        host=settings.host,
        port=settings.port,
        loop="auto",
        reload=False,
    )
