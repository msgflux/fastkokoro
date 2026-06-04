from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro


def _session(model_path: Path, custom_lib: Path | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(
        os.environ.get("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", "6")
    )
    if custom_lib is not None:
        options.register_custom_ops_library(str(custom_lib))
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _engine(model_path: Path, custom_lib: Path | None) -> FastKokoro:
    settings = Settings.from_env()
    engine = FastKokoro(settings)
    engine.session = _session(model_path, custom_lib)
    engine._onnx_output_name = engine.session.get_outputs()[0].name
    return engine


def _audio(engine: FastKokoro, text: str, voice: str) -> np.ndarray:
    samples, _ = engine._create_samples(
        text,
        voice=engine._voice_styles[voice],
        lang="pt-br",
        speed=1.0,
    )
    return samples.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--custom-lib", type=Path, required=True)
    parser.add_argument(
        "--text", default="Ola, este e um teste curto em portugues brasileiro."
    )
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    base = _engine(args.base, None)
    fused = _engine(args.fused, args.custom_lib)

    base_audio = _audio(base, args.text, args.voice)
    fused_audio = _audio(fused, args.text, args.voice)
    size = min(base_audio.size, fused_audio.size)
    diff = base_audio[:size] - fused_audio[:size]
    print(
        {
            "base_samples": int(base_audio.size),
            "fused_samples": int(fused_audio.size),
            "max_abs": float(np.max(np.abs(diff))) if size else None,
            "mean_abs": float(np.mean(np.abs(diff))) if size else None,
            "rms": float(np.sqrt(np.mean(diff * diff))) if size else None,
        }
    )

    for name, engine in (("base", base), ("fused", fused)):
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            _audio(engine, args.text, args.voice)
            times.append(time.perf_counter() - start)
        print(
            {
                "name": name,
                "p50": float(np.percentile(times, 50)),
                "p90": float(np.percentile(times, 90)),
                "runs": [round(t, 4) for t in times],
            }
        )


if __name__ == "__main__":
    main()
