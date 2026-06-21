from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import onnx

from fastkokoro.config import Settings

logger = logging.getLogger("uvicorn.error")

GPU_PROVIDER_PREFIXES = (
    "CUDA",
    "Tensorrt",
    "ROCM",
    "MIGraphX",
    "Dml",
    "OpenVINO",
    "CoreML",
)


def resolve_cpu_simplified_model_path(
    model_path: Path,
    settings: Settings,
    providers: list[str],
) -> Path:
    if not _should_simplify_for_cpu(providers):
        return model_path
    if not model_path.exists():
        return model_path

    cache_path = _simplified_cache_path(model_path, settings)
    if cache_path.exists():
        logger.info("Using cached ONNX-simplified CPU model: %s", cache_path)
        return cache_path

    try:
        simplified_path = simplify_onnx_model(model_path, cache_path)
    except Exception:
        logger.exception(
            "Failed to simplify ONNX model for CPU; using original model: %s",
            model_path,
        )
        return model_path
    return simplified_path


def simplify_onnx_model(model_path: Path, output_path: Path) -> Path:
    from onnxsim import simplify

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Simplifying ONNX model for CPU: input=%s output=%s",
        model_path,
        output_path,
    )
    model, check = simplify(str(model_path), check_n=0)
    if not check:
        raise RuntimeError(f"onnxsim validation failed for {model_path}")
    onnx.save(model, output_path)
    logger.info("ONNX-simplified CPU model written: %s", output_path)
    return output_path


def _should_simplify_for_cpu(providers: list[str]) -> bool:
    if "CPUExecutionProvider" not in providers:
        return False
    return not any(
        provider.startswith(GPU_PROVIDER_PREFIXES) for provider in providers
    )


def _simplified_cache_path(model_path: Path, settings: Settings) -> Path:
    stat = model_path.stat()
    key = f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return settings.cache_dir / "onnx" / f"{model_path.stem}.sim.{digest}.onnx"
