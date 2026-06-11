from __future__ import annotations

import logging
from pathlib import Path

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

from fastkokoro.config import Settings

logger = logging.getLogger("uvicorn.error")


def _missing_runtime_error() -> RuntimeError:
    return RuntimeError(
        "ONNX Runtime is not installed. Install fastkokoro with one runtime "
        "extra: `pip install fastkokoro[cpu]` for CPU or "
        "`pip install fastkokoro[gpu]` for GPU."
    )


def _require_ort():
    if ort is None:
        raise _missing_runtime_error()
    return ort


def _check_gpu_shadowed() -> None:
    runtime = _require_ort()
    try:
        import importlib.metadata

        importlib.metadata.distribution("onnxruntime-gpu")
    except (importlib.metadata.PackageNotFoundError, ImportError):
        return
    if any(
        p.startswith(("CUDA", "TensorRT", "ROCM"))
        for p in runtime.get_available_providers()
    ):
        return
    logger.warning(
        "onnxruntime-gpu installed but GPU providers not detected. "
        "Run: pip install --upgrade --force-reinstall --no-deps onnxruntime-gpu"
    )


GRAPH_OPTIMIZATION_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def create_session(model_path: Path, settings: Settings):
    runtime = _require_ort()
    _check_gpu_shadowed()
    available = runtime.get_available_providers()
    providers = (
        available if settings.onnx_auto_providers else list(settings.onnx_providers)
    )
    if settings.onnx_adain_fusion:
        _validate_adain_fusion_providers(providers)
    if settings.onnx_conv_adain_fusion:
        _validate_conv_adain_fusion_providers(providers)
    if settings.onnx_adain_fusion and settings.onnx_conv_adain_fusion:
        raise ValueError(
            "FASTKOKORO_ONNX_ADAIN_FUSION and FASTKOKORO_ONNX_CONV_ADAIN_FUSION "
            "cannot be enabled at the same time"
        )
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise ValueError(
            "Requested ONNX Runtime provider(s) are not available: "
            f"{', '.join(missing)}. Available providers: {', '.join(available)}"
        )
    provider_options = [
        settings.onnx_provider_options.get(provider, {}) for provider in providers
    ]

    runtime.set_default_logger_severity(settings.onnx_log_severity_level)
    session_options = runtime.SessionOptions()
    graph_level = GRAPH_OPTIMIZATION_LEVELS[settings.onnx_graph_optimization_level]
    session_options.graph_optimization_level = getattr(
        runtime.GraphOptimizationLevel,
        graph_level,
    )
    session_options.log_severity_level = settings.onnx_log_severity_level
    if settings.onnx_intra_op_num_threads is not None:
        session_options.intra_op_num_threads = settings.onnx_intra_op_num_threads
    if settings.onnx_inter_op_num_threads is not None:
        session_options.inter_op_num_threads = settings.onnx_inter_op_num_threads
    if settings.onnx_adain_fusion:
        assert settings.onnx_adain_custom_op_library is not None
        session_options.register_custom_ops_library(
            str(settings.onnx_adain_custom_op_library)
        )
    if settings.onnx_conv_adain_fusion:
        assert settings.onnx_conv_adain_custom_op_library is not None
        session_options.register_custom_ops_library(
            str(settings.onnx_conv_adain_custom_op_library)
        )

    session = runtime.InferenceSession(
        str(model_path),
        providers=providers,
        provider_options=provider_options,
        sess_options=session_options,
    )
    logger.info(
        "ONNX Runtime session initialized: model=%s requested_providers=%s "
        "provider_options=%s active_providers=%s available_providers=%s "
        "graph_optimization_level=%s",
        model_path,
        providers,
        provider_options,
        session.get_providers(),
        available,
        settings.onnx_graph_optimization_level,
    )
    return session


def _validate_adain_fusion_providers(providers: list[str]) -> None:
    if providers != ["CPUExecutionProvider"]:
        raise ValueError(
            "FASTKOKORO_ONNX_ADAIN_FUSION is currently supported only with "
            "CPUExecutionProvider. Custom AdaIN runs as a CPU custom op and can "
            "cause provider copies or regressions with GPU/OpenVINO providers."
        )


def _validate_conv_adain_fusion_providers(providers: list[str]) -> None:
    if providers != ["CPUExecutionProvider"]:
        raise ValueError(
            "FASTKOKORO_ONNX_CONV_ADAIN_FUSION is currently supported only with "
            "CPUExecutionProvider. Custom ConvAdaIN runs as a CPU custom op and "
            "can cause provider copies or regressions with GPU/OpenVINO providers."
        )
