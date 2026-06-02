from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from fastkokoro.config import Settings

logger = logging.getLogger("uvicorn.error")

GRAPH_OPTIMIZATION_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def create_session(model_path: Path, settings: Settings) -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = (
        available if settings.onnx_auto_providers else list(settings.onnx_providers)
    )
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise ValueError(
            "Requested ONNX Runtime provider(s) are not available: "
            f"{', '.join(missing)}. Available providers: {', '.join(available)}"
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[
        settings.onnx_graph_optimization_level
    ]
    if settings.onnx_log_severity_level is not None:
        session_options.log_severity_level = settings.onnx_log_severity_level
    if settings.onnx_intra_op_num_threads is not None:
        session_options.intra_op_num_threads = settings.onnx_intra_op_num_threads
    if settings.onnx_inter_op_num_threads is not None:
        session_options.inter_op_num_threads = settings.onnx_inter_op_num_threads

    session_providers = provider_configs(providers, settings)
    session = ort.InferenceSession(
        str(model_path),
        providers=session_providers,
        sess_options=session_options,
    )
    logger.info(
        "ONNX Runtime session initialized: model=%s requested_providers=%s "
        "active_providers=%s available_providers=%s graph_optimization_level=%s",
        model_path,
        session_providers,
        session.get_providers(),
        available,
        settings.onnx_graph_optimization_level,
    )
    return session


def provider_configs(
    providers: list[str],
    settings: Settings,
) -> list[str | tuple[str, dict[str, str]]]:
    if not settings.onnx_cuda_graph:
        return providers

    configured: list[str | tuple[str, dict[str, str]]] = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            configured.append((provider, {"enable_cuda_graph": "1"}))
        else:
            configured.append(provider)
    return configured
