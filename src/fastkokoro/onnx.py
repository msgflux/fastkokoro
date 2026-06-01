from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

from fastkokoro.config import Settings


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
    if settings.onnx_intra_op_num_threads is not None:
        session_options.intra_op_num_threads = settings.onnx_intra_op_num_threads
    if settings.onnx_inter_op_num_threads is not None:
        session_options.inter_op_num_threads = settings.onnx_inter_op_num_threads

    return ort.InferenceSession(
        str(model_path),
        providers=providers,
        sess_options=session_options,
    )
