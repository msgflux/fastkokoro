from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fastkokoro.config import Settings
from fastkokoro.onnx import create_session


def _settings(**overrides):
    values = dict(
        model_repo="repo",
        model_file="model.onnx",
        voices_file="voices.bin",
        voices_index_file="voices.txt",
        model_path=None,
        voices_path=None,
        cache_dir=Path("/tmp/cache"),
        default_voice="af_heart",
        default_lang="en-us",
        host="0.0.0.0",
        port=8880,
        onnx_providers=("CPUExecutionProvider",),
        onnx_auto_providers=False,
        onnx_intra_op_num_threads=None,
        onnx_inter_op_num_threads=None,
        warmup=False,
        warmup_text="hello",
        stream_strategy="sentence",
        stream_audio_frame_ms=200,
    )
    values.update(overrides)
    return Settings(**values)


def test_create_session_uses_configured_providers():
    session = Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["AzureExecutionProvider", "CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.InferenceSession", return_value=session) as init,
    ):
        result = create_session(Path("model.onnx"), _settings())

    assert result is session
    assert init.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert session.get_providers.called


def test_create_session_auto_uses_all_available_providers():
    session = Mock()
    session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.InferenceSession", return_value=session) as init,
    ):
        create_session(Path("model.onnx"), _settings(onnx_auto_providers=True))

    assert init.call_args.kwargs["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_create_session_rejects_missing_provider():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        pytest.raises(ValueError, match="CUDAExecutionProvider"),
    ):
        create_session(
            Path("model.onnx"),
            _settings(onnx_providers=("CUDAExecutionProvider",)),
        )


def test_create_session_applies_thread_options():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.InferenceSession") as init,
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_intra_op_num_threads=4,
                onnx_inter_op_num_threads=2,
            ),
        )

    session_options = init.call_args.kwargs["sess_options"]
    assert session_options.intra_op_num_threads == 4
    assert session_options.inter_op_num_threads == 2
