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
        onnx_provider_options={},
        onnx_auto_providers=False,
        onnx_intra_op_num_threads=None,
        onnx_inter_op_num_threads=None,
        onnx_graph_optimization_level="all",
        onnx_log_severity_level=3,
        onnx_io_binding=False,
        onnx_io_binding_device="auto",
        onnx_weight_only_nbits=None,
        onnx_weight_only_block_size=128,
        onnx_weight_only_accuracy_level=4,
        onnx_weight_only_symmetric=True,
        onnx_adain_fusion=False,
        onnx_adain_model_path=None,
        onnx_adain_custom_op_library=None,
        onnx_conv_adain_fusion=False,
        onnx_conv_adain_model_path=None,
        onnx_conv_adain_custom_op_library=None,
        warmup_multi_shape=False,
        onnx_ttfc_shape_buckets=(6, 8, 9, 10, 11, 12, 16, 24),
        jit=False,
        warmup=False,
        warmup_text="hello",
        stream_strategy="sentence",
        stream_audio_frame_ms=200,
        stream_max_segment_chars=80,
        stream_max_segment_words=12,
        stream_schedule_max_segment_chars=96,
        stream_schedule_max_segment_words=12,
        stream_cpu_schedule_max_segment_chars=48,
        stream_cpu_schedule_max_segment_words=4,
        cors_allow_origins=(),
        cors_allow_methods=("GET", "POST", "OPTIONS"),
        cors_allow_headers=("*",),
        cors_allow_credentials=False,
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
    assert init.call_args.kwargs["provider_options"] == [{}]
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
    assert init.call_args.kwargs["provider_options"] == [{}, {}]


def test_create_session_applies_provider_options():
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
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                onnx_provider_options={
                    "CUDAExecutionProvider": {"device_id": "0"},
                },
            ),
        )

    assert init.call_args.kwargs["provider_options"] == [
        {"device_id": "0"},
        {},
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


def test_create_session_applies_graph_optimization_level():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.InferenceSession") as init,
    ):
        create_session(
            Path("model.onnx"),
            _settings(onnx_graph_optimization_level="extended"),
        )

    session_options = init.call_args.kwargs["sess_options"]
    assert session_options.graph_optimization_level.name == "ORT_ENABLE_EXTENDED"


def test_create_session_applies_log_severity_level():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.set_default_logger_severity") as set_severity,
        patch("fastkokoro.onnx.ort.InferenceSession") as init,
    ):
        create_session(
            Path("model.onnx"),
            _settings(onnx_log_severity_level=3),
        )

    session_options = init.call_args.kwargs["sess_options"]
    assert session_options.log_severity_level == 3
    set_severity.assert_called_once_with(3)


def test_create_session_registers_adain_custom_op_library():
    class FakeSessionOptions:
        def __init__(self):
            self._registered_custom_ops_library = []
            self.graph_optimization_level = None
            self.log_severity_level = None
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

        def register_custom_ops_library(self, path):
            self._registered_custom_ops_library.append(path)

    custom_op_library = Path("/tmp/libfastkokoro_adain.so")
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.SessionOptions", FakeSessionOptions),
        patch("fastkokoro.onnx.ort.InferenceSession") as init,
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_adain_fusion=True,
                onnx_adain_custom_op_library=custom_op_library,
            ),
        )

    session_options = init.call_args.kwargs["sess_options"]
    assert session_options._registered_custom_ops_library == [str(custom_op_library)]


def test_create_session_rejects_adain_fusion_with_non_cpu_provider():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        pytest.raises(ValueError, match="ADAIN_FUSION"),
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                onnx_adain_fusion=True,
                onnx_adain_custom_op_library=Path("/tmp/libfastkokoro_adain.so"),
            ),
        )


def test_create_session_registers_conv_adain_custom_op_library():
    class FakeSessionOptions:
        def __init__(self):
            self._registered_custom_ops_library = []
            self.graph_optimization_level = None
            self.log_severity_level = None
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

        def register_custom_ops_library(self, path):
            self._registered_custom_ops_library.append(path)

    custom_op_library = Path("/tmp/libfastkokoro_conv_adain.so")
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("fastkokoro.onnx.ort.SessionOptions", FakeSessionOptions),
        patch("fastkokoro.onnx.ort.InferenceSession") as init,
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_conv_adain_fusion=True,
                onnx_conv_adain_custom_op_library=custom_op_library,
            ),
        )

    session_options = init.call_args.kwargs["sess_options"]
    assert session_options._registered_custom_ops_library == [str(custom_op_library)]


def test_create_session_rejects_conv_adain_fusion_with_non_cpu_provider():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        pytest.raises(ValueError, match="CONV_ADAIN_FUSION"),
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                onnx_conv_adain_fusion=True,
                onnx_conv_adain_custom_op_library=Path(
                    "/tmp/libfastkokoro_conv_adain.so"
                ),
            ),
        )


def test_create_session_rejects_adain_and_conv_adain_together():
    with (
        patch(
            "fastkokoro.onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        pytest.raises(ValueError, match="cannot be enabled at the same time"),
    ):
        create_session(
            Path("model.onnx"),
            _settings(
                onnx_adain_fusion=True,
                onnx_adain_custom_op_library=Path("/tmp/libfastkokoro_adain.so"),
                onnx_conv_adain_fusion=True,
                onnx_conv_adain_custom_op_library=Path(
                    "/tmp/libfastkokoro_conv_adain.so"
                ),
            ),
        )
