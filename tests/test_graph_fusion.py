from pathlib import Path

import pytest

from fastkokoro.config import Settings
from fastkokoro.graph_fusion import resolve_adain_fused_model_path


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
        cors_allow_origins=("*",),
        cors_allow_methods=("GET", "POST", "OPTIONS"),
        cors_allow_headers=("*",),
        cors_allow_credentials=False,
    )
    values.update(overrides)
    return Settings(**values)


def test_resolve_adain_fused_model_path_returns_original_when_disabled():
    model_path = Path("/tmp/model.onnx")

    assert resolve_adain_fused_model_path(model_path, _settings()) == model_path


def test_resolve_adain_fused_model_path_requires_custom_op_library():
    with pytest.raises(ValueError, match="ADAIN_CUSTOM_OP_LIBRARY"):
        resolve_adain_fused_model_path(
            Path("/tmp/model.onnx"),
            _settings(onnx_adain_fusion=True),
        )


def test_resolve_adain_fused_model_path_uses_explicit_model(tmp_path):
    custom_op_library = tmp_path / "libfastkokoro_adain.so"
    adain_model = tmp_path / "model.adain.onnx"
    custom_op_library.touch()
    adain_model.touch()

    assert (
        resolve_adain_fused_model_path(
            tmp_path / "model.onnx",
            _settings(
                onnx_adain_fusion=True,
                onnx_adain_custom_op_library=custom_op_library,
                onnx_adain_model_path=adain_model,
            ),
        )
        == adain_model
    )
