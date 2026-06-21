from pathlib import Path
from unittest.mock import patch

from fastkokoro.config import Settings
from fastkokoro.quantization import resolve_quantized_model_path


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
        onnx_ttfc_attention_mask_bucket=None,
        onnx_ttfc_model_path=None,
        onnx_ttfc_warm_session=False,
        onnx_ttfc_warm_texts=("Hello.", "Good morning."),
        onnx_ttfc_warm_token_counts=(1, 2, 3),
        jit=False,
        warmup=False,
        warmup_text=(
            "Hello there. This is a warmup request for streaming speech generation."
        ),
        warmup_request=False,
        runtime_tail_trim_ms=150,
        runtime_tail_fade_ms=72,
        profile=False,
        profile_dir=Path("/tmp/cache/profiles"),
        profile_warmup=False,
        profile_requests=False,
        stream_strategy="sentence",
        stream_adaptive_max_chars=50,
        stream_adaptive_cpu_max_chars=12,
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


def test_resolve_quantized_model_path_returns_original_when_disabled():
    model_path = Path("/models/kokoro.onnx")

    result = resolve_quantized_model_path(model_path, _settings())

    assert result == model_path


def test_resolve_quantized_model_path_uses_cached_model(tmp_path):
    model_path = tmp_path / "kokoro.onnx"
    model_path.touch()
    settings = _settings(cache_dir=tmp_path, onnx_weight_only_nbits=8)
    expected = tmp_path / "quantized" / "kokoro-matmul-nbits8-b128-acc4-sym.onnx"
    expected.parent.mkdir()
    expected.touch()

    with patch("fastkokoro.quantization._quantize_matmul_nbits") as quantize:
        result = resolve_quantized_model_path(model_path, settings)

    assert result == expected
    assert not quantize.called


def test_resolve_quantized_model_path_generates_missing_model(tmp_path):
    model_path = tmp_path / "kokoro.onnx"
    model_path.touch()
    settings = _settings(cache_dir=tmp_path, onnx_weight_only_nbits=4)

    def fake_quantize(_model_path, output_path, _settings):
        output_path.touch()

    with patch(
        "fastkokoro.quantization._quantize_matmul_nbits",
        side_effect=fake_quantize,
    ) as quantize:
        result = resolve_quantized_model_path(model_path, settings)

    assert result.exists()
    assert result.name == "kokoro-matmul-nbits4-b128-acc4-sym.onnx"
    assert quantize.called
