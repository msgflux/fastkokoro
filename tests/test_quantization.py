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
        onnx_auto_providers=False,
        onnx_intra_op_num_threads=None,
        onnx_inter_op_num_threads=None,
        onnx_graph_optimization_level="all",
        onnx_io_binding=False,
        onnx_io_binding_device="auto",
        onnx_weight_only_nbits=None,
        onnx_weight_only_block_size=128,
        onnx_weight_only_accuracy_level=4,
        onnx_weight_only_symmetric=True,
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
