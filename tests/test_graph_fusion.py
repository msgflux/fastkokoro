from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from fastkokoro.config import Settings
from fastkokoro.graph_fusion import (
    _make_portable_atan2_nodes,
    resolve_adain_fused_model_path,
)


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
        onnx_ttfc_model_path=None,
        jit=False,
        warmup=False,
        warmup_text=(
            "Hello there. This is a warmup request for streaming speech generation."
        ),
        warmup_request=False,
        runtime_tail_trim_ms=150,
        runtime_tail_fade_ms=72,
        runtime_part_trim_padding_ms=80,
        profile=False,
        profile_dir=Path("/tmp/cache/profiles"),
        profile_warmup=False,
        profile_requests=False,
        stream_strategy="sentence",
        stream_adaptive_max_chars=50,
        stream_adaptive_cpu_max_chars=12,
        stream_audio_frame_ms=200,
        stream_boundary_silence_ms=0,
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


def test_portable_atan2_polynomial_matches_numpy(tmp_path):
    runtime = pytest.importorskip("onnxruntime")
    shape = [2, 7]
    graph = helper.make_graph(
        _make_portable_atan2_nodes("Atan2Poly", "imag", "real", "phase"),
        "atan2_poly",
        [
            helper.make_tensor_value_info("imag", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info("real", TensorProto.FLOAT16, shape),
        ],
        [helper.make_tensor_value_info("phase", TensorProto.FLOAT16, shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    model_path = tmp_path / "atan2-poly.onnx"
    onnx.save(model, model_path)

    imag = np.array(
        [[0.0, 1.0, -1.0, 1.0, -1.0, 0.25, -4.0]] * 2,
        dtype=np.float16,
    )
    real = np.array(
        [[0.0, 1.0, 1.0, -1.0, -1.0, -2.0, 0.125]] * 2,
        dtype=np.float16,
    )
    session = runtime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    actual = session.run(None, {"imag": imag, "real": real})[0]
    expected = np.arctan2(imag.astype(np.float32), real.astype(np.float32)).astype(
        np.float16
    )

    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0)
