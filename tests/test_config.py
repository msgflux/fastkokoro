from fastkokoro.config import (
    DEFAULT_JIT,
    DEFAULT_ONNX_ADAIN_FUSION,
    DEFAULT_ONNX_CONV_ADAIN_FUSION,
    DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
    DEFAULT_ONNX_INTRA_OP_NUM_THREADS,
    DEFAULT_ONNX_IO_BINDING,
    DEFAULT_ONNX_IO_BINDING_DEVICE,
    DEFAULT_ONNX_TTFC_SHAPE_BUCKETS,
    DEFAULT_WARMUP_MULTI_SHAPE,
    Settings,
)


def test_settings_parses_onnx_providers(monkeypatch):
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_PROVIDERS",
        "CUDAExecutionProvider, CPUExecutionProvider",
    )
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_PROVIDER_OPTIONS",
        '{"CUDAExecutionProvider":{"device_id":"0"}}',
    )
    monkeypatch.setenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", "4")
    monkeypatch.setenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", "2")
    monkeypatch.setenv("FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL", "extended")
    monkeypatch.setenv("FASTKOKORO_ONNX_IO_BINDING", "false")
    monkeypatch.setenv("FASTKOKORO_ONNX_IO_BINDING_DEVICE", "cuda")
    monkeypatch.setenv("FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS", "8")
    monkeypatch.setenv("FASTKOKORO_ONNX_WEIGHT_ONLY_BLOCK_SIZE", "64")
    monkeypatch.setenv("FASTKOKORO_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL", "2")
    monkeypatch.setenv("FASTKOKORO_ONNX_WEIGHT_ONLY_SYMMETRIC", "false")
    monkeypatch.setenv("FASTKOKORO_ONNX_ADAIN_FUSION", "true")
    monkeypatch.setenv("FASTKOKORO_ONNX_ADAIN_MODEL_PATH", "/tmp/adain.onnx")
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY",
        "/tmp/libfastkokoro_adain.so",
    )
    monkeypatch.setenv("FASTKOKORO_ONNX_CONV_ADAIN_FUSION", "true")
    monkeypatch.setenv("FASTKOKORO_ONNX_CONV_ADAIN_MODEL_PATH", "/tmp/conv_adain.onnx")
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY",
        "/tmp/libfastkokoro_conv_adain.so",
    )
    monkeypatch.setenv("FASTKOKORO_ONNX_LOG_SEVERITY_LEVEL", "2")
    monkeypatch.setenv("FASTKOKORO_STREAM_MAX_SEGMENT_CHARS", "40")
    monkeypatch.setenv("FASTKOKORO_STREAM_MAX_SEGMENT_WORDS", "6")
    monkeypatch.setenv("FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_CHARS", "120")
    monkeypatch.setenv("FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_WORDS", "10")
    monkeypatch.setenv("FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS", "56")
    monkeypatch.setenv("FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS", "4")
    monkeypatch.setenv("FASTKOKORO_WARMUP_MULTI_SHAPE", "true")
    monkeypatch.setenv("FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS", "8,6,8,16")
    monkeypatch.setenv("FASTKOKORO_JIT", "false")

    settings = Settings.from_env()

    assert settings.onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert settings.onnx_provider_options == {
        "CUDAExecutionProvider": {"device_id": "0"}
    }
    assert settings.onnx_intra_op_num_threads == 4
    assert settings.onnx_inter_op_num_threads == 2
    assert settings.onnx_graph_optimization_level == "extended"
    assert settings.onnx_io_binding is False
    assert settings.onnx_io_binding_device == "cuda"
    assert settings.onnx_weight_only_nbits == 8
    assert settings.onnx_weight_only_block_size == 64
    assert settings.onnx_weight_only_accuracy_level == 2
    assert settings.onnx_weight_only_symmetric is False
    assert settings.onnx_adain_fusion is True
    assert str(settings.onnx_adain_model_path) == "/tmp/adain.onnx"
    assert str(settings.onnx_adain_custom_op_library) == "/tmp/libfastkokoro_adain.so"
    assert settings.onnx_conv_adain_fusion is True
    assert str(settings.onnx_conv_adain_model_path) == "/tmp/conv_adain.onnx"
    assert (
        str(settings.onnx_conv_adain_custom_op_library)
        == "/tmp/libfastkokoro_conv_adain.so"
    )
    assert settings.onnx_log_severity_level == 2
    assert settings.stream_max_segment_chars == 40
    assert settings.stream_max_segment_words == 6
    assert settings.stream_schedule_max_segment_chars == 120
    assert settings.stream_schedule_max_segment_words == 10
    assert settings.stream_cpu_schedule_max_segment_chars == 56
    assert settings.stream_cpu_schedule_max_segment_words == 4
    assert settings.warmup_multi_shape is True
    assert settings.onnx_ttfc_shape_buckets == (6, 8, 16)
    assert settings.jit is False


def test_settings_defaults_to_cpu_provider(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_FUSION", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_MODEL_PATH", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_CONV_ADAIN_FUSION", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_CONV_ADAIN_MODEL_PATH", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDER_OPTIONS", raising=False)
    monkeypatch.delenv("FASTKOKORO_WARMUP_MULTI_SHAPE", raising=False)
    monkeypatch.delenv("FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS", raising=False)
    monkeypatch.delenv("FASTKOKORO_CORS_ALLOW_ORIGINS", raising=False)

    settings = Settings.from_env()

    assert settings.onnx_providers == ("CPUExecutionProvider",)
    assert settings.onnx_provider_options == {}
    assert settings.onnx_auto_providers is True
    assert settings.onnx_intra_op_num_threads == DEFAULT_ONNX_INTRA_OP_NUM_THREADS
    assert settings.onnx_inter_op_num_threads == 1
    assert settings.onnx_graph_optimization_level == (
        DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL
    )
    assert settings.onnx_log_severity_level == 3
    assert settings.onnx_io_binding == DEFAULT_ONNX_IO_BINDING
    assert settings.onnx_io_binding_device == DEFAULT_ONNX_IO_BINDING_DEVICE
    assert settings.onnx_weight_only_nbits is None
    assert settings.onnx_adain_fusion == DEFAULT_ONNX_ADAIN_FUSION
    assert settings.onnx_adain_model_path is None
    assert settings.onnx_adain_custom_op_library is None
    assert settings.onnx_conv_adain_fusion == DEFAULT_ONNX_CONV_ADAIN_FUSION
    assert settings.onnx_conv_adain_model_path is None
    assert settings.onnx_conv_adain_custom_op_library is None
    assert settings.warmup_multi_shape == DEFAULT_WARMUP_MULTI_SHAPE
    assert settings.onnx_ttfc_shape_buckets == DEFAULT_ONNX_TTFC_SHAPE_BUCKETS
    assert settings.jit == DEFAULT_JIT
    assert settings.stream_strategy == "adaptive"
    assert settings.stream_adaptive_max_chars == 50
    assert settings.stream_adaptive_cpu_max_chars == 12
    assert settings.stream_max_segment_chars == 32
    assert settings.stream_max_segment_words == 2
    assert settings.stream_schedule_max_segment_chars == 96
    assert settings.stream_schedule_max_segment_words == 12
    assert settings.stream_cpu_schedule_max_segment_chars == 48
    assert settings.stream_cpu_schedule_max_segment_words == 4
    assert settings.cors_allow_origins == ("*",)


def test_settings_allows_ort_default_thread_options(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", "")
    monkeypatch.setenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", "")

    settings = Settings.from_env()

    assert settings.onnx_intra_op_num_threads is None
    assert settings.onnx_inter_op_num_threads is None


def test_settings_parses_auto_providers(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_AUTO_PROVIDERS", "true")

    settings = Settings.from_env()

    assert settings.onnx_auto_providers is True


def test_settings_rejects_invalid_provider_options(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_PROVIDER_OPTIONS", "[]")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_ONNX_PROVIDER_OPTIONS" in str(exc)
    else:
        raise AssertionError("expected invalid provider options to fail")


def test_settings_parses_stream_options(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_STREAM_STRATEGY", "chunk")
    monkeypatch.setenv("FASTKOKORO_STREAM_AUDIO_FRAME_MS", "80")

    settings = Settings.from_env()

    assert settings.stream_strategy == "chunk"
    assert settings.stream_audio_frame_ms == 80


def test_settings_rejects_invalid_stream_strategy(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_STREAM_STRATEGY", "invalid")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_STREAM_STRATEGY" in str(exc)
    else:
        raise AssertionError("expected invalid stream strategy to fail")


def test_settings_rejects_invalid_graph_optimization_level(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL", "invalid")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL" in str(exc)
    else:
        raise AssertionError("expected invalid graph optimization level to fail")


def test_settings_rejects_invalid_iobinding_device(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_IO_BINDING_DEVICE", "invalid")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_ONNX_IO_BINDING_DEVICE" in str(exc)
    else:
        raise AssertionError("expected invalid IOBinding device to fail")


def test_settings_rejects_invalid_weight_only_nbits(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS", "3")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS" in str(exc)
    else:
        raise AssertionError("expected invalid weight-only nbits to fail")


def test_settings_rejects_invalid_ttfc_shape_buckets(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS", "8,0,16")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS" in str(exc)
    else:
        raise AssertionError("expected invalid ttfc shape buckets to fail")


def test_settings_parses_cors(monkeypatch):
    monkeypatch.setenv(
        "FASTKOKORO_CORS_ALLOW_ORIGINS",
        "http://localhost:3000, https://example.com",
    )
    monkeypatch.setenv("FASTKOKORO_CORS_ALLOW_METHODS", "GET, POST")
    monkeypatch.setenv("FASTKOKORO_CORS_ALLOW_HEADERS", "Authorization, Content-Type")
    monkeypatch.setenv("FASTKOKORO_CORS_ALLOW_CREDENTIALS", "true")

    settings = Settings.from_env()

    assert settings.cors_allow_origins == (
        "http://localhost:3000",
        "https://example.com",
    )
    assert settings.cors_allow_methods == ("GET", "POST")
    assert settings.cors_allow_headers == ("Authorization", "Content-Type")
    assert settings.cors_allow_credentials is True
