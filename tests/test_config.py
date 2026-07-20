from fastkokoro.config import (
    DEFAULT_JIT,
    DEFAULT_ONNX_ADAIN_FUSION,
    DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
    DEFAULT_ONNX_INTRA_OP_NUM_THREADS,
    DEFAULT_ONNX_IO_BINDING,
    DEFAULT_ONNX_IO_BINDING_DEVICE,
    DEFAULT_ONNX_TTFC_MODEL_PATH,
    DEFAULT_PROFILE,
    DEFAULT_RUNTIME_PART_TRIM_PADDING_MS,
    DEFAULT_RUNTIME_TAIL_FADE_MS,
    DEFAULT_RUNTIME_TAIL_TRIM_MS,
    DEFAULT_STREAM_BOUNDARY_SILENCE_MS,
    DEFAULT_WARMUP_TEXT,
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
    monkeypatch.setenv("FASTKOKORO_ONNX_LOG_SEVERITY_LEVEL", "2")
    monkeypatch.setenv("FASTKOKORO_STREAM_MAX_SEGMENT_CHARS", "40")
    monkeypatch.setenv("FASTKOKORO_STREAM_MAX_SEGMENT_WORDS", "6")
    monkeypatch.setenv("FASTKOKORO_STREAM_BOUNDARY_SILENCE_MS", "120")
    monkeypatch.setenv("FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_CHARS", "120")
    monkeypatch.setenv("FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_WORDS", "10")
    monkeypatch.setenv("FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS", "56")
    monkeypatch.setenv("FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS", "4")
    monkeypatch.setenv("FASTKOKORO_ONNX_TTFC_MODEL_PATH", "/tmp/ttfc.onnx")
    monkeypatch.setenv("FASTKOKORO_JIT", "false")
    monkeypatch.setenv("FASTKOKORO_WARMUP_REQUEST", "true")
    monkeypatch.setenv("FASTKOKORO_RUNTIME_TAIL_TRIM_MS", "120")
    monkeypatch.setenv("FASTKOKORO_RUNTIME_TAIL_FADE_MS", "48")
    monkeypatch.setenv("FASTKOKORO_RUNTIME_PART_TRIM_PADDING_MS", "96")
    monkeypatch.setenv("FASTKOKORO_PROFILE", "true")
    monkeypatch.setenv("FASTKOKORO_PROFILE_DIR", "/tmp/profiles")
    monkeypatch.setenv("FASTKOKORO_PROFILE_WARMUP", "false")
    monkeypatch.setenv("FASTKOKORO_PROFILE_REQUESTS", "true")

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
    assert settings.onnx_log_severity_level == 2
    assert settings.stream_max_segment_chars == 40
    assert settings.stream_max_segment_words == 6
    assert settings.stream_boundary_silence_ms == 120
    assert settings.stream_schedule_max_segment_chars == 120
    assert settings.stream_schedule_max_segment_words == 10
    assert settings.stream_cpu_schedule_max_segment_chars == 56
    assert settings.stream_cpu_schedule_max_segment_words == 4
    assert str(settings.onnx_ttfc_model_path) == "/tmp/ttfc.onnx"
    assert settings.jit is False
    assert settings.warmup_request is True
    assert settings.runtime_tail_trim_ms == 120
    assert settings.runtime_tail_fade_ms == 48
    assert settings.runtime_part_trim_padding_ms == 96
    assert settings.profile is True
    assert str(settings.profile_dir) == "/tmp/profiles"
    assert settings.profile_warmup is False
    assert settings.profile_requests is True


def test_settings_defaults_to_cpu_provider(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_FUSION", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_MODEL_PATH", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDER_OPTIONS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_TTFC_MODEL_PATH", raising=False)
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
    assert settings.model_repo == "msgflux/Kokoro-82M-streaming-onnx"
    assert settings.model_file == "onnx/kokoro-82m-streaming-b96-fp16.onnx"
    assert settings.voices_file == "voices.npz"
    assert settings.warmup_text == DEFAULT_WARMUP_TEXT
    assert settings.onnx_ttfc_model_path == DEFAULT_ONNX_TTFC_MODEL_PATH
    assert settings.jit == DEFAULT_JIT
    assert settings.warmup_request is False
    assert settings.runtime_tail_trim_ms == DEFAULT_RUNTIME_TAIL_TRIM_MS
    assert settings.runtime_tail_fade_ms == DEFAULT_RUNTIME_TAIL_FADE_MS
    assert (
        settings.runtime_part_trim_padding_ms
        == DEFAULT_RUNTIME_PART_TRIM_PADDING_MS
    )
    assert settings.profile is DEFAULT_PROFILE
    assert settings.profile_warmup is DEFAULT_PROFILE
    assert settings.profile_requests is DEFAULT_PROFILE
    assert settings.profile_dir == settings.cache_dir / "profiles"
    assert settings.stream_strategy == "sentence"
    assert settings.stream_adaptive_max_chars == 50
    assert settings.stream_adaptive_cpu_max_chars == 12
    assert settings.stream_boundary_silence_ms == DEFAULT_STREAM_BOUNDARY_SILENCE_MS
    assert settings.stream_max_segment_chars is None
    assert settings.stream_max_segment_words is None
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


def test_settings_profile_subflags_default_to_master_switch(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_WARMUP_REQUEST", "true")
    monkeypatch.setenv("FASTKOKORO_PROFILE", "true")

    settings = Settings.from_env()

    assert settings.profile is True
    assert settings.profile_warmup is True
    assert settings.profile_requests is True
