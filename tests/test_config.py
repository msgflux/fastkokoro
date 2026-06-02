from fastkokoro.config import (
    DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
    DEFAULT_ONNX_INTRA_OP_NUM_THREADS,
    DEFAULT_ONNX_IO_BINDING,
    Settings,
)


def test_settings_parses_onnx_providers(monkeypatch):
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_PROVIDERS",
        "CUDAExecutionProvider, CPUExecutionProvider",
    )
    monkeypatch.setenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", "4")
    monkeypatch.setenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", "2")
    monkeypatch.setenv("FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL", "extended")
    monkeypatch.setenv("FASTKOKORO_ONNX_IO_BINDING", "false")

    settings = Settings.from_env()

    assert settings.onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert settings.onnx_intra_op_num_threads == 4
    assert settings.onnx_inter_op_num_threads == 2
    assert settings.onnx_graph_optimization_level == "extended"
    assert settings.onnx_io_binding is False


def test_settings_defaults_to_cpu_provider(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", raising=False)

    settings = Settings.from_env()

    assert settings.onnx_providers == ("CPUExecutionProvider",)
    assert settings.onnx_auto_providers is False
    assert settings.onnx_intra_op_num_threads == DEFAULT_ONNX_INTRA_OP_NUM_THREADS
    assert settings.onnx_inter_op_num_threads == 1
    assert settings.onnx_graph_optimization_level == (
        DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL
    )
    assert settings.onnx_io_binding == DEFAULT_ONNX_IO_BINDING


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


def test_settings_parses_stream_options(monkeypatch):
    monkeypatch.setenv("FASTKOKORO_STREAM_STRATEGY", "kokoro")
    monkeypatch.setenv("FASTKOKORO_STREAM_AUDIO_FRAME_MS", "80")

    settings = Settings.from_env()

    assert settings.stream_strategy == "kokoro"
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
