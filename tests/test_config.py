from fastkokoro.config import Settings


def test_settings_parses_onnx_providers(monkeypatch):
    monkeypatch.setenv(
        "FASTKOKORO_ONNX_PROVIDERS",
        "CUDAExecutionProvider, CPUExecutionProvider",
    )
    monkeypatch.setenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS", "4")
    monkeypatch.setenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS", "2")

    settings = Settings.from_env()

    assert settings.onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert settings.onnx_intra_op_num_threads == 4
    assert settings.onnx_inter_op_num_threads == 2


def test_settings_defaults_to_cpu_provider(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDERS", raising=False)

    settings = Settings.from_env()

    assert settings.onnx_providers == ("CPUExecutionProvider",)
    assert settings.onnx_auto_providers is False


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
