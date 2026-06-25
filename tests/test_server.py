from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from fastkokoro.config import Settings
from fastkokoro.server import create_app, iter_warmup_request_texts


class FakeEngine:
    def __init__(self):
        self.settings = settings()
        self.session = Mock()
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        self.warmup_calls = 0
        self.stream_calls = []

    def voices(self) -> list[str]:
        return ["af_heart", "pf_dora"]

    def resolve_request(self, voice: str | None, lang: str | None) -> tuple[str, str]:
        if lang == "bad":
            raise ValueError("Unsupported language")
        if voice == "pf_dora" or lang in {"p", "pt-br"}:
            return voice or "pf_dora", "pt-br"
        return voice or "af_heart", "en-us"

    def create(self, text: str, **kwargs) -> bytes:
        return b"audio"

    def warmup(self) -> None:
        self.warmup_calls += 1

    async def create_stream(self, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
        self.stream_calls.append((text, kwargs))
        yield b"chunk-1"
        yield b"chunk-2"


def settings(**overrides):
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


def test_health():
    client = TestClient(create_app(FakeEngine()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metrics_endpoint_reports_http_requests():
    client = TestClient(create_app(FakeEngine()))

    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["http"]["requests"] >= 1
    assert data["http"]["by_path"]["/health"] == 1
    assert data["speech"]["requests"] == 0
    assert data["runtime"]["active_providers"] == ["CPUExecutionProvider"]


def test_cors_preflight():
    app = create_app(
        FakeEngine(),
        settings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "POST", "OPTIONS"),
            cors_allow_headers=("Authorization", "Content-Type"),
        ),
    )
    client = TestClient(app)

    response = client.options(
        "/v1/audio/speech",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_default_allows_any_origin():
    client = TestClient(create_app(FakeEngine()))

    response = client.options(
        "/v1/audio/speech",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_models():
    client = TestClient(create_app(FakeEngine()))

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}
    assert model_ids == {"kokoro"}


def test_voices():
    client = TestClient(create_app(FakeEngine()))

    response = client.get("/v1/audio/voices")

    assert response.status_code == 200
    assert response.json() == {"voices": ["af_heart", "pf_dora"]}


def test_speech_non_streaming():
    client = TestClient(create_app(FakeEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    assert response.content == b"audio"


def test_speech_defaults_to_pcm_response_format():
    client = TestClient(create_app(FakeEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")

    metrics = client.get("/metrics").json()
    assert metrics["speech"]["requests"] == 1
    assert metrics["speech"]["bytes"] == len(b"audio")
    assert metrics["speech"]["latency_seconds_last"] >= 0


def test_speech_accepts_portuguese_alias():
    client = TestClient(create_app(FakeEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "ola",
            "voice": "pf_dora",
            "response_format": "pcm",
            "lang": "p",
        },
    )

    assert response.status_code == 200
    assert response.content == b"audio"


def test_speech_rejects_unsupported_language():
    client = TestClient(create_app(FakeEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "lang": "bad",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported language"


def test_speech_rejects_unsupported_model():
    client = TestClient(create_app(FakeEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "unknown",
            "input": "hello",
            "voice": "af_heart",
        },
    )

    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


def test_speech_streaming():
    client = TestClient(create_app(FakeEngine()))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "pcm",
            "stream": True,
        },
    ) as response:
        content = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert content == b"chunk-1chunk-2"
    assert response.headers["content-type"].startswith("audio/pcm")

    metrics = client.get("/metrics").json()
    assert metrics["speech"]["requests"] == 1
    assert metrics["speech"]["streaming_requests"] == 1
    assert metrics["speech"]["chunks"] == 2
    assert metrics["speech"]["bytes"] == len(content)
    assert metrics["speech"]["first_chunk_observations"] == 1
    assert metrics["speech"]["first_chunk_latency_seconds_last"] >= 0


def test_speech_profiling_writes_request_artifacts(tmp_path):
    client = TestClient(
        create_app(
            FakeEngine(),
            settings(
                profile=True,
                profile_dir=tmp_path,
                profile_requests=True,
            ),
        )
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    profiles = sorted(tmp_path.glob("*speech-non-streaming*.prof"))
    summaries = sorted(tmp_path.glob("*speech-non-streaming*.txt"))
    assert len(profiles) == 1
    assert len(summaries) == 1
    assert "cumtime" in summaries[0].read_text()


def test_warmup_profiling_writes_startup_artifacts(tmp_path):
    engine = FakeEngine()
    engine.settings = settings(
        warmup=True,
        profile=True,
        profile_dir=tmp_path,
        profile_warmup=True,
    )

    with TestClient(create_app(engine, engine.settings)):
        pass

    assert engine.warmup_calls == 1
    profiles = sorted(tmp_path.glob("*startup-warmup*.prof"))
    summaries = sorted(tmp_path.glob("*startup-warmup*.txt"))
    assert len(profiles) == 1
    assert len(summaries) == 1


def test_startup_warmup_request_consumes_first_stream_chunk():
    engine = FakeEngine()
    engine.settings = settings(warmup_request=True)

    with TestClient(create_app(engine, engine.settings)):
        pass

    assert len(engine.stream_calls) == 1
    text, kwargs = engine.stream_calls[0]
    assert (
        text == "Hello there. This is a warmup request for streaming speech generation."
    )
    assert kwargs["voice"] == "af_heart"
    assert kwargs["response_format"] == "pcm"
    assert kwargs["lang"] == "en-us"


def test_startup_warmup_request_does_not_record_speech_metrics():
    engine = FakeEngine()
    engine.settings = settings(warmup_request=True)

    with TestClient(create_app(engine, engine.settings)) as client:
        metrics = client.get("/metrics").json()

    assert metrics["speech"]["requests"] == 0


def test_iter_warmup_request_texts_returns_configured_text_once():
    texts = list(
        iter_warmup_request_texts(
            settings(
                warmup_text="Custom warmup sentence.",
            )
        )
    )

    assert texts == ["Custom warmup sentence."]


def test_startup_warmup_request_uses_single_configured_text():
    engine = FakeEngine()
    engine.settings = settings(
        warmup_request=True,
        warmup_text="Custom warmup sentence.",
    )

    with TestClient(create_app(engine, engine.settings)):
        pass

    assert len(engine.stream_calls) == 1
    assert engine.stream_calls[0][0] == "Custom warmup sentence."
