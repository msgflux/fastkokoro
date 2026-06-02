from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi.testclient import TestClient

from fastkokoro.config import Settings
from fastkokoro.server import create_app


class FakeEngine:
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

    async def create_stream(self, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
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
        onnx_auto_providers=False,
        onnx_intra_op_num_threads=None,
        onnx_inter_op_num_threads=None,
        warmup=False,
        warmup_text="hello",
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
