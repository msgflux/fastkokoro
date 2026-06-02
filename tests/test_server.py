from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

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


def test_health():
    client = TestClient(create_app(FakeEngine()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


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
    assert response.headers["content-type"].startswith("audio/pcm")


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
