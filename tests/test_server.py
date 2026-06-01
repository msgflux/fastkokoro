from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from fastkokoro.server import create_app


class FakeEngine:
    def voices(self) -> list[str]:
        return ["af_heart", "pf_dora"]

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
    assert {"kokoro", "tts-1", "gpt-4o-mini-tts"} <= model_ids


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
