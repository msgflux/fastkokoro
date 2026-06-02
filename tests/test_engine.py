from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro


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
        warmup=False,
        warmup_text="hello",
        stream_strategy="sentence",
        stream_audio_frame_ms=1,
    )
    values.update(overrides)
    return Settings(**values)


class FakeKokoro:
    def __init__(self):
        self.created_texts = []

    def get_voices(self):
        return ["af_heart"]

    def create(self, text, *, voice, speed, lang):
        self.created_texts.append(text)
        return np.ones(48, dtype=np.float32), 24000

    async def create_stream(self, text, *, voice, speed, lang):
        yield np.ones(48, dtype=np.float32), 24000


def _engine(settings):
    engine = object.__new__(FastKokoro)
    engine.settings = settings
    engine.model_path = Path("model.onnx")
    engine.voices_path = Path("voices.bin")
    engine.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    engine.kokoro = FakeKokoro()
    return engine


@pytest.mark.asyncio
async def test_sentence_stream_splits_text_and_pcm_frames():
    engine = _engine(_settings(stream_strategy="sentence", stream_audio_frame_ms=1))

    chunks = [
        chunk
        async for chunk in engine.create_stream(
            "Hello. World.",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == ["Hello.", "World."]
    assert len(chunks) == 4
    assert all(len(chunk) == 48 for chunk in chunks)


@pytest.mark.asyncio
async def test_kokoro_stream_strategy_uses_upstream_stream():
    engine = _engine(_settings(stream_strategy="kokoro"))

    chunks = [
        chunk
        async for chunk in engine.create_stream(
            "Hello. World.",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == []
    assert len(chunks) == 1
