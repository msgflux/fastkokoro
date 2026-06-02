from pathlib import Path
from threading import local
from types import SimpleNamespace

import numpy as np
import pytest

from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro, split_phonemes_for_model


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
        onnx_graph_optimization_level="all",
        onnx_io_binding=False,
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
        self.tokenizer = FakeTokenizer(self.created_texts)

    def get_voices(self):
        return ["af_heart"]

    def create(self, text, *, voice, speed, lang):
        self.created_texts.append(text)
        return np.ones(48, dtype=np.float32), 24000

    async def create_stream(self, text, *, voice, speed, lang):
        yield np.ones(48, dtype=np.float32), 24000


class FakeTokenizer:
    def __init__(self, created_texts):
        self.created_texts = created_texts

    def phonemize(self, text, lang):
        self.created_texts.append(text)
        return text

    def tokenize(self, phonemes):
        return [ord(char) for char in phonemes]


def _engine(settings):
    engine = object.__new__(FastKokoro)
    engine.settings = settings
    engine.model_path = Path("model.onnx")
    engine.voices_path = Path("voices.bin")
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_inputs=lambda: [SimpleNamespace(name="tokens")],
        get_outputs=lambda: [SimpleNamespace(name="audio")],
        run=lambda output_names, inputs: [np.ones(48, dtype=np.float32)],
    )
    engine.kokoro = FakeKokoro()
    engine._voices = tuple(engine.kokoro.get_voices())
    engine._voice_set = frozenset(engine._voices)
    engine._voice_styles = {
        voice: np.ones((512, 256), dtype=np.float32) for voice in engine._voices
    }
    engine._onnx_input_names = frozenset(
        item.name for item in engine.session.get_inputs()
    )
    engine._onnx_output_name = engine.session.get_outputs()[0].name
    engine._token_input_name = (
        "input_ids" if "input_ids" in engine._onnx_input_names else "tokens"
    )
    engine._onnx_input_buffers = local()
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


@pytest.mark.asyncio
async def test_phrase_stream_splits_text_on_commas():
    engine = _engine(_settings(stream_strategy="phrase", stream_audio_frame_ms=1))

    [
        chunk
        async for chunk in engine.create_stream(
            "Hello, World.",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == ["Hello,", "World."]


def test_create_uses_cached_onnx_input_names():
    calls = []
    engine = _engine(_settings())
    engine._onnx_input_names = frozenset({"input_ids"})
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_inputs=lambda: calls.append("get_inputs"),
        run=lambda output_names, inputs: [np.ones(48, dtype=np.float32)],
    )

    engine.create("Hello.", voice="af_heart", lang="en-us", response_format="pcm")

    assert calls == []


def test_create_reuses_preallocated_token_inputs():
    captured = []
    engine = _engine(_settings())

    def run(output_names, inputs):
        captured.append(inputs)
        return [np.ones(48, dtype=np.float32)]

    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        run=run,
    )

    engine.create("Hi.", voice="af_heart", lang="en-us", response_format="pcm")
    engine.create("Hello.", voice="af_heart", lang="en-us", response_format="pcm")

    first_tokens = captured[0]["tokens"]
    second_tokens = captured[1]["tokens"]
    assert first_tokens.base is second_tokens.base
    assert first_tokens.shape == (1, 5)
    assert second_tokens.shape == (1, 8)


def test_create_can_run_with_iobinding():
    class FakeBinding:
        def __init__(self):
            self.inputs = {}
            self.output_name = None

        def bind_cpu_input(self, name, value):
            self.inputs[name] = value

        def bind_output(self, name):
            self.output_name = name

        def copy_outputs_to_cpu(self):
            return [np.ones(48, dtype=np.float32)]

    binding = FakeBinding()
    calls = []
    engine = _engine(_settings(onnx_io_binding=True))
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        io_binding=lambda: binding,
        run_with_iobinding=lambda value: calls.append(value),
    )

    engine.create("Hello.", voice="af_heart", lang="en-us", response_format="pcm")

    assert calls == [binding]
    assert set(binding.inputs) == {"tokens", "style", "speed"}
    assert binding.output_name == "audio"


def test_split_phonemes_for_model_prefers_punctuation_boundaries():
    phonemes = "a" * 500 + ". " + "b" * 20

    batches = split_phonemes_for_model(phonemes)

    assert batches == ["a" * 500 + ".", "b" * 20]
