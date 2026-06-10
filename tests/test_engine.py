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
        onnx_conv_adain_fusion=False,
        onnx_conv_adain_model_path=None,
        onnx_conv_adain_custom_op_library=None,
        warmup_multi_shape=False,
        onnx_ttfc_shape_buckets=(6, 8, 9, 10, 11, 12, 16, 24),
        jit=False,
        warmup=False,
        warmup_text="hello",
        stream_strategy="sentence",
        stream_adaptive_max_chars=50,
        stream_adaptive_cpu_max_chars=12,
        stream_audio_frame_ms=1,
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
    engine._output_buffers = local()
    engine._phonemize_cache = {}
    return engine


def _set_providers(engine, providers):
    engine.session = SimpleNamespace(
        get_providers=lambda: providers,
        get_inputs=lambda: [SimpleNamespace(name="tokens")],
        get_outputs=lambda: [SimpleNamespace(name="audio")],
        run=lambda output_names, inputs: [np.ones(48, dtype=np.float32)],
    )


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


@pytest.mark.asyncio
async def test_chunk_stream_limits_segment_size():
    engine = _engine(
        _settings(
            stream_strategy="chunk",
            stream_audio_frame_ms=1,
            stream_max_segment_chars=80,
            stream_max_segment_words=2,
            stream_schedule_max_segment_chars=80,
            stream_schedule_max_segment_words=2,
            stream_cpu_schedule_max_segment_chars=80,
            stream_cpu_schedule_max_segment_words=2,
        )
    )

    [
        chunk
        async for chunk in engine.create_stream(
            "one two three four five",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == ["one two", "three four", "five"]


@pytest.mark.asyncio
async def test_chunk_stream_uses_cpu_schedule_limits():
    engine = _engine(
        _settings(
            stream_strategy="chunk",
            stream_audio_frame_ms=1,
            stream_max_segment_chars=80,
            stream_max_segment_words=2,
            stream_schedule_max_segment_chars=96,
            stream_schedule_max_segment_words=12,
            stream_cpu_schedule_max_segment_chars=80,
            stream_cpu_schedule_max_segment_words=4,
        )
    )

    [
        chunk
        async for chunk in engine.create_stream(
            (
                "one two three four five six seven eight nine ten "
                "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen"
            ),
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == [
        "one two",
        "three four five six",
        "seven eight nine ten",
        "eleven twelve thirteen fourteen",
        "fifteen sixteen seventeen eighteen",
    ]


@pytest.mark.asyncio
async def test_chunk_stream_uses_gpu_schedule_limits():
    engine = _engine(
        _settings(
            stream_strategy="chunk",
            stream_audio_frame_ms=1,
            stream_max_segment_chars=80,
            stream_max_segment_words=2,
            stream_schedule_max_segment_chars=96,
            stream_schedule_max_segment_words=12,
            stream_cpu_schedule_max_segment_chars=80,
            stream_cpu_schedule_max_segment_words=4,
        )
    )
    _set_providers(engine, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    [
        chunk
        async for chunk in engine.create_stream(
            (
                "one two three four five six seven eight nine ten "
                "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen"
            ),
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == [
        "one two",
        "three four five six",
        "seven eight nine ten eleven twelve thirteen fourteen",
        "fifteen sixteen seventeen eighteen",
    ]


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


def test_create_inserts_pause_silence_without_model_call():
    engine = _engine(_settings())

    audio = engine.create(
        "Hi [pause:0.5s] bye",
        voice="af_heart",
        lang="en-us",
        response_format="pcm",
    )

    assert engine.kokoro.created_texts == ["Hi", "bye"]
    assert len(audio) == (48 + 12000 + 48) * 2


def test_invalid_pause_tag_is_read_as_text():
    engine = _engine(_settings())

    engine.create(
        "Hi [pause=0.5] bye",
        voice="af_heart",
        lang="en-us",
        response_format="pcm",
    )

    assert engine.kokoro.created_texts == ["Hi [pause=0.5] bye"]


@pytest.mark.asyncio
async def test_stream_inserts_pause_silence():
    engine = _engine(_settings(stream_strategy="phrase", stream_audio_frame_ms=1))

    chunks = [
        chunk
        async for chunk in engine.create_stream(
            "Hi [pause:0.001s] bye",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == ["Hi", "bye"]
    assert [len(chunk) for chunk in chunks] == [48, 48, 48, 48, 48]


def test_warm_ttfc_shape_buckets_runs_selected_shapes():
    runs = []
    engine = _engine(
        _settings(
            warmup_multi_shape=True,
            onnx_ttfc_shape_buckets=(6, 8),
        )
    )

    def run(output_names, inputs):
        runs.append(inputs["tokens"].shape[1])
        return [np.ones(48, dtype=np.float32)]

    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        run=run,
    )

    engine._warm_ttfc_shape_buckets()

    assert 6 in runs
    assert 8 in runs
    assert len(runs) >= 2


def test_create_samples_with_buffer_pool_grows_and_merges(monkeypatch):
    engine = _engine(_settings())
    parts = iter(
        [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
            np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        ]
    )

    monkeypatch.setattr(
        "fastkokoro.engine.split_phonemes_for_model",
        lambda _: ["p1", "p2", "p3"],
    )
    engine._run_onnx_audio = lambda *_args, **_kwargs: next(parts)

    samples, sample_rate = engine._create_samples(
        "unused",
        voice=engine._voice_styles["af_heart"],
        speed=1.0,
        lang="en-us",
        is_phonemes=True,
        trim=False,
    )

    np.testing.assert_array_equal(
        samples,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
    )
    assert sample_rate == 24000


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


def test_create_can_bind_cuda_inputs(monkeypatch):
    class FakeBinding:
        def __init__(self):
            self.inputs = {}
            self.output = None

        def bind_ortvalue_input(self, name, ortvalue):
            self.inputs[name] = ortvalue

        def bind_output(self, name, device_type="cpu", device_id=0):
            self.output = (name, device_type, device_id)

        def copy_outputs_to_cpu(self):
            return [np.ones(48, dtype=np.float32)]

    binding = FakeBinding()
    ortvalues = []
    engine = _engine(_settings(onnx_io_binding=True, onnx_io_binding_device="cuda"))

    def make_ortvalue(value, device_type, device_id):
        ortvalue = (value, device_type, device_id)
        ortvalues.append(ortvalue)
        return ortvalue

    monkeypatch.setattr(
        "fastkokoro.engine.ort.OrtValue.ortvalue_from_numpy",
        make_ortvalue,
    )
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        io_binding=lambda: binding,
        run_with_iobinding=lambda value: None,
    )

    engine.create("Hello.", voice="af_heart", lang="en-us", response_format="pcm")

    assert set(binding.inputs) == {"tokens", "style", "speed"}
    assert binding.output == ("audio", "cpu", 0)
    assert {ortvalue[1] for ortvalue in ortvalues} == {"cuda"}


def test_cuda_iobinding_falls_back_to_cpu(monkeypatch):
    class FakeBinding:
        def __init__(self):
            self.cpu_inputs = {}
            self.output_name = None

        def bind_cpu_input(self, name, value):
            self.cpu_inputs[name] = value

        def bind_output(self, name, **kwargs):
            self.output_name = name

        def copy_outputs_to_cpu(self):
            return [np.ones(48, dtype=np.float32)]

    bindings = [FakeBinding(), FakeBinding()]
    calls = []
    engine = _engine(_settings(onnx_io_binding=True, onnx_io_binding_device="cuda"))
    monkeypatch.setattr(
        "fastkokoro.engine.ort.OrtValue.ortvalue_from_numpy",
        lambda value, device_type, device_id: (_ for _ in ()).throw(
            RuntimeError("cuda unavailable")
        ),
    )
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        io_binding=lambda: bindings.pop(0),
        run_with_iobinding=lambda value: calls.append(value),
    )

    engine.create("Hello.", voice="af_heart", lang="en-us", response_format="pcm")

    assert len(calls) == 1
    assert set(calls[0].cpu_inputs) == {"tokens", "style", "speed"}


def test_split_phonemes_for_model_prefers_punctuation_boundaries():
    phonemes = "a" * 500 + ". " + "b" * 20

    batches = split_phonemes_for_model(phonemes)

    assert batches == ["a" * 500 + ".", "b" * 20]


def test_split_phonemes_for_model_uses_mlx_punctuation_priority():
    phonemes = ("a" * 250) + ", " + ("b" * 249) + "? " + ("c" * 20)

    batches = split_phonemes_for_model(phonemes)

    assert batches == [("a" * 250) + ", " + ("b" * 249) + "?", "c" * 20]


def test_split_phonemes_for_model_splits_oversized_unpunctuated_text():
    phonemes = "a" * 700

    batches = split_phonemes_for_model(phonemes)

    assert "".join(batches) == phonemes
    assert all(len(batch) <= 510 for batch in batches)
