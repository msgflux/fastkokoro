from pathlib import Path
from threading import local
from types import SimpleNamespace

import numpy as np
import pytest

import fastkokoro.engine as engine_module
from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro, OnnxSessionProfile, split_phonemes_for_model


class FakeOrtValue:
    @staticmethod
    def ortvalue_from_numpy(value, device_type, device_id):
        return value, device_type, device_id


@pytest.fixture(autouse=True)
def fake_engine_onnxruntime(monkeypatch):
    monkeypatch.setattr(
        engine_module,
        "ort",
        SimpleNamespace(OrtValue=FakeOrtValue),
    )


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
        stream_audio_frame_ms=1,
        stream_max_segment_chars=None,
        stream_max_segment_words=None,
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
    engine.ttfc_session = None
    engine._ttfc_onnx_profile = None
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
    engine._token_input_width = 512
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name=engine._onnx_output_name,
        token_input_name=engine._token_input_name,
        token_input_width=engine._token_input_width,
        token_input_static=False,
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


def _set_static_token_width(engine, width):
    input_names = frozenset({"input_ids", "style", "speed", "input_lengths"})
    engine._onnx_input_names = input_names
    engine._token_input_name = "input_ids"
    engine._token_input_width = width
    engine._onnx_profile = OnnxSessionProfile(
        input_names=input_names,
        output_name="audio",
        token_input_name=engine._token_input_name,
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )


def test_stream_initial_schedule_limits_scale_with_bucket():
    engine = _engine(_settings())

    cases = [
        (16, (24, 1)),
        (24, (24, 2)),
        (32, (48, 4)),
        (48, (72, 6)),
        (64, (96, 8)),
        (128, (96, 12)),
    ]
    for token_width, expected in cases:
        engine._token_input_width = token_width

        assert engine._stream_initial_schedule_limits(96, 12) == expected


def test_stream_initial_schedule_limits_respect_explicit_settings():
    engine = _engine(
        _settings(
            stream_max_segment_chars=80,
            stream_max_segment_words=3,
        )
    )
    engine._token_input_width = 48

    assert engine._stream_initial_schedule_limits(96, 12) == (80, 3)


def test_stream_schedule_limits_cap_fixed_bucket_word_capacity():
    engine = _engine(
        _settings(
            stream_schedule_max_segment_chars=96,
            stream_schedule_max_segment_words=12,
        )
    )
    _set_providers(engine, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    _set_static_token_width(engine, 24)
    assert engine._stream_schedule_limits() == (48, 3)

    _set_static_token_width(engine, 48)
    assert engine._stream_schedule_limits() == (96, 6)


def test_stream_initial_schedule_limits_cap_explicit_settings_to_bucket_capacity():
    engine = _engine(
        _settings(
            stream_max_segment_chars=120,
            stream_max_segment_words=12,
        )
    )
    _set_static_token_width(engine, 48)

    assert engine._stream_initial_schedule_limits(120, 12) == (96, 6)


def test_runtime_tail_trim_scales_with_bucket():
    engine = _engine(_settings(runtime_tail_trim_ms=150, runtime_tail_fade_ms=72))

    engine._token_input_width = 24
    assert engine._runtime_tail_trim_ms() == 150
    assert engine._runtime_tail_fade_ms() == 72

    engine._token_input_width = 48
    assert engine._runtime_tail_trim_ms() == 220
    assert engine._runtime_tail_fade_ms() == 96


def test_runtime_tail_trim_respects_explicit_settings():
    engine = _engine(_settings(runtime_tail_trim_ms=180, runtime_tail_fade_ms=80))
    engine._token_input_width = 48

    assert engine._runtime_tail_trim_ms() == 180
    assert engine._runtime_tail_fade_ms() == 80


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
async def test_kokoro_stream_strategy_uses_local_engine_path():
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

    assert engine.kokoro.created_texts == ["Hello. World."]
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


@pytest.mark.asyncio
async def test_adaptive_stream_scales_initial_segment_with_gpu_bucket():
    engine = _engine(
        _settings(
            stream_strategy="adaptive",
            stream_audio_frame_ms=1,
            stream_schedule_max_segment_chars=96,
            stream_schedule_max_segment_words=12,
        )
    )
    engine._token_input_width = 48
    _set_providers(engine, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    [
        chunk
        async for chunk in engine.create_stream(
            (
                "one two three four five six seven eight nine ten eleven twelve "
                "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
            ),
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert engine.kokoro.created_texts == [
        "one two three four five six",
        (
            "seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen"
        ),
        "nineteen",
    ]


def test_stream_text_segments_respect_static_onnx_token_width():
    engine = _engine(
        _settings(
            stream_strategy="adaptive",
            stream_adaptive_max_chars=50,
        )
    )
    _set_static_token_width(engine, 8)

    segments = engine._stream_text_control_segments(
        "one two three",
        lang="en-us",
    )

    assert [segment.text for segment in segments] == ["one", "two", "three"]
    assert all(
        engine._text_token_count(segment.text, "en-us") <= 6 for segment in segments
    )


@pytest.mark.asyncio
async def test_stream_never_sends_more_tokens_than_fixed_onnx_width():
    observed_input_lengths = []
    engine = _engine(
        _settings(
            stream_strategy="adaptive",
            stream_adaptive_max_chars=50,
        )
    )
    _set_static_token_width(engine, 8)
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", shape=[1, 8]),
            SimpleNamespace(name="style"),
            SimpleNamespace(name="speed"),
            SimpleNamespace(name="input_lengths"),
        ],
        get_outputs=lambda: [SimpleNamespace(name="audio")],
        run=lambda output_names, inputs: (
            observed_input_lengths.append(int(inputs["input_lengths"][0]))
            or [np.ones(48, dtype=np.float32)]
        ),
    )

    [
        chunk
        async for chunk in engine.create_stream(
            "one two three",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert observed_input_lengths == [5, 5, 7]
    assert all(length <= 8 for length in observed_input_lengths)


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


@pytest.mark.asyncio
async def test_stream_first_audio_segment_uses_ttfc_session():
    engine = _engine(
        _settings(
            stream_strategy="chunk",
            stream_audio_frame_ms=1,
            stream_max_segment_words=1,
            stream_cpu_schedule_max_segment_words=1,
        )
    )
    ttfc_session = SimpleNamespace(name="ttfc")
    sessions = []

    def run_onnx(phonemes, voice, speed, *, session=None):
        sessions.append(session)
        return np.ones(24, dtype=np.float32)

    engine.ttfc_session = ttfc_session
    engine._run_onnx_audio = run_onnx

    chunks = [
        chunk
        async for chunk in engine.create_stream(
            "Hello world",
            voice="af_heart",
            lang="en-us",
            response_format="pcm",
        )
    ]

    assert chunks
    assert sessions == [ttfc_session, None]


def test_build_onnx_inputs_keeps_dynamic_model_at_real_token_length():
    engine = _engine(_settings())

    inputs = engine._build_onnx_inputs(
        [10, 20, 30],
        engine._voice_styles["af_heart"],
        1.0,
    )

    assert inputs["tokens"].shape == (1, 5)
    assert "attention_mask" not in inputs
    np.testing.assert_array_equal(
        inputs["tokens"],
        np.array([[0, 10, 20, 30, 0]], dtype=np.int64),
    )


def test_build_onnx_inputs_pads_fixed_attention_mask_model():
    engine = _engine(_settings())
    engine.session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", shape=[1, 8]),
            SimpleNamespace(name="attention_mask", shape=[1, 8]),
        ],
        get_outputs=lambda: [SimpleNamespace(name="audio")],
        run=lambda output_names, inputs: [np.ones(48, dtype=np.float32)],
    )
    engine._onnx_input_names = frozenset(
        item.name for item in engine.session.get_inputs()
    )
    engine._token_input_name = "input_ids"
    engine._token_input_width = 8
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name="audio",
        token_input_name=engine._token_input_name,
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )

    inputs = engine._build_onnx_inputs(
        [10, 20, 30],
        engine._voice_styles["af_heart"],
        1.0,
    )

    assert inputs["input_ids"].shape == (1, 8)
    assert inputs["attention_mask"].shape == (1, 8)
    np.testing.assert_array_equal(
        inputs["input_ids"],
        np.array([[0, 10, 20, 30, 0, 0, 0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        inputs["attention_mask"],
        np.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int64),
    )


def test_build_onnx_inputs_pads_fixed_model_without_attention_mask():
    engine = _engine(_settings())
    engine._onnx_input_names = frozenset(
        {"input_ids", "style", "speed", "input_lengths"}
    )
    engine._token_input_name = "input_ids"
    engine._token_input_width = 8
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name="audio",
        token_input_name=engine._token_input_name,
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )

    inputs = engine._build_onnx_inputs(
        [10, 20, 30],
        engine._voice_styles["af_heart"],
        1.0,
    )

    assert inputs["input_ids"].shape == (1, 8)
    assert inputs["input_lengths"].shape == (1,)
    assert inputs["speed"].dtype == np.float32
    assert "attention_mask" not in inputs
    np.testing.assert_array_equal(
        inputs["input_ids"],
        np.array([[0, 10, 20, 30, 0, 0, 0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        inputs["input_lengths"],
        np.array([5], dtype=np.int64),
    )


def test_build_onnx_inputs_uses_ttfc_session_profile():
    engine = _engine(_settings())
    ttfc_session = SimpleNamespace(name="ttfc")
    engine.ttfc_session = ttfc_session
    engine._ttfc_onnx_profile = OnnxSessionProfile(
        input_names=frozenset({"input_ids", "style", "speed", "attention_mask"}),
        output_name="audio",
        token_input_name="input_ids",
        token_input_width=8,
        token_input_static=True,
    )

    inputs = engine._build_onnx_inputs(
        [10, 20, 30],
        engine._voice_styles["af_heart"],
        1.0,
        profile=engine._onnx_profile_for_session(ttfc_session),
    )

    assert inputs["input_ids"].shape == (1, 8)
    assert inputs["attention_mask"].shape == (1, 8)
    assert "tokens" not in inputs


def test_split_for_onnx_token_width_preserves_dynamic_model_batches():
    engine = _engine(_settings())

    assert engine._split_for_onnx_token_width("hello world") == ["hello world"]


def test_split_for_onnx_token_width_uses_fixed_mask_width():
    engine = _engine(_settings())
    engine._onnx_input_names = frozenset({"tokens", "attention_mask"})
    engine._token_input_width = 7
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name="audio",
        token_input_name="tokens",
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )

    assert engine._split_for_onnx_token_width("one two three") == [
        "one",
        "two",
        "three",
    ]


def test_split_for_onnx_token_width_uses_fixed_width_without_attention_mask():
    engine = _engine(_settings())
    engine._onnx_input_names = frozenset({"input_ids"})
    engine._token_input_width = 7
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name="audio",
        token_input_name="input_ids",
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )

    assert engine._split_for_onnx_token_width("one two three") == [
        "one",
        "two",
        "three",
    ]


def test_split_for_onnx_token_width_splits_oversized_piece():
    engine = _engine(_settings())
    engine._onnx_input_names = frozenset({"tokens", "attention_mask"})
    engine._token_input_width = 5
    engine._onnx_profile = OnnxSessionProfile(
        input_names=engine._onnx_input_names,
        output_name="audio",
        token_input_name="tokens",
        token_input_width=engine._token_input_width,
        token_input_static=True,
    )

    assert engine._split_for_onnx_token_width("abcdef") == ["abc", "def"]


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
