from fastkokoro.streaming import (
    split_chunks,
    split_pcm_frames,
    split_phrases,
    split_scheduled_chunks,
    split_sentences,
)


def test_split_sentences_keeps_sentence_punctuation():
    assert split_sentences("Ola. Tudo bem? Sim!  ") == [
        "Ola.",
        "Tudo bem?",
        "Sim!",
    ]


def test_split_sentences_returns_whole_text_without_punctuation():
    assert split_sentences("texto sem pontuacao") == ["texto sem pontuacao"]


def test_split_phrases_splits_commas_and_sentence_punctuation():
    assert split_phrases("Ola, tudo bem? Sim.") == ["Ola,", "tudo bem?", "Sim."]


def test_split_chunks_splits_on_punctuation():
    assert split_chunks("Ola, tudo bem? Sim.", max_chars=80, max_words=12) == [
        "Ola,",
        "tudo bem?",
        "Sim.",
    ]


def test_split_chunks_limits_words_without_punctuation():
    assert split_chunks(
        "um dois tres quatro cinco seis",
        max_chars=80,
        max_words=2,
    ) == ["um dois", "tres quatro", "cinco seis"]


def test_split_chunks_limits_chars_without_punctuation():
    assert split_chunks(
        "primeiro segundo terceiro",
        max_chars=15,
        max_words=12,
    ) == ["primeiro", "segundo", "terceiro"]


def test_split_scheduled_chunks_grows_segment_size():
    assert split_scheduled_chunks(
        "um dois tres quatro cinco seis sete oito nove dez",
        initial_max_chars=80,
        initial_max_words=1,
        max_chars=80,
        max_words=4,
    ) == ["um", "dois tres", "quatro cinco seis sete", "oito nove dez"]


def test_split_pcm_frames_uses_even_sample_boundaries():
    audio = bytes(range(20))

    frames = list(split_pcm_frames(audio, frame_ms=1, sample_rate=1000))

    assert frames == [
        audio[0:2],
        audio[2:4],
        audio[4:6],
        audio[6:8],
        audio[8:10],
        audio[10:12],
        audio[12:14],
        audio[14:16],
        audio[16:18],
        audio[18:20],
    ]
