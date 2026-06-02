from fastkokoro.streaming import split_pcm_frames, split_sentences


def test_split_sentences_keeps_sentence_punctuation():
    assert split_sentences("Ola. Tudo bem? Sim!  ") == [
        "Ola.",
        "Tudo bem?",
        "Sim!",
    ]


def test_split_sentences_returns_whole_text_without_punctuation():
    assert split_sentences("texto sem pontuacao") == ["texto sem pontuacao"]


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
