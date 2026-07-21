from fastkokoro.kokoro import DEFAULT_VOCAB, Tokenizer


def test_phonemize_uses_misaki_for_japanese(monkeypatch):
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB
    tokenizer._ja_g2p = lambda text: ("koɲɲiʨiβa.", None)

    def reject_espeak(*args, **kwargs):
        raise AssertionError("Japanese must not use the eSpeak backend")

    monkeypatch.setattr("fastkokoro.kokoro.phonemizer.phonemize", reject_espeak)

    assert tokenizer.phonemize("こんにちは。", "ja") == "koɲɲiʨiβa."


def test_phonemize_uses_misaki_for_chinese(monkeypatch):
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB
    tokenizer._zh_g2p = lambda text: ("ni↓ xau↓.", None)

    def reject_espeak(*args, **kwargs):
        raise AssertionError("Chinese must not use the eSpeak backend")

    monkeypatch.setattr("fastkokoro.kokoro.phonemizer.phonemize", reject_espeak)

    assert tokenizer.phonemize("你好。", "zh") == "ni↓ xau↓."
