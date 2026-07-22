import pytest

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


def test_phonemize_accepts_english_custom_pronunciation(monkeypatch):
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB

    outputs = {"Say": "sA", "now": "nW"}
    monkeypatch.setattr(
        tokenizer,
        "_phonemize_espeak",
        lambda text, lang: outputs[text],
    )

    phonemes = tokenizer.phonemize(
        "Say [Kokoro](/kˈOkəɹO/) now",
        "en-us",
    )

    assert phonemes == "sA kˈOkəɹO nW"


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("-1", "wˌɜd"),
        ("-2", "wɜd"),
        ("+1", "ˌɔɹ"),
        ("+2", "ˈɔɹ"),
    ],
)
def test_phonemize_applies_english_stress_controls(
    monkeypatch,
    feature,
    expected,
):
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB

    outputs = {"word": "wˈɜd", "or": "ɔɹ"}
    monkeypatch.setattr(
        tokenizer,
        "_phonemize_espeak",
        lambda text, lang: outputs[text],
    )
    label = "word" if feature.startswith("-") else "or"

    assert tokenizer.phonemize(f"[{label}]({feature})", "en-us") == expected


def test_phonemize_rejects_unsupported_custom_phonemes():
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB

    with pytest.raises(ValueError, match="Unsupported custom phoneme symbols"):
        tokenizer.phonemize("[sound](/kɬ/)", "en-us")


def test_phonemize_does_not_apply_english_controls_to_other_languages(monkeypatch):
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = DEFAULT_VOCAB
    calls = []

    def phonemize(text, lang):
        calls.append((text, lang))
        return "ola"

    monkeypatch.setattr(tokenizer, "_phonemize_espeak", phonemize)

    assert tokenizer.phonemize("[olá](/ola/)", "pt-br") == "ola"
    assert calls == [("[olá](/ola/)", "pt-br")]
