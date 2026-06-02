import pytest

from fastkokoro.voices import normalize_language, validate_voice_language


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("a", "en-us"),
        ("en-us", "en-us"),
        ("b", "en-gb"),
        ("en-gb", "en-gb"),
        ("j", "ja"),
        ("ja-jp", "ja"),
        ("z", "zh"),
        ("zh-cn", "zh"),
        ("e", "es"),
        ("es", "es"),
        ("f", "fr-fr"),
        ("fr-fr", "fr-fr"),
        ("h", "hi"),
        ("hi", "hi"),
        ("i", "it"),
        ("it", "it"),
        ("p", "pt-br"),
        ("pt-br", "pt-br"),
    ],
)
def test_normalize_language_aliases(alias: str, expected: str):
    assert normalize_language(alias, None, "a") == expected


def test_normalize_language_from_voice():
    assert normalize_language(None, "pf_dora", "a") == "pt-br"


def test_normalize_default_english_runtime_language():
    assert normalize_language(None, None, "en-us") == "en-us"


def test_validate_voice_language_rejects_mismatch():
    with pytest.raises(ValueError, match="belongs to language"):
        validate_voice_language("pf_dora", "en-us", {"pf_dora"})
