import pytest

from fastkokoro.voices import normalize_language, validate_voice_language


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("a", "a"),
        ("en-us", "a"),
        ("b", "b"),
        ("en-gb", "b"),
        ("j", "j"),
        ("ja-jp", "j"),
        ("z", "z"),
        ("zh-cn", "z"),
        ("e", "e"),
        ("es", "e"),
        ("f", "f"),
        ("fr-fr", "f"),
        ("h", "h"),
        ("hi", "h"),
        ("i", "i"),
        ("it", "i"),
        ("p", "pt-br"),
        ("pt-br", "pt-br"),
    ],
)
def test_normalize_language_aliases(alias: str, expected: str):
    assert normalize_language(alias, None, "a") == expected


def test_normalize_language_from_voice():
    assert normalize_language(None, "pf_dora", "a") == "pt-br"


def test_validate_voice_language_rejects_mismatch():
    with pytest.raises(ValueError, match="belongs to language"):
        validate_voice_language("pf_dora", "a", {"pf_dora"})
