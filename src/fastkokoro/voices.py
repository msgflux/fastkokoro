from __future__ import annotations

from dataclasses import dataclass

KOKORO_MODEL_ID = "kokoro"
OPENAI_MODEL_ALIASES = frozenset({"tts-1", "gpt-4o-mini-tts"})
SUPPORTED_MODEL_IDS = frozenset({KOKORO_MODEL_ID, *OPENAI_MODEL_ALIASES})


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    aliases: tuple[str, ...]
    voices: tuple[str, ...]


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        code="a",
        aliases=("a", "en-us", "en_us", "american", "american-english"),
        voices=(
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
        ),
    ),
    LanguageSpec(
        code="b",
        aliases=("b", "en-gb", "en_gb", "british", "british-english"),
        voices=(
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
        ),
    ),
    LanguageSpec(
        code="j",
        aliases=("j", "ja", "ja-jp", "ja_jp", "japanese"),
        voices=("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
    ),
    LanguageSpec(
        code="z",
        aliases=("z", "zh", "zh-cn", "zh_cn", "mandarin", "mandarin-chinese"),
        voices=(
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ),
    ),
    LanguageSpec(
        code="e",
        aliases=("e", "es", "es-es", "es_es", "spanish"),
        voices=("ef_dora", "em_alex", "em_santa"),
    ),
    LanguageSpec(
        code="f",
        aliases=("f", "fr", "fr-fr", "fr_fr", "french"),
        voices=("ff_siwis",),
    ),
    LanguageSpec(
        code="h",
        aliases=("h", "hi", "hi-in", "hi_in", "hindi"),
        voices=("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
    ),
    LanguageSpec(
        code="i",
        aliases=("i", "it", "it-it", "it_it", "italian"),
        voices=("if_sara", "im_nicola"),
    ),
    LanguageSpec(
        code="pt-br",
        aliases=("p", "pt", "pt-br", "pt_br", "brazilian-portuguese", "portuguese"),
        voices=("pf_dora", "pm_alex", "pm_santa"),
    ),
)

LANGUAGE_BY_ALIAS = {
    alias: language for language in LANGUAGES for alias in language.aliases
}
VOICE_TO_LANGUAGE = {
    voice: language for language in LANGUAGES for voice in language.voices
}
SUPPORTED_LANGUAGE_ALIASES = frozenset(LANGUAGE_BY_ALIAS)


def normalize_language(lang: str | None, voice: str | None, default_lang: str) -> str:
    raw_lang = lang or ""
    key = raw_lang.strip().lower().replace("_", "-")
    if key:
        language = LANGUAGE_BY_ALIAS.get(key)
        if language is None:
            raise ValueError(
                "Unsupported language. Supported values: "
                + ", ".join(sorted(SUPPORTED_LANGUAGE_ALIASES))
            )
        return language.code

    if voice and voice in VOICE_TO_LANGUAGE:
        return VOICE_TO_LANGUAGE[voice].code

    default_key = default_lang.strip().lower().replace("_", "-")
    language = LANGUAGE_BY_ALIAS.get(default_key)
    if language is None:
        raise ValueError(f"Unsupported default language: {default_lang}")
    return language.code


def validate_voice_language(voice: str, lang: str, available_voices: set[str]) -> None:
    if voice not in available_voices:
        raise ValueError(f"Voice {voice!r} is not available")

    language = VOICE_TO_LANGUAGE.get(voice)
    if language is not None and language.code != lang:
        raise ValueError(
            f"Voice {voice!r} belongs to language {language.code!r}, "
            f"but request language resolved to {lang!r}"
        )
