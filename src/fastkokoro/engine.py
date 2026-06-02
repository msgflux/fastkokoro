from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import numpy as np
from kokoro_onnx import Kokoro

from fastkokoro.assets import resolve_model_path, resolve_voices_path
from fastkokoro.audio import AudioFormat, encode_audio
from fastkokoro.config import Settings
from fastkokoro.onnx import create_session
from fastkokoro.voices import normalize_language, validate_voice_language

logger = logging.getLogger("uvicorn.error")


class FastKokoro:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_path = resolve_model_path(self.settings)
        self.voices_path = resolve_voices_path(self.settings)
        self.session = create_session(self.model_path, self.settings)
        self.kokoro = Kokoro.from_session(self.session, str(self.voices_path))
        logger.info(
            "fastkokoro engine initialized: model_repo=%s model_file=%s "
            "model_path=%s voices_path=%s active_providers=%s "
            "default_voice=%s default_lang=%s warmup=%s",
            self.settings.model_repo,
            self.settings.model_file,
            self.model_path,
            self.voices_path,
            self.session.get_providers(),
            self.settings.default_voice,
            self.settings.default_lang,
            self.settings.warmup,
        )

    def voices(self) -> list[str]:
        return self.kokoro.get_voices()

    def warmup(self) -> None:
        self.create(
            self.settings.warmup_text,
            voice=self.settings.default_voice,
            response_format="pcm",
            lang=self.settings.default_lang,
        )

    def resolve_request(self, voice: str | None, lang: str | None) -> tuple[str, str]:
        resolved_voice = voice or self.settings.default_voice
        resolved_lang = normalize_language(
            lang, resolved_voice, self.settings.default_lang
        )
        validate_voice_language(resolved_voice, resolved_lang, set(self.voices()))
        return resolved_voice, resolved_lang

    def create(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: AudioFormat = "mp3",
        lang: str | None = None,
    ) -> bytes:
        resolved_voice, resolved_lang = self.resolve_request(voice, lang)

        samples, sample_rate = self.kokoro.create(
            text,
            voice=resolved_voice,
            speed=speed,
            lang=resolved_lang,
        )
        return encode_audio(samples, sample_rate, response_format)

    async def create_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: AudioFormat = "pcm",
        lang: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        resolved_voice, resolved_lang = self.resolve_request(voice, lang)

        stream = self.kokoro.create_stream(
            text,
            voice=resolved_voice,
            speed=speed,
            lang=resolved_lang,
        )
        async for samples, sample_rate in stream:
            yield encode_audio(
                samples.astype(np.float32),
                sample_rate,
                response_format,
            )
