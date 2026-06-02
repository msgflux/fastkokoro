from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import numpy as np
from kokoro_onnx import Kokoro

from fastkokoro.assets import resolve_model_path, resolve_voices_path
from fastkokoro.audio import AudioFormat, encode_audio
from fastkokoro.config import Settings
from fastkokoro.onnx import create_session
from fastkokoro.streaming import split_pcm_frames, split_phrases, split_sentences
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
            "default_voice=%s default_lang=%s warmup=%s stream_strategy=%s "
            "stream_audio_frame_ms=%s",
            self.settings.model_repo,
            self.settings.model_file,
            self.model_path,
            self.voices_path,
            self.session.get_providers(),
            self.settings.default_voice,
            self.settings.default_lang,
            self.settings.warmup,
            self.settings.stream_strategy,
            self.settings.stream_audio_frame_ms,
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

        return self._create_resolved(
            text,
            voice=resolved_voice,
            speed=speed,
            response_format=response_format,
            lang=resolved_lang,
        )

    def _create_resolved(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        response_format: AudioFormat,
        lang: str,
    ) -> bytes:
        samples, sample_rate = self.kokoro.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang,
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

        if self.settings.stream_strategy == "kokoro":
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
            return

        if self.settings.stream_strategy == "phrase":
            segments = split_phrases(text)
        else:
            segments = split_sentences(text)

        for segment in segments:
            audio = self._create_resolved(
                segment,
                voice=resolved_voice,
                speed=speed,
                response_format=response_format,
                lang=resolved_lang,
            )
            if response_format != "pcm":
                yield audio
                continue

            for frame in split_pcm_frames(
                audio,
                self.settings.stream_audio_frame_ms,
            ):
                yield frame
