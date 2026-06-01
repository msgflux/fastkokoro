from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
from kokoro_onnx import Kokoro

from fastkokoro.assets import resolve_model_path, resolve_voices_path
from fastkokoro.audio import AudioFormat, encode_audio
from fastkokoro.config import Settings
from fastkokoro.onnx import create_session


class FastKokoro:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_path = resolve_model_path(self.settings)
        self.voices_path = resolve_voices_path(self.settings)
        self.session = create_session(self.model_path, self.settings)
        self.kokoro = Kokoro.from_session(self.session, str(self.voices_path))

    def voices(self) -> list[str]:
        return self.kokoro.get_voices()

    def create(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: AudioFormat = "mp3",
        lang: str | None = None,
    ) -> bytes:
        samples, sample_rate = self.kokoro.create(
            text,
            voice=voice or self.settings.default_voice,
            speed=speed,
            lang=lang or self.settings.default_lang,
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
        stream = self.kokoro.create_stream(
            text,
            voice=voice or self.settings.default_voice,
            speed=speed,
            lang=lang or self.settings.default_lang,
        )
        async for samples, sample_rate in stream:
            yield encode_audio(
                samples.astype(np.float32),
                sample_rate,
                response_format,
            )
