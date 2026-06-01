from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from fastkokoro.audio import AudioFormat


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str = Field(min_length=1)
    voice: str = "af_heart"
    response_format: AudioFormat = "mp3"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    stream: bool | None = None
    lang: str | None = None


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastkokoro"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
