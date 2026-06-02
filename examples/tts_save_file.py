# /// script
# dependencies = [
#   "openai>=2.0.0",
# ]
# ///
# ruff: noqa: I001
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


BASE_URL = os.getenv("FASTKOKORO_BASE_URL", "http://localhost:8880/v1")
API_KEY = os.getenv("FASTKOKORO_API_KEY", "fastkokoro")
OUTPUT_PATH = Path(os.getenv("FASTKOKORO_TTS_OUTPUT", "speech.wav"))


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice=os.getenv("FASTKOKORO_VOICE", "pf_dora"),
        input=os.getenv("FASTKOKORO_TEXT", "Ola, tudo bem?"),
        response_format="wav",
    ) as response:
        response.stream_to_file(OUTPUT_PATH)

    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
