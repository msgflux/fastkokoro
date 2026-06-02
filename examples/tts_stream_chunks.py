# /// script
# dependencies = [
#   "openai>=2.0.0",
# ]
# ///
# ruff: noqa: I001
from __future__ import annotations

import os

from openai import OpenAI


BASE_URL = os.getenv("FASTKOKORO_BASE_URL", "http://localhost:8880/v1")
API_KEY = os.getenv("FASTKOKORO_API_KEY", "fastkokoro")


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice=os.getenv("FASTKOKORO_VOICE", "pf_dora"),
        input=os.getenv("FASTKOKORO_TEXT", "Ola, tudo bem?"),
        response_format="pcm",
    ) as response:
        total_bytes = 0
        for index, chunk in enumerate(response.iter_bytes(), start=1):
            total_bytes += len(chunk)
            print(f"chunk={index} bytes={len(chunk)} total={total_bytes}")


if __name__ == "__main__":
    main()
