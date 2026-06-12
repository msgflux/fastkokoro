#!/usr/bin/env python3
"""Generate a synthetic voice-encoder dataset from FastKokoro voicepacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from tqdm import tqdm

from fastkokoro.engine import FastKokoro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/voice-encoder/config.example.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config.get("voices"), list) or not config["voices"]:
        raise ValueError("config must define a non-empty 'voices' list")
    if not isinstance(config.get("texts"), list) or not config["texts"]:
        raise ValueError("config must define a non-empty 'texts' list")
    return config


def token_count(engine: FastKokoro, text: str, lang: str) -> tuple[str, int]:
    phonemes = engine._phonemize_cached(text, lang)
    tokens = engine.kokoro.tokenizer.tokenize(phonemes)
    return phonemes, len(tokens)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    output_dir = Path(config["output_dir"])
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    engine = FastKokoro()
    rows: list[dict[str, Any]] = []
    jobs = []
    for voice in config["voices"]:
        for item in config["texts"]:
            try:
                engine.resolve_request(voice, item["lang"])
            except ValueError:
                continue
            jobs.append((voice, item))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    for index, (voice, item) in enumerate(tqdm(jobs, desc="synthesizing"), start=1):
        text = item["text"]
        lang = item["lang"]
        resolved_voice, resolved_lang = engine.resolve_request(voice, lang)
        phonemes, count = token_count(engine, text, resolved_lang)
        style = engine._voice_styles[resolved_voice]
        style_index = min(count, style.shape[0] - 1)
        samples, sample_rate = engine._create_samples(
            text,
            voice=style,
            speed=1.0,
            lang=resolved_lang,
        )

        audio_path = audio_dir / f"{index:06d}_{resolved_voice}_{resolved_lang}.wav"
        sf.write(audio_path, np.asarray(samples, dtype=np.float32), sample_rate)
        rows.append(
            {
                "audio_path": str(audio_path),
                "voice": resolved_voice,
                "lang": resolved_lang,
                "text": text,
                "phonemes": phonemes,
                "token_count": count,
                "style_index": style_index,
                "style_shape": list(style.shape),
                "sample_rate": sample_rate,
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", rows)
    print(f"wrote {len(rows)} rows to {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
