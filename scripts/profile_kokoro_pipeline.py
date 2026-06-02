from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from fastkokoro.engine import FastKokoro

TEXTS = {
    "tiny": "Ola.",
    "short": "Ola, tudo bem?",
    "medium_first": "Ola, tudo bem?",
    "medium": (
        "Ola, tudo bem? Este e um teste de sintese de voz em portugues brasileiro. "
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total de geracao."
    ),
}


@dataclass(frozen=True)
class ProfileRun:
    text_name: str
    text_chars: int
    phonemes: int
    tokens: int
    audio_samples: int
    phonemize_seconds: float
    tokenize_seconds: float
    inference_seconds: float
    total_seconds: float
    active_providers: list[str]


@dataclass(frozen=True)
class ProfileSummary:
    text_name: str
    text_chars: int
    iterations: int
    phonemes: int
    tokens: int
    audio_samples: int
    phonemize_avg_seconds: float
    phonemize_p50_seconds: float
    tokenize_avg_seconds: float
    tokenize_p50_seconds: float
    inference_avg_seconds: float
    inference_p50_seconds: float
    inference_min_seconds: float
    total_avg_seconds: float
    total_p50_seconds: float
    active_providers: list[str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=TEXTS, default="short")
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="p")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()

    engine = FastKokoro()
    if args.warmup:
        engine.warmup()

    text = TEXTS[args.text]
    runs = [
        profile_once(engine, text, args.text, args.voice, args.lang, args.speed)
        for _ in range(args.iterations)
    ]
    if args.json_lines:
        for run in runs:
            print(json.dumps(asdict(run)), flush=True)
    print(json.dumps(asdict(summarize(runs))), flush=True)


def profile_once(
    engine: FastKokoro,
    text: str,
    text_name: str,
    voice_name: str,
    lang: str,
    speed: float,
) -> ProfileRun:
    voice, resolved_lang = engine.resolve_request(voice_name, lang)
    voice_style = engine.kokoro.get_voice_style(voice)

    start = time.perf_counter()
    phonemes = engine.kokoro.tokenizer.phonemize(text, resolved_lang)
    phonemized = time.perf_counter()

    tokens = np.array(engine.kokoro.tokenizer.tokenize(phonemes), dtype=np.int64)
    tokenized = time.perf_counter()

    style = voice_style[len(tokens)]
    input_names = [item.name for item in engine.session.get_inputs()]
    if "input_ids" in input_names:
        inputs = {
            "input_ids": [[0, *tokens, 0]],
            "style": np.array(style, dtype=np.float32),
            "speed": np.array([speed], dtype=np.int32),
        }
    else:
        inputs = {
            "tokens": [[0, *tokens, 0]],
            "style": style,
            "speed": np.ones(1, dtype=np.float32) * speed,
        }
    audio = engine.session.run(None, inputs)[0]
    finished = time.perf_counter()

    return ProfileRun(
        text_name=text_name,
        text_chars=len(text),
        phonemes=len(phonemes),
        tokens=len(tokens),
        audio_samples=len(audio),
        phonemize_seconds=phonemized - start,
        tokenize_seconds=tokenized - phonemized,
        inference_seconds=finished - tokenized,
        total_seconds=finished - start,
        active_providers=engine.session.get_providers(),
    )


def summarize(runs: list[ProfileRun]) -> ProfileSummary:
    first = runs[0]
    phonemize = [run.phonemize_seconds for run in runs]
    tokenize = [run.tokenize_seconds for run in runs]
    inference = [run.inference_seconds for run in runs]
    total = [run.total_seconds for run in runs]
    return ProfileSummary(
        text_name=first.text_name,
        text_chars=first.text_chars,
        iterations=len(runs),
        phonemes=first.phonemes,
        tokens=first.tokens,
        audio_samples=first.audio_samples,
        phonemize_avg_seconds=statistics.fmean(phonemize),
        phonemize_p50_seconds=statistics.median(phonemize),
        tokenize_avg_seconds=statistics.fmean(tokenize),
        tokenize_p50_seconds=statistics.median(tokenize),
        inference_avg_seconds=statistics.fmean(inference),
        inference_p50_seconds=statistics.median(inference),
        inference_min_seconds=min(inference),
        total_avg_seconds=statistics.fmean(total),
        total_p50_seconds=statistics.median(total),
        active_providers=first.active_providers,
    )


if __name__ == "__main__":
    main()
