#!/usr/bin/env python3
"""Stream FastKokoro speech to raw PCM, optionally playing it live."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TEXT = (
    "FastKokoro streams speech from a Docker GPU server. "
    "The client starts receiving raw PCM chunks before the full text is synthesized."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8880/v1/audio/speech",
        help="FastKokoro speech endpoint.",
    )
    parser.add_argument("--api-key", default="fastkokoro")
    parser.add_argument("--model", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("demo-output/demo-stream.pcm")
    )
    parser.add_argument("--wav-output", type=Path, default=None)
    parser.add_argument(
        "--play",
        action="store_true",
        help="Mirror the stream to ffplay while saving the raw PCM file.",
    )
    return parser.parse_args()


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8").strip()
    if args.text:
        return args.text
    return DEFAULT_TEXT


def open_ffplay(sample_rate: int) -> subprocess.Popen[bytes] | None:
    if not shutil.which("ffplay"):
        raise SystemExit("ffplay not found. Install ffmpeg or omit --play.")
    return subprocess.Popen(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "warning",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-",
        ],
        stdin=subprocess.PIPE,
    )


def convert_to_wav(pcm_path: Path, wav_path: Path, sample_rate: int) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install ffmpeg or omit --wav-output.")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            str(pcm_path),
            str(wav_path),
        ],
        check=True,
    )


def stream_speech(args: argparse.Namespace) -> None:
    text = read_text(args)
    payload = {
        "model": args.model,
        "voice": args.voice,
        "input": text,
        "response_format": "pcm",
        "speed": args.speed,
        "stream": True,
    }
    if args.lang:
        payload["lang"] = args.lang

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    player = open_ffplay(args.sample_rate) if args.play else None
    first_chunk_at: float | None = None
    started_at = time.perf_counter()
    total_bytes = 0
    chunks = 0

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise SystemExit(f"Unexpected HTTP status: {response.status}")
            with args.output.open("wb") as output:
                while chunk := response.read(args.chunk_size):
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    chunks += 1
                    total_bytes += len(chunk)
                    output.write(chunk)
                    output.flush()
                    if player and player.stdin:
                        player.stdin.write(chunk)
                        player.stdin.flush()
                    print(
                        f"chunk={chunks} bytes={len(chunk)} total={total_bytes}",
                        file=sys.stderr,
                    )
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc
    finally:
        if player and player.stdin:
            player.stdin.close()
            player.wait(timeout=10)

    elapsed = time.perf_counter() - started_at
    ttfc = "n/a" if first_chunk_at is None else f"{first_chunk_at - started_at:.4f}s"
    print(
        f"summary chunks={chunks} bytes={total_bytes} ttfc={ttfc} "
        f"elapsed={elapsed:.4f}s",
        file=sys.stderr,
    )

    if args.wav_output:
        convert_to_wav(args.output, args.wav_output, args.sample_rate)
        print(f"wav={args.wav_output}", file=sys.stderr)


def main() -> None:
    stream_speech(parse_args())


if __name__ == "__main__":
    main()
