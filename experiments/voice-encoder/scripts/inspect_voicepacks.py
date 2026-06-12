#!/usr/bin/env python3
"""Inspect Kokoro/FastKokoro voicepack tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voices = np.load(args.voices_path)
    names = sorted(voices.keys())
    print(f"voices={len(names)} path={args.voices_path}")

    for name in names[: args.limit]:
        style = np.asarray(voices[name], dtype=np.float32)
        print(
            f"{name}: shape={style.shape} "
            f"mean={style.mean():.6f} std={style.std():.6f} "
            f"l2={np.linalg.norm(style):.6f}"
        )


if __name__ == "__main__":
    main()
