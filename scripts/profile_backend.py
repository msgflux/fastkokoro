#!/usr/bin/env python3
"""Profile backend-level hotspots in fastkokoro (not ONNX Runtime)."""
from __future__ import annotations

import cProfile
import pstats
import time

from fastkokoro.engine import FastKokoro
from fastkokoro.streaming import split_scheduled_chunks, split_sentences, split_phrases

TEXT = "Hello, how are you? This is a test of speech synthesis."

def profile_engine_init():
    e = FastKokoro()
    e.warmup()
    return e

def profile_generate(engine):
    audio = engine.create(TEXT, voice="af_heart", speed=1.0, response_format="pcm")
    return audio

def profile_streaming(engine):
    chunks = list(split_scheduled_chunks(TEXT))
    for seg in chunks:
        audio = engine._create_resolved(seg, voice="af_heart", speed=1.0, response_format="pcm", lang="en-us")

def main():
    print("=== Profiling engine init + warmup ===", flush=True)
    prof = cProfile.Profile()
    prof.enable()
    engine = profile_engine_init()
    prof.disable()
    ps = pstats.Stats(prof).sort_stats("cumtime")
    ps.print_stats(30)

    print("\n=== Profiling single generate ===", flush=True)
    prof = cProfile.Profile()
    prof.enable()
    audio = profile_generate(engine)
    prof.disable()
    ps = pstats.Stats(prof).sort_stats("cumtime")
    ps.print_stats(30)

if __name__ == "__main__":
    main()
