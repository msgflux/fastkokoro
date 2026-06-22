from __future__ import annotations

import argparse

import uvicorn

from fastkokoro.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    settings = Settings.from_env()
    uvicorn.run(
        "fastkokoro.server:app",
        host=settings.host,
        port=settings.port,
        loop="auto",
        reload=False,
    )
