from __future__ import annotations

import uvicorn

from fastkokoro.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "fastkokoro.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
