from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download

from fastkokoro.config import Settings


def resolve_model_path(settings: Settings) -> Path:
    if settings.model_path is not None:
        return settings.model_path

    path = hf_hub_download(
        repo_id=settings.model_repo,
        filename=settings.model_file,
        cache_dir=settings.cache_dir,
    )
    return Path(path)


def resolve_voices_path(settings: Settings) -> Path:
    if settings.voices_path is not None:
        return settings.voices_path

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.cache_dir / "voices-v1.0.bin"
    if not destination.exists():
        urlretrieve(settings.voices_url, destination)
    return destination
