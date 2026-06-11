from __future__ import annotations

from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from fastkokoro.config import Settings

VOICE_STYLE_SHAPE = (510, 1, 256)


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

    voices_bin = Path(
        hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.voices_file,
            cache_dir=settings.cache_dir,
        )
    )
    voices_index = Path(
        hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.voices_index_file,
            cache_dir=settings.cache_dir,
        )
    )
    return convert_raw_voices_to_npz(voices_bin, voices_index, settings.cache_dir)


def convert_raw_voices_to_npz(
    voices_bin: Path, voices_index: Path, cache_dir: Path
) -> Path:
    destination = cache_dir / "voices-fastkokoro.npz"
    if destination.exists():
        return destination

    names = parse_voice_names(voices_index)
    raw = np.fromfile(voices_bin, dtype=np.float32)
    expected_values = len(names) * np.prod(VOICE_STYLE_SHAPE)
    if raw.size != expected_values:
        raise ValueError(
            "Unexpected voices.bin shape: "
            f"got {raw.size} float32 values for {len(names)} voices, "
            f"expected {expected_values}."
        )

    styles = raw.reshape((len(names), *VOICE_STYLE_SHAPE))
    voices = {name: styles[index] for index, name in enumerate(names)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **voices)
    return destination


def parse_voice_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        _, name = line.split("=", 1)
        names.append(name)
    return names
