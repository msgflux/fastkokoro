import numpy as np

from fastkokoro.assets import VOICE_STYLE_SHAPE, convert_raw_voices_to_npz


def test_convert_raw_voices_to_npz(tmp_path):
    voices_bin = tmp_path / "voices.bin"
    voices_index = tmp_path / "voices.txt"
    cache_dir = tmp_path / "cache"
    names = ["af_heart", "pf_dora"]
    values = np.arange(
        len(names) * np.prod(VOICE_STYLE_SHAPE),
        dtype=np.float32,
    )

    values.tofile(voices_bin)
    voices_index.write_text("0=af_heart\n1=pf_dora\n", encoding="utf-8")

    converted = convert_raw_voices_to_npz(voices_bin, voices_index, cache_dir)

    assert converted == cache_dir / "voices-fastkokoro.npz"
    loaded = np.load(converted)
    assert loaded.files == names
    assert loaded["af_heart"].shape == VOICE_STYLE_SHAPE
    assert loaded["pf_dora"].shape == VOICE_STYLE_SHAPE
