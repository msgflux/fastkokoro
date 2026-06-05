import numpy as np

from fastkokoro.audio import encode_audio, trim_audio_part


def test_encode_audio_pcm_numpy_path_matches_expected():
    samples = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)

    encoded = encode_audio(samples, 24000, "pcm", use_pcm_jit=False)

    expected = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    assert encoded == expected


def test_encode_audio_pcm_jit_matches_numpy():
    samples = np.array([-0.9, -0.1, 0.0, 0.1, 0.9], dtype=np.float32)

    encoded_jit = encode_audio(samples, 24000, "pcm", use_pcm_jit=True)
    encoded_numpy = encode_audio(samples, 24000, "pcm", use_pcm_jit=False)

    assert encoded_jit == encoded_numpy


def test_trim_audio_part_jit_trims_leading_and_trailing_silence():
    samples = np.array([0.0, 0.0, 0.02, 0.5, 0.03, 0.0, 0.0], dtype=np.float32)

    trimmed = trim_audio_part(samples, use_jit=True)

    np.testing.assert_array_equal(
        trimmed,
        np.array([0.02, 0.5, 0.03], dtype=np.float32),
    )


def test_trim_audio_part_numpy_path_uses_kokoro_trim():
    samples = np.array([0.0, 0.0, 0.02, 0.5, 0.03, 0.0, 0.0], dtype=np.float32)

    trimmed = trim_audio_part(samples, use_jit=False)

    assert len(trimmed) > 0
