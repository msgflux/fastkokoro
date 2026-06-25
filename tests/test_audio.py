import numpy as np

from fastkokoro.audio import encode_audio, trim_audio_part, trim_audio_tail


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


def test_trim_audio_part_jit_matches_numpy_frame_rms_path():
    samples = np.concatenate(
        [
            np.zeros(2048, dtype=np.float32),
            np.linspace(0.0, 0.5, 4096, dtype=np.float32),
            np.zeros(2048, dtype=np.float32),
        ]
    )

    trimmed_jit = trim_audio_part(samples, use_jit=True)
    trimmed_numpy = trim_audio_part(samples, use_jit=False)

    np.testing.assert_array_equal(trimmed_jit, trimmed_numpy)


def test_trim_audio_part_numpy_path_uses_kokoro_trim():
    samples = np.array([0.0, 0.0, 0.02, 0.5, 0.03, 0.0, 0.0], dtype=np.float32)

    trimmed = trim_audio_part(samples, use_jit=False)

    assert len(trimmed) > 0


def test_trim_audio_tail_removes_tail_and_fades_boundary():
    samples = np.ones(240, dtype=np.float32)

    trimmed = trim_audio_tail(samples, sample_rate=1000, trim_ms=40, fade_ms=10)

    assert trimmed.shape == (200,)
    assert trimmed[-1] == 0.0
    assert 0.0 < trimmed[-5] < 1.0


def test_trim_audio_tail_disabled_returns_float32_view():
    samples = np.ones(10, dtype=np.float64)

    trimmed = trim_audio_tail(samples, sample_rate=1000, trim_ms=0, fade_ms=10)

    assert trimmed.dtype == np.float32
    np.testing.assert_array_equal(trimmed, np.ones(10, dtype=np.float32))
