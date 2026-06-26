import pytest

torch = pytest.importorskip("torch")

from scripts.export_kokoro_torch_ttfc import KokoroTTFCExportWrapper  # noqa: E402


def test_mask_waveform_to_duration_zeros_after_active_samples_with_fade():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=12,
        output_samples_per_frame=2,
        output_fade_samples=4,
    )
    waveform = torch.ones(12)
    duration = torch.tensor([2, 1])

    masked = wrapper.mask_waveform_to_duration(waveform, duration)

    assert torch.all(masked[:2] == 1.0)
    assert torch.all(masked[6:] == 0.0)
    assert torch.all(masked[2:6] <= 1.0)
    assert torch.all(masked[2:6] >= 0.0)


def test_mask_waveform_to_duration_allows_tail_margin_before_fade_out():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=12,
        output_samples_per_frame=2,
        output_fade_samples=4,
        output_tail_margin_samples=2,
    )
    waveform = torch.ones(12)
    duration = torch.tensor([2, 1])

    masked = wrapper.mask_waveform_to_duration(waveform, duration)

    assert torch.all(masked[:4] == 1.0)
    assert torch.all(masked[8:] == 0.0)
