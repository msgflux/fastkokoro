from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

torch = pytest.importorskip("torch")

from scripts.export_kokoro_torch_ttfc import (  # noqa: E402
    FixedLengthAwareBiLSTM,
    KokoroTTFCExportWrapper,
    fold_constant_reciprocals,
    validate_output_geometry,
)


def test_fixed_length_aware_bilstm_matches_packed_sequence():
    torch.manual_seed(7)
    source = torch.nn.LSTM(
        input_size=5,
        hidden_size=4,
        batch_first=True,
        bidirectional=True,
    ).eval()
    inputs = torch.randn(2, 12, 5)
    lengths = torch.tensor([7, 3], dtype=torch.long)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        inputs,
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, _ = source(packed)
    expected, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=inputs.shape[1],
    )

    actual, _ = FixedLengthAwareBiLSTM(source)(inputs, lengths)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_fixed_length_aware_bilstm_clamps_lengths_to_fixed_width():
    torch.manual_seed(11)
    source = torch.nn.LSTM(
        input_size=3,
        hidden_size=2,
        batch_first=True,
        bidirectional=True,
    ).eval()
    inputs = torch.randn(1, 6, 3)
    expected, _ = source(inputs)

    actual, _ = FixedLengthAwareBiLSTM(source)(inputs, torch.tensor([9]))

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


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


def test_mask_waveform_to_duration_uses_short_margin_by_input_length():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=20,
        output_samples_per_frame=2,
        output_tail_margin_samples=6,
        output_short_tail_margin_samples=2,
        output_short_tail_margin_max_tokens=4,
    )
    waveform = torch.ones(20)
    duration = torch.tensor([2, 1])

    short = wrapper.mask_waveform_to_duration(
        waveform,
        duration,
        input_lengths=torch.tensor([4]),
    )
    long = wrapper.mask_waveform_to_duration(
        waveform,
        duration,
        input_lengths=torch.tensor([5]),
    )

    assert torch.all(short[:8] == 1.0)
    assert torch.all(short[8:] == 0.0)
    assert torch.all(long[:12] == 1.0)
    assert torch.all(long[12:] == 0.0)


def test_mask_waveform_to_duration_uses_three_margin_tiers():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=30,
        output_samples_per_frame=2,
        output_tail_margin_samples=8,
        output_short_tail_margin_samples=2,
        output_short_tail_margin_max_tokens=4,
        output_medium_tail_margin_samples=5,
        output_medium_tail_margin_max_tokens=7,
    )
    waveform = torch.ones(30)
    duration = torch.tensor([2, 1])

    short = wrapper.mask_waveform_to_duration(
        waveform, duration, input_lengths=torch.tensor([4])
    )
    medium = wrapper.mask_waveform_to_duration(
        waveform, duration, input_lengths=torch.tensor([7])
    )
    long = wrapper.mask_waveform_to_duration(
        waveform, duration, input_lengths=torch.tensor([8])
    )

    assert torch.all(short[:8] == 1.0)
    assert torch.all(short[8:] == 0.0)
    assert torch.all(medium[:11] == 1.0)
    assert torch.all(medium[11:] == 0.0)
    assert torch.all(long[:14] == 1.0)
    assert torch.all(long[14:] == 0.0)


def test_finalize_waveform_only_pads_missing_samples():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=12,
    )
    wrapper.configure_output_padding(native_output_samples=10)

    waveform = wrapper.finalize_waveform(torch.ones(10), torch.ones(1))

    assert wrapper._fixed_output_padding_samples == 2
    assert wrapper._fixed_output_crop_samples == 0
    assert waveform.shape == (12,)
    assert torch.all(waveform[:10] == 1.0)
    assert torch.all(waveform[10:] == 0.0)


def test_finalize_waveform_crops_without_padding():
    wrapper = KokoroTTFCExportWrapper(
        torch.nn.Identity(),
        fixed_output_samples=12,
    )
    wrapper.configure_output_padding(native_output_samples=14)

    waveform = wrapper.finalize_waveform(torch.ones(14), torch.ones(1))

    assert wrapper._fixed_output_padding_samples == 0
    assert wrapper._fixed_output_crop_samples == 2
    assert waveform.shape == (12,)


def test_fold_constant_reciprocals_replaces_node_with_initializer():
    alpha = numpy_helper.from_array(
        np.array([2.0, 4.0], dtype=np.float16),
        name="alpha",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Reciprocal", ["alpha"], ["inverse_alpha"]),
            helper.make_node("Mul", ["input", "inverse_alpha"], ["output"]),
        ],
        "reciprocal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2])],
        [alpha],
    )
    model = helper.make_model(graph)

    folded = fold_constant_reciprocals(model)

    assert folded == 1
    assert [node.op_type for node in model.graph.node] == ["Mul"]
    inverse = next(
        initializer
        for initializer in model.graph.initializer
        if initializer.name == "inverse_alpha"
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(inverse),
        np.array([0.5, 0.25], dtype=np.float16),
    )
    onnx.checker.check_model(model)


def test_output_geometry_allows_masked_native_vocoder_tail_crop():
    args = SimpleNamespace(
        fixed_output_samples=104400,
        fixed_alignment_frames=200,
        output_samples_per_frame=480,
        output_tail_margin_samples=8400,
    )

    validate_output_geometry(args, native_output_samples=120000)


def test_output_geometry_rejects_crop_inside_reachable_mask():
    args = SimpleNamespace(
        fixed_output_samples=104399,
        fixed_alignment_frames=200,
        output_samples_per_frame=480,
        output_tail_margin_samples=8400,
    )

    with pytest.raises(ValueError, match="shorter than the reachable duration mask"):
        validate_output_geometry(args, native_output_samples=120000)


def test_output_geometry_allows_native_ratio_with_reserved_tail_frames():
    args = SimpleNamespace(
        fixed_output_samples=163200,
        fixed_alignment_frames=272,
        output_samples_per_frame=600,
        output_tail_margin_samples=8400,
    )

    validate_output_geometry(args, native_output_samples=163200)
