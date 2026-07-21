from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import sys
from collections import Counter
from contextlib import suppress
from pathlib import Path

import onnx
import torch
import torch.nn.functional as functional
from onnx import numpy_helper

BLOCKED_TTFC_OPS = {"NonZero", "ScatterND", "STFT", "Range"}


class FixedLengthAwareBiLSTM(torch.nn.Module):
    """Run a bidirectional LSTM at fixed width without padding contamination."""

    def __init__(self, source: torch.nn.LSTM) -> None:
        super().__init__()
        if not source.batch_first or not source.bidirectional or source.num_layers != 1:
            raise ValueError(
                "fixed length-aware export requires a one-layer, batch-first "
                "bidirectional LSTM"
            )
        if source.proj_size:
            raise ValueError("projected LSTMs are not supported")

        options = {
            "input_size": source.input_size,
            "hidden_size": source.hidden_size,
            "num_layers": 1,
            "bias": source.bias,
            "batch_first": True,
            "bidirectional": False,
        }
        self.forward_lstm = torch.nn.LSTM(**options)
        self.backward_lstm = torch.nn.LSTM(**options)
        parameter = next(source.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)
        self._copy_direction(source, self.forward_lstm, reverse=False)
        self._copy_direction(source, self.backward_lstm, reverse=True)

    @staticmethod
    def _copy_direction(
        source: torch.nn.LSTM,
        target: torch.nn.LSTM,
        *,
        reverse: bool,
    ) -> None:
        suffix = "_reverse" if reverse else ""
        with torch.no_grad():
            target.weight_ih_l0.copy_(getattr(source, f"weight_ih_l0{suffix}"))
            target.weight_hh_l0.copy_(getattr(source, f"weight_hh_l0{suffix}"))
            if source.bias:
                target.bias_ih_l0.copy_(getattr(source, f"bias_ih_l0{suffix}"))
                target.bias_hh_l0.copy_(getattr(source, f"bias_hh_l0{suffix}"))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, sequence_length, feature_count = x.shape
        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)
        lengths = torch.clamp(lengths, min=1, max=sequence_length)

        positions = torch.arange(
            sequence_length,
            device=x.device,
            dtype=lengths.dtype,
        ).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        valid_features = valid.unsqueeze(-1)
        masked_x = x.masked_fill(~valid_features, 0.0)

        self.forward_lstm.flatten_parameters()
        forward_output, _ = self.forward_lstm(masked_x)

        reverse_indices = lengths.unsqueeze(1) - 1 - positions
        reverse_indices = torch.clamp(
            reverse_indices,
            min=0,
            max=sequence_length - 1,
        )
        reverse_input_indices = reverse_indices.unsqueeze(-1).expand(
            batch_size,
            sequence_length,
            feature_count,
        )
        reverse_input = torch.gather(masked_x, 1, reverse_input_indices)
        reverse_input = reverse_input.masked_fill(~valid_features, 0.0)
        self.backward_lstm.flatten_parameters()
        reverse_output, _ = self.backward_lstm(reverse_input)
        reverse_output_indices = reverse_indices.unsqueeze(-1).expand(
            batch_size,
            sequence_length,
            self.backward_lstm.hidden_size,
        )
        backward_output = torch.gather(reverse_output, 1, reverse_output_indices)

        output = torch.cat([forward_output, backward_output], dim=-1)
        output = output.masked_fill(~valid_features, 0.0)
        return output, None


class KokoroTTFCExportWrapper(torch.nn.Module):
    def __init__(
        self,
        kmodel: torch.nn.Module,
        *,
        fixed_output_samples: int | None = None,
        fixed_alignment_frames: int | None = None,
        output_samples_per_frame: int | None = None,
        output_fade_samples: int = 0,
        output_tail_margin_samples: int = 0,
        output_short_tail_margin_samples: int | None = None,
        output_short_tail_margin_max_tokens: int | None = None,
        output_medium_tail_margin_samples: int | None = None,
        output_medium_tail_margin_max_tokens: int | None = None,
        static_alignment: bool = False,
        length_aware: bool = False,
        internal_dtype: torch.dtype = torch.float32,
        decoder_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.kmodel = kmodel
        self.fixed_output_samples = fixed_output_samples
        self.fixed_alignment_frames = fixed_alignment_frames
        self.output_samples_per_frame = output_samples_per_frame
        self.output_fade_samples = output_fade_samples
        self.output_tail_margin_samples = output_tail_margin_samples
        self.output_short_tail_margin_samples = output_short_tail_margin_samples
        self.output_short_tail_margin_max_tokens = output_short_tail_margin_max_tokens
        self.output_medium_tail_margin_samples = output_medium_tail_margin_samples
        self.output_medium_tail_margin_max_tokens = (
            output_medium_tail_margin_max_tokens
        )
        self.static_alignment = static_alignment
        self.length_aware = length_aware
        self.internal_dtype = internal_dtype
        self.decoder_dtype = decoder_dtype
        self._fixed_output_padding_samples: int | None = None
        self._fixed_output_crop_samples: int | None = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.synthesize(
            input_ids,
            ref_s,
            speed,
            input_lengths,
        )
        waveform = self.finalize_waveform(
            waveform,
            duration,
            input_lengths=input_lengths,
        )
        return waveform.float(), duration

    def synthesize(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        ref_s = ref_s.to(self.internal_dtype)
        speed = speed.to(self.internal_dtype)
        if self.static_alignment:
            waveform, duration = self.forward_static_alignment(
                input_ids,
                ref_s,
                speed,
                input_lengths,
            )
        elif self.length_aware:
            if input_lengths is None:
                input_lengths = torch.full(
                    (input_ids.shape[0],),
                    input_ids.shape[-1],
                    device=input_ids.device,
                    dtype=torch.long,
                )
            waveform, duration = self.forward_with_lengths(
                input_ids,
                ref_s,
                speed,
                input_lengths,
            )
        else:
            waveform, duration = self.kmodel.forward_with_tokens(
                input_ids,
                ref_s,
                speed,
            )
        return waveform, duration

    def configure_output_padding(self, native_output_samples: int) -> None:
        if self.fixed_output_samples is None:
            self._fixed_output_padding_samples = 0
            self._fixed_output_crop_samples = 0
            return
        self._fixed_output_padding_samples = max(
            self.fixed_output_samples - native_output_samples,
            0,
        )
        self._fixed_output_crop_samples = max(
            native_output_samples - self.fixed_output_samples,
            0,
        )

    def finalize_waveform(
        self,
        waveform: torch.FloatTensor,
        duration: torch.LongTensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        if self.fixed_output_samples is not None:
            padding_samples = self._fixed_output_padding_samples
            if padding_samples is None:
                padding_samples = max(
                    self.fixed_output_samples - int(waveform.shape[-1]),
                    0,
                )
            if padding_samples:
                waveform = functional.pad(waveform, (0, padding_samples))
            crop_samples = self._fixed_output_crop_samples
            if crop_samples is None:
                crop_samples = max(
                    int(waveform.shape[-1]) - self.fixed_output_samples,
                    0,
                )
            if padding_samples or crop_samples:
                waveform = waveform[..., : self.fixed_output_samples]
            if self.output_samples_per_frame is not None:
                waveform = self.mask_waveform_to_duration(
                    waveform,
                    duration,
                    input_lengths=input_lengths,
                )
        return waveform

    def mask_waveform_to_duration(
        self,
        waveform: torch.FloatTensor,
        duration: torch.LongTensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        sample_count = self.fixed_output_samples or waveform.shape[-1]
        active_frames = duration.sum()
        if self.fixed_alignment_frames is not None:
            max_frames = torch.tensor(
                self.fixed_alignment_frames,
                device=waveform.device,
                dtype=active_frames.dtype,
            )
            active_frames = torch.minimum(active_frames, max_frames)
        active_samples = active_frames * self.output_samples_per_frame
        max_samples = torch.tensor(
            sample_count,
            device=waveform.device,
            dtype=active_samples.dtype,
        )
        active_samples = torch.minimum(active_samples, max_samples)
        margin_adjustment = torch.zeros_like(active_samples)
        if input_lengths is not None:
            if (
                self.output_medium_tail_margin_samples is not None
                and self.output_medium_tail_margin_max_tokens is not None
            ):
                medium_margin_adjustment = torch.tensor(
                    self.output_medium_tail_margin_samples
                    - self.output_tail_margin_samples,
                    device=waveform.device,
                    dtype=active_samples.dtype,
                )
                margin_adjustment = torch.where(
                    input_lengths[0] <= self.output_medium_tail_margin_max_tokens,
                    medium_margin_adjustment,
                    margin_adjustment,
                )
            if (
                self.output_short_tail_margin_samples is not None
                and self.output_short_tail_margin_max_tokens is not None
            ):
                short_margin_adjustment = torch.tensor(
                    self.output_short_tail_margin_samples
                    - self.output_tail_margin_samples,
                    device=waveform.device,
                    dtype=active_samples.dtype,
                )
                margin_adjustment = torch.where(
                    input_lengths[0] <= self.output_short_tail_margin_max_tokens,
                    short_margin_adjustment,
                    margin_adjustment,
                )
        if self.output_tail_margin_samples > 0:
            active_samples = active_samples + margin_adjustment
        if self.output_tail_margin_samples > 0:
            positions = torch.arange(
                -self.output_tail_margin_samples,
                sample_count - self.output_tail_margin_samples,
                device=waveform.device,
            )
        else:
            positions = torch.arange(sample_count, device=waveform.device)
        if self.output_fade_samples <= 0:
            return waveform * (positions < active_samples).to(waveform.dtype)

        fade_start = active_samples - self.output_fade_samples
        fade_start = torch.clamp(fade_start, min=0)
        fade_progress = (positions - fade_start).to(waveform.dtype) / float(
            self.output_fade_samples
        )
        fade_progress = torch.clamp(fade_progress, min=0.0, max=1.0)
        gain = 0.5 * (1.0 + torch.cos(fade_progress * torch.pi))
        gain = torch.where(
            positions < fade_start,
            torch.ones_like(gain),
            gain,
        )
        gain = torch.where(
            positions < active_samples,
            gain,
            torch.zeros_like(gain),
        )
        return waveform * gain

    def forward_static_alignment(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        del speed
        batch_size, token_count = input_ids.shape
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,),
                token_count,
                device=input_ids.device,
                dtype=torch.long,
            )
        positions = torch.arange(token_count, device=input_ids.device)
        text_mask = positions.unsqueeze(0).expand(batch_size, -1) >= (
            input_lengths.unsqueeze(1)
        )

        bert_dur = self.kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        pred_dur = (~text_mask).long().squeeze(0)
        pred_aln_trg = torch.eye(
            token_count,
            device=input_ids.device,
            dtype=d.dtype,
        ).unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = self.run_f0_noise(en, s, pred_dur.sum().reshape(1))
        t_en = self.kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.run_decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def run_f0_noise(
        self,
        x: torch.FloatTensor,
        style: torch.FloatTensor,
        frame_lengths: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        predictor = self.kmodel.predictor
        if not isinstance(predictor.shared, FixedLengthAwareBiLSTM):
            return predictor.F0Ntrain(x, style)

        shared, _ = predictor.shared(x.transpose(-1, -2), frame_lengths)
        f0 = shared.transpose(-1, -2)
        for block in predictor.F0:
            f0 = block(f0, style)
        f0 = predictor.F0_proj(f0)
        noise = shared.transpose(-1, -2)
        for block in predictor.N:
            noise = block(noise, style)
        noise = predictor.N_proj(noise)
        return f0.squeeze(1), noise.squeeze(1)

    def forward_with_lengths(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        batch_size, token_count = input_ids.shape
        positions = torch.arange(token_count, device=input_ids.device)
        text_mask = positions.unsqueeze(0).expand(batch_size, -1) >= (
            input_lengths.unsqueeze(1)
        )
        bert_dur = self.kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        duration_lstm = self.kmodel.predictor.lstm
        if isinstance(duration_lstm, FixedLengthAwareBiLSTM):
            x, _ = duration_lstm(d, input_lengths)
        else:
            x, _ = duration_lstm(d)
        duration = self.kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        pred_dur = torch.where(
            text_mask.squeeze(0),
            torch.zeros_like(pred_dur),
            pred_dur,
        )
        pred_aln_trg = self.build_alignment(token_count, pred_dur, d.dtype)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = self.run_f0_noise(en, s, pred_dur.sum().reshape(1))
        t_en = self.kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.run_decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def run_decoder(
        self,
        asr: torch.FloatTensor,
        f0_pred: torch.FloatTensor,
        n_pred: torch.FloatTensor,
        style: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.kmodel.decoder(
            asr.to(self.decoder_dtype),
            f0_pred.to(self.decoder_dtype),
            n_pred.to(self.decoder_dtype),
            style.to(self.decoder_dtype),
        )

    def build_alignment(
        self,
        token_count: int,
        pred_dur: torch.LongTensor,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if self.fixed_alignment_frames is not None:
            frame_positions = torch.arange(
                self.fixed_alignment_frames,
                device=self.kmodel.device,
            )
            ends = torch.cumsum(pred_dur, dim=0).unsqueeze(1)
            starts = (ends.squeeze(1) - pred_dur).unsqueeze(1)
            alignment = (frame_positions.unsqueeze(0) >= starts) & (
                frame_positions.unsqueeze(0) < ends
            )
            return alignment.unsqueeze(0).to(dtype=dtype, device=self.kmodel.device)

        indices = torch.repeat_interleave(
            torch.arange(token_count, device=self.kmodel.device),
            pred_dur,
        )
        frame_count = indices.shape[0]
        frame_indices = torch.arange(frame_count, device=self.kmodel.device)
        pred_aln_trg = torch.zeros(
            (token_count, frame_count),
            device=self.kmodel.device,
            dtype=dtype,
        )
        pred_aln_trg[indices, frame_indices] = 1
        return pred_aln_trg.unsqueeze(0).to(self.kmodel.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a fixed-input Kokoro TTFC ONNX from hexgrad/kokoro PyTorch."
    )
    parser.add_argument(
        "--kokoro-repo",
        type=Path,
        default=Path("demo-output/reexport/hexgrad-kokoro"),
        help="Local checkout of https://github.com/hexgrad/kokoro.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo-output/reexport/kokoro-ttfc-b24.onnx"),
    )
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bucket", type=int, default=24)
    parser.add_argument("--fixed-output-samples", type=int)
    parser.add_argument("--fixed-alignment-frames", type=int)
    parser.add_argument("--output-samples-per-frame", type=int)
    parser.add_argument("--output-fade-samples", type=int, default=0)
    parser.add_argument("--output-tail-margin-samples", type=int, default=0)
    parser.add_argument("--output-short-tail-margin-samples", type=int)
    parser.add_argument("--output-short-tail-margin-max-tokens", type=int)
    parser.add_argument("--output-medium-tail-margin-samples", type=int)
    parser.add_argument("--output-medium-tail-margin-max-tokens", type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=(
            "fp32",
            "fp16",
            "decoder-fp16",
            "decoder-fp16-istft-fp32",
            "decoder-fp16-post-fp32",
            "decoder-fp16-generator-fp32",
        ),
        help="Internal PyTorch precision used during export.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--legacy-export",
        action="store_true",
        help="Use the legacy tracer-based torch.onnx exporter instead of torch.export.",
    )
    parser.add_argument(
        "--static-alignment",
        action="store_true",
        help="Bypass dynamic duration alignment with one frame per token.",
    )
    parser.add_argument(
        "--length-aware",
        action="store_true",
        help=(
            "Export input_lengths and mask padded bucket slots from duration/alignment."
        ),
    )
    parser.add_argument(
        "--patch-fixed-lstm",
        action="store_true",
        help=(
            "Patch Kokoro bidirectional LSTMs to avoid pack_padded_sequence "
            "for fixed shapes."
        ),
    )
    parser.add_argument(
        "--patch-fixed-lstm-scope",
        choices=("all", "duration-text", "duration"),
        default="all",
        help=(
            "Select which bidirectional LSTMs respect real sequence lengths. "
            "The duration predictor is always included."
        ),
    )
    parser.add_argument(
        "--patch-deterministic-source",
        action="store_true",
        help=(
            "Patch the NSF source module to avoid random/scatter-heavy sine reset "
            "logic."
        ),
    )
    parser.add_argument(
        "--patch-deterministic-sine-source",
        action="store_true",
        help=(
            "Patch the NSF source module to keep deterministic sine excitation "
            "while removing random source noise."
        ),
    )
    parser.add_argument(
        "--patch-scatterless-sine-source",
        action="store_true",
        help=(
            "Patch SineGen in-place phase updates into slice/cat operations to "
            "avoid ScatterND while preserving source randomness."
        ),
    )
    parser.add_argument(
        "--patch-split-adain",
        action="store_true",
        help=(
            "Split AdaIN affine projections into gamma/beta Linear modules to "
            "avoid chunk-generated Slice subgraphs."
        ),
    )
    parser.add_argument(
        "--patch-albert-sdpa-bool-mask-scale",
        action="store_true",
        help=(
            "Patch Albert SDPA export to use a boolean 4D attention mask and "
            "an explicit attention scale, avoiding dynamic Shape/Sqrt scale "
            "subgraphs while keeping Albert in fp32."
        ),
    )
    parser.add_argument(
        "--fold-constant-reciprocals",
        action="store_true",
        help=(
            "Replace Reciprocal nodes whose inputs are initializers with folded "
            "initializers. This avoids unsupported fp16 constant folding in older "
            "ONNX Runtime releases."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for export tracing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_export_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    add_checkout_to_path(args.kokoro_repo)
    if args.patch_albert_sdpa_bool_mask_scale:
        patch_albert_sdpa_bool_mask_scale_for_export()
    from kokoro import KModel  # noqa: PLC0415

    kmodel_kwargs: dict[str, object] = {
        "repo_id": args.repo_id,
        "disable_complex": True,
    }
    if args.config is not None:
        kmodel_kwargs["config"] = str(args.config)
    if args.checkpoint is not None:
        kmodel_kwargs["model"] = str(args.checkpoint)

    device = torch.device(args.device)
    kmodel = KModel(**kmodel_kwargs).eval().to(device)
    if args.patch_fixed_lstm:
        patch_fixed_lstm_for_export(kmodel, scope=args.patch_fixed_lstm_scope)
    if args.patch_deterministic_source:
        patch_deterministic_source_for_export(kmodel)
    if args.patch_deterministic_sine_source:
        patch_deterministic_sine_source_for_export(kmodel)
    if args.patch_scatterless_sine_source:
        patch_scatterless_sine_source_for_export(kmodel)
    if args.patch_split_adain:
        patch_split_adain_for_export(kmodel)
    internal_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    decoder_dtype = (
        torch.float16
        if args.precision
        in {
            "fp16",
            "decoder-fp16",
            "decoder-fp16-istft-fp32",
            "decoder-fp16-post-fp32",
            "decoder-fp16-generator-fp32",
        }
        else torch.float32
    )
    if args.precision == "fp16":
        kmodel = kmodel.half()
        keep_source_module_float(kmodel)
    elif args.precision in {
        "decoder-fp16",
        "decoder-fp16-istft-fp32",
        "decoder-fp16-post-fp32",
        "decoder-fp16-generator-fp32",
    }:
        kmodel.decoder = kmodel.decoder.half()
        keep_source_module_float(kmodel)
        if args.precision == "decoder-fp16-istft-fp32":
            keep_generator_istft_float(kmodel)
        elif args.precision == "decoder-fp16-post-fp32":
            keep_generator_post_float(kmodel)
        elif args.precision == "decoder-fp16-generator-fp32":
            keep_generator_float(kmodel)
    model = KokoroTTFCExportWrapper(
        kmodel,
        fixed_output_samples=args.fixed_output_samples,
        fixed_alignment_frames=args.fixed_alignment_frames,
        output_samples_per_frame=args.output_samples_per_frame,
        output_fade_samples=args.output_fade_samples,
        output_tail_margin_samples=args.output_tail_margin_samples,
        output_short_tail_margin_samples=args.output_short_tail_margin_samples,
        output_short_tail_margin_max_tokens=(args.output_short_tail_margin_max_tokens),
        output_medium_tail_margin_samples=args.output_medium_tail_margin_samples,
        output_medium_tail_margin_max_tokens=(
            args.output_medium_tail_margin_max_tokens
        ),
        static_alignment=args.static_alignment,
        length_aware=args.length_aware,
        internal_dtype=internal_dtype,
        decoder_dtype=decoder_dtype,
    ).eval()

    input_ids = torch.arange(args.bucket, dtype=torch.long, device=device).unsqueeze(0)
    input_ids[:, 0] = 0
    input_ids[:, -1] = 0
    input_lengths = torch.full((1,), args.bucket, dtype=torch.long, device=device)
    ref_s = torch.randn(1, 256, dtype=torch.float32, device=device)
    speed = torch.ones(1, dtype=torch.float32, device=device)
    export_args = (
        (input_ids, ref_s, speed, input_lengths)
        if args.length_aware or args.static_alignment
        else (input_ids, ref_s, speed)
    )
    input_names = (
        ["input_ids", "style", "speed", "input_lengths"]
        if args.length_aware or args.static_alignment
        else ["input_ids", "style", "speed"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        native_waveform, duration = model.synthesize(*export_args)
        native_output_samples = int(native_waveform.shape[-1])
        model.configure_output_padding(native_output_samples)
        input_lengths = export_args[3] if len(export_args) > 3 else None
        waveform = model.finalize_waveform(
            native_waveform,
            duration,
            input_lengths=input_lengths,
        ).float()
    print(f"native_waveform_samples={native_output_samples}")
    print(f"torch_waveform_shape={tuple(waveform.shape)}")
    print(f"torch_duration_shape={tuple(duration.shape)}")
    print_output_capacity_report(args, native_output_samples)
    validate_output_geometry(args, native_output_samples)

    torch.onnx.export(
        model,
        args=export_args,
        f=str(args.output),
        export_params=True,
        input_names=input_names,
        output_names=["waveform", "duration"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=not args.legacy_export,
    )

    exported = onnx.load(args.output)
    if args.patch_albert_sdpa_bool_mask_scale:
        cleanup_albert_sdpa_bool_mask_scale_export(exported)
    folded_reciprocals = 0
    if args.fold_constant_reciprocals:
        folded_reciprocals = fold_constant_reciprocals(exported)
    set_export_metadata(exported, args, native_output_samples)
    onnx.checker.check_model(exported)
    onnx.save(exported, args.output)
    ops = Counter(node.op_type for node in exported.graph.node)
    print(f"onnx_path={args.output}")
    print(f"onnx_nodes={len(exported.graph.node)}")
    top_ops = ", ".join(f"{op}:{count}" for op, count in ops.most_common(20))
    print(f"onnx_top_ops={top_ops}")
    print(f"folded_constant_reciprocals={folded_reciprocals}")
    blocked = {op: ops[op] for op in sorted(BLOCKED_TTFC_OPS) if ops[op]}
    print(f"blocked_ops={blocked}")
    return 2 if blocked else 0


def validate_export_args(args: argparse.Namespace) -> None:
    if args.bucket <= 2:
        raise ValueError("--bucket must leave room for start/end pad tokens")
    for name in (
        "fixed_output_samples",
        "fixed_alignment_frames",
        "output_samples_per_frame",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("output_fade_samples", "output_tail_margin_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    adaptive_margin_values = (
        args.output_short_tail_margin_samples,
        args.output_short_tail_margin_max_tokens,
    )
    if any(value is not None for value in adaptive_margin_values) and not all(
        value is not None for value in adaptive_margin_values
    ):
        raise ValueError(
            "adaptive short tail margin requires both "
            "--output-short-tail-margin-samples and "
            "--output-short-tail-margin-max-tokens"
        )
    if args.output_short_tail_margin_samples is not None:
        if args.output_short_tail_margin_samples < 0:
            raise ValueError("--output-short-tail-margin-samples must be non-negative")
        if args.output_short_tail_margin_samples > args.output_tail_margin_samples:
            raise ValueError(
                "--output-short-tail-margin-samples cannot exceed "
                "--output-tail-margin-samples"
            )
        if args.output_short_tail_margin_max_tokens <= 0:
            raise ValueError("--output-short-tail-margin-max-tokens must be positive")
        if not args.length_aware:
            raise ValueError("adaptive short tail margin requires --length-aware")
    adaptive_medium_margin_values = (
        args.output_medium_tail_margin_samples,
        args.output_medium_tail_margin_max_tokens,
    )
    if any(value is not None for value in adaptive_medium_margin_values) and not all(
        value is not None for value in adaptive_medium_margin_values
    ):
        raise ValueError(
            "adaptive medium tail margin requires both "
            "--output-medium-tail-margin-samples and "
            "--output-medium-tail-margin-max-tokens"
        )
    if args.output_medium_tail_margin_samples is not None:
        if args.output_medium_tail_margin_samples < 0:
            raise ValueError("--output-medium-tail-margin-samples must be non-negative")
        if args.output_medium_tail_margin_samples > args.output_tail_margin_samples:
            raise ValueError(
                "--output-medium-tail-margin-samples cannot exceed "
                "--output-tail-margin-samples"
            )
        if args.output_medium_tail_margin_max_tokens <= 0:
            raise ValueError("--output-medium-tail-margin-max-tokens must be positive")
        if not args.length_aware:
            raise ValueError("adaptive medium tail margin requires --length-aware")
    if (
        args.output_short_tail_margin_samples is not None
        and args.output_medium_tail_margin_samples is not None
    ):
        if (
            args.output_short_tail_margin_samples
            > args.output_medium_tail_margin_samples
        ):
            raise ValueError(
                "short tail margin cannot exceed adaptive medium tail margin"
            )
        if (
            args.output_short_tail_margin_max_tokens
            >= args.output_medium_tail_margin_max_tokens
        ):
            raise ValueError(
                "short tail margin token limit must be lower than the medium limit"
            )
    if args.fixed_alignment_frames is not None and not (
        args.length_aware or args.static_alignment
    ):
        raise ValueError(
            "--fixed-alignment-frames requires --length-aware or --static-alignment"
        )
    if args.output_samples_per_frame is not None and args.fixed_output_samples is None:
        raise ValueError("--output-samples-per-frame requires --fixed-output-samples")
    if (args.output_fade_samples or args.output_tail_margin_samples) and (
        args.output_samples_per_frame is None
    ):
        raise ValueError("output fade/tail masking requires --output-samples-per-frame")
    if (
        args.fixed_output_samples is not None
        and args.output_tail_margin_samples > args.fixed_output_samples
    ):
        raise ValueError(
            "--output-tail-margin-samples cannot exceed --fixed-output-samples"
        )
    source_patches = (
        args.patch_deterministic_source,
        args.patch_deterministic_sine_source,
        args.patch_scatterless_sine_source,
    )
    if sum(source_patches) > 1:
        raise ValueError("select at most one source patch")


def print_output_capacity_report(
    args: argparse.Namespace,
    native_output_samples: int,
) -> None:
    fixed_output_samples = args.fixed_output_samples or native_output_samples
    output_padding_samples = max(fixed_output_samples - native_output_samples, 0)
    output_crop_samples = max(native_output_samples - fixed_output_samples, 0)
    print(f"fixed_output_padding_samples={output_padding_samples}")
    print(f"fixed_output_crop_samples={output_crop_samples}")

    if args.fixed_alignment_frames is None or args.output_samples_per_frame is None:
        return
    aligned_content_samples = (
        args.fixed_alignment_frames * args.output_samples_per_frame
    )
    native_samples_per_alignment_frame, native_remainder_samples = divmod(
        native_output_samples,
        args.fixed_alignment_frames,
    )
    print(f"native_samples_per_alignment_frame={native_samples_per_alignment_frame}")
    print(f"native_alignment_remainder_samples={native_remainder_samples}")
    print(f"content_mask_alignment_span_samples={aligned_content_samples}")
    if args.output_samples_per_frame != native_samples_per_alignment_frame:
        print(
            "note=content mask samples per duration frame intentionally differs "
            "from the native vocoder tensor ratio "
            f"({args.output_samples_per_frame} != "
            f"{native_samples_per_alignment_frame}); validate the suppressed "
            "vocoder tail by listening"
        )
    available_active_samples = max(
        fixed_output_samples - args.output_tail_margin_samples,
        0,
    )
    safe_duration_frames = min(
        args.fixed_alignment_frames,
        available_active_samples // args.output_samples_per_frame,
    )
    print(f"safe_duration_frames={safe_duration_frames}")
    if output_padding_samples:
        print(
            "warning=fixed output exceeds native vocoder output; the extra "
            f"{output_padding_samples} samples are zeros"
        )
    if output_crop_samples:
        print(
            "note=fixed output suppresses the final "
            f"{output_crop_samples} native vocoder tensor samples"
        )


def validate_output_geometry(
    args: argparse.Namespace,
    native_output_samples: int,
) -> None:
    if (
        args.fixed_output_samples is None
        or args.fixed_alignment_frames is None
        or args.output_samples_per_frame is None
    ):
        return
    maximum_masked_samples = (
        args.fixed_alignment_frames * args.output_samples_per_frame
        + args.output_tail_margin_samples
    )
    required_output_samples = min(native_output_samples, maximum_masked_samples)
    if args.fixed_output_samples < required_output_samples:
        raise ValueError(
            "--fixed-output-samples is shorter than the reachable duration mask "
            f"({args.fixed_output_samples} < {required_output_samples}); increase "
            "the output length or reduce alignment/tail capacity"
        )


def fold_constant_reciprocals(model: onnx.ModelProto) -> int:
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    folded_initializers = []
    kept_nodes = []
    folded = 0
    for node in model.graph.node:
        initializer = initializers.get(node.input[0]) if node.input else None
        if (
            node.op_type != "Reciprocal"
            or node.domain
            or initializer is None
            or len(node.output) != 1
            or node.output[0] in initializers
        ):
            kept_nodes.append(node)
            continue
        values = numpy_helper.to_array(initializer)
        if values.dtype.kind != "f":
            kept_nodes.append(node)
            continue
        reciprocal = (1.0 / values).astype(values.dtype)
        folded_initializers.append(
            numpy_helper.from_array(reciprocal, name=node.output[0])
        )
        folded += 1

    if not folded:
        return 0
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(folded_initializers)
    return folded


def set_export_metadata(
    model: onnx.ModelProto,
    args: argparse.Namespace,
    native_output_samples: int,
) -> None:
    metadata = {
        "fastkokoro.bucket": str(args.bucket),
        "fastkokoro.precision": args.precision,
        "fastkokoro.opset": str(args.opset),
        "fastkokoro.exporter": "legacy" if args.legacy_export else "dynamo",
        "fastkokoro.native_output_samples": str(native_output_samples),
        "fastkokoro.torch_version": torch.__version__,
    }
    if args.patch_fixed_lstm:
        metadata["fastkokoro.fixed_lstm_scope"] = args.patch_fixed_lstm_scope
    optional_values = {
        "fastkokoro.fixed_alignment_frames": args.fixed_alignment_frames,
        "fastkokoro.fixed_output_samples": args.fixed_output_samples,
        "fastkokoro.output_samples_per_frame": args.output_samples_per_frame,
        "fastkokoro.output_tail_margin_samples": args.output_tail_margin_samples,
        "fastkokoro.output_short_tail_margin_samples": (
            args.output_short_tail_margin_samples
        ),
        "fastkokoro.output_short_tail_margin_max_tokens": (
            args.output_short_tail_margin_max_tokens
        ),
        "fastkokoro.output_medium_tail_margin_samples": (
            args.output_medium_tail_margin_samples
        ),
        "fastkokoro.output_medium_tail_margin_max_tokens": (
            args.output_medium_tail_margin_max_tokens
        ),
        "fastkokoro.output_fade_samples": args.output_fade_samples,
    }
    if args.fixed_alignment_frames is not None:
        native_ratio, native_remainder = divmod(
            native_output_samples,
            args.fixed_alignment_frames,
        )
        if native_remainder == 0:
            optional_values["fastkokoro.native_samples_per_alignment_frame"] = (
                native_ratio
            )
    if args.output_samples_per_frame is not None:
        optional_values["fastkokoro.content_samples_per_duration_frame"] = (
            args.output_samples_per_frame
        )
    metadata.update(
        {key: str(value) for key, value in optional_values.items() if value is not None}
    )
    dependency_metadata = {
        "transformers": "fastkokoro.transformers_version",
        "onnx": "fastkokoro.onnx_version",
        "numpy": "fastkokoro.numpy_version",
        "huggingface-hub": "fastkokoro.huggingface_hub_version",
        "loguru": "fastkokoro.loguru_version",
        "misaki": "fastkokoro.misaki_version",
    }
    for package, key in dependency_metadata.items():
        with suppress(importlib.metadata.PackageNotFoundError):
            metadata[key] = importlib.metadata.version(package)
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(metadata)
    onnx.helper.set_model_props(model, existing)


def add_checkout_to_path(path: Path) -> None:
    if not (path / "kokoro" / "model.py").exists():
        raise FileNotFoundError(f"Invalid kokoro checkout: {path}")
    sys.path.insert(0, str(path.resolve()))
    if importlib.util.find_spec("kokoro") is None:
        raise RuntimeError(f"Unable to import kokoro from {path}")


def patch_fixed_lstm_for_export(
    kmodel: torch.nn.Module,
    *,
    scope: str = "all",
) -> None:
    import torch.nn.functional as functional  # noqa: PLC0415
    from kokoro.modules import AdaLayerNorm  # noqa: PLC0415

    if scope not in {"all", "duration-text", "duration"}:
        raise ValueError(f"Unsupported fixed LSTM patch scope: {scope}")

    def text_encoder_forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x = x.masked_fill(m, 0.0)
        for conv in self.cnn:
            x = conv(x)
            x = x.masked_fill(m, 0.0)
        x = x.transpose(1, 2)
        if isinstance(self.lstm, FixedLengthAwareBiLSTM):
            x, _ = self.lstm(x, input_lengths)
        else:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x = x.masked_fill(m, 0.0)
        return x

    def duration_encoder_forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        style_expanded = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style_expanded], axis=-1)
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, style_expanded.permute(1, 2, 0)], axis=1)
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x, _ = block(x, text_lengths)
                x = functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

    if scope in {"all", "duration-text"}:
        kmodel.text_encoder.lstm = FixedLengthAwareBiLSTM(kmodel.text_encoder.lstm)
    for index, block in enumerate(kmodel.predictor.text_encoder.lstms):
        if isinstance(block, torch.nn.LSTM):
            kmodel.predictor.text_encoder.lstms[index] = FixedLengthAwareBiLSTM(
                block
            )
    kmodel.predictor.lstm = FixedLengthAwareBiLSTM(kmodel.predictor.lstm)
    if scope == "all":
        kmodel.predictor.shared = FixedLengthAwareBiLSTM(kmodel.predictor.shared)

    kmodel.text_encoder.forward = text_encoder_forward.__get__(
        kmodel.text_encoder,
        type(kmodel.text_encoder),
    )
    kmodel.predictor.text_encoder.forward = duration_encoder_forward.__get__(
        kmodel.predictor.text_encoder,
        type(kmodel.predictor.text_encoder),
    )


def keep_source_module_float(kmodel: torch.nn.Module) -> None:
    source = kmodel.decoder.generator.m_source
    source.float()
    original_forward = source.forward

    def source_forward_fp32(self, x):
        output_dtype = x.dtype
        sine_merge, noise, uv = original_forward(x.float())
        return (
            sine_merge.to(output_dtype),
            noise.to(output_dtype),
            uv.to(output_dtype),
        )

    source.forward = source_forward_fp32.__get__(source, type(source))


def keep_generator_post_float(kmodel: torch.nn.Module) -> None:
    generator = kmodel.decoder.generator
    generator.conv_post.float()
    generator.stft.float()

    def generator_forward_post_fp32(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _noi_source, _uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source.float())
            har = torch.cat([har_spec, har_phase], dim=1).to(x.dtype)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                block_output = self.resblocks[i * self.num_kernels + j](x, s)
                xs = block_output if xs is None else xs + block_output
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x).float()
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)

    generator.forward = generator_forward_post_fp32.__get__(generator, type(generator))


def keep_generator_istft_float(kmodel: torch.nn.Module) -> None:
    generator = kmodel.decoder.generator
    generator.stft.float()

    def generator_forward_istft_fp32(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _noi_source, _uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source.float())
            har = torch.cat([har_spec, har_phase], dim=1).to(x.dtype)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                block_output = self.resblocks[i * self.num_kernels + j](x, s)
                xs = block_output if xs is None else xs + block_output
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x).float()
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)

    generator.forward = generator_forward_istft_fp32.__get__(generator, type(generator))


def keep_generator_float(kmodel: torch.nn.Module) -> None:
    generator = kmodel.decoder.generator
    generator.float()
    original_forward = generator.forward

    def generator_forward_fp32(self, x, s, f0):
        return original_forward(x.float(), s.float(), f0.float())

    generator.forward = generator_forward_fp32.__get__(generator, type(generator))


def patch_deterministic_source_for_export(kmodel: torch.nn.Module) -> None:
    source = kmodel.decoder.generator.m_source

    def source_forward(self, x):
        batch, length, _ = x.shape
        harmonic_dim = self.l_sin_gen.harmonic_num + 1
        sine_wavs = torch.zeros(
            batch,
            length,
            harmonic_dim,
            device=x.device,
            dtype=x.dtype,
        )
        uv = (x > self.l_sin_gen.voiced_threshold).to(dtype=x.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.zeros_like(uv)
        return sine_merge, noise, uv

    source.forward = source_forward.__get__(source, type(source))


def patch_deterministic_sine_source_for_export(kmodel: torch.nn.Module) -> None:
    source = kmodel.decoder.generator.m_source
    sine_generator = source.l_sin_gen

    def sine_forward(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rad_values = functional.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = functional.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def sine_gen_forward(self, f0):
        harmonic_factors = torch.arange(
            1,
            self.harmonic_num + 2,
            device=f0.device,
            dtype=f0.dtype,
        ).view(1, 1, -1)
        sine_waves = self._f02sine(f0 * harmonic_factors) * self.sine_amp
        uv = self._f02uv(f0).to(dtype=f0.dtype)
        sine_waves = sine_waves * uv
        noise = torch.zeros_like(sine_waves)
        return sine_waves, uv, noise

    def source_forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.zeros_like(uv)
        return sine_merge, noise, uv

    sine_generator._f02sine = sine_forward.__get__(
        sine_generator,
        type(sine_generator),
    )
    sine_generator.forward = sine_gen_forward.__get__(
        sine_generator,
        type(sine_generator),
    )
    source.forward = source_forward.__get__(source, type(source))


def patch_scatterless_sine_source_for_export(kmodel: torch.nn.Module) -> None:
    sine_generator = kmodel.decoder.generator.m_source.l_sin_gen

    def sine_forward(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        random_tail = torch.rand(
            f0_values.shape[0],
            f0_values.shape[2] - 1,
            device=f0_values.device,
            dtype=f0_values.dtype,
        )
        rand_ini = torch.cat(
            [
                torch.zeros(
                    f0_values.shape[0],
                    1,
                    device=f0_values.device,
                    dtype=f0_values.dtype,
                ),
                random_tail,
            ],
            dim=1,
        )
        first_frame = rad_values[:, :1, :] + rand_ini.unsqueeze(1)
        rad_values = torch.cat([first_frame, rad_values[:, 1:, :]], dim=1)
        rad_values = functional.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = functional.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    sine_generator._f02sine = sine_forward.__get__(
        sine_generator,
        type(sine_generator),
    )


def patch_split_adain_for_export(kmodel: torch.nn.Module) -> None:
    for module in kmodel.modules():
        if module.__class__.__name__ == "AdaIN1d":
            split_adain1d_projection(module)
        elif module.__class__.__name__ == "AdaLayerNorm":
            split_adalayernorm_projection(module)


def patch_albert_sdpa_bool_mask_scale_for_export() -> None:
    from transformers import AlbertModel  # noqa: PLC0415
    from transformers.modeling_outputs import (
        BaseModelOutputWithPooling,  # noqa: PLC0415
    )
    from transformers.models.albert.modeling_albert import (  # noqa: PLC0415
        AlbertAttention,
        AlbertEmbeddings,
        AlbertSdpaAttention,
    )

    class AlbertExportTokenTypeIds(torch.autograd.Function):
        @staticmethod
        def forward(ctx, position_ids, batch_size: int, seq_length: int):
            del ctx
            return torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device=position_ids.device,
            )

        @staticmethod
        def symbolic(g, position_ids, batch_size, seq_length):
            del batch_size
            bucket = int(seq_length)
            zero_buffer = g.op(
                "Constant",
                value_t=torch.zeros((1, 512), dtype=torch.long),
            )
            shape = g.op(
                "Constant",
                value_t=torch.tensor([1, -1], dtype=torch.long),
            )
            rank = g.op("Constant", value_t=torch.tensor([2], dtype=torch.long))
            ones = g.op(
                "ConstantOfShape",
                rank,
                value_t=torch.tensor([1], dtype=torch.long),
            )
            neg_one = g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long))
            neg_ones = g.op("Mul", ones, neg_one)
            equal = g.op("Equal", shape, neg_ones)
            expand_shape = g.op("Where", equal, ones, shape)
            expanded = g.op("Expand", zero_buffer, expand_shape)
            gathered = g.op("GatherElements", expanded, position_ids, axis_i=1)

            final_shape = g.op(
                "Constant",
                value_t=torch.tensor([1, bucket], dtype=torch.long),
            )
            final_rank = g.op("Constant", value_t=torch.tensor([2], dtype=torch.long))
            final_ones = g.op(
                "ConstantOfShape",
                final_rank,
                value_t=torch.tensor([1], dtype=torch.long),
            )
            final_neg_ones = g.op("Mul", final_ones, neg_one)
            final_equal = g.op("Equal", final_shape, final_neg_ones)
            final_expand_shape = g.op("Where", final_equal, final_ones, final_shape)
            return g.op("Expand", gathered, final_expand_shape)

    class AlbertExportBoolMask4d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attention_mask, batch_size: int, seq_length: int):
            del ctx
            keep_mask = attention_mask.to(dtype=torch.bool)
            query_keep = (
                torch.arange(seq_length, device=attention_mask.device).view(
                    1,
                    1,
                    seq_length,
                    1,
                )
                >= 0
            )
            return (query_keep & keep_mask[:, None, None, :]).expand(
                batch_size,
                1,
                seq_length,
                seq_length,
            )

        @staticmethod
        def symbolic(g, attention_mask, batch_size, seq_length):
            del batch_size
            bucket = int(seq_length)
            empty_shape = g.op(
                "Constant",
                value_t=torch.empty(0, dtype=torch.long),
            )
            true_scalar = g.op(
                "ConstantOfShape",
                empty_shape,
                value_t=torch.tensor([True], dtype=torch.bool),
            )
            query_positions = g.op(
                "Constant",
                value_t=torch.arange(bucket, dtype=torch.long).view(
                    1,
                    1,
                    bucket,
                    1,
                ),
            )
            zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.long))
            query_keep = g.op("GreaterOrEqual", query_positions, zero)
            query_keep = g.op("Cast", query_keep, to_i=9)
            query_keep = g.op("And", true_scalar, query_keep)

            mask_shape = g.op("Shape", attention_mask)
            src_dim = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
            src_len = g.op("Gather", mask_shape, src_dim, axis_i=0)
            flat_mask = g.op("Flatten", attention_mask, axis_i=2)
            batch_zeros = g.op(
                "Constant",
                value_t=torch.zeros((1, 1, 1, 1), dtype=torch.long),
            )
            batch_offsets = g.op("Mul", batch_zeros, src_len)
            source_positions = g.op(
                "Constant",
                value_t=torch.arange(bucket, dtype=torch.long).view(
                    1,
                    1,
                    1,
                    bucket,
                ),
            )
            gather_indices = g.op("Add", source_positions, batch_offsets)
            gathered = g.op("Gather", flat_mask, gather_indices, axis_i=0)
            gather_shape = g.op("Shape", gather_indices)
            flatten_shape = g.op(
                "Constant",
                value_t=torch.tensor([-1], dtype=torch.long),
            )
            gathered_flat = g.op("Reshape", gathered, flatten_shape)
            reshape_shape = g.op("Concat", gather_shape, axis_i=0)
            gathered_mask = g.op("Reshape", gathered_flat, reshape_shape)
            gathered_mask = g.op("Cast", gathered_mask, to_i=9)
            keep_mask = g.op("And", query_keep, gathered_mask)

            expand_shape = g.op(
                "Constant",
                value_t=torch.tensor([1, -1, bucket, bucket], dtype=torch.long),
            )
            rank = g.op("Constant", value_t=torch.tensor([4], dtype=torch.long))
            ones = g.op(
                "ConstantOfShape",
                rank,
                value_t=torch.tensor([1], dtype=torch.long),
            )
            neg_one = g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long))
            neg_ones = g.op("Mul", ones, neg_one)
            equal = g.op("Equal", expand_shape, neg_ones)
            final_shape = g.op("Where", equal, ones, expand_shape)
            return g.op("Expand", keep_mask, final_shape)

    def embeddings_forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]
        if token_type_ids is None:
            token_type_ids = AlbertExportTokenTypeIds.apply(
                position_ids,
                int(input_shape[0]),
                int(seq_length),
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

    def sdpa_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        if (
            self.position_embedding_type != "absolute"
            or output_attentions
            or head_mask is not None
        ):
            return AlbertAttention.forward(
                self,
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions,
            )

        batch_size, seq_len, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if (
            getattr(self, "require_contiguous_qkv", False)
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.attention_head_size),
        )
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(
            batch_size,
            seq_len,
            self.all_head_size,
        )
        projected_context_layer = self.dense(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(
            hidden_states + projected_context_layer_dropout
        )
        return (layernormed_context_layer,)

    def albert_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape,
                device=device,
                dtype=torch.long,
            )

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        use_sdpa_attention_mask = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        if use_sdpa_attention_mask:
            extended_attention_mask = AlbertExportBoolMask4d.apply(
                attention_mask.to(dtype=torch.bool),
                int(batch_size),
                int(seq_length),
            )
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                self.dtype
            ).min

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler_activation(self.pooler(sequence_output[:, 0]))
            if self.pooler is not None
            else None
        )
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    AlbertSdpaAttention.forward = sdpa_forward
    AlbertEmbeddings.forward = embeddings_forward
    AlbertModel.forward = albert_forward


def cleanup_albert_sdpa_bool_mask_scale_export(model: onnx.ModelProto) -> None:
    replacements = {
        "/bert/embeddings/Cast_output_0": "/bert/embeddings/Expand_1_output_0",
        "/bert/embeddings/Constant_5_output_0": (
            "/bert/embeddings/Constant_4_output_0"
        ),
        "/bert/embeddings/Constant_9_output_0": (
            "/bert/embeddings/Constant_8_output_0"
        ),
        "/bert/Constant_10_output_0": "/bert/Constant_9_output_0",
    }
    remove_node_names = {
        "/bert/embeddings/Cast",
        "/bert/embeddings/Constant_5",
        "/bert/embeddings/Constant_9",
        "/bert/Constant_10",
    }
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]

    kept_nodes = [
        node for node in model.graph.node if node.name not in remove_node_names
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def split_adain1d_projection(module: torch.nn.Module) -> None:
    channels = module.fc.out_features // 2
    module.fc_gamma = build_linear_slice(module.fc, 0, channels)
    module.fc_beta = build_linear_slice(module.fc, channels, channels * 2)

    def forward(self, x, s):
        gamma = self.fc_gamma(s).view(s.shape[0], channels, 1)
        beta = self.fc_beta(s).view(s.shape[0], channels, 1)
        return (1 + gamma) * self.norm(x) + beta

    module.forward = forward.__get__(module, type(module))


def split_adalayernorm_projection(module: torch.nn.Module) -> None:
    channels = module.channels
    module.fc_gamma = build_linear_slice(module.fc, 0, channels)
    module.fc_beta = build_linear_slice(module.fc, channels, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        gamma = self.fc_gamma(s).view(s.shape[0], channels, 1).transpose(1, -1)
        beta = self.fc_beta(s).view(s.shape[0], channels, 1).transpose(1, -1)
        x = functional.layer_norm(x, (channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

    module.forward = forward.__get__(module, type(module))


def build_linear_slice(
    linear: torch.nn.Linear,
    start: int,
    end: int,
) -> torch.nn.Linear:
    sliced = torch.nn.Linear(
        linear.in_features,
        end - start,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        sliced.weight.copy_(linear.weight[start:end])
        if linear.bias is not None:
            sliced.bias.copy_(linear.bias[start:end])
    return sliced


if __name__ == "__main__":
    raise SystemExit(main())
