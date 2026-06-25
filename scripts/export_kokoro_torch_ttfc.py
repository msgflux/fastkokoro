from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch
import torch.nn.functional as functional

BLOCKED_TTFC_OPS = {"NonZero", "ScatterND", "STFT", "Range"}


class KokoroTTFCExportWrapper(torch.nn.Module):
    def __init__(
        self,
        kmodel: torch.nn.Module,
        *,
        fixed_output_samples: int | None = None,
        fixed_alignment_frames: int | None = None,
        output_samples_per_frame: int | None = None,
        output_fade_samples: int = 0,
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
        self.static_alignment = static_alignment
        self.length_aware = length_aware
        self.internal_dtype = internal_dtype
        self.decoder_dtype = decoder_dtype

    def forward(
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
        if self.fixed_output_samples is not None:
            waveform = functional.pad(waveform, (0, self.fixed_output_samples))
            waveform = waveform[..., : self.fixed_output_samples]
            if self.output_samples_per_frame is not None:
                waveform = self.mask_waveform_to_duration(waveform, duration)
        return waveform.float(), duration

    def mask_waveform_to_duration(
        self,
        waveform: torch.FloatTensor,
        duration: torch.LongTensor,
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
        positions = torch.arange(sample_count, device=waveform.device)
        if self.output_fade_samples <= 0:
            return waveform * (positions < active_samples).to(waveform.dtype)

        fade_start = active_samples - self.output_fade_samples
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
        f0_pred, n_pred = self.kmodel.predictor.F0Ntrain(en, s)
        t_en = self.kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.run_decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

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
        x, _ = self.kmodel.predictor.lstm(d)
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
        f0_pred, n_pred = self.kmodel.predictor.F0Ntrain(en, s)
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
        help="Patch Kokoro text LSTMs to avoid pack_padded_sequence for fixed shapes.",
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
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for export tracing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bucket <= 2:
        raise ValueError("--bucket must leave room for start/end pad tokens")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    add_checkout_to_path(args.kokoro_repo)
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
        patch_fixed_lstm_for_export(kmodel)
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
        waveform, duration = model(*export_args)
    print(f"torch_waveform_shape={tuple(waveform.shape)}")
    print(f"torch_duration_shape={tuple(duration.shape)}")

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
    onnx.checker.check_model(exported)
    ops = Counter(node.op_type for node in exported.graph.node)
    print(f"onnx_path={args.output}")
    print(f"onnx_nodes={len(exported.graph.node)}")
    top_ops = ", ".join(f"{op}:{count}" for op, count in ops.most_common(20))
    print(f"onnx_top_ops={top_ops}")
    blocked = {op: ops[op] for op in sorted(BLOCKED_TTFC_OPS) if ops[op]}
    print(f"blocked_ops={blocked}")
    return 2 if blocked else 0


def add_checkout_to_path(path: Path) -> None:
    if not (path / "kokoro" / "model.py").exists():
        raise FileNotFoundError(f"Invalid kokoro checkout: {path}")
    sys.path.insert(0, str(path.resolve()))
    if importlib.util.find_spec("kokoro") is None:
        raise RuntimeError(f"Unable to import kokoro from {path}")


def patch_fixed_lstm_for_export(kmodel: torch.nn.Module) -> None:
    import torch.nn.functional as functional  # noqa: PLC0415
    from kokoro.modules import AdaLayerNorm  # noqa: PLC0415

    def text_encoder_forward(self, x, input_lengths, m):
        del input_lengths
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x = x.masked_fill(m, 0.0)
        for conv in self.cnn:
            x = conv(x)
            x = x.masked_fill(m, 0.0)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x = x.masked_fill(m, 0.0)
        return x

    def duration_encoder_forward(self, x, style, text_lengths, m):
        del text_lengths
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
                block.flatten_parameters()
                x, _ = block(x)
                x = functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

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
