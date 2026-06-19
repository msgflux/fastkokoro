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
        static_alignment: bool = False,
        length_aware: bool = False,
    ) -> None:
        super().__init__()
        self.kmodel = kmodel
        self.fixed_output_samples = fixed_output_samples
        self.fixed_alignment_frames = fixed_alignment_frames
        self.static_alignment = static_alignment
        self.length_aware = length_aware

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
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
        return waveform, duration

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
        audio = self.kmodel.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
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
        pred_aln_trg = self.build_alignment(token_count, pred_dur)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = self.kmodel.predictor.F0Ntrain(en, s)
        t_en = self.kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.kmodel.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def build_alignment(
        self,
        token_count: int,
        pred_dur: torch.LongTensor,
    ) -> torch.FloatTensor:
        indices = torch.repeat_interleave(
            torch.arange(token_count, device=self.kmodel.device),
            pred_dur,
        )
        if self.fixed_alignment_frames is None:
            frame_count = indices.shape[0]
            frame_indices = torch.arange(frame_count, device=self.kmodel.device)
            pred_aln_trg = torch.zeros(
                (token_count, frame_count),
                device=self.kmodel.device,
            )
            pred_aln_trg[indices, frame_indices] = 1
            return pred_aln_trg.unsqueeze(0).to(self.kmodel.device)

        frame_count = self.fixed_alignment_frames
        indices = indices[:frame_count]
        frame_indices = torch.arange(indices.shape[0], device=self.kmodel.device)
        pred_aln_trg = torch.zeros(
            (token_count, frame_count),
            device=self.kmodel.device,
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
            "Export input_lengths and mask padded bucket slots from "
            "duration/alignment."
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
    model = KokoroTTFCExportWrapper(
        kmodel,
        fixed_output_samples=args.fixed_output_samples,
        fixed_alignment_frames=args.fixed_alignment_frames,
        static_alignment=args.static_alignment,
        length_aware=args.length_aware,
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


if __name__ == "__main__":
    raise SystemExit(main())
