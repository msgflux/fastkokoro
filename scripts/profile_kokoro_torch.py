from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastkokoro.assets import resolve_voices_path  # noqa: E402
from fastkokoro.config import Settings  # noqa: E402
from fastkokoro.kokoro import Tokenizer  # noqa: E402
from scripts.export_kokoro_torch_ttfc import (  # noqa: E402
    FixedLengthAwareBiLSTM,
    patch_fixed_lstm_for_export,
)


@dataclass(frozen=True)
class TimedResult:
    audio: torch.Tensor
    duration: torch.Tensor
    timings_ms: dict[str, float]


class TimedKokoro(torch.nn.Module):
    def __init__(
        self,
        kmodel: torch.nn.Module,
        *,
        length_aware: bool,
        generator_kernels: int | None = None,
        skip_noise_res: bool = False,
    ) -> None:
        super().__init__()
        self.kmodel = kmodel
        self.length_aware = length_aware
        self.generator_kernels = generator_kernels
        self.skip_noise_res = skip_noise_res

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> TimedResult:
        timings: dict[str, float] = {}
        batch_size, token_count = input_ids.shape
        positions = torch.arange(token_count, device=input_ids.device)
        if self.length_aware:
            text_mask = positions.unsqueeze(0).expand(batch_size, -1) >= (
                input_lengths.unsqueeze(1)
            )
        else:
            full_lengths = torch.full(
                (batch_size,),
                token_count,
                device=input_ids.device,
                dtype=torch.long,
            )
            input_lengths = full_lengths
            text_mask = positions.unsqueeze(0).expand(batch_size, -1) >= (
                input_lengths.unsqueeze(1)
            )

        bert_dur = self._time(
            timings,
            "bert",
            lambda: self.kmodel.bert(input_ids, attention_mask=(~text_mask).int()),
        )
        d_en = self._time(
            timings,
            "bert_encoder",
            lambda: self.kmodel.bert_encoder(bert_dur).transpose(-1, -2),
        )
        s = ref_s[:, 128:]
        d = self._time(
            timings,
            "duration_text_encoder",
            lambda: self.kmodel.predictor.text_encoder(
                d_en,
                s,
                input_lengths,
                text_mask,
            ),
        )
        x = self._time(
            timings,
            "duration_lstm",
            lambda: self._duration_lstm(d, input_lengths),
        )
        duration = self._time(
            timings,
            "duration_projection",
            lambda: self.kmodel.predictor.duration_proj(x),
        )
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if self.length_aware:
            pred_dur = torch.where(
                text_mask.squeeze(0),
                torch.zeros_like(pred_dur),
                pred_dur,
            )
        pred_aln_trg = self._time(
            timings,
            "alignment",
            lambda: self._build_alignment(token_count, pred_dur),
        )
        en = self._time(
            timings,
            "duration_to_frames",
            lambda: d.transpose(-1, -2) @ pred_aln_trg,
        )
        f0_pred, n_pred = self._time(
            timings,
            "f0_noise",
            lambda: self.kmodel.predictor.F0Ntrain(en, s),
        )
        t_en = self._time(
            timings,
            "text_encoder",
            lambda: self.kmodel.text_encoder(input_ids, input_lengths, text_mask),
        )
        asr = self._time(timings, "text_to_frames", lambda: t_en @ pred_aln_trg)
        audio = self._decoder(timings, asr, f0_pred, n_pred, ref_s[:, :128])
        return TimedResult(audio=audio, duration=pred_dur, timings_ms=timings)

    def _decoder(
        self,
        timings: dict[str, float],
        asr: torch.Tensor,
        f0_pred: torch.Tensor,
        noise_pred: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        decoder = self.kmodel.decoder
        f0 = self._time(
            timings,
            "decoder_f0_conv",
            lambda: decoder.F0_conv(f0_pred.unsqueeze(1)),
        )
        noise = self._time(
            timings,
            "decoder_noise_conv",
            lambda: decoder.N_conv(noise_pred.unsqueeze(1)),
        )
        x = torch.cat([asr, f0, noise], axis=1)
        x = self._time(timings, "decoder_encode", lambda: decoder.encode(x, style))
        asr_res = self._time(timings, "decoder_asr_res", lambda: decoder.asr_res(asr))
        res = True
        for index, block in enumerate(decoder.decode):
            if res:
                x = torch.cat([x, asr_res, f0, noise], axis=1)
            x = self._time(
                timings,
                f"decoder_decode_{index}",
                lambda block=block, x=x: block(x, style),
            )
            if block.upsample_type != "none":
                res = False
        return self._generator(timings, x, style, f0_pred).squeeze()

    def _generator(
        self,
        timings: dict[str, float],
        x: torch.Tensor,
        style: torch.Tensor,
        f0_pred: torch.Tensor,
    ) -> torch.Tensor:
        generator = self.kmodel.decoder.generator

        def build_harmonic():
            with torch.no_grad():
                f0 = generator.f0_upsamp(f0_pred[:, None]).transpose(1, 2)
                har_source, _, _ = generator.m_source(f0)
                har_source = har_source.transpose(1, 2).squeeze(1)
                har_spec, har_phase = generator.stft.transform(har_source)
                return torch.cat([har_spec, har_phase], dim=1)

        harmonic = self._time(timings, "generator_harmonic_stft", build_harmonic)
        for index in range(generator.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self._time(
                timings,
                f"generator_noise_conv_{index}",
                lambda index=index: generator.noise_convs[index](harmonic),
            )
            if not self.skip_noise_res:
                x_source = self._time(
                    timings,
                    f"generator_noise_res_{index}",
                    lambda index=index, x_source=x_source: generator.noise_res[index](
                        x_source,
                        style,
                    ),
                )
            x = self._time(
                timings,
                f"generator_upsample_{index}",
                lambda index=index, x=x: generator.ups[index](x),
            )
            if index == generator.num_upsamples - 1:
                x = generator.reflection_pad(x)
            x = x + x_source
            xs = None
            kernel_count = self.generator_kernels or generator.num_kernels
            for kernel_index in range(kernel_count):
                block_index = index * generator.num_kernels + kernel_index
                block_out = self._time(
                    timings,
                    f"generator_resblock_{index}_{kernel_index}",
                    lambda block_index=block_index, x=x: generator.resblocks[
                        block_index
                    ](x, style),
                )
                xs = block_out if xs is None else xs + block_out
            x = xs / kernel_count
        x = torch.nn.functional.leaky_relu(x)
        x = self._time(timings, "generator_conv_post", lambda: generator.conv_post(x))
        spec = torch.exp(x[:, : generator.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, generator.post_n_fft // 2 + 1 :, :])
        return self._time(
            timings,
            "generator_inverse_stft",
            lambda: generator.stft.inverse(spec, phase),
        )

    def _duration_lstm(
        self,
        d: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        lstm = self.kmodel.predictor.lstm
        if isinstance(lstm, FixedLengthAwareBiLSTM):
            x, _ = lstm(d, input_lengths)
            return x
        lstm.flatten_parameters()
        x, _ = lstm(d)
        return x

    def _build_alignment(
        self,
        token_count: int,
        pred_dur: torch.Tensor,
    ) -> torch.Tensor:
        indices = torch.repeat_interleave(
            torch.arange(token_count, device=self.kmodel.device),
            pred_dur,
        )
        pred_aln_trg = torch.zeros(
            (token_count, indices.shape[0]),
            device=self.kmodel.device,
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        return pred_aln_trg.unsqueeze(0).to(self.kmodel.device)

    @staticmethod
    def _time(
        timings: dict[str, float],
        name: str,
        fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ):
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) * 1000
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Kokoro PyTorch submodules.")
    parser.add_argument(
        "--kokoro-repo",
        type=Path,
        default=Path("demo-output/reexport/hexgrad-kokoro"),
    )
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text", default="Oi, tudo bem?")
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="pt-br")
    parser.add_argument("--bucket", type=int, default=24)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--length-aware", action="store_true")
    parser.add_argument("--patch-fixed-lstm", action="store_true")
    parser.add_argument(
        "--generator-kernels",
        type=int,
        choices=(1, 2, 3),
        help="Use only the first N generator resblocks per upsample.",
    )
    parser.add_argument("--skip-noise-res", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    add_checkout_to_path(args.kokoro_repo)
    from kokoro import KModel  # noqa: PLC0415

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    kwargs: dict[str, object] = {"repo_id": args.repo_id, "disable_complex": True}
    if args.config is not None:
        kwargs["config"] = str(args.config)
    if args.checkpoint is not None:
        kwargs["model"] = str(args.checkpoint)
    kmodel = KModel(**kwargs).eval().to(device)
    if args.patch_fixed_lstm:
        patch_fixed_lstm_for_export(kmodel)

    model = TimedKokoro(
        kmodel,
        length_aware=args.length_aware,
        generator_kernels=args.generator_kernels,
        skip_noise_res=args.skip_noise_res,
    ).eval()
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    input_ids, ref_s, speed, input_lengths = build_inputs(args, device)
    with torch.inference_mode():
        for _ in range(args.warmups):
            model(input_ids, ref_s, speed, input_lengths)
        rows = []
        for _ in range(args.runs):
            start = time.perf_counter()
            result = model(input_ids, ref_s, speed, input_lengths)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1000
            rows.append((total_ms, result))

    totals = np.array([row[0] for row in rows], dtype=np.float64)
    keys = rows[-1][1].timings_ms.keys()
    print(f"text={args.text!r} device={args.device} bucket={args.bucket}")
    print(
        "mode="
        f"length_aware={args.length_aware} "
        f"patch_fixed_lstm={args.patch_fixed_lstm} "
        f"generator_kernels={args.generator_kernels} "
        f"skip_noise_res={args.skip_noise_res} "
        f"compile={args.compile}"
    )
    print(f"duration={rows[-1][1].duration.cpu().numpy().tolist()}")
    print(f"audio_samples={rows[-1][1].audio.numel()}")
    print(
        f"total_ms median={np.median(totals):.2f} "
        f"min={totals.min():.2f} max={totals.max():.2f}"
    )
    for key in keys:
        values = np.array([row[1].timings_ms[key] for row in rows], dtype=np.float64)
        print(
            f"{key:24s} median={np.median(values):8.2f} "
            f"min={values.min():8.2f} max={values.max():8.2f}"
        )
    return 0


def build_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    voices = np.load(resolve_voices_path(Settings.from_env()))
    tokenizer = Tokenizer()
    phonemes = tokenizer.phonemize(args.text, lang=args.lang)
    tokens = tokenizer.tokenize(phonemes)
    max_tokens = args.bucket - 2
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    input_ids = torch.zeros((1, args.bucket), dtype=torch.long, device=device)
    input_ids[0, 1 : 1 + len(tokens)] = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    )
    style_index = min(max(len(tokens) - 1, 0), voices[args.voice].shape[0] - 1)
    style = voices[args.voice][style_index]
    ref_s = torch.tensor(style.reshape(1, 256), dtype=torch.float32, device=device)
    speed = torch.ones(1, dtype=torch.float32, device=device)
    input_lengths = torch.tensor([len(tokens) + 2], dtype=torch.long, device=device)
    print(
        f"phonemes={phonemes!r} token_count={len(tokens)} "
        f"input_lengths={input_lengths.cpu().tolist()}"
    )
    return input_ids, ref_s, speed, input_lengths


def add_checkout_to_path(path: Path) -> None:
    if not (path / "kokoro" / "model.py").exists():
        raise FileNotFoundError(f"Invalid kokoro checkout: {path}")
    sys.path.insert(0, str(path.resolve()))
    if importlib.util.find_spec("kokoro") is None:
        raise RuntimeError(f"Unable to import kokoro from {path}")


if __name__ == "__main__":
    raise SystemExit(main())
