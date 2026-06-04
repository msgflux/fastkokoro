from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import onnxruntime as ort

from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro

TEXTS = {
    "tiny": "Ola.",
    "short": "Ola, tudo bem?",
    "medium_first": "Ola, tudo bem?",
    "generator": "em portugues brasileiro.",
}


@dataclass(frozen=True)
class Hotspot:
    name: str
    op_name: str
    calls: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    max_ms: float


@dataclass(frozen=True)
class OpSummary:
    op_name: str
    calls: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    max_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=TEXTS, default="short")
    parser.add_argument("--custom-text")
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="p")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--profile-prefix", default="/tmp/fastkokoro-ort-profile")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = args.custom_text or TEXTS[args.text]
    settings = Settings.from_env()
    engine = FastKokoro(settings)
    engine.session.end_profiling()
    engine.session = _create_profiled_session(
        engine.model_path,
        settings,
        args.profile_prefix,
    )
    engine._onnx_input_names = frozenset(
        item.name for item in engine.session.get_inputs()
    )
    engine._onnx_output_name = engine.session.get_outputs()[0].name
    engine._token_input_name = (
        "input_ids" if "input_ids" in engine._onnx_input_names else "tokens"
    )

    for _ in range(args.iterations):
        engine.create(
            text,
            voice=args.voice,
            lang=args.lang,
            speed=args.speed,
            response_format="pcm",
        )

    profile_path = Path(engine.session.end_profiling())
    events = _load_node_events(profile_path)
    node_hotspots = _summarize_nodes(events)[: args.top]
    op_hotspots = _summarize_ops(events)[: args.top]
    payload = {
        "profile_path": str(profile_path),
        "text": text,
        "iterations": args.iterations,
        "node_hotspots": [asdict(item) for item in node_hotspots],
        "op_hotspots": [asdict(item) for item in op_hotspots],
    }
    if args.json:
        print(json.dumps(payload))
        return

    print(f"profile: {profile_path}")
    print(f"text: {text!r}")
    print("\nTop nodes:")
    for item in node_hotspots:
        print(
            f"{item.total_ms:9.3f} ms  {item.calls:3d}x  "
            f"{item.avg_ms:8.3f} avg  {item.op_name:18s} {item.name}"
        )
    print("\nTop op types:")
    for item in op_hotspots:
        print(
            f"{item.total_ms:9.3f} ms  {item.calls:3d}x  "
            f"{item.avg_ms:8.3f} avg  {item.op_name}"
        )


def _create_profiled_session(
    model_path: Path,
    settings: Settings,
    profile_prefix: str,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = profile_prefix
    options.log_severity_level = settings.onnx_log_severity_level
    options.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[settings.onnx_graph_optimization_level]
    if settings.onnx_intra_op_num_threads is not None:
        options.intra_op_num_threads = settings.onnx_intra_op_num_threads
    if settings.onnx_inter_op_num_threads is not None:
        options.inter_op_num_threads = settings.onnx_inter_op_num_threads
    return ort.InferenceSession(
        str(model_path),
        providers=list(settings.onnx_providers),
        sess_options=options,
    )


def _load_node_events(profile_path: Path) -> list[dict]:
    with profile_path.open() as file:
        events = json.load(file)
    return [
        event
        for event in events
        if event.get("cat") == "Node" and event.get("dur", 0) > 0
    ]


def _event_op_name(event: dict) -> str:
    args = event.get("args", {})
    return args.get("op_name") or args.get("provider") or "unknown"


def _summarize_nodes(events: list[dict]) -> list[Hotspot]:
    durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        durations[(event.get("name", "unknown"), _event_op_name(event))].append(
            event["dur"] / 1000
        )
    return sorted(
        (
            Hotspot(
                name=name,
                op_name=op_name,
                calls=len(values),
                total_ms=sum(values),
                avg_ms=statistics.fmean(values),
                p50_ms=statistics.median(values),
                max_ms=max(values),
            )
            for (name, op_name), values in durations.items()
        ),
        key=lambda item: item.total_ms,
        reverse=True,
    )


def _summarize_ops(events: list[dict]) -> list[OpSummary]:
    durations: dict[str, list[float]] = defaultdict(list)
    calls: Counter[str] = Counter()
    for event in events:
        op_name = _event_op_name(event)
        durations[op_name].append(event["dur"] / 1000)
        calls[op_name] += 1
    return sorted(
        (
            OpSummary(
                op_name=op_name,
                calls=calls[op_name],
                total_ms=sum(values),
                avg_ms=statistics.fmean(values),
                p50_ms=statistics.median(values),
                max_ms=max(values),
            )
            for op_name, values in durations.items()
        ),
        key=lambda item: item.total_ms,
        reverse=True,
    )


if __name__ == "__main__":
    main()
