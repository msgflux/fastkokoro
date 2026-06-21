from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--top", default=20, type=int)
    args = parser.parse_args()

    model = onnx.load(args.model)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    nodes = [event for event in profile if event.get("cat") == "Node"]

    print("== Top profile groups ==")
    groups, op_totals, op_counts = _aggregate_profile(nodes)
    sorted_groups = sorted(groups.items(), key=lambda item: item[1], reverse=True)
    for name, duration in sorted_groups[: args.top]:
        print(f"{duration:10d} us  {name}")

    print("\n== Generator block internals ==")
    block_totals, block_counts = _aggregate_generator_blocks(nodes)
    for name, duration in sorted(
        block_totals.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{duration:10d} us  {block_counts[name]:5d} events  {name}")

    print("\n== Op totals ==")
    for name, duration in sorted(
        op_totals.items(), key=lambda item: item[1], reverse=True
    )[: args.top]:
        print(f"{duration:10d} us  {op_counts[name]:5d} events  {name}")

    print("\n== Generator Conv shapes ==")
    for node in model.graph.node:
        if not _is_generator_block_conv(node):
            continue
        weight = initializers.get(node.input[1])
        shape = list(weight.dims) if weight is not None else None
        print(f"{node.name}  weight={shape}")

    print("\n== Candidate ranking ==")
    print(
        "1. AdaIN + Snake pre-conv fusion: replaces InstanceNorm/Mul/Add/Sin/Pow/"
        "Reciprocal with one CUDA op while keeping cuDNN Conv."
    )
    print(
        "2. Atan2 phase fusion: removes the CPU Atan boundary; already validated as "
        "a small but real copy reduction."
    )
    print(
        "3. Full residual step fusion: AdaIN + Snake + Conv + residual add. Highest "
        "ceiling, but requires custom specialized Conv1d."
    )
    print(
        "4. Source generator fusion: fuses sine/noise source math. Lower ceiling than "
        "resblocks, but removes dynamic/random-heavy graph islands."
    )


def _aggregate_profile(
    nodes: list[dict],
) -> tuple[dict[str, int], dict[str, int], Counter[str]]:
    groups: dict[str, int] = defaultdict(int)
    op_totals: dict[str, int] = defaultdict(int)
    op_counts: Counter[str] = Counter()
    for event in nodes:
        name = event.get("name", "").removesuffix("_kernel_time")
        op_name = event.get("args", {}).get("op_name") or "None"
        duration = int(event.get("dur", 0))
        groups[_profile_group(name)] += duration
        op_totals[op_name] += duration
        op_counts[op_name] += 1
    return groups, op_totals, op_counts


def _aggregate_generator_blocks(
    nodes: list[dict],
) -> tuple[dict[str, int], Counter[str]]:
    totals: dict[str, int] = defaultdict(int)
    counts: Counter[str] = Counter()
    for event in nodes:
        name = event.get("name", "").removesuffix("_kernel_time")
        op_name = event.get("args", {}).get("op_name") or "None"
        if "/decoder/generator/" not in name:
            continue
        if "/resblocks." in name or "/noise_res." in name:
            category = _block_category(op_name)
        elif "/ups." in name:
            category = "ups ConvTranspose"
        elif "/conv_post" in name:
            category = "conv_post Conv"
        elif "/m_source" in name:
            category = "m_source"
        elif name.startswith("/decoder/generator/Atan") or "Atan2Cuda" in name:
            category = "atan2 phase"
        else:
            continue
        totals[category] += int(event.get("dur", 0))
        counts[category] += 1
    return totals, counts


def _profile_group(name: str) -> str:
    parts = [part for part in name.split("/") if part]
    if not parts:
        return "other"
    if parts[0] == "decoder" and len(parts) > 2 and parts[1] == "generator":
        if len(parts) > 3 and parts[2] in {"resblocks", "noise_res"}:
            return "/".join(parts[:4])
        if len(parts) > 2 and parts[2] in {"ups", "noise_convs"}:
            return "/".join(parts[:4]) if len(parts) > 3 else "/".join(parts[:3])
        return "/".join(parts[:3])
    if parts[0] == "text_encoder" and len(parts) > 1:
        return "/".join(parts[:3]) if len(parts) > 2 else "/".join(parts[:2])
    if parts[0] == "decoder" and len(parts) > 1:
        return "/".join(parts[:3]) if len(parts) > 2 else "/".join(parts[:2])
    return parts[0]


def _block_category(op_name: str) -> str:
    if op_name == "Conv":
        return "block Conv"
    if op_name == "InstanceNormalization":
        return "block InstanceNorm"
    if op_name in {"Gemm", "MatMul"}:
        return "block style Gemm/MatMul"
    if op_name in {"Sin", "Pow", "Reciprocal"}:
        return "block Snake math"
    if op_name in {"Add", "Mul"}:
        return "block elementwise affine/add/mul"
    if op_name in {"Reshape", "Transpose", "Concat", "Cast"}:
        return "block shape/layout"
    return f"block {op_name}"


def _is_generator_block_conv(node: onnx.NodeProto) -> bool:
    return (
        node.op_type == "Conv"
        and "/decoder/generator/" in node.name
        and ("/resblocks." in node.name or "/noise_res." in node.name)
    )


if __name__ == "__main__":
    main()
