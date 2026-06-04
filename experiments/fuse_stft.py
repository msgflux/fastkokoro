from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def fuse(input_path: Path, output_path: Path) -> int:
    model = onnx.load(input_path)
    fused = 0
    nodes = []
    for node in model.graph.node:
        if node.op_type == "STFT" and node.name == "/decoder/decoder/generator/STFT":
            nodes.append(
                helper.make_node(
                    "KokoroSTFT",
                    inputs=[node.input[0], node.input[2]],
                    outputs=list(node.output),
                    name=node.name + "_KokoroSTFT",
                    domain="fastkokoro",
                    frame_step=5,
                )
            )
            fused += 1
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if fused == 0:
        raise ValueError("No generator STFT node found")
    onnx.save_model(model, output_path, save_as_external_data=True)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"fused={fuse(args.input, args.output)} output={args.output}")


if __name__ == "__main__":
    main()
