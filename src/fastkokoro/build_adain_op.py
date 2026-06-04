from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the optional native AdaIN ONNX Runtime custom op."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output shared library path. Defaults under FASTKOKORO_CACHE_DIR/native.",
    )
    parser.add_argument(
        "--cc",
        default=os.getenv("CC", "gcc"),
        help="C compiler executable. Defaults to CC or gcc.",
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Build without OpenMP. Useful when the platform compiler lacks OpenMP.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print an export line for FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY.",
    )
    args = parser.parse_args()

    output = build_adain_custom_op(
        output=args.output,
        cc=args.cc,
        openmp=not args.no_openmp,
    )

    print(output)
    if args.print_env:
        quoted = shlex.quote(str(output))
        print(f"export FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY={quoted}")


def build_command(*, output: Path, cc: str, openmp: bool) -> list[str]:
    native_dir = Path(__file__).parent / "native"
    source = native_dir / "adain_custom_op.c"
    include_dir = native_dir
    command = [
        cc,
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        f"-I{include_dir}",
        str(source),
        "-o",
        str(output),
        "-lm",
    ]
    if openmp:
        command.insert(3, "-fopenmp")
    return command


def build_adain_custom_op(
    *, output: Path | None = None, cc: str = "gcc", openmp: bool = True
) -> Path:
    output = output or default_output_path(default_cache_dir())
    output.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(output=output, cc=cc, openmp=openmp)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if not openmp:
            raise
        fallback = build_command(output=output, cc=cc, openmp=False)
        print(
            "OpenMP build failed; retrying without -fopenmp.",
            file=sys.stderr,
        )
        subprocess.run(fallback, check=True)
    return output


def default_output_path(cache_dir: Path) -> Path:
    return cache_dir / "native" / f"libfastkokoro_adain{shared_library_suffix()}"


def default_cache_dir() -> Path:
    return Path(os.getenv("FASTKOKORO_CACHE_DIR", "~/.cache/fastkokoro")).expanduser()


def shared_library_suffix() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return ".dylib"
    if system == "windows":
        return ".dll"
    return ".so"


if __name__ == "__main__":
    main()
