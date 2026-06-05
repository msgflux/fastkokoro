from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
from contextlib import ExitStack
from importlib import resources
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the optional native Conv1dAdaIn ONNX Runtime custom op."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output shared library path. Defaults under FASTKOKORO_CACHE_DIR/native."
        ),
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
        help="Print an export line for FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY.",
    )
    args = parser.parse_args()

    output = build_conv_adain_custom_op(
        output=args.output,
        cc=args.cc,
        openmp=not args.no_openmp,
    )

    print(output)
    if args.print_env:
        quoted = shlex.quote(str(output))
        print(f"export FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY={quoted}")


def build_conv_adain_custom_op(
    *, output: Path | None = None, cc: str = "gcc", openmp: bool = True
) -> Path:
    output = output or default_output_path(default_cache_dir())
    output.parent.mkdir(parents=True, exist_ok=True)

    with native_paths() as (source, include_dir):
        command = _build_command(
            output=output,
            cc=cc,
            openmp=openmp,
            source=source,
            include_dir=include_dir,
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if not openmp:
                raise
            fallback = _build_command(
                output=output,
                cc=cc,
                openmp=False,
                source=source,
                include_dir=include_dir,
            )
            print(
                "OpenMP build failed; retrying without -fopenmp.",
                file=sys.stderr,
            )
            subprocess.run(fallback, check=True)
    return output


def _build_command(
    *,
    output: Path,
    cc: str,
    openmp: bool,
    source: Path,
    include_dir: Path,
) -> list[str]:
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


class native_paths:
    def __enter__(self) -> tuple[Path, Path]:
        self._stack = ExitStack()
        native = resources.files("fastkokoro.native")
        source = self._stack.enter_context(
            resources.as_file(native / "conv_adain_custom_op.c")
        )
        include_header = self._stack.enter_context(
            resources.as_file(native / "onnxruntime_c_api.h")
        )
        self._stack.enter_context(resources.as_file(native / "onnxruntime_ep_c_api.h"))
        return source, include_header.parent

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._stack.close()
        return False


def default_output_path(cache_dir: Path) -> Path:
    return cache_dir / "native" / f"libfastkokoro_conv_adain{shared_library_suffix()}"


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
