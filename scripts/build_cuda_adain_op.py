from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
from contextlib import ExitStack
from importlib import resources
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the experimental CUDA AdaIN ONNX Runtime custom op."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output shared library path. Defaults under FASTKOKORO_CACHE_DIR/native.",
    )
    parser.add_argument(
        "--nvcc",
        default=os.getenv("NVCC", "nvcc"),
        help="CUDA compiler executable. Defaults to NVCC or nvcc.",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=None,
        help=(
            "NVCC gencode architecture, e.g. sm_86. Can be repeated. "
            "Defaults to FASTKOKORO_CUDA_ARCH or sm_75,sm_86."
        ),
    )
    parser.add_argument(
        "--ort-api-version",
        type=int,
        default=int(os.getenv("FASTKOKORO_ORT_API_VERSION", "18")),
        help=(
            "ONNX Runtime C API version to request. Defaults to 18 for "
            "onnxruntime-gpu 1.18.x compatibility."
        ),
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print an export line for FASTKOKORO_ONNX_CUDA_ADAIN_CUSTOM_OP_LIBRARY.",
    )
    args = parser.parse_args()

    output = build_cuda_adain_custom_op(
        output=args.output,
        nvcc=args.nvcc,
        archs=tuple(args.arch or parse_archs(os.getenv("FASTKOKORO_CUDA_ARCH"))),
        ort_api_version=args.ort_api_version,
    )

    print(output)
    if args.print_env:
        quoted = shlex.quote(str(output))
        print(f"export FASTKOKORO_ONNX_CUDA_ADAIN_CUSTOM_OP_LIBRARY={quoted}")


def build_cuda_adain_custom_op(
    *,
    output: Path | None = None,
    nvcc: str = "nvcc",
    archs: tuple[str, ...] = ("sm_75", "sm_86"),
    ort_api_version: int = 18,
) -> Path:
    output = output or default_output_path(default_cache_dir())
    output.parent.mkdir(parents=True, exist_ok=True)

    with native_paths() as (source, include_dir):
        command = build_command(
            output=output,
            nvcc=nvcc,
            source=source,
            include_dir=include_dir,
            archs=archs,
            ort_api_version=ort_api_version,
        )
        subprocess.run(command, check=True)
    return output


def build_command(
    *,
    output: Path,
    nvcc: str,
    source: Path,
    include_dir: Path,
    archs: tuple[str, ...],
    ort_api_version: int,
) -> list[str]:
    command = [
        nvcc,
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        f"-DFASTKOKORO_ORT_API_VERSION={ort_api_version}",
        f"-I{include_dir}",
        str(source),
        "-o",
        str(output),
    ]
    for arch in archs:
        code = arch.removeprefix("sm_")
        command.extend(["-gencode", f"arch=compute_{code},code={arch}"])
    return command


class native_paths:
    def __enter__(self) -> tuple[Path, Path]:
        self._stack = ExitStack()
        native = resources.files("fastkokoro.native")
        source = self._stack.enter_context(
            resources.as_file(native / "adain_cuda_custom_op.cu")
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
    return cache_dir / "native" / f"libfastkokoro_cuda_adain{shared_library_suffix()}"


def default_cache_dir() -> Path:
    return Path(os.getenv("FASTKOKORO_CACHE_DIR", "~/.cache/fastkokoro")).expanduser()


def shared_library_suffix() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return ".dylib"
    if system == "windows":
        return ".dll"
    return ".so"


def parse_archs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("sm_75", "sm_86")
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    main()
