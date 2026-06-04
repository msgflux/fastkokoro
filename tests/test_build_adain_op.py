from unittest.mock import Mock, patch

import pytest

from fastkokoro import build_adain_op


def test_default_output_path_uses_platform_suffix(tmp_path):
    output = build_adain_op.default_output_path(tmp_path)

    assert output.parent == tmp_path / "native"
    assert output.name.startswith("libfastkokoro_adain")
    assert output.suffix in {".dll", ".dylib", ".so"}


def test_build_command_includes_packaged_native_sources(tmp_path):
    output = tmp_path / "libfastkokoro_adain.so"

    command = build_adain_op.build_command(output=output, cc="cc", openmp=True)

    assert command[0] == "cc"
    assert "-fopenmp" in command
    assert str(output) in command
    assert any(value.endswith("adain_custom_op.c") for value in command)
    assert any(value.startswith("-I") and value.endswith("native") for value in command)


def test_main_retries_without_openmp(tmp_path):
    output = tmp_path / "libfastkokoro_adain.so"
    calls = []

    def fake_run(command, check):
        calls.append(command)
        if len(calls) == 1:
            raise build_adain_op.subprocess.CalledProcessError(1, command)
        return Mock()

    with (
        patch("fastkokoro.build_adain_op.subprocess.run", fake_run),
        patch(
            "fastkokoro.build_adain_op.sys.argv",
            ["fastkokoro-build-adain-op", "--output", str(output)],
        ),
    ):
        build_adain_op.main()

    assert "-fopenmp" in calls[0]
    assert "-fopenmp" not in calls[1]


def test_main_keeps_no_openmp_failures(tmp_path):
    output = tmp_path / "libfastkokoro_adain.so"

    with (
        patch(
            "fastkokoro.build_adain_op.subprocess.run",
            side_effect=build_adain_op.subprocess.CalledProcessError(1, "cc"),
        ),
        patch(
            "fastkokoro.build_adain_op.sys.argv",
            ["fastkokoro-build-adain-op", "--no-openmp", "--output", str(output)],
        ),
        pytest.raises(build_adain_op.subprocess.CalledProcessError),
    ):
        build_adain_op.main()
