from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_job_runner_submits_ordinary_job_with_stage_and_editable_consist(
    tmp_path: Path,
) -> None:
    """The wrapper expands settings, exports CONSIST_SRC_DIR, and forwards stage."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    settings = pilates_dir / "settings.yaml"
    settings.write_text("beam:\n  memory: ${BEAM_MEMORY}\n", encoding="utf-8")
    stage = pilates_dir / "restart.yaml"
    stage.write_text("stage: 1\n", encoding="utf-8")
    consist_source = tmp_path / "consist-src"
    consist_source.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "captured-sbatch-args"
    captured_env = tmp_path / "captured-consist-env"
    _write_executable(
        fake_bin / "mkdir",
        """#!/bin/sh
if [ "$2" = "/global/scratch/users/test-user/pilates_logs" ]; then
    exit 0
fi
exec /bin/mkdir "$@"
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/sh
printf '%s\\n' "$@" > "$SBATCH_ARGS_FILE"
printf '%s\\n' "$CONSIST_SRC_DIR" > "$CONSIST_ENV_FILE"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PILATES_DIR": str(pilates_dir),
        "SBATCH_ARGS_FILE": str(captured_args),
        "CONSIST_ENV_FILE": str(captured_env),
        "USER": "test-user",
    }

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-s",
            "restart.yaml",
            "-a",
            "test-account",
            "--editable-consist",
            str(consist_source),
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert captured_env.read_text(encoding="utf-8").strip() == str(consist_source)
    generated_settings = next(pilates_dir.glob("settings_*.yaml"))
    assert generated_settings.read_text(encoding="utf-8") == "beam:\n  memory: 180g\n"
    submitted_args = captured_args.read_text(encoding="utf-8").splitlines()
    assert submitted_args[-3:] == [
        str(pilates_dir / "hpc/job.sh"),
        str(generated_settings),
        str(stage),
    ]


def test_job_runner_rejects_invalid_editable_consist_checkout(
    tmp_path: Path,
) -> None:
    """--editable-consist must be an existing absolute directory."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    (pilates_dir / "settings.yaml").write_text("run: {}\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--editable-consist",
            "relative-consist",
        ],
        cwd=_PROJECT_ROOT,
        env={**os.environ, "PILATES_DIR": str(pilates_dir)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert (
        "--editable-consist must name an existing absolute Consist checkout"
        in completed.stderr
    )


def test_job_runs_ordinary_workflow_entrypoint(tmp_path: Path) -> None:
    """job.sh installs editable Consist when present and invokes run.py."""

    pilates_dir = tmp_path / "pilates"
    requirements = pilates_dir / "hpc" / "requirements-hpc.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("", encoding="utf-8")
    settings = pilates_dir / "settings.yaml"
    settings.write_text("base: base.yaml\n", encoding="utf-8")
    consist_source = tmp_path / "consist"
    consist_source.mkdir()
    venv = tmp_path / "venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    calls = tmp_path / "python-calls"
    _write_executable(
        venv_bin / "python3",
        """#!/bin/sh
printf '%s\\n' "$*" >> "$PYTHON_CALLS_FILE"
""",
    )
    (venv_bin / "activate").write_text(
        f'export PATH="{venv_bin}:$PATH"\n',
        encoding="utf-8",
    )
    (venv / ".last_requirements_hash").write_text(
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "module", "#!/bin/sh\nexit 0\n")

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job.sh"),
            str(settings),
        ],
        cwd=_PROJECT_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PILATES_DIR": str(pilates_dir),
            "PILATES_VENV_PATH": str(venv),
            "PILATES_REQUIREMENTS_FILE": str(requirements),
            "CONSIST_SRC_DIR": str(consist_source),
            "PYTHON_CALLS_FILE": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    python_calls = calls.read_text(encoding="utf-8").splitlines()
    assert any(
        call == f"-m pip install --no-build-isolation --no-deps -e {consist_source}"
        for call in python_calls
    )
    assert f"-u run.py -c {settings}" in python_calls


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("settings.yaml", "stage.yaml", "extra-arg"),
    ],
)
def test_job_requires_valid_arity(arguments: tuple[str, ...]) -> None:
    """job.sh requires <settings_file> and optional [stage_file]."""

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job.sh"),
            *arguments,
        ],
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Usage:" in completed.stdout + completed.stderr


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
