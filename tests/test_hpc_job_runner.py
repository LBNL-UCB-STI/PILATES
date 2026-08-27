from __future__ import annotations

import os
from pathlib import Path
import subprocess


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_job_runner_stages_and_exports_native_structural_canary(
    tmp_path: Path,
) -> None:
    """The submit wrapper gives one job an isolated, writable canary manifest."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run:\n  label: canary\n", encoding="utf-8")
    seed_manifest = pilates_dir / "seed-canary.json"
    seed_manifest.write_text('{"schema_version": 2}\n', encoding="utf-8")
    evidence_root = tmp_path / "canary-evidence"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_manifest = tmp_path / "captured-manifest-path"
    captured_args = tmp_path / "captured-sbatch-args"
    _write_executable(
        fake_bin / "mkdir",
        """#!/bin/sh
if [ "$2" = "/global/scratch/users/canary-user/pilates_logs" ]; then
    exit 0
fi
exec /bin/mkdir "$@"
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/sh
printf '%s\\n' "$@" > "$SBATCH_ARGS_FILE"
printf '%s\\n' "$PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST" > "$SBATCH_MANIFEST_FILE"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PILATES_DIR": str(pilates_dir),
        "PILATES_NATIVE_STRUCTURAL_CANARY_ROOT": str(evidence_root),
        "SBATCH_ARGS_FILE": str(captured_args),
        "SBATCH_MANIFEST_FILE": str(captured_manifest),
        "USER": "canary-user",
    }

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--native-structural-canary",
            "seed-canary.json",
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    staged_manifests = list(evidence_root.glob("*/canary.json"))
    assert len(staged_manifests) == 1
    staged_manifest = staged_manifests[0]
    assert staged_manifest.read_text(encoding="utf-8") == '{"schema_version": 2}\n'
    assert captured_manifest.read_text(encoding="utf-8").strip() == str(staged_manifest)
    assert (staged_manifest.parent / "generated/settings.yaml").read_text(
        encoding="utf-8"
    ) == "run:\n  label: canary\n"
    assert "--export=ALL,JOB_LOG_FILE_PATH=" in captured_args.read_text(
        encoding="utf-8"
    )


def test_job_runner_rejects_native_structural_canary_without_seed_manifest() -> None:
    """A canary request must not silently fall back to the normal job path."""

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-a",
            "test-account",
            "--native-structural-canary",
        ],
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "requires a seed manifest" in completed.stderr


def test_job_runner_assembles_beam_preprocess_acceptance_submission(
    tmp_path: Path,
) -> None:
    """The wrapper passes the original manifest and one retained evidence root."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run:\n  label: acceptance\n", encoding="utf-8")
    manifest = pilates_dir / "inputs.json"
    manifest.write_text('{"cohort": {"year": 2019}}\n', encoding="utf-8")
    evidence_root = tmp_path / "acceptance-evidence"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "captured-sbatch-args"
    _write_executable(
        fake_bin / "mkdir",
        """#!/bin/sh
if [ "$2" = "/global/scratch/users/acceptance-user/pilates_logs" ]; then
    exit 0
fi
exec /bin/mkdir "$@"
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/sh
printf '%s\\n' "$@" > "$SBATCH_ARGS_FILE"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PILATES_DIR": str(pilates_dir),
        "PILATES_BEAM_PREPROCESS_ACCEPTANCE_ROOT": str(evidence_root),
        "SBATCH_ARGS_FILE": str(captured_args),
        "USER": "acceptance-user",
    }

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--beam-preprocess-acceptance",
            "inputs.json",
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence_dirs = [path for path in evidence_root.iterdir() if path.is_dir()]
    assert len(evidence_dirs) == 1
    evidence_dir = evidence_dirs[0]
    assert (evidence_dir / "submitted-input-manifest.json").read_text(
        encoding="utf-8"
    ) == manifest.read_text(encoding="utf-8")
    assert (evidence_dir / "generated-settings.yaml").read_text(
        encoding="utf-8"
    ) == settings.read_text(encoding="utf-8")
    submitted_args = captured_args.read_text(encoding="utf-8").splitlines()
    selector_index = submitted_args.index("--beam-preprocess-acceptance")
    assert submitted_args[selector_index + 2] == str(manifest)
    assert submitted_args[selector_index + 3] == str(evidence_dir)


def test_job_runner_rejects_acceptance_with_restart_stage(tmp_path: Path) -> None:
    """Acceptance cannot acquire an ordinary restart-stage positional argument."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    (pilates_dir / "settings.yaml").write_text("run: {}\n", encoding="utf-8")
    (pilates_dir / "inputs.json").write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--beam-preprocess-acceptance",
            "inputs.json",
            "-s",
            "restart.yaml",
        ],
        cwd=_PROJECT_ROOT,
        env={**os.environ, "PILATES_DIR": str(pilates_dir)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "cannot be combined with -s/--stage" in completed.stderr


def test_job_acceptance_mode_runs_driver_once_and_exits(tmp_path: Path) -> None:
    """The allocated-job selector is terminal and cannot fall through to run.py."""

    pilates_dir = tmp_path / "pilates"
    requirements = pilates_dir / "hpc" / "requirements-hpc.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("", encoding="utf-8")
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run: {}\n", encoding="utf-8")
    manifest = pilates_dir / "inputs.json"
    manifest.write_text("{}\n", encoding="utf-8")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
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
            "--beam-preprocess-acceptance",
            str(settings),
            str(manifest),
            str(evidence_root),
        ],
        cwd=_PROJECT_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PILATES_DIR": str(pilates_dir),
            "PILATES_VENV_PATH": str(venv),
            "PILATES_REQUIREMENTS_FILE": str(requirements),
            "CONSIST_SRC_DIR": str(tmp_path / "missing-consist"),
            "PYTHON_CALLS_FILE": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    python_calls = calls.read_text(encoding="utf-8").splitlines()
    driver_calls = [
        call
        for call in python_calls
        if call.startswith("-m pilates.runtime.beam_preprocess_acceptance ")
    ]
    assert driver_calls == [
        "-m pilates.runtime.beam_preprocess_acceptance "
        f"--settings {settings} --manifest {manifest} --evidence-root {evidence_root}"
    ]
    assert not any("run.py" in call for call in python_calls)


def test_job_acceptance_mode_requires_exact_arity() -> None:
    """Missing any of settings, manifest, or evidence root fails before setup."""

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job.sh"),
            "--beam-preprocess-acceptance",
            "settings.yaml",
            "inputs.json",
        ],
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "<settings_file> <input_manifest> <evidence_root>" in completed.stderr


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
