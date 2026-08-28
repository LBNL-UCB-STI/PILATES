from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


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


def test_job_runner_assembles_activitysim_run_acceptance_submission(
    tmp_path: Path,
) -> None:
    """The wrapper preserves one four-role manifest and one evidence root."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run:\n  label: activitysim-acceptance\n", encoding="utf-8")
    manifest = pilates_dir / "inputs.json"
    manifest.write_text('{"inputs": {}}\n', encoding="utf-8")
    evidence_root = tmp_path / "activitysim-acceptance-evidence"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "captured-sbatch-args"
    _write_executable(
        fake_bin / "mkdir",
        """#!/bin/sh
if [ "$2" = "/global/scratch/users/activitysim-acceptance-user/pilates_logs" ]; then
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
        "PILATES_ACTIVITYSIM_RUN_ACCEPTANCE_ROOT": str(evidence_root),
        "SBATCH_ARGS_FILE": str(captured_args),
        "USER": "activitysim-acceptance-user",
    }

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--activitysim-run-acceptance",
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
    submitted_args = captured_args.read_text(encoding="utf-8").splitlines()
    selector_index = submitted_args.index("--activitysim-run-acceptance")
    assert submitted_args[selector_index + 2] == str(manifest)
    assert submitted_args[selector_index + 3] == str(evidence_dir)


def test_job_runner_assembles_urbansim_h5_snapshot_acceptance_submission(
    tmp_path: Path,
) -> None:
    """The HDF5 harness retains inputs and passes its terminal-job contract."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run:\n  label: h5-acceptance\n", encoding="utf-8")
    manifest = pilates_dir / "inputs.json"
    manifest.write_text(
        '{"inputs": {"usim_datastore_h5": "input.h5"}}\n', encoding="utf-8"
    )
    evidence_root = tmp_path / "h5-acceptance-evidence"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "captured-sbatch-args"
    _write_executable(
        fake_bin / "mkdir",
        """#!/bin/sh
if [ "$2" = "/global/scratch/users/h5-acceptance-user/pilates_logs" ]; then
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
        "PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_ROOT": str(evidence_root),
        "SBATCH_ARGS_FILE": str(captured_args),
        "USER": "h5-acceptance-user",
    }

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--urbansim-h5-snapshot-acceptance",
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
    selector_index = submitted_args.index("--urbansim-h5-snapshot-acceptance")
    assert submitted_args[selector_index + 1 :] == [
        str(
            settings.parent
            / next(path.name for path in pilates_dir.glob("settings_*.yaml"))
        ),
        str(manifest),
        str(evidence_dir),
    ]


@pytest.mark.parametrize(
    ("additional_args", "expected_message"),
    [
        (
            ["--beam-preprocess-acceptance", "inputs.json"],
            "cannot be combined with --beam-preprocess-acceptance",
        ),
        (
            ["--native-structural-canary", "seed-canary.json"],
            "cannot be combined with --native-structural-canary",
        ),
        (["-s", "restart.yaml"], "cannot be combined with -s/--stage"),
    ],
)
def test_job_runner_rejects_urbansim_h5_snapshot_acceptance_combinations(
    tmp_path: Path,
    additional_args: list[str],
    expected_message: str,
) -> None:
    """The HDF5 selector is terminal and mutually exclusive."""

    pilates_dir = tmp_path / "pilates"
    pilates_dir.mkdir()
    (pilates_dir / "settings.yaml").write_text("run: {}\n", encoding="utf-8")
    (pilates_dir / "inputs.json").write_text("{}\n", encoding="utf-8")
    (pilates_dir / "seed-canary.json").write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job_runner.sh"),
            "-c",
            "settings.yaml",
            "-a",
            "test-account",
            "--urbansim-h5-snapshot-acceptance",
            "inputs.json",
            *additional_args,
        ],
        cwd=_PROJECT_ROOT,
        env={**os.environ, "PILATES_DIR": str(pilates_dir)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert expected_message in completed.stderr


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
        if call.startswith("-u -m pilates.runtime.beam_preprocess_acceptance ")
    ]
    assert driver_calls == [
        "-u -m pilates.runtime.beam_preprocess_acceptance "
        f"--settings {settings} --manifest {manifest} --evidence-root {evidence_root}"
    ]
    assert not any("run.py" in call for call in python_calls)


def test_job_activitysim_acceptance_installs_required_release_not_editable(
    tmp_path: Path,
) -> None:
    """ActivitySim acceptance never substitutes an adjacent editable Consist."""

    pilates_dir = tmp_path / "pilates"
    requirements = pilates_dir / "hpc" / "requirements-hpc.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("", encoding="utf-8")
    settings = pilates_dir / "settings.yaml"
    settings.write_text("run: {}\n", encoding="utf-8")
    manifest = pilates_dir / "inputs.json"
    manifest.write_text('{"released_consist_version": "9.8.7"}\n', encoding="utf-8")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    consist_source = tmp_path / "adjacent-consist"
    consist_source.mkdir()
    venv = tmp_path / "venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    calls = tmp_path / "python-calls"
    _write_executable(
        venv_bin / "python3",
        """#!/bin/sh
printf '%s\\n' "$*" >> "$PYTHON_CALLS_FILE"
if [ "$1" = "-c" ]; then
    printf '%s\\n' '9.8.7'
fi
""",
    )
    (venv_bin / "activate").write_text(
        f'export PATH="{venv_bin}:$PATH"\n', encoding="utf-8"
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
            "--activitysim-run-acceptance",
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
        call == "-m pip install --upgrade --force-reinstall consist==9.8.7"
        for call in python_calls
    )
    assert not any("-m pip install -e" in call for call in python_calls)


def test_job_urbansim_h5_snapshot_acceptance_runs_capture_then_reconcile(
    tmp_path: Path,
) -> None:
    """The HDF5 job executes both terminal drivers without reaching run.py."""

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
    consist_source = tmp_path / "consist"
    (consist_source / "src" / "consist").mkdir(parents=True)
    _write_executable(
        venv_bin / "python3",
        """#!/bin/sh
printf '%s\\n' "$*" >> "$PYTHON_CALLS_FILE"
if [ "$1" = "-" ]; then
    if [ -z "${PILATES_DIR+x}" ]; then
        echo 'PILATES_DIR was not exported to the runtime record writer' >&2
        exit 1
    fi
    printf '%s\\n' '{"consist":{"editable_source":"fake","import_path":"fake","revision":"fake-revision"},"pilates":{"revision":"fake-revision"},"python":{"executable":"fake"}}' > "$CONSIST_ACCEPTANCE_RUNTIME_RECORD"
fi
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
    _write_executable(fake_bin / "git", "#!/bin/sh\nprintf '%s\\n' fake-revision\n")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'PILATES_DIR="$1"; export -n PILATES_DIR; source "$2" --urbansim-h5-snapshot-acceptance "$3" "$4" "$5"',
            "job.sh",
            str(pilates_dir),
            str(_PROJECT_ROOT / "hpc/job.sh"),
            str(settings),
            str(manifest),
            str(evidence_root),
        ],
        cwd=_PROJECT_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
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
    driver_calls = [
        call
        for call in python_calls
        if call.startswith("-u -m pilates.runtime.urbansim_h5_snapshot_acceptance ")
    ]
    assert driver_calls == [
        "-u -m pilates.runtime.urbansim_h5_snapshot_acceptance "
        f"capture --settings {settings} --manifest {manifest} --evidence-root {evidence_root}",
        "-u -m pilates.runtime.urbansim_h5_snapshot_acceptance "
        f"reconcile --evidence-root {evidence_root}",
    ]
    assert not any("run.py" in call for call in python_calls)
    runtime_environment = json.loads(
        (evidence_root / "runtime-environment.json").read_text(encoding="utf-8")
    )
    assert runtime_environment["consist"]["revision"] == "fake-revision"


def test_job_urbansim_h5_snapshot_acceptance_rejects_missing_editable_consist(
    tmp_path: Path,
) -> None:
    """This acceptance cannot fall back to a packaged Consist installation."""

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
            "--urbansim-h5-snapshot-acceptance",
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

    assert completed.returncode != 0
    assert "requires an existing editable Consist checkout" in completed.stderr
    assert not calls.exists()


def test_job_urbansim_h5_snapshot_acceptance_rejects_unverified_editable_import(
    tmp_path: Path,
) -> None:
    """An editable install alone is insufficient when import verification fails."""

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
if [ "$1" = "-" ]; then
    echo 'simulated unexpected Consist import path' >&2
    exit 1
fi
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
    _write_executable(fake_bin / "git", "#!/bin/sh\nprintf '%s\\n' fake-revision\n")

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job.sh"),
            "--urbansim-h5-snapshot-acceptance",
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
            "CONSIST_SRC_DIR": str(consist_source),
            "PYTHON_CALLS_FILE": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "simulated unexpected Consist import path" in completed.stderr
    assert not any(
        "pilates.runtime.urbansim_h5_snapshot_acceptance" in call
        for call in calls.read_text(encoding="utf-8").splitlines()
    )


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


@pytest.mark.parametrize(
    "arguments",
    [
        ("settings.yaml", "inputs.json"),
        ("settings.yaml", "inputs.json", "evidence", "unexpected"),
    ],
)
def test_hdf5_snapshot_acceptance_requires_exact_arity(
    arguments: tuple[str, ...],
) -> None:
    """Malformed HDF5 acceptance invocation fails before runtime setup."""

    completed = subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "hpc/job.sh"),
            "--urbansim-h5-snapshot-acceptance",
            *arguments,
        ],
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "<settings_file> <input_manifest> <evidence_root>" in completed.stderr
    assert "Setting up HPC runtime environment" not in completed.stdout


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
