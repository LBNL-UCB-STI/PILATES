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


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
