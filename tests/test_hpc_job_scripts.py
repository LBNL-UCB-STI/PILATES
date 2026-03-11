from __future__ import annotations

import os
import subprocess
from pathlib import Path
from textwrap import dedent


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def test_job_runner_generates_settings_and_skips_stage_for_fresh_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "hpc/job_runner.sh"
    project_dir = tmp_path / "project"
    fake_bin = tmp_path / "fake-bin"
    sbatch_args = tmp_path / "sbatch_args.txt"

    (project_dir / "hpc").mkdir(parents=True, exist_ok=True)
    (project_dir / "hpc/job.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (project_dir / "settings.yaml").write_text(
        "beam:\n  memory: ${BEAM_MEMORY}\n",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "sbatch",
        f"""
        #!/bin/bash
        printf '%s\n' "$@" > "{sbatch_args}"
        exit 0
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PILATES_DIR"] = str(project_dir)
    env["PILATES_LOG_DIR"] = str(tmp_path / "logs")
    env["USER"] = "tester"

    result = subprocess.run(
        ["bash", str(script_path), "-c", "settings.yaml"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    generated_configs = sorted(project_dir.glob("settings_*.yaml"))
    assert len(generated_configs) == 1
    assert "180g" in generated_configs[0].read_text(encoding="utf-8")

    sbatch_lines = sbatch_args.read_text(encoding="utf-8").splitlines()
    assert sbatch_lines[-2] == str(project_dir / "hpc/job.sh")
    assert sbatch_lines[-1] == str(generated_configs[0])
    assert not any("current_stage_" in line for line in sbatch_lines)


def _run_job_sh(tmp_path: Path, *, create_consist_repo: bool):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "hpc/job.sh"
    project_dir = tmp_path / "project"
    fake_bin = tmp_path / "fake-bin"
    py_log = tmp_path / "python_calls.txt"
    run_log = tmp_path / "run_args.txt"
    consist_state = tmp_path / "consist_installed.txt"

    (project_dir / "hpc").mkdir(parents=True, exist_ok=True)
    (project_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (project_dir / "run.py").write_text("print('stub run')\n", encoding="utf-8")
    (project_dir / "hpc/requirements-hpc.txt").write_text(
        "numpy<2.0\n",
        encoding="utf-8",
    )
    config_path = project_dir / "settings.yaml"
    config_path.write_text(
        dedent(
            """
            run:
              region: test
            shared: {}
            infrastructure: {}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    stage_path = project_dir / "current_stage.yaml"
    stage_path.write_text("stage: test\n", encoding="utf-8")
    if create_consist_repo:
        (project_dir / "consist").mkdir(parents=True, exist_ok=True)

    fake_python = fake_bin / "python3"
    _write_executable(
        fake_python,
        f"""
        #!/bin/bash
        set -euo pipefail
        if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "venv" ]; then
            target="${{3:?missing venv target}}"
            mkdir -p "$target/bin"
            cp "$0" "$target/bin/python3"
            printf 'VIRTUAL_ENV="%s"\nPATH="%s/bin:%s"\nexport VIRTUAL_ENV PATH\n' "$target" "$target" "$PATH" > "$target/bin/activate"
            exit 0
        fi
        if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "pip" ]; then
            printf 'pip %s\n' "$*" >> "{py_log}"
            case "$*" in
                *" install -e "*|*" install consist=="*)
                    touch "{consist_state}"
                    ;;
            esac
            exit 0
        fi
        if [ "${{1:-}}" = "--version" ]; then
            echo "Python 3.11.6"
            exit 0
        fi
        if [ "${{1:-}}" = "-c" ]; then
            case "${{2:-}}" in
                *"from consist import create_tracker"*)
                    if [ -f "{consist_state}" ]; then
                        exit 0
                    fi
                    exit 1
                    ;;
            esac
            exit 0
        fi
        printf '%s\n' "$*" >> "{run_log}"
        exit 0
        """,
    )
    _write_executable(
        fake_bin / "free",
        """
        #!/bin/bash
        echo "Mem: stub"
        """,
    )
    _write_executable(
        fake_bin / "squeue",
        """
        #!/bin/bash
        exit 0
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PILATES_DIR"] = str(project_dir)
    env["USER"] = "tester"

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'function module() {{ return 0; }}; export -f module; "{script_path}" "{config_path}" "{stage_path}"',
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    return result, project_dir, py_log, run_log


def test_job_sh_prefers_editable_consist_when_repo_exists(tmp_path: Path) -> None:
    result, project_dir, py_log, run_log = _run_job_sh(
        tmp_path,
        create_consist_repo=True,
    )

    assert result.returncode == 0, result.stderr
    assert (project_dir / "PILATES-env/bin/python3").exists()

    pip_calls = py_log.read_text(encoding="utf-8")
    assert "install --upgrade pip setuptools wheel" in pip_calls
    assert f"install -r {project_dir / 'hpc/requirements-hpc.txt'}" in pip_calls
    assert f"install -e {project_dir / 'consist'}" in pip_calls
    assert "install consist==0.1.0" not in pip_calls

    run_args = run_log.read_text(encoding="utf-8")
    assert f"run.py -c {project_dir / 'settings.yaml'} -S {project_dir / 'current_stage.yaml'}" in run_args


def test_job_sh_installs_consist_from_pypi_when_repo_missing(tmp_path: Path) -> None:
    result, project_dir, py_log, _run_log = _run_job_sh(
        tmp_path,
        create_consist_repo=False,
    )

    assert result.returncode == 0, result.stderr

    pip_calls = py_log.read_text(encoding="utf-8")
    assert "install consist==0.1.0" in pip_calls
    assert "install -e" not in pip_calls
