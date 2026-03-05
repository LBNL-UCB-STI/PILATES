from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

import pilates_consist_analysis.api as api_module
import pilates_consist_analysis.cli as cli_module
from pilates_consist_analysis.api import AnalysisSession
from pilates_consist_analysis.runtime import validate_run_tagging


@dataclass
class FakeRun:
    id: str
    model_name: str | None = None
    scenario_id: str | None = None
    year: int | None = None
    iteration: int | None = None
    parent_run_id: str | None = None
    metadata: dict | None = None


class TrackerStub:
    def __init__(self, runs):
        self._runs = list(runs)

    def run_set(self, label, limit=200000):
        del label, limit
        return list(self._runs)


def _archive_paths(tmp_path: Path) -> tuple[Path, Path]:
    archive_run_dir = tmp_path / "archive"
    project_root = tmp_path / "project"
    (archive_run_dir / ".consist").mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / ".consist" / "consist.duckdb").write_bytes(b"")
    return archive_run_dir, project_root


def test_validate_run_tagging_counts_missing_required_fields():
    tracker = TrackerStub(
        [
            FakeRun(
                id="r1",
                model_name="activitysim_run",
                scenario_id="baseline",
                year=2018,
                iteration=0,
            ),
            FakeRun(
                id="r2",
                model_name="beam_sim",
                scenario_id=None,
                year=2018,
                iteration=0,
            ),
            FakeRun(
                id="r3",
                model_name="beam_sim",
                scenario_id="baseline",
                year=None,
                iteration=0,
            ),
            FakeRun(
                id="r4",
                model_name="",
                scenario_id="baseline",
                year=2018,
                iteration=None,
            ),
        ]
    )

    warnings = validate_run_tagging(tracker)

    assert "run_tagging.scenario_id.missing=1/4" in warnings
    assert "run_tagging.year.missing=1/4" in warnings
    assert "run_tagging.iteration.missing=1/4" in warnings
    assert "run_tagging.model.missing=1/4" in warnings


def test_validate_run_tagging_reports_beam_parent_missing_and_mismatch():
    tracker = TrackerStub(
        [
            FakeRun(
                id="asim-1",
                model_name="activitysim",
                scenario_id="baseline",
                year=2030,
                iteration=2,
            ),
            FakeRun(
                id="beam-1",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                parent_run_id=None,
            ),
            FakeRun(
                id="beam-2",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                parent_run_id="not-asim-1",
            ),
        ]
    )

    warnings = validate_run_tagging(tracker)

    assert (
        "run_tagging.beam_parent_missing=1/2 (expected ActivitySim parent_run_id hints)"
        in warnings
    )
    assert (
        "run_tagging.beam_parent_mismatch=1/2 (compared to ActivitySim sibling run id)"
        in warnings
    )


def test_validate_run_tagging_handles_empty_runs():
    warnings = validate_run_tagging(TrackerStub([]))
    assert warnings == ["No runs available for run-tag validation."]


def test_analysis_session_inspect_and_assert_run_tagging(monkeypatch, tmp_path):
    archive_run_dir, project_root = _archive_paths(tmp_path)
    monkeypatch.setattr(api_module, "validate_run_tagging", lambda _tracker: [])
    session = AnalysisSession(
        archive_run_dir=archive_run_dir,
        project_root=project_root,
        tracker=object(),
    )

    monkeypatch.setattr(
        api_module,
        "validate_run_tagging",
        lambda _tracker: ["run_tagging.model.missing=1/1"],
    )
    inspect_df = session.inspect_run_tagging()
    assert bool(inspect_df.iloc[0]["healthy"]) is False
    assert int(inspect_df.iloc[0]["issue_count"]) == 1

    def _raise_on_assert(_tracker, *, strict):
        del strict
        raise RuntimeError("run tagging failed")

    monkeypatch.setattr(api_module, "assert_run_tagging", _raise_on_assert)
    with pytest.raises(RuntimeError, match="run tagging failed"):
        session.assert_run_tagging()


def test_analysis_session_open_strict_tagging_raises(monkeypatch, tmp_path):
    archive_run_dir, project_root = _archive_paths(tmp_path)
    monkeypatch.setattr(api_module, "create_analysis_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(
        api_module,
        "validate_run_tagging",
        lambda _tracker: ["run_tagging.year.missing=1/1"],
    )

    with pytest.raises(RuntimeError, match="Run tagging validation failed on open"):
        AnalysisSession.open(
            archive_run_dir=archive_run_dir,
            project_root=project_root,
            strict_tagging=True,
        )


def test_cli_run_tagging_command_is_registered():
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "run-tagging",
            "--archive-run-dir",
            "/tmp/archive",
            "--project-root",
            "/tmp/project",
        ]
    )

    assert args.command == "run-tagging"
    assert args.func is cli_module.cmd_run_tagging
    assert args.strict is False
    assert args.fail_on_issues is False


def test_cli_run_tagging_non_strict_fail_mode_ignores_empty_runs_warning(
    monkeypatch, tmp_path
):
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "run-tagging",
            "--archive-run-dir",
            str(tmp_path / "archive"),
            "--project-root",
            str(tmp_path / "project"),
            "--fail-on-issues",
        ]
    )

    monkeypatch.setattr(cli_module, "_build_tracker", lambda _args: object())
    monkeypatch.setattr(
        cli_module,
        "validate_run_tagging",
        lambda _tracker: ["No runs available for run-tag validation."],
    )
    monkeypatch.setattr(
        cli_module, "assert_run_tagging", lambda _tracker, *, strict: []  # pragma: no cover
    )

    payloads: list[dict] = []
    monkeypatch.setattr(cli_module, "_print_json", lambda payload: payloads.append(payload))

    exit_code = cli_module.cmd_run_tagging(args)
    assert exit_code == 0
    assert payloads
    assert payloads[0]["issues"] == []
    assert payloads[0]["strict"] is False


def test_cli_run_tagging_strict_fail_mode_exits_2(monkeypatch, tmp_path):
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "run-tagging",
            "--archive-run-dir",
            str(tmp_path / "archive"),
            "--project-root",
            str(tmp_path / "project"),
            "--strict",
            "--fail-on-issues",
        ]
    )

    monkeypatch.setattr(cli_module, "_build_tracker", lambda _args: object())

    def _assert_raises(_tracker, *, strict):
        assert strict is True
        raise RuntimeError("strict tagging failed")

    monkeypatch.setattr(cli_module, "assert_run_tagging", _assert_raises)
    monkeypatch.setattr(
        cli_module,
        "validate_run_tagging",
        lambda _tracker: ["No runs available for run-tag validation."],
    )

    payloads: list[dict] = []
    monkeypatch.setattr(cli_module, "_print_json", lambda payload: payloads.append(payload))

    exit_code = cli_module.cmd_run_tagging(args)
    assert exit_code == 2
    assert payloads
    assert payloads[0]["issues"] == ["No runs available for run-tag validation."]
    assert payloads[0]["strict"] is True
