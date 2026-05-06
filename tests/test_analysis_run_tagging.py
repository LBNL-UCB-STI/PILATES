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
from pilates_consist_analysis.runtime import (
    assert_run_tagging_consistent,
    assert_run_tagging_report,
    get_run_tagging_issues,
    inspect_run_tagging,
    run_tagging_to_frame,
    validate_run_tagging,
)


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


def test_inspect_run_tagging_reports_structured_counts_and_linkage():
    tracker = TrackerStub(
        [
            FakeRun(
                id="asim-1",
                model_name="activitysim",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                parent_run_id=None,
            ),
            FakeRun(
                id="asim-2",
                model_name="activitysim",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                parent_run_id="not-beam-prev",
            ),
            FakeRun(
                id="beam-prev",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=1,
                parent_run_id="asim-prev",
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
                parent_run_id="asim-2",
            ),
            FakeRun(
                id="missing-scenario",
                model_name="beam",
                scenario_id=None,
                year=2030,
                iteration=0,
            ),
            FakeRun(
                id="missing-year",
                model_name="beam",
                scenario_id="baseline",
                year=None,
                iteration=0,
            ),
            FakeRun(
                id="missing-iteration",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=None,
            ),
            FakeRun(
                id="missing-model",
                model_name="",
                scenario_id="baseline",
                year=2030,
                iteration=0,
            ),
        ]
    )

    report = inspect_run_tagging(tracker)
    assert report["total_runs"] == 9
    assert report["missing_counts"] == {
        "scenario_id": 1,
        "year": 1,
        "iteration": 1,
        "model": 1,
    }
    assert report["linkage_counts"] == {
        "beam_parent_checked": 2,
        "beam_parent_missing": 1,
        "beam_parent_mismatch": 1,
        "asim_parent_checked": 2,
        "asim_parent_missing": 1,
        "asim_parent_mismatch": 1,
    }

    warnings = validate_run_tagging(tracker)
    assert warnings == report["warnings"]

    assert "run_tagging.scenario_id.missing=1/9" in warnings
    assert "run_tagging.year.missing=1/9" in warnings
    assert "run_tagging.iteration.missing=1/9" in warnings
    assert "run_tagging.model.missing=1/9" in warnings
    assert (
        "run_tagging.beam_parent_missing=1/2 (expected ActivitySim parent_run_id hints)"
        in warnings
    )
    assert (
        "run_tagging.beam_parent_mismatch=1/2 (compared to ActivitySim sibling run id)"
        in warnings
    )
    assert (
        "run_tagging.asim_parent_missing=1/2 (expected previous-iteration BEAM parent_run_id)"
        in warnings
    )
    assert (
        "run_tagging.asim_parent_mismatch=1/2 (compared to previous-iteration BEAM sibling run ids)"
        in warnings
    )


def test_validate_run_tagging_handles_empty_runs():
    report = inspect_run_tagging(TrackerStub([]))
    assert report["total_runs"] == 0
    assert report["missing_counts"] == {
        "scenario_id": 0,
        "year": 0,
        "iteration": 0,
        "model": 0,
    }
    assert report["linkage_counts"] == {
        "beam_parent_checked": 0,
        "beam_parent_missing": 0,
        "beam_parent_mismatch": 0,
        "asim_parent_checked": 0,
        "asim_parent_missing": 0,
        "asim_parent_mismatch": 0,
    }
    assert report["warnings"] == ["No runs available for run-tag validation."]
    assert validate_run_tagging(TrackerStub([])) == [
        "No runs available for run-tag validation."
    ]


def test_run_tagging_issues_and_assert_helpers():
    payload = {
        "total_runs": 1,
        "missing_counts": {
            "scenario_id": 0,
            "year": 0,
            "iteration": 0,
            "model": 1,
        },
        "linkage_counts": {
            "beam_parent_checked": 0,
            "beam_parent_missing": 0,
            "beam_parent_mismatch": 0,
            "asim_parent_checked": 0,
            "asim_parent_missing": 0,
            "asim_parent_mismatch": 0,
        },
        "warnings": ["run_tagging.model.missing=1/1"],
    }

    assert get_run_tagging_issues(payload, strict=False) == []
    assert get_run_tagging_issues(payload, strict=True) == [
        "run_tagging.model.missing=1/1"
    ]
    assert (
        get_run_tagging_issues(
            ["No runs available for run-tag validation."],
            strict=False,
        )
        == []
    )
    assert get_run_tagging_issues(
        ["No runs available for run-tag validation."],
        strict=True,
    ) == ["No runs available for run-tag validation."]

    assert_run_tagging_report(payload, strict=False, raise_on_issues=False)
    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        assert_run_tagging_report(payload, strict=True, raise_on_issues=False)
    assert_run_tagging_report(payload, strict=False, raise_on_issues=True)

    requested_fail_payload = {
        **payload,
        "missing_counts": {
            "scenario_id": 0,
            "year": 1,
            "iteration": 0,
            "model": 0,
        },
    }
    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        assert_run_tagging_report(
            requested_fail_payload,
            strict=False,
            raise_on_issues=True,
        )

    frame = run_tagging_to_frame(payload, strict=True)
    assert bool(frame.iloc[0]["healthy"]) is False
    assert int(frame.iloc[0]["missing_model"]) == 1
    assert int(frame.iloc[0]["issue_count"]) == 1


def test_assert_run_tagging_consistent_raises_when_requested():
    tracker = TrackerStub(
        [
            FakeRun(
                id="bad",
                model_name="activitysim",
                scenario_id=None,
                year=2030,
                iteration=1,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        assert_run_tagging_consistent(
            tracker,
            strict=False,
            raise_on_issues=True,
        )


def test_analysis_session_run_tagging_report_and_assert(monkeypatch, tmp_path):
    archive_run_dir, project_root = _archive_paths(tmp_path)
    monkeypatch.setattr(
        api_module,
        "inspect_run_tagging_report",
        lambda _tracker: {
            "total_runs": 1,
            "missing_counts": {
                "scenario_id": 0,
                "year": 0,
                "iteration": 0,
                "model": 1,
            },
            "linkage_counts": {
                "beam_parent_checked": 0,
                "beam_parent_missing": 0,
                "beam_parent_mismatch": 0,
                "asim_parent_checked": 0,
                "asim_parent_missing": 0,
                "asim_parent_mismatch": 0,
            },
            "warnings": ["run_tagging.model.missing=1/1"],
        },
    )
    session = AnalysisSession(
        archive_run_dir=archive_run_dir,
        project_root=project_root,
        tracker=object(),
    )
    assert session.run_tagging_report()["total_runs"] == 1
    assert session.tagging_issues == []

    inspect_df = session.inspect_run_tagging(strict=True)
    assert bool(inspect_df.iloc[0]["healthy"]) is False
    assert int(inspect_df.iloc[0]["issue_count"]) == 1

    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        session.assert_run_tagging(strict=True)
    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        session.assert_run_tagging_consistent(strict=True, raise_on_issues=True)


def test_analysis_session_open_strict_or_fail_modes_raise(monkeypatch, tmp_path):
    archive_run_dir, project_root = _archive_paths(tmp_path)
    monkeypatch.setattr(
        api_module, "create_analysis_tracker", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        api_module,
        "inspect_run_tagging_report",
        lambda _tracker: {
            "total_runs": 1,
            "missing_counts": {
                "scenario_id": 0,
                "year": 1,
                "iteration": 0,
                "model": 0,
            },
            "linkage_counts": {
                "beam_parent_checked": 0,
                "beam_parent_missing": 0,
                "beam_parent_mismatch": 0,
                "asim_parent_checked": 0,
                "asim_parent_missing": 0,
                "asim_parent_mismatch": 0,
            },
            "warnings": ["run_tagging.year.missing=1/1"],
        },
    )

    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        AnalysisSession.open(
            archive_run_dir=archive_run_dir,
            project_root=project_root,
            strict_tagging=True,
        )
    with pytest.raises(RuntimeError, match="Run tagging validation failed"):
        AnalysisSession.open(
            archive_run_dir=archive_run_dir,
            project_root=project_root,
            fail_on_tagging_issues=True,
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
    assert args.output_format == "json"


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
        "inspect_run_tagging",
        lambda _tracker: {
            "total_runs": 0,
            "missing_counts": {
                "scenario_id": 0,
                "year": 0,
                "iteration": 0,
                "model": 0,
            },
            "linkage_counts": {
                "beam_parent_checked": 0,
                "beam_parent_missing": 0,
                "beam_parent_mismatch": 0,
                "asim_parent_checked": 0,
                "asim_parent_missing": 0,
                "asim_parent_mismatch": 0,
            },
            "warnings": ["No runs available for run-tag validation."],
        },
    )

    payloads: list[dict] = []
    monkeypatch.setattr(
        cli_module, "_print_json", lambda payload: payloads.append(payload)
    )

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
    monkeypatch.setattr(
        cli_module,
        "inspect_run_tagging",
        lambda _tracker: {
            "total_runs": 1,
            "missing_counts": {
                "scenario_id": 0,
                "year": 1,
                "iteration": 0,
                "model": 0,
            },
            "linkage_counts": {
                "beam_parent_checked": 0,
                "beam_parent_missing": 0,
                "beam_parent_mismatch": 0,
                "asim_parent_checked": 0,
                "asim_parent_missing": 0,
                "asim_parent_mismatch": 0,
            },
            "warnings": ["run_tagging.year.missing=1/1"],
        },
    )

    payloads: list[dict] = []
    monkeypatch.setattr(
        cli_module, "_print_json", lambda payload: payloads.append(payload)
    )

    exit_code = cli_module.cmd_run_tagging(args)
    assert exit_code == 2
    assert payloads
    assert payloads[0]["issues"] == ["run_tagging.year.missing=1/1"]
    assert payloads[0]["strict"] is True


def test_cli_run_tagging_table_output_includes_optional_lines(
    monkeypatch, tmp_path, capsys
):
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "run-tagging",
            "--archive-run-dir",
            str(tmp_path / "archive"),
            "--project-root",
            str(tmp_path / "project"),
            "--output-format",
            "table",
            "--include-warnings",
            "--include-issues",
            "--strict",
        ]
    )
    monkeypatch.setattr(cli_module, "_build_tracker", lambda _args: object())
    monkeypatch.setattr(
        cli_module,
        "inspect_run_tagging",
        lambda _tracker: {
            "total_runs": 1,
            "missing_counts": {
                "scenario_id": 0,
                "year": 1,
                "iteration": 0,
                "model": 0,
            },
            "linkage_counts": {
                "beam_parent_checked": 0,
                "beam_parent_missing": 0,
                "beam_parent_mismatch": 0,
                "asim_parent_checked": 0,
                "asim_parent_missing": 0,
                "asim_parent_mismatch": 0,
            },
            "warnings": ["run_tagging.year.missing=1/1"],
        },
    )

    exit_code = cli_module.cmd_run_tagging(args)
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Warnings:" in output
    assert "Issues:" in output
    assert "missing_year" in output
