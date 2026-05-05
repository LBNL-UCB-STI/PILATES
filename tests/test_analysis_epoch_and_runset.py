from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis.epochs import build_epoch_panel
from pilates_consist_analysis.runset import runset_from_runs


@dataclass
class FakeRun:
    id: str
    year: int | None
    iteration: int | None
    scenario_id: str | None
    model_name: str
    status: str
    created_at: datetime
    parent_run_id: str | None = None
    ended_at: datetime | None = None
    metadata: dict | None = None


class TrackerStub:
    def __init__(self, runs):
        self._runs = list(runs)

    def run_set(self, label, model=None, limit=200000):
        del label, limit
        if model is None:
            return list(self._runs)
        return [run for run in self._runs if run.model_name == model]


def _run(run_id: str, *, minutes: int = 0, **kwargs) -> FakeRun:
    return FakeRun(
        id=run_id,
        created_at=datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=minutes),
        ended_at=None,
        metadata={},
        **kwargs,
    )


def test_runset_converged_selects_max_completed_iteration_per_default_group():
    runs = [
        _run(
            "a-2020-i0-beam",
            year=2020,
            iteration=0,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2020-i1-beam-failed",
            year=2020,
            iteration=1,
            scenario_id="a",
            model_name="beam",
            status="failed",
        ),
        _run(
            "a-2020-i2-beam",
            year=2020,
            iteration=2,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2020-i2-urbansim",
            year=2020,
            iteration=2,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2020-missing-iteration",
            year=2020,
            iteration=None,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2021-i1-beam",
            year=2021,
            iteration=1,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2021-i2-beam-running",
            year=2021,
            iteration=2,
            scenario_id="a",
            model_name="beam",
            status="running",
        ),
        _run(
            "b-2021-i1-beam",
            year=2021,
            iteration=1,
            scenario_id="b",
            model_name="beam",
            status="completed",
        ),
        _run(
            "b-2021-i2-beam",
            year=2021,
            iteration=2,
            scenario_id="b",
            model_name="beam",
            status="completed",
        ),
    ]
    runset = runset_from_runs(runs, name="test-runset")

    with pytest.warns(RuntimeWarning, match="missing iteration"):
        converged = runset.converged()

    assert {run.id for run in converged} == {
        "a-2020-i2-beam",
        "a-2020-i2-urbansim",
        "a-2021-i1-beam",
        "b-2021-i2-beam",
    }


def test_build_epoch_panel_groups_by_year_iteration_and_scenario():
    runs = [
        _run(
            "a-2030-i0-usim",
            year=2030,
            iteration=0,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2030-i0-beam",
            year=2030,
            iteration=0,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2030-i1-usim",
            year=2030,
            iteration=1,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2030-i1-beam",
            year=2030,
            iteration=1,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "b-2030-i1-usim",
            year=2030,
            iteration=1,
            scenario_id="b",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "b-2030-i1-beam",
            year=2030,
            iteration=1,
            scenario_id="b",
            model_name="beam",
            status="completed",
        ),
    ]
    panel = build_epoch_panel(TrackerStub(runs), models=["urbansim", "beam"])

    assert [(epoch.year, epoch.outer_iteration, epoch.scenario_id) for epoch in panel] == [
        (2030, 0, "a"),
        (2030, 1, "a"),
        (2030, 1, "b"),
    ]


def test_converged_epochs_returns_latest_complete_epoch_per_year_and_scenario():
    runs = [
        _run(
            "a-2030-i0-usim",
            year=2030,
            iteration=0,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2030-i0-beam",
            year=2030,
            iteration=0,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2030-i1-usim",
            year=2030,
            iteration=1,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2030-i1-beam",
            year=2030,
            iteration=1,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "a-2030-i2-usim",
            year=2030,
            iteration=2,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2030-i2-beam-running",
            year=2030,
            iteration=2,
            scenario_id="a",
            model_name="beam",
            status="running",
        ),
        _run(
            "a-2031-i0-usim",
            year=2031,
            iteration=0,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2031-i0-beam",
            year=2031,
            iteration=0,
            scenario_id="a",
            model_name="beam",
            status="completed",
        ),
        _run(
            "b-2030-i0-usim",
            year=2030,
            iteration=0,
            scenario_id="b",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "b-2030-i0-beam",
            year=2030,
            iteration=0,
            scenario_id="b",
            model_name="beam",
            status="completed",
        ),
    ]
    converged_panel = build_epoch_panel(
        TrackerStub(runs),
        models=["urbansim", "beam"],
    ).converged_epochs()

    assert {(epoch.year, epoch.scenario_id): epoch.outer_iteration for epoch in converged_panel} == {
        (2030, "a"): 1,
        (2030, "b"): 0,
        (2031, "a"): 0,
    }
    assert all(epoch.is_complete for epoch in converged_panel)


def test_epoch_raises_for_missing_or_ambiguous_year():
    runs = [
        _run(
            "a-2035-i0-usim",
            year=2035,
            iteration=0,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
        _run(
            "a-2035-i1-usim",
            year=2035,
            iteration=1,
            scenario_id="a",
            model_name="urbansim",
            status="completed",
        ),
    ]
    panel = build_epoch_panel(TrackerStub(runs), models=["urbansim"])

    with pytest.raises(ValueError, match="No epoch found for year=1999"):
        panel.epoch(1999)

    with pytest.raises(ValueError, match="Multiple epochs found for year=2035"):
        panel.epoch(2035)

