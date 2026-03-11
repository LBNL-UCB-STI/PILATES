from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

import pilates_consist_analysis.scenario_compare as scenario_compare
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


def _run(run_id: str, *, minutes: int = 0, **kwargs) -> FakeRun:
    return FakeRun(
        id=run_id,
        created_at=datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=minutes),
        ended_at=None,
        metadata={},
        **kwargs,
    )


def test_compare_scenarios_converged_raises_when_overlap_lacks_complete_candidates():
    left = runset_from_runs(
        [
            _run(
                "left-2020-i1",
                year=2020,
                iteration=1,
                scenario_id="base",
                model_name="beam",
                status="completed",
            )
        ],
        name="left",
    )
    right = runset_from_runs(
        [
            _run(
                "right-2020-missing-iteration",
                year=2020,
                iteration=None,
                scenario_id="base",
                model_name="beam",
                status="completed",
            )
        ],
        name="right",
    )

    with pytest.warns(RuntimeWarning, match="missing iteration"):
        with pytest.raises(
            ValueError,
            match="Converged scenario compare requires complete epoch candidates on both sides",
        ) as exc:
            scenario_compare.compare_scenarios(
                tracker=object(),
                run_set_left=left,
                run_set_right=right,
                datasets=["linkstats"],
                align_on="year",
                use_converged=True,
            )

    message = str(exc.value)
    assert "Missing on right: 2020" in message
    assert "--converged-group-by" in message
    assert "disable converged mode" in message


def test_compare_scenarios_converged_allows_valid_candidates(monkeypatch):
    left = runset_from_runs(
        [
            _run(
                "left-2020-i0",
                year=2020,
                iteration=0,
                scenario_id="base",
                model_name="beam",
                status="completed",
                minutes=0,
            ),
            _run(
                "left-2020-i1",
                year=2020,
                iteration=1,
                scenario_id="base",
                model_name="beam",
                status="completed",
                minutes=1,
            ),
            _run(
                "left-2021-i0",
                year=2021,
                iteration=0,
                scenario_id="base",
                model_name="beam",
                status="completed",
                minutes=2,
            ),
        ],
        name="left",
    )
    right = runset_from_runs(
        [
            _run(
                "right-2020-i1",
                year=2020,
                iteration=1,
                scenario_id="policy",
                model_name="beam",
                status="completed",
                minutes=0,
            ),
            _run(
                "right-2020-i2",
                year=2020,
                iteration=2,
                scenario_id="policy",
                model_name="beam",
                status="completed",
                minutes=1,
            ),
            _run(
                "right-2021-i1",
                year=2021,
                iteration=1,
                scenario_id="policy",
                model_name="beam",
                status="completed",
                minutes=2,
            ),
        ],
        name="right",
    )

    calls = []

    def _fake_dataset_frame(_tracker, *, dataset, runset, year, iteration):
        del _tracker, year, iteration
        calls.append((dataset, len(list(runset))))
        return pd.DataFrame(
            [
                {
                    "run_id": run.id,
                    "parent_run_id": run.parent_run_id,
                    "year": run.year,
                    "iteration": run.iteration,
                    "trip_count": 1.0,
                }
                for run in runset
            ]
        )

    monkeypatch.setattr(scenario_compare, "_build_dataset_frame", _fake_dataset_frame)
    monkeypatch.setattr(scenario_compare, "_build_config_diff", lambda *args, **kwargs: pd.DataFrame())

    comparison = scenario_compare.compare_scenarios(
        tracker=object(),
        run_set_left=left,
        run_set_right=right,
        datasets=["linkstats"],
        align_on="year",
        use_converged=True,
    )

    assert comparison.aligned_keys == [2020, 2021]
    assert list(comparison.dataset_summaries["dataset"]) == ["linkstats"]
    assert "linkstats" in comparison.dataset_frames
    assert calls == [("linkstats", 3), ("linkstats", 3)]

