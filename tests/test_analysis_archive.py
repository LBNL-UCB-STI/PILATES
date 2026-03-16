from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

import pilates_consist_analysis.archive as archive_module
from pilates_consist_analysis import open_archive
from pilates_consist_analysis.run_index import build_run_index


@dataclass
class FakeRun:
    id: str
    created_at: datetime
    status: str = "completed"
    model_name: str | None = None
    scenario_id: str | None = None
    year: int | None = None
    iteration: int | None = None
    parent_run_id: str | None = None
    metadata: dict | None = None
    ended_at: datetime | None = None
    name: str | None = None


class TrackerStub:
    def __init__(self, runs):
        self._runs = list(runs)
        self.db = None

    def run_set(self, label, limit=200000):
        del label, limit
        return list(self._runs)


class QueryResultStub:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def df(self) -> pd.DataFrame:
        return self._frame.copy()


class DbStub:
    def __init__(self, frames_by_view_name: dict[str, pd.DataFrame]):
        self._frames_by_view_name = {
            key: value.copy() for key, value in frames_by_view_name.items()
        }

    def query(self, sql: str) -> QueryResultStub:
        normalized = str(sql).strip()
        for view_name, frame in self._frames_by_view_name.items():
            if view_name in normalized:
                if "LIMIT 1" in normalized:
                    return QueryResultStub(frame.head(1))
                return QueryResultStub(frame)
        raise KeyError(f"Unhandled SQL in test stub: {normalized}")


@dataclass
class EpochStub:
    year: int
    outer_iteration: int
    scenario_id: str | None
    runs: dict

    def run_ids(self) -> dict[str, str]:
        return {
            key: str(getattr(run, "id", ""))
            for key, run in self.runs.items()
            if getattr(run, "id", None)
        }


class PanelStub:
    def __init__(self, epochs):
        self._epochs = list(epochs)

    def __iter__(self):
        return iter(self._epochs)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for epoch in self._epochs:
            for model, run in epoch.runs.items():
                rows.append(
                    {
                        "year": epoch.year,
                        "outer_iteration": epoch.outer_iteration,
                        "scenario_id": epoch.scenario_id,
                        "model": model,
                        "run_id": run.id,
                    }
                )
        return pd.DataFrame(rows)

    def converged_epochs(self):
        latest: dict[tuple[int, str | None], EpochStub] = {}
        for epoch in self._epochs:
            key = (epoch.year, epoch.scenario_id)
            current = latest.get(key)
            if current is None or epoch.outer_iteration > current.outer_iteration:
                latest[key] = epoch
        return PanelStub(list(latest.values()))


class ViewsStub:
    def __init__(self, tracker):
        self._tracker = tracker
        self.trips = "v_trips"
        self.persons = "v_persons"
        self.households = "v_households"
        self.land_use = "v_land_use"
        self.linkstats = "v_linkstats"
        self.skim_summary = pd.DataFrame(
            [
                {
                    "concept_key": "omx_skims",
                    "run_id": "asim-run",
                    "year": 2030,
                    "iteration": 1,
                    "matrix_name": "DIST",
                    "n_rows": 3,
                    "n_cols": 3,
                }
            ]
        )

    def query(self, sql: str) -> pd.DataFrame:
        result = self._tracker.db.query(sql.format(views=self))
        return result.df()


class SessionStub:
    def __init__(self, tracker, archive_run_dir: Path, db_path: Path, epochs):
        self.tracker = tracker
        self.archive_run_dir = archive_run_dir
        self.db_path = db_path
        self.tagging_issues = []
        self.tagging_warnings = []
        self._panel = PanelStub(epochs)

    def epochs(self, *, scenario_id=None, models=None):
        del models
        if scenario_id is None:
            return self._panel
        filtered = [
            epoch for epoch in self._panel if str(epoch.scenario_id or "") == str(scenario_id)
        ]
        return PanelStub(filtered)

    def converged_epoch(self, *, year, scenario_id=None, models=None):
        del models
        panel = self.epochs(scenario_id=scenario_id).converged_epochs()
        matches = [epoch for epoch in panel if epoch.year == year]
        if not matches:
            raise ValueError("No converged epoch found.")
        return matches[0]

    def views(self, epoch):
        del epoch
        return ViewsStub(self.tracker)


def _run(run_id: str, *, minutes: int = 0, **kwargs) -> FakeRun:
    payload = dict(kwargs)
    payload.setdefault("metadata", {})
    return FakeRun(
        id=run_id,
        created_at=datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=minutes),
        ended_at=None,
        **payload,
    )


def test_build_run_index_normalizes_metadata_and_sources():
    tracker = TrackerStub(
        [
            _run(
                "beam-a",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                parent_run_id="asim-a",
                status="completed",
            ),
            _run(
                "asim-a",
                model_name=None,
                scenario_id=None,
                year=None,
                iteration=None,
                metadata={
                    "facet": {
                        "scenario_id": "baseline",
                        "year": 2030,
                        "iteration": 2,
                        "model": "activitysim",
                        "seed": 17,
                    }
                },
                status="completed",
            ),
        ]
    )

    run_index = build_run_index(
        tracker,
        archive_run_dir=Path("/tmp/archive"),
    )
    frame = run_index.frame

    assert list(frame["run_id"]) == ["asim-a", "beam-a"]
    asim_row = frame.loc[frame["run_id"] == "asim-a"].iloc[0]
    beam_row = frame.loc[frame["run_id"] == "beam-a"].iloc[0]

    assert asim_row["scenario_id"] == "baseline"
    assert asim_row["scenario_id_source"] == "metadata.facet.scenario_id"
    assert int(asim_row["year"]) == 2030
    assert asim_row["year_source"] == "metadata.facet.year"
    assert int(asim_row["iteration"]) == 2
    assert asim_row["model"] == "activitysim"
    assert int(asim_row["seed"]) == 17
    assert asim_row["seed_source"] == "metadata.facet.seed"
    assert bool(asim_row["is_converged_candidate"]) is True
    assert beam_row["scenario_id_source"] == "run_attr"
    assert bool(beam_row["has_parent"]) is True


def test_run_index_filter_scopes_scenarios_and_years():
    tracker = TrackerStub(
        [
            _run(
                "baseline-2030",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=1,
            ),
            _run(
                "baseline-2031",
                model_name="activitysim",
                scenario_id="baseline",
                year=2031,
                iteration=0,
            ),
            _run(
                "policy-2030",
                model_name="beam",
                scenario_id="policy",
                year=2030,
                iteration=1,
            ),
        ]
    )

    run_index = build_run_index(tracker)
    assert run_index.scenarios() == ["baseline", "policy"]
    assert run_index.years(scenario_id="baseline") == [2030, 2031]
    assert run_index.models(scenario_id="baseline") == ["activitysim", "beam"]

    filtered = run_index.filter(scenario_id="policy", year=2030)
    assert list(filtered["run_id"]) == ["policy-2030"]


def test_open_archive_exposes_notebook_friendly_discovery(monkeypatch, tmp_path):
    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "consist.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")

    runs = [
        _run(
            "baseline-2030-i0-beam",
            model_name="beam",
            scenario_id="baseline",
            year=2030,
            iteration=0,
        ),
        _run(
            "baseline-2030-i1-beam",
            model_name="beam",
            scenario_id="baseline",
            year=2030,
            iteration=1,
        ),
        _run(
            "baseline-2031-i0-asim",
            model_name="activitysim",
            scenario_id="baseline",
            year=2031,
            iteration=0,
        ),
        _run(
            "policy-2030-i0-beam",
            model_name="beam",
            scenario_id="policy",
            year=2030,
            iteration=0,
        ),
    ]
    tracker = TrackerStub(runs)
    tracker.db = DbStub(
        {
            "v_trips": pd.DataFrame(
                [{"person_id": 1, "trip_mode": "DRIVE", "depart": 8}]
            ),
            "v_persons": pd.DataFrame(
                [{"person_id": 1, "household_id": 10, "value_of_time": 15.0}]
            ),
            "v_households": pd.DataFrame([{"household_id": 10, "income": 75000}]),
            "v_land_use": pd.DataFrame([{"TAZ": 1, "TOTEMP": 20}]),
            "v_linkstats": pd.DataFrame([{"link": 100, "volume": 42.0}]),
        }
    )
    epochs = [
        EpochStub(2030, 0, "baseline", {"beam": runs[0]}),
        EpochStub(2030, 1, "baseline", {"beam": runs[1]}),
        EpochStub(2031, 0, "baseline", {"activitysim": runs[2]}),
        EpochStub(2030, 0, "policy", {"beam": runs[3]}),
    ]
    session = SessionStub(
        tracker=tracker,
        archive_run_dir=archive_run_dir,
        db_path=db_path,
        epochs=epochs,
    )

    monkeypatch.setattr(archive_module.AnalysisSession, "open", lambda **kwargs: session)
    monkeypatch.setattr(
        archive_module,
        "get_db_health",
        lambda *_args, **_kwargs: {"healthy": True},
    )
    monkeypatch.setattr(
        archive_module,
        "get_db_health_issues",
        lambda *_args, **_kwargs: [],
    )

    archive = open_archive(archive_run_dir, project_root=tmp_path / "project")

    assert archive.scenarios() == ["baseline", "policy"]
    assert archive.years() == [2030, 2031]
    assert archive.models() == ["activitysim", "beam"]
    summary = archive.summary()
    assert bool(summary.iloc[0]["db_healthy"]) is True
    assert int(summary.iloc[0]["scenario_count"]) == 2

    baseline = archive.scenario("baseline")
    assert baseline.years() == [2030, 2031]
    assert list(baseline.runs(year=2030)["run_id"]) == [
        "baseline-2030-i0-beam",
        "baseline-2030-i1-beam",
    ]

    converged = baseline.epoch(year=2030, converged=True)
    assert int(converged.outer_iteration) == 1
    assert converged.run_ids() == {"beam": "baseline-2030-i1-beam"}
    assert converged.raw is epochs[1]
    assert "linkstats" in converged.tables.available()
    assert list(converged.tables.linkstats()["link"]) == [100]
    assert list(converged.tables.trips(limit=1)["trip_mode"]) == ["DRIVE"]
    assert list(converged.tables.persons(columns=["person_id"])["person_id"]) == [1]
    assert int(converged.tables.skim_summary().iloc[0]["n_rows"]) == 3
    assert list(converged.sql("SELECT * FROM {views.trips} LIMIT 1")["person_id"]) == [1]
    assert converged.tables.view_name("linkstats") == "v_linkstats"

    converged_frame = baseline.epochs(converged=True)
    assert set(converged_frame["run_id"]) == {
        "baseline-2030-i1-beam",
        "baseline-2031-i0-asim",
    }
