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
import pilates_consist_analysis.comparison_api as comparison_module
from pilates_consist_analysis import open_archive
from pilates_consist_analysis.run_index import build_run_index
from pilates_consist_analysis.runset import runset_from_runs
from pilates_consist_analysis.scenario_compare import ScenarioComparison


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
        self._compare_factory = None

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

    def runset_from_ids(self, run_ids, *, name="runset"):
        selected = [run for run in self.tracker._runs if run.id in set(run_ids)]
        return runset_from_runs(selected, name=name, tracker=self.tracker)

    def compare_scenarios(self, left, right, **kwargs):
        if callable(self._compare_factory):
            return self._compare_factory(left, right, **kwargs)
        raise NotImplementedError("SessionStub.compare_scenarios requires _compare_factory.")


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


def test_build_run_index_prefers_run_attrs_over_facet_copies():
    tracker = TrackerStub(
        [
            _run(
                "asim-new",
                model_name="activitysim",
                scenario_id="baseline",
                year=2030,
                iteration=2,
                metadata={
                    "facet": {
                        "scenario_id": "old-baseline",
                        "year": 1999,
                        "iteration": 9,
                        "model": "stale-model",
                        "seed": 123,
                    }
                },
                status="completed",
            ),
        ]
    )

    frame = build_run_index(tracker).frame
    row = frame.loc[frame["run_id"] == "asim-new"].iloc[0]

    assert row["scenario_id"] == "baseline"
    assert row["scenario_id_source"] == "run_attr"
    assert int(row["year"]) == 2030
    assert row["year_source"] == "run_attr"
    assert int(row["iteration"]) == 2
    assert row["iteration_source"] == "run_attr"
    assert row["model"] == "activitysim"
    assert row["model_source"] == "run_attr"
    assert int(row["seed"]) == 123
    assert row["seed_source"] == "metadata.facet.seed"
    run_index = build_run_index(tracker)
    assert run_index.source_usage == {
        "scenario_id": {"run_attr": 1},
        "year": {"run_attr": 1},
        "iteration": {"run_attr": 1},
        "model": {"run_attr": 1},
        "seed": {"metadata.facet.seed": 1},
    }


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


def test_archive_compare_exposes_notebook_friendly_comparison(monkeypatch, tmp_path):
    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "consist.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")

    runs = [
        _run(
            "baseline-2030-i0",
            model_name="beam",
            scenario_id="baseline",
            year=2030,
            iteration=0,
        ),
        _run(
            "baseline-2030-i1",
            model_name="beam",
            scenario_id="baseline",
            year=2030,
            iteration=1,
        ),
        _run(
            "policy-2030-i0",
            model_name="beam",
            scenario_id="policy",
            year=2030,
            iteration=0,
        ),
        _run(
            "policy-2030-i2",
            model_name="beam",
            scenario_id="policy",
            year=2030,
            iteration=2,
        ),
    ]
    tracker = TrackerStub(runs)
    epochs = [
        EpochStub(2030, 0, "baseline", {"beam": runs[0]}),
        EpochStub(2030, 1, "baseline", {"beam": runs[1]}),
        EpochStub(2030, 0, "policy", {"beam": runs[2]}),
        EpochStub(2030, 2, "policy", {"beam": runs[3]}),
    ]
    session = SessionStub(
        tracker=tracker,
        archive_run_dir=archive_run_dir,
        db_path=db_path,
        epochs=epochs,
    )

    def _comparison_factory(left, right, **kwargs):
        del kwargs
        return ScenarioComparison(
            left_name=getattr(left, "label", "baseline"),
            right_name=getattr(right, "label", "policy"),
            left_run_ids=[run.id for run in left],
            right_run_ids=[run.id for run in right],
            aligned_on="year",
            aligned_keys=[2030],
            config_diff=pd.DataFrame(
                [
                    {
                        "align_on": "year",
                        "on_value": 2030,
                        "key": "beam.param",
                        "namespace": "beam",
                        "status": "changed",
                        "left": "1",
                        "right": "2",
                        "run_id_left": "baseline-2030-i1",
                        "run_id_right": "policy-2030-i2",
                        "left_status": "completed",
                        "right_status": "completed",
                    }
                ]
            ),
            dataset_summaries=pd.DataFrame(
                [
                    {
                        "dataset": "asim_trips",
                        "row_count": 2,
                        "overlap_rows": 2,
                    },
                    {
                        "dataset": "linkstats",
                        "row_count": 1,
                        "overlap_rows": 1,
                    },
                ]
            ),
            dataset_frames={
                "linkstats": pd.DataFrame(
                    [
                        {
                            "year": 2030,
                            "iteration": 1,
                            "volume_baseline": 10.0,
                            "volume_policy": 12.0,
                            "volume_delta": 2.0,
                        }
                    ]
                ),
                "asim_trips": pd.DataFrame(
                    [
                        {
                            "year": 2030,
                            "iteration": 1,
                            "total_trips_baseline": 100,
                            "total_trips_policy": 110,
                            "total_trips_delta": 10,
                        }
                    ]
                ),
            },
        )

    session._compare_factory = _comparison_factory

    class TripsDatasetStub:
        def __init__(self):
            self.mode_counts = pd.DataFrame(
                [
                    {
                        "comparison_group": "baseline-2030-i1",
                        "run_id": "baseline-2030-i1",
                        "year": 2030,
                        "iteration": 1,
                        "trip_mode": "DRIVE",
                        "trip_count": 60,
                        "mode_share": 0.6,
                    },
                    {
                        "comparison_group": "policy-2030-i2",
                        "run_id": "policy-2030-i2",
                        "year": 2030,
                        "iteration": 2,
                        "trip_mode": "DRIVE",
                        "trip_count": 50,
                        "mode_share": 0.5,
                    },
                    {
                        "comparison_group": "baseline-2030-i1",
                        "run_id": "baseline-2030-i1",
                        "year": 2030,
                        "iteration": 1,
                        "trip_mode": "WALK",
                        "trip_count": 40,
                        "mode_share": 0.4,
                    },
                    {
                        "comparison_group": "policy-2030-i2",
                        "run_id": "policy-2030-i2",
                        "year": 2030,
                        "iteration": 2,
                        "trip_mode": "WALK",
                        "trip_count": 60,
                        "mode_share": 0.5,
                    },
                ]
            )
            self.purpose_mode_counts = pd.DataFrame(
                [
                    {
                        "comparison_group": "baseline-2030-i1",
                        "run_id": "baseline-2030-i1",
                        "year": 2030,
                        "iteration": 1,
                        "primary_purpose": "work",
                        "trip_mode": "DRIVE",
                        "trip_count": 35,
                    },
                    {
                        "comparison_group": "policy-2030-i2",
                        "run_id": "policy-2030-i2",
                        "year": 2030,
                        "iteration": 2,
                        "primary_purpose": "work",
                        "trip_mode": "DRIVE",
                        "trip_count": 30,
                    },
                ]
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
    monkeypatch.setattr(
        comparison_module,
        "build_activitysim_trips_dataset",
        lambda *args, **kwargs: TripsDatasetStub(),
    )

    archive = open_archive(archive_run_dir, project_root=tmp_path / "project")
    comparison = archive.compare(
        "baseline",
        "policy",
        year=2030,
        converged=True,
        datasets=["linkstats", "asim_trips"],
    )

    summary = comparison.summary()
    assert int(summary.iloc[0]["aligned_key_count"]) == 1
    assert bool(summary.iloc[0]["use_converged"]) is True
    assert comparison.aligned_keys == [2030]

    config_diff = comparison.config_diff()
    assert list(config_diff["key"]) == ["beam.param"]

    linkstats = comparison.linkstats_summary()
    assert list(linkstats["volume_delta"]) == [2.0]

    mode_shares = comparison.mode_shares()
    assert set(mode_shares["trip_mode"]) == {"DRIVE", "WALK"}
    assert "mode_share_delta" in mode_shares.columns

    trip_purposes = comparison.trip_purposes()
    assert set(trip_purposes["primary_purpose"]) == {"work"}
    assert "trip_count_delta" in trip_purposes.columns

    scenario_comparison = archive.scenario("baseline").compare(
        "policy",
        year=2030,
        converged=True,
        datasets=["linkstats"],
    )
    assert scenario_comparison.left_name == "baseline"
    assert scenario_comparison.right_name == "policy"
