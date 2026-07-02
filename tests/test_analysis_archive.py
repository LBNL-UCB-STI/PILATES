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

import pilates_consist_analysis.archive as archive_module
from pilates_consist_analysis import delta, delta_change, difference, open_archive, rank
from pilates_consist_analysis.run_index import build_run_index
from pilates_consist_analysis.runset import runset_from_runs


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
        self._artifacts = []

    def run_set(self, label, limit=200000):
        del label, limit
        return list(self._runs)

    def find_artifacts_by_params(self, *, params, namespace=None, limit=1000):
        del namespace, limit
        wanted = {}
        for param in params:
            key, _, value = str(param).partition("=")
            wanted[key.split(".")[-1]] = value
        output = []
        for artifact in self._artifacts:
            meta = getattr(artifact, "meta", {}) or {}
            facet = meta.get("facet", {}) if isinstance(meta, dict) else {}
            if all(str(facet.get(key, "")) == value for key, value in wanted.items()):
                output.append(artifact)
        return output


@dataclass
class FakeArtifact:
    id: str
    key: str
    run_id: str
    uri: str
    abs_path: str
    meta: dict


class FakeColumn:
    def __init__(self, table, name):
        self.table = table
        self.name = name

    def isin(self, values):
        return ("isin", self.name, set(values))

    def __eq__(self, other):
        return ("eq", self.name, other)


class FakeGroupedTable:
    def __init__(self, table, by):
        self.table = table
        self.by = list(by)

    def agg(self, **measures):
        rows = []
        grouped = self.table._frame.groupby(self.by, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(self.by, keys))
            for name, expr in measures.items():
                if expr == ("count",):
                    row[name] = len(group)
                elif isinstance(expr, tuple) and expr[0] == "sum":
                    row[name] = group[expr[1]].sum()
                else:
                    raise AssertionError(f"Unhandled fake aggregate expression: {expr!r}")
            rows.append(row)
        return FakeIbisTable(pd.DataFrame(rows))


class FakeIbisTable:
    _pilates_fake_table = True

    def __init__(self, frame):
        self._frame = frame.copy()

    @property
    def columns(self):
        return list(self._frame.columns)

    def __getitem__(self, name):
        return FakeColumn(self, name)

    def count(self):
        return ("count",)

    def sum(self, column):
        return ("sum", column)

    def filter(self, predicate):
        op, column, value = predicate
        if op == "isin":
            return FakeIbisTable(self._frame.loc[self._frame[column].isin(value)])
        if op == "eq":
            return FakeIbisTable(self._frame.loc[self._frame[column] == value])
        raise AssertionError(f"Unhandled fake predicate: {predicate!r}")

    def mutate(self, **values):
        frame = self._frame.copy()
        for name, value in values.items():
            if isinstance(value, FakeColumn):
                frame[name] = frame[value.name]
            else:
                frame[name] = value
        return FakeIbisTable(frame)

    def group_by(self, by):
        return FakeGroupedTable(self, by)

    def to_pandas(self):
        return self._frame.copy()


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
            epoch
            for epoch in self._panel
            if str(epoch.scenario_id or "") == str(scenario_id)
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
        raise NotImplementedError(
            "SessionStub.compare_scenarios requires _compare_factory."
        )


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


def test_build_run_index_keeps_grouping_stable_across_old_and_new_metadata_shapes():
    tracker = TrackerStub(
        [
            _run(
                "old-asim",
                model_name=None,
                scenario_id=None,
                year=None,
                iteration=None,
                metadata={
                    "facet": {
                        "scenario_id": "baseline",
                        "year": 2030,
                        "iteration": 1,
                        "model": "activitysim",
                        "seed": 17,
                    }
                },
                status="completed",
            ),
            _run(
                "new-beam",
                model_name="beam",
                scenario_id="baseline",
                year=2030,
                iteration=1,
                parent_run_id="old-asim",
                metadata={"facet": {"scenario_id": "stale-copy", "year": 1999}},
                status="completed",
            ),
            _run(
                "new-usim",
                model_name="urbansim",
                scenario_id="baseline",
                year=2031,
                iteration=0,
                status="completed",
            ),
            _run(
                "old-atlas",
                model_name=None,
                scenario_id=None,
                year=None,
                iteration=None,
                metadata={
                    "facet": {
                        "scenario_id": "baseline",
                        "year": 2031,
                        "iteration": 0,
                        "model": "atlas",
                    }
                },
                status="completed",
            ),
        ]
    )

    run_index = build_run_index(tracker, archive_run_dir=Path("/tmp/archive"))
    frame = run_index.frame
    tuples = set(
        tuple(row)
        for row in frame.loc[
            :, ["scenario_id", "year", "iteration", "model"]
        ].itertuples(index=False, name=None)
    )

    assert tuples == {
        ("baseline", 2030, 1, "activitysim"),
        ("baseline", 2030, 1, "beam"),
        ("baseline", 2031, 0, "urbansim"),
        ("baseline", 2031, 0, "atlas"),
    }
    assert run_index.source_usage["scenario_id"] == {
        "metadata.facet.scenario_id": 2,
        "run_attr": 2,
    }
    assert run_index.source_usage["year"] == {
        "metadata.facet.year": 2,
        "run_attr": 2,
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

    monkeypatch.setattr(
        archive_module.AnalysisSession, "open", lambda **kwargs: session
    )
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
    assert list(converged.sql("SELECT * FROM {views.trips} LIMIT 1")["person_id"]) == [
        1
    ]
    assert converged.tables.view_name("linkstats") == "v_linkstats"

    converged_frame = baseline.epochs(converged=True)
    assert set(converged_frame["run_id"]) == {
        "baseline-2030-i1-beam",
        "baseline-2031-i0-asim",
    }


def test_archive_table_returns_faceted_ibis_table(monkeypatch, tmp_path):
    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "consist.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")
    artifact_path = archive_run_dir / "activitysim" / "output" / "trips.parquet"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("placeholder", encoding="utf-8")

    tracker = TrackerStub(
        [
            _run(
                "asim-2020",
                model_name="activitysim",
                scenario_id="baseline",
                year=2020,
                iteration=0,
            )
        ]
    )
    tracker._artifacts = [
        FakeArtifact(
            id="artifact-1",
            key="trips_asim_out",
            run_id="asim-2020",
            uri="workspace://activitysim/output/trips.parquet",
            abs_path=str(artifact_path),
            meta={"facet": {"artifact_family": "trips", "pricing_policy": "none"}},
        )
    ]
    session = SessionStub(
        tracker=tracker,
        archive_run_dir=archive_run_dir,
        db_path=db_path,
        epochs=[],
    )

    captured = {}

    def fake_grouped_view(**kwargs):
        captured.update(kwargs)
        return FakeIbisTable(
            pd.DataFrame(
                [
                    {
                        "trip_id": 1,
                        "facet_scenario_id": "baseline",
                        "facet_year": 2020,
                        "facet_iteration": 0,
                        "facet_pricing_policy": "none",
                    },
                    {
                        "trip_id": 2,
                        "facet_scenario_id": "baseline",
                        "facet_year": 2030,
                        "facet_iteration": 0,
                        "facet_pricing_policy": "none",
                    },
                    {
                        "trip_id": 3,
                        "facet_scenario_id": "policy",
                        "facet_year": 2040,
                        "facet_iteration": 0,
                        "facet_pricing_policy": "cordon",
                    },
                ]
            )
        )

    monkeypatch.setattr(
        archive_module.AnalysisSession, "open", lambda **kwargs: session
    )
    monkeypatch.setattr(archive_module, "_open_grouped_ibis_table", fake_grouped_view)

    archive = open_archive(archive_run_dir, project_root=tmp_path / "project")
    table = archive.table(
        "activitysim.trips",
        facets=["scenario_id", "year", "iteration", "pricing_policy"],
        where={"year": [2020, 2030]},
    )

    assert captured["artifact_id"] == "artifact-1"
    assert captured["namespace"] == "activitysim"
    assert "artifact_family=trips" in captured["params"]
    assert captured["attach_facets"] == [
        "scenario_id",
        "year",
        "iteration",
        "pricing_policy",
    ]
    frame = table.to_pandas()
    assert set(["scenario_id", "year", "iteration", "pricing_policy"]).issubset(
        frame.columns
    )
    assert list(frame["year"]) == [2020, 2030]


def test_archive_measure_groups_by_arbitrary_facets():
    table = FakeIbisTable(
        pd.DataFrame(
            [
                {"scenario_id": "baseline", "pricing_policy": "none", "year": 2020},
                {"scenario_id": "baseline", "pricing_policy": "none", "year": 2020},
                {"scenario_id": "baseline", "pricing_policy": "none", "year": 2030},
                {"scenario_id": "policy", "pricing_policy": "cordon", "year": 2030},
            ]
        )
    )
    archive = archive_module.Archive.__new__(archive_module.Archive)

    measured = archive.measure(
        table,
        by=["scenario_id", "pricing_policy", "year"],
        measures={"trips": lambda t: t.count()},
    )

    frame = measured.to_pandas().sort_values(
        ["scenario_id", "pricing_policy", "year"]
    )
    assert frame.to_dict("records") == [
        {
            "scenario_id": "baseline",
            "pricing_policy": "none",
            "year": 2020,
            "trips": 2,
        },
        {
            "scenario_id": "baseline",
            "pricing_policy": "none",
            "year": 2030,
            "trips": 1,
        },
        {
            "scenario_id": "policy",
            "pricing_policy": "cordon",
            "year": 2030,
            "trips": 1,
        },
    ]


def test_faceted_helpers_handle_scenario_and_parameter_sweep_comparisons():
    measured = FakeIbisTable(
        pd.DataFrame(
            [
                {"scenario_id": "baseline", "pricing_policy": "none", "year": 2020, "trips": 10},
                {"scenario_id": "baseline", "pricing_policy": "none", "year": 2030, "trips": 14},
                {"scenario_id": "policy-a", "pricing_policy": "cordon", "year": 2030, "trips": 20},
                {"scenario_id": "policy-b", "pricing_policy": "mileage", "year": 2030, "trips": 18},
            ]
        )
    )

    year_delta = delta(
        measured,
        value="trips",
        over="year",
        by=["scenario_id", "pricing_policy"],
    ).to_pandas()
    baseline_2030 = year_delta.loc[
        (year_delta["scenario_id"] == "baseline") & (year_delta["year"] == 2030)
    ].iloc[0]
    assert int(baseline_2030["trips_delta"]) == 4

    sweep = difference(
        measured,
        value="trips",
        compare="pricing_policy",
        baseline="none",
        at={"year": 2030},
    ).to_pandas().sort_values("pricing_policy")
    assert sweep.loc[sweep["pricing_policy"] == "cordon", "trips_difference"].iloc[0] == 6
    assert sweep.loc[sweep["pricing_policy"] == "mileage", "trips_difference"].iloc[0] == 4

    ranked = rank(
        measured,
        value="trips",
        by=["year"],
        descending=True,
    ).to_pandas()
    assert list(ranked.loc[ranked["year"] == 2030].sort_values("trips_rank")["pricing_policy"]) == [
        "cordon",
        "mileage",
        "none",
    ]


def test_delta_change_computes_change_in_iteration_delta():
    measured = FakeIbisTable(
        pd.DataFrame(
            [
                {"scenario_id": "baseline", "year": 2030, "iteration": 0, "trips": 100},
                {"scenario_id": "baseline", "year": 2030, "iteration": 1, "trips": 130},
                {"scenario_id": "baseline", "year": 2030, "iteration": 2, "trips": 145},
            ]
        )
    )

    changed = delta_change(
        measured,
        value="trips",
        over="iteration",
        by=["scenario_id", "year"],
    ).to_pandas()

    row = changed.loc[changed["iteration"] == 2].iloc[0]
    assert int(row["trips_delta"]) == 15
    assert int(row["trips_delta_change"]) == -15


def test_open_archive_local_cache_copies_db_and_selected_artifacts(monkeypatch, tmp_path):
    persistent = tmp_path / "persistent" / "run"
    db_path = persistent / ".consist" / "consist.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("db", encoding="utf-8")
    artifact_path = persistent / "activitysim" / "output" / "trips.parquet"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("trips", encoding="utf-8")
    local_cache = tmp_path / "local-cache"

    sessions = []

    def fake_open(**kwargs):
        archive_dir = Path(kwargs["archive_run_dir"])
        tracker = TrackerStub([])
        tracker._artifacts = [
            FakeArtifact(
                id="artifact-1",
                key="trips_asim_out",
                run_id="asim-2020",
                uri="workspace://activitysim/output/trips.parquet",
                abs_path=str(artifact_path),
                meta={"facet": {"artifact_family": "trips"}},
            )
        ]
        session = SessionStub(
            tracker=tracker,
            archive_run_dir=archive_dir,
            db_path=Path(kwargs["db_path"]),
            epochs=[],
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(archive_module.AnalysisSession, "open", fake_open)
    monkeypatch.setattr(
        archive_module,
        "_open_grouped_ibis_table",
        lambda **_kwargs: FakeIbisTable(pd.DataFrame([{"facet_year": 2030}])),
    )

    archive = open_archive(
        persistent,
        project_root=tmp_path / "project",
        local_cache=local_cache,
    )
    archive.table("activitysim.trips", facets=["year"])

    assert sessions[0].archive_run_dir == local_cache / "run"
    assert sessions[0].db_path == local_cache / "run" / ".consist" / "consist.duckdb"
    assert sessions[0].db_path.read_text(encoding="utf-8") == "db"
    assert (
        local_cache / "run" / "activitysim" / "output" / "trips.parquet"
    ).read_text(encoding="utf-8") == "trips"
    manifest = archive.localization_manifest()
    assert manifest["artifacts"][0]["artifact_id"] == "artifact-1"
    assert manifest["artifacts"][0]["local_path"].endswith(
        "activitysim/output/trips.parquet"
    )


