from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

import pilates_consist_analysis.cli as cli_module
import pilates_consist_analysis.handoff as handoff


@dataclass
class FakeArtifact:
    id: str
    key: str
    uri: str
    meta: dict
    container_uri: str | None = None
    abs_path: str | None = None
    run_id: str | None = None
    driver: str | None = None


@dataclass
class FakeRunArtifacts:
    inputs: dict
    outputs: dict


@dataclass
class FakeRunRecord:
    id: str
    meta: dict
    year: int | None = None
    iteration: int | None = None


class FakeQueryResult:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def df(self) -> pd.DataFrame:
        return self._frame.copy()


class FakeDB:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.sql_calls: list[str] = []

    def query(self, sql: str):
        self.sql_calls.append(sql)
        return FakeQueryResult(self.frame)


class FakeTracker:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.db = FakeDB(frame if frame is not None else pd.DataFrame())
        self.access_mode = "standard"
        self.started: list[dict] = []
        self.logged: list[dict] = []
        self.ingested: list[str] = []
        self._run_artifacts: dict[str, FakeRunArtifacts] = {}
        self._runs: dict[str, FakeRunRecord] = {}

    @contextmanager
    def start_run(self, run_id: str, model: str, **kwargs):
        self.started.append({"run_id": run_id, "model": model, **kwargs})
        yield self

    def log_artifact(self, path: str, key: str, direction: str, driver=None, **meta):
        self.logged.append(
            {
                "path": path,
                "key": key,
                "direction": direction,
                "driver": driver,
                "meta": dict(meta),
            }
        )
        return FakeArtifact(
            id=f"art-{len(self.logged)}",
            key=key,
            uri=f"workspace://{Path(path).name}",
            container_uri=f"workspace://{Path(path).name}",
            abs_path=str(Path(path).resolve()),
            run_id=(self.started[-1]["run_id"] if self.started else None),
            driver=driver,
            meta={"is_ingested": False},
        )

    def ingest(self, artifact: FakeArtifact, profile_schema: bool = True):
        del profile_schema
        artifact.meta["is_ingested"] = True
        self.ingested.append(artifact.key)
        return {"ok": True}

    def get_artifacts_for_run(self, run_id: str):
        return self._run_artifacts.get(
            run_id, FakeRunArtifacts(inputs={}, outputs={})
        )

    def get_run(self, run_id: str):
        return self._runs.get(run_id)

    def find_artifacts(self, creator=None, consumer=None, key=None, limit=50):
        del creator, consumer, key, limit
        return []

    def resolve_uri(self, uri: str):
        return uri


class FakeRun:
    def __init__(self, run_id: str):
        self.id = run_id


class FakeRunSet:
    def __init__(self, run_ids):
        self._run_ids = list(run_ids)

    def filter(self, **kwargs):
        del kwargs
        return self

    def converged(self, group_by=None):
        del group_by
        return self

    def latest(self, group_by=None):
        del group_by
        return self

    def __iter__(self):
        for run_id in self._run_ids:
            yield FakeRun(run_id)


def test_parse_artifact_arg_supports_key_equals_path():
    spec = handoff.parse_artifact_arg(
        "trips=/tmp/trips.csv.gz",
        direction="output",
        driver="csv",
        artifact_family="trips",
    )
    assert spec.key == "trips"
    assert str(spec.path) == "/tmp/trips.csv.gz"
    assert spec.direction == "output"
    assert spec.driver == "csv"
    assert spec.artifact_family == "trips"


def test_parse_artifact_ref_arg_supports_run_key_format():
    spec = handoff.parse_artifact_ref_arg("run-123:trips")
    assert spec.source_run_id == "run-123"
    assert spec.source_key == "trips"
    assert spec.path is None


def test_ingest_artifacts_logs_and_ingests(tmp_path):
    csv_path = tmp_path / "trips.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    tracker = FakeTracker()
    payload = handoff.ingest_artifacts(
        tracker,
        [
            handoff.ArtifactIngestSpec(
                path=csv_path,
                key="trips",
                artifact_family="trips",
                driver="csv",
            )
        ],
        run_id="ingest-run",
        model="analysis_ingest",
        scenario_id="baseline-2030",
        year=2030,
        iteration=3,
        seed=123,
    )

    assert payload["run_id"] == "ingest-run"
    assert payload["artifact_count"] == 1
    assert tracker.started
    assert tracker.logged[0]["key"] == "trips"
    assert tracker.logged[0]["meta"]["artifact_family"] == "trips"
    assert tracker.ingested == ["trips"]


def test_ingest_artifacts_resolves_source_run_key(tmp_path):
    csv_path = tmp_path / "trips.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    tracker = FakeTracker()
    tracker._runs["source-run"] = FakeRunRecord(id="source-run", meta={})
    tracker._run_artifacts["source-run"] = FakeRunArtifacts(
        inputs={},
        outputs={
            "trips": FakeArtifact(
                id="src-art-1",
                key="trips",
                uri="workspace://trips.csv",
                container_uri="workspace://trips.csv",
                abs_path=str(csv_path.resolve()),
                run_id="source-run",
                driver="csv",
                meta={"artifact_family": "trips"},
            )
        },
    )

    payload = handoff.ingest_artifacts(
        tracker,
        [
            handoff.ArtifactIngestSpec(
                source_run_id="source-run",
                source_key="trips",
            )
        ],
        run_id="ingest-run",
    )

    assert payload["artifact_count"] == 1
    assert tracker.logged[0]["path"] == str(csv_path.resolve())
    assert tracker.logged[0]["meta"]["artifact_family"] == "trips"


def test_ingest_artifacts_requires_standard_mode(tmp_path):
    csv_path = tmp_path / "trips.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    tracker = FakeTracker()
    tracker.access_mode = "analysis"

    with pytest.raises(RuntimeError, match="access_mode='standard'"):
        handoff.ingest_artifacts(
            tracker,
            [handoff.ArtifactIngestSpec(path=csv_path, key="trips")],
        )


def test_export_sql_query_writes_csv(tmp_path):
    frame = pd.DataFrame({"trip_mode": ["car", "walk"], "n": [10, 5]})
    tracker = FakeTracker(frame=frame)

    output_path = tmp_path / "mode_counts.csv"
    payload = handoff.export_sql_query(
        tracker,
        sql="SELECT * FROM v_trips",
        output_path=output_path,
        output_format="csv",
        limit=1,
    )

    assert output_path.exists()
    exported = pd.read_csv(output_path)
    assert len(exported) == 1
    assert payload["row_count"] == 1
    assert tracker.db.sql_calls == ["SELECT * FROM v_trips"]


def test_list_run_artifacts_resolves_paths(tmp_path):
    csv_path = tmp_path / "trips.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    tracker = FakeTracker()
    tracker._runs["run-1"] = FakeRunRecord(id="run-1", meta={})
    tracker._run_artifacts["run-1"] = FakeRunArtifacts(
        inputs={},
        outputs={
            "trips": FakeArtifact(
                id="src-art-1",
                key="trips",
                uri="workspace://trips.csv",
                container_uri="workspace://trips.csv",
                abs_path=str(csv_path.resolve()),
                run_id="run-1",
                driver="csv",
                meta={"artifact_family": "trips"},
            )
        },
    )

    frame = handoff.list_run_artifacts(tracker, run_id="run-1")
    assert len(frame) == 1
    assert frame.iloc[0]["key"] == "trips"
    assert bool(frame.iloc[0]["path_exists"]) is True


def test_list_run_artifacts_includes_tagged_and_content_years(tmp_path):
    skim_path = (
        tmp_path
        / "activitysim"
        / "output"
        / "inputs-year-2035-iteration-0"
        / "skims.zarr"
    )
    skim_path.parent.mkdir(parents=True, exist_ok=True)
    skim_path.write_text("placeholder", encoding="utf-8")

    tracker = FakeTracker()
    tracker._runs["run-1"] = FakeRunRecord(id="run-1", meta={}, year=2041, iteration=0)
    tracker._run_artifacts["run-1"] = FakeRunArtifacts(
        inputs={},
        outputs={
            "asim_input_skims_zarr_archived": FakeArtifact(
                id="src-art-1",
                key="asim_input_skims_zarr_archived",
                uri="workspace://activitysim/output/inputs-year-2035-iteration-0/skims.zarr",
                container_uri="workspace://activitysim/output/inputs-year-2035-iteration-0/skims.zarr",
                abs_path=str(skim_path.resolve()),
                run_id="run-1",
                driver="zarr",
                meta={"artifact_family": "skims"},
            )
        },
    )

    frame = handoff.list_run_artifacts(tracker, run_id="run-1")
    assert int(frame.iloc[0]["tagged_year"]) == 2041
    assert int(frame.iloc[0]["tagged_iteration"]) == 0
    assert int(frame.iloc[0]["content_year"]) == 2035
    assert int(frame.iloc[0]["content_iteration"]) == 0
    assert frame.iloc[0]["content_path_kind"] == "activitysim_input_snapshot"


def test_resolve_urbansim_activitysim_boundary_h5s_discovers_next_input_snapshot(tmp_path):
    data_dir = tmp_path / "archive" / "urbansim" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "model_data_2029.h5").write_bytes(b"")
    (data_dir / "input_data_for_2035_outputs.h5").write_bytes(b"")
    (data_dir / "custom_mpo_06197001_model_data.h5").write_bytes(b"")

    frame = handoff.resolve_urbansim_activitysim_boundary_h5s(
        tmp_path / "archive",
        forecast_year=2029,
    )

    pre = frame.loc[frame["boundary_role"].eq("pre_urbansim_forecast_output")].iloc[0]
    post = frame.loc[frame["boundary_role"].eq("post_activitysim_next_input")].iloc[0]
    rolling = frame.loc[frame["boundary_role"].eq("rolling_urbansim_input")].iloc[0]

    assert int(pre["year"]) == 2029
    assert pre["path"].endswith("model_data_2029.h5")
    assert bool(pre["path_exists"]) is True

    assert int(post["year"]) == 2035
    assert post["path"].endswith("input_data_for_2035_outputs.h5")
    assert bool(post["path_exists"]) is True

    assert rolling["path"].endswith("custom_mpo_06197001_model_data.h5")
    assert bool(rolling["path_exists"]) is True


def test_export_scenario_bundle_uses_runset_and_export(monkeypatch, tmp_path):
    tracker = FakeTracker()

    calls: dict[str, object] = {}

    def _fake_runset_from_query(**kwargs):
        calls["runset_query"] = kwargs
        return FakeRunSet(["run-a", "run-b"])

    def _fake_export_bundle(_tracker, **kwargs):
        calls["export"] = kwargs
        return {"artifact_count": 2, "out_path": str(kwargs["out_path"])}

    monkeypatch.setattr(handoff, "runset_from_query", _fake_runset_from_query)
    monkeypatch.setattr(handoff, "export_bundle", _fake_export_bundle)

    payload = handoff.export_scenario_bundle(
        tracker,
        archive_run_dir=tmp_path,
        out_path=tmp_path / "subset.duckdb",
        scenario_id="baseline-2030",
        limit=50,
    )

    assert calls["runset_query"]["limit"] == 50
    assert calls["export"]["run_ids"] == ["run-a", "run-b"]
    assert payload["selected_run_ids"] == ["run-a", "run-b"]


def test_cli_registers_new_handoff_commands():
    parser = cli_module.build_parser()
    subparser_action = next(
        action for action in parser._subparsers._group_actions if hasattr(action, "choices")
    )
    choices = set(subparser_action.choices.keys())

    assert "ingest-artifacts" in choices
    assert "export-scenario-db" in choices
    assert "export-sql" in choices
    assert "export-asim-inputs" in choices

    ingest_args = parser.parse_args(
        [
            "ingest-artifacts",
            "--archive-run-dir",
            "/tmp/archive",
            "--project-root",
            "/tmp/project",
            "--artifact",
            "k=/tmp/f.csv",
        ]
    )
    assert ingest_args.access_mode == "standard"


def test_cli_export_sql_invokes_pipeline(monkeypatch, tmp_path):
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "export-sql",
            "--archive-run-dir",
            str(tmp_path / "archive"),
            "--project-root",
            str(tmp_path / "project"),
            "--sql",
            "SELECT 1",
            "--output-path",
            str(tmp_path / "out.csv"),
        ]
    )

    called = {}
    monkeypatch.setattr(cli_module, "_build_tracker", lambda _args: object())

    def _fake_export_sql_query(tracker, **kwargs):
        del tracker
        called.update(kwargs)
        return {"output_path": kwargs["output_path"], "row_count": 0}

    monkeypatch.setattr(cli_module, "export_sql_query", _fake_export_sql_query)

    exit_code = cli_module.cmd_export_sql(args)
    assert exit_code == 0
    assert called["sql"] == "SELECT 1"
    assert called["output_format"] == "csv"


def test_cli_ingest_artifacts_accepts_artifact_from_run(monkeypatch, tmp_path):
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "ingest-artifacts",
            "--archive-run-dir",
            str(tmp_path / "archive"),
            "--project-root",
            str(tmp_path / "project"),
            "--artifact-from-run",
            "run-1:trips",
        ]
    )

    monkeypatch.setattr(cli_module, "_build_tracker", lambda _args: object())
    captured = {}

    def _fake_ingest_artifacts(tracker, artifact_specs, **kwargs):
        del tracker, kwargs
        captured["artifact_specs"] = artifact_specs
        return {"run_id": "ingest-run", "artifact_count": len(artifact_specs)}

    monkeypatch.setattr(cli_module, "ingest_artifacts", _fake_ingest_artifacts)
    exit_code = cli_module.cmd_ingest_artifacts(args)
    assert exit_code == 0
    assert len(captured["artifact_specs"]) == 1
    spec = captured["artifact_specs"][0]
    assert spec.source_run_id == "run-1"
    assert spec.source_key == "trips"
