import types

import pytest

from pilates.generic.model import Model, provenance_logging
from pilates.generic.records import FileRecord, RecordStore
from pilates.generic.runner import GenericRunner
from pilates.utils import consist_runtime as cr


class _StubState:
    def __init__(self):
        database = types.SimpleNamespace(use_consist=True)
        shared = types.SimpleNamespace(database=database)
        self.full_settings = types.SimpleNamespace(shared=shared)
        self.current_year = 2017
        self.current_inner_iter = 0
        self.current_major_stage = None
        self.current_sub_stage = None
        self.forecast_year = 2017


def test_noop_scenario_run_returns_outputs_mapping():
    store = RecordStore(
        recordList=[
            FileRecord(file_path="/tmp/out.txt", short_name="out", description="out")
        ]
    )

    def _run():
        return store

    with cr.scenario("noop-test", enabled=False) as scenario:
        result = scenario.run(fn=_run)

    assert result.outputs == {"out": "/tmp/out.txt"}
    assert result.cache_hit is False


def test_provenance_logging_requires_active_run_when_enabled(monkeypatch):
    state = _StubState()

    class DummyModel(Model):
        @provenance_logging
        def do(self, input_store: RecordStore) -> RecordStore:
            return RecordStore(recordList=[])

    monkeypatch.setattr(cr, "consist_available", lambda _: True)
    monkeypatch.setattr(cr, "current_run", lambda: None)

    model = DummyModel("dummy", state)
    with pytest.raises(RuntimeError, match="active run context"):
        model.do(RecordStore())


def test_provenance_logging_uses_input_store_kwarg(monkeypatch):
    state = _StubState()
    calls = []

    class DummyModel(Model):
        @provenance_logging
        def do(self, input_store: RecordStore) -> RecordStore:
            return RecordStore(
                recordList=[
                    FileRecord(
                        file_path="/tmp/out.txt", short_name="out", description="out"
                    )
                ]
            )

    monkeypatch.setattr(cr, "consist_available", lambda _: True)
    monkeypatch.setattr(cr, "current_run", lambda: object())

    def _log_artifacts(mapping, **meta):
        calls.append((mapping, meta))

    monkeypatch.setattr(cr, "log_artifacts", _log_artifacts)
    monkeypatch.setattr(cr, "log_meta", lambda **_: None)

    input_store = RecordStore(
        recordList=[
            FileRecord(
                file_path="/tmp/in.txt", short_name="input_store", description="in"
            )
        ]
    )
    model = DummyModel("dummy", state)
    model.do(input_store=input_store)

    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[0][0] == {"input_store": "/tmp/in.txt"}
    assert calls[1][1]["direction"] == "output"
    assert calls[1][0] == {"out": "/tmp/out.txt"}


def test_decorator_logs_artifacts_with_consist(monkeypatch, tmp_path):
    consist = pytest.importorskip("consist")
    from consist import Tracker

    run_dir = tmp_path / "consist_runs"
    tracker = Tracker(
        run_dir=run_dir,
        db_path=str(tmp_path / "consist_test.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )

    calls = []

    def _log_artifacts(mapping, **meta):
        calls.append((mapping, meta))
        return mapping

    monkeypatch.setattr(consist, "log_artifacts", _log_artifacts)

    state = _StubState()

    class DummyModel(Model):
        @provenance_logging
        def do(self, input_store: RecordStore) -> RecordStore:
            return RecordStore(
                recordList=[
                    FileRecord(
                        file_path=str(tmp_path / "out.txt"),
                        short_name="out",
                        description="out",
                    )
                ]
            )

    input_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(tmp_path / "in.txt"),
                short_name="input_store",
                description="in",
            )
        ]
    )

    model = DummyModel("dummy", state)
    with cr.use_tracker(tracker):
        with cr.scenario("decorator-test", tracker=tracker) as scenario:
            scenario.run(
                fn=lambda: model.do(input_store=input_store),
                name="dummy_step",
                model="dummy",
                year=2017,
                iteration=0,
                load_inputs=False,
            )

    assert calls
    directions = [meta.get("direction") for _, meta in calls]
    assert "input" in directions
    assert "output" in directions


def test_run_container_uses_current_tracker(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist import Tracker

    run_dir = tmp_path / "consist_runs"
    tracker = Tracker(
        run_dir=run_dir,
        db_path=str(tmp_path / "consist_test.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )

    called = {}

    def _fake_run_container(**kwargs):
        called.update(kwargs)
        return True

    monkeypatch.setattr(
        "consist.integrations.containers.run_container", _fake_run_container
    )

    settings = types.SimpleNamespace(
        shared=types.SimpleNamespace(database=types.SimpleNamespace(use_consist=True)),
        infrastructure=types.SimpleNamespace(
            container_manager="docker",
            docker_config=types.SimpleNamespace(pull_latest=False),
        ),
    )

    with cr.use_tracker(tracker):
        with tracker.start_run("container-test", model="dummy"):
            ok = GenericRunner.run_container(
                client=None,
                settings=settings,
                image="dummy-image",
                volumes={str(tmp_path): {"bind": "/app", "mode": "rw"}},
                command="echo hello",
                model_name="dummy",
                input_artifacts=[],
                output_paths=[],
            )

    assert ok is True
    assert called.get("tracker") is tracker
