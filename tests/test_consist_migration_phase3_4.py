import types

import consist
import pytest
from consist.types import CacheOptions, ExecutionOptions

from pilates.generic.model import Model
from pilates.generic.records import FileRecord, RecordStore
from pilates.generic.runner import GenericRunner
from pilates.utils import consist_runtime as cr


class _StubState:
    def __init__(self):
        database = types.SimpleNamespace()
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
        result = scenario.run(fn=_run, output_paths={"out": "/tmp/out.txt"})

    output = result.outputs["out"]
    assert isinstance(output, consist.NoopArtifact)
    assert output.key == "out"
    assert str(output.path) == "/tmp/out.txt"
    assert isinstance(result, consist.NoopRunResult)
    assert result.run is not None
    assert result.cache_hit is False


def test_model_keeps_name_and_state():
    state = _StubState()
    model = Model("dummy", state)

    assert model.model_name == "dummy"
    assert model.state is state


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
        shared=types.SimpleNamespace(database=types.SimpleNamespace()),
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


def test_scenario_run_hash_includes_coupler_input_keys(tmp_path):
    pytest.importorskip("consist")
    from consist import Tracker

    tracker = Tracker(
        run_dir=tmp_path / "consist_runs",
        db_path=str(tmp_path / "consist_test.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )

    plans_path = tmp_path / "plans.parquet"
    consumer_out = tmp_path / "consumer.out"
    plans_path.write_text("plans-v1")
    consume_calls = {"count": 0}

    def _consume():
        consume_calls["count"] += 1
        consumer_out.write_text(f"consume-{consume_calls['count']}")

    with cr.scenario("hash-inputs-test", tracker=tracker) as scenario:
        scenario.run(
            fn=lambda: None,
            name="seed_plans",
            model="seed_plans",
            output_paths={"plans_beam_in": str(plans_path)},
            cache_options=CacheOptions(cache_mode="off"),
        )

        first = scenario.run(
            fn=_consume,
            name="beam_run_consumer",
            model="beam_run",
            year=2018,
            iteration=0,
            phase="run",
            input_keys=["plans_beam_in"],
            output_paths={"consumer_out": str(consumer_out)},
        )
        second = scenario.run(
            fn=_consume,
            name="beam_run_consumer",
            model="beam_run",
            year=2018,
            iteration=0,
            phase="run",
            input_keys=["plans_beam_in"],
            output_paths={"consumer_out": str(consumer_out)},
        )

        plans_path.write_text("plans-v2")
        scenario.run(
            fn=lambda: None,
            name="seed_plans_refresh",
            model="seed_plans",
            output_paths={"plans_beam_in": str(plans_path)},
            cache_options=CacheOptions(cache_mode="off"),
        )
        third = scenario.run(
            fn=_consume,
            name="beam_run_consumer",
            model="beam_run",
            year=2018,
            iteration=0,
            phase="run",
            input_keys=["plans_beam_in"],
            output_paths={"consumer_out": str(consumer_out)},
        )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    assert consume_calls["count"] == 2
