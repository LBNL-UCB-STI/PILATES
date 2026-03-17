from types import SimpleNamespace
import json
import os
from pathlib import Path
import re

import pytest
from consist import MaterializationResult
from consist.types import CacheOptions

from pilates.runtime import launcher as run_module
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils import consist_db_snapshot as snapshot_module
from workflow_state import WorkflowState


class DummyWorkspace:
    def __init__(self, full_path="/tmp/bootstrap", settings=None):
        self.full_path = full_path
        self.settings = settings
        self.input_data = {}
        self.output_data = {}

    def get_usim_mutable_data_dir(self):
        return os.path.join(self.full_path, "urbansim", "data")

    def get_asim_mutable_data_dir(self):
        return os.path.join(self.full_path, "activitysim", "data")

    def get_asim_mutable_configs_dir(self):
        return os.path.join(self.full_path, "activitysim", "configs")

    def get_asim_output_dir(self):
        return os.path.join(self.full_path, "activitysim", "output")

    def get_atlas_mutable_input_dir(self):
        return os.path.join(self.full_path, "atlas", "atlas_input")

    def get_beam_mutable_data_dir(self):
        return os.path.join(self.full_path, "beam", "input")


class DummyInitialization:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, _settings, workspace):
        rec_in = RecordStore(
            recordList=[
                FileRecord(
                    unique_id="in1",
                    short_name="bootstrap_in",
                    file_path="/tmp/source",
                    metadata={"model": "beam", "bootstrap_direction": "input"},
                )
            ]
        )
        rec_out = RecordStore(
            recordList=[
                FileRecord(
                    unique_id="out1",
                    short_name="bootstrap_out",
                    file_path="/tmp/dest",
                    metadata={"model": "beam", "bootstrap_direction": "output"},
                )
            ]
        )
        combined = RecordStore()
        combined += rec_in
        combined += rec_out
        return combined


class DummyTracker:
    def __init__(self, responses, materialization_results=None):
        self.responses = list(responses)
        self.materialization_results = list(materialization_results or [])
        self.calls = []
        self.materialization_calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        if response["execute_fn"] and kwargs.get("fn") is not None:
            kwargs["fn"]()
        return SimpleNamespace(
            cache_hit=response["cache_hit"],
            run=SimpleNamespace(id=response["run_id"]),
        )

    def materialize_run_outputs(self, **kwargs):
        self.materialization_calls.append(kwargs)
        if not self.materialization_results:
            raise AssertionError("missing prepared materialization result")
        return self.materialization_results.pop(0)


class DummySnapshotTracker:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def snapshot_db(self, dest_path: str, checkpoint: bool = True):
        self.calls.append({"dest_path": dest_path, "checkpoint": checkpoint})
        if self.fail:
            raise RuntimeError("simulated snapshot failure")
        output_path = Path(dest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("db", encoding="utf-8")
        metadata = {
            "run_id": "run-test",
            "snapshot_ts_utc": "2026-02-24T00:00:00Z",
        }
        (output_path.parent / snapshot_module.snapshot_meta_filename(output_path.name)).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


def _settings(cache_enabled=True):
    return SimpleNamespace(run=SimpleNamespace(bootstrap_cache_enabled=cache_enabled))


def _state():
    return SimpleNamespace(start_year=2017)


def test_run_bootstrap_phase_cache_miss_executes_once(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_probe"}
        ]
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 1
    first_call = tracker.calls[0]
    assert "scenario_id:seattle-baseline" in first_call["tags"]
    assert "seed:12345" in first_call["tags"]
    assert "year:2017" in first_call["tags"]
    assert "iteration:0" in first_call["tags"]
    assert "model:initialization" in first_call["tags"]
    assert first_call["facet"]["scenario_id"] == "seattle-baseline"
    assert first_call["facet"]["seed"] == 12345
    assert first_call["facet"]["year"] == 2017
    assert first_call["facet"]["iteration"] == 0
    assert first_call["facet"]["model"] == "initialization"
    assert result["bootstrap_cache_hit"] is False
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["run_reference"] == {"probe_run_id": "bootstrap_probe"}


def test_run_bootstrap_phase_cache_hit_materializes_without_rerun(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    materialization_result = MaterializationResult(
        materialized_from_filesystem={"bootstrap_initialization": "/tmp/dest"},
        skipped_existing=["existing-cache"],
    )
    tracker = DummyTracker(
        responses=[
            {"cache_hit": True, "execute_fn": False, "run_id": "bootstrap_probe"},
        ],
        materialization_results=[materialization_result],
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 1
    assert result["bootstrap_cache_hit"] is True
    assert result["fallback_rerun"] is False
    assert result["staged_artifact_summary"]["copied_records_total"] == 0
    assert result["run_reference"] == {"probe_run_id": "bootstrap_probe"}
    assert result["materialization"]["complete"] is True
    assert result["materialization"]["materialized_from_filesystem_count"] == 1
    assert result["materialization"]["skipped_existing_count"] == 1
    assert tracker.materialization_calls == [
        {
            "run_id": "bootstrap_probe",
            "target_root": workspace.full_path,
            "source_root": None,
            "preserve_existing": True,
        }
    ]


def test_run_bootstrap_phase_cache_hit_partial_materialization_triggers_fallback_rerun(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    partial_materialization = MaterializationResult(
        materialized_from_filesystem={"bootstrap_initialization": "/tmp/dest"},
        skipped_missing_source=["missing-record"],
        failed=[("bootstrap_initialization", "missing source")],
    )
    tracker = DummyTracker(
        responses=[
            {"cache_hit": True, "execute_fn": False, "run_id": "bootstrap_probe"},
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_fallback"},
        ],
        materialization_results=[partial_materialization],
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 2
    fallback_options = tracker.calls[1]["cache_options"]
    assert isinstance(fallback_options, CacheOptions)
    assert result["bootstrap_cache_hit"] is True
    assert result["fallback_rerun"] is True
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["run_reference"] == {
        "probe_run_id": "bootstrap_probe",
        "materialization_run_id": "bootstrap_fallback",
    }
    assert result["materialization"]["complete"] is False
    assert result["materialization"]["skipped_missing_source_count"] == 1
    assert result["materialization"]["failed_count"] == 1
    assert tracker.materialization_calls == [
        {
            "run_id": "bootstrap_probe",
            "target_root": workspace.full_path,
            "source_root": None,
            "preserve_existing": True,
        }
    ]


def test_run_bootstrap_phase_cache_disabled_uses_cache_off(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_off"}
        ]
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=False),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=None,
    )

    assert len(tracker.calls) == 1
    first_call = tracker.calls[0]
    assert "scenario_id:seattle-baseline" in first_call["tags"]
    assert all(not tag.startswith("seed:") for tag in first_call["tags"])
    assert first_call["facet"]["scenario_id"] == "seattle-baseline"
    assert "seed" not in first_call["facet"]
    cache_options = tracker.calls[0]["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.cache_mode == "off"
    assert result["bootstrap_cache_hit"] is False
    assert result["run_reference"] == {"probe_run_id": "bootstrap_off"}


def test_seed_bootstrap_artifacts_to_coupler_publishes_beam_defaults(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plans = scenario_dir / "plans.parquet"
    households = scenario_dir / "households.parquet"
    persons = scenario_dir / "persons.parquet"
    plans.write_text("plans", encoding="utf-8")
    households.write_text("households", encoding="utf-8")
    persons.write_text("persons", encoding="utf-8")
    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_config_exchange_folder(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(name, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None


def test_seed_bootstrap_artifacts_to_coupler_prefers_yaml_scenario_folder_when_present(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (default_dir / name).write_text(f"default-{name}", encoding="utf-8")

    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(f"config-{name}", encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") == str(default_dir / "plans.parquet")
    assert coupler.get("households_beam_in") == str(default_dir / "households.parquet")
    assert coupler.get("persons_beam_in") == str(default_dir / "persons.parquet")


@pytest.mark.parametrize("extension", ["csv", "csv.gz"])
def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_csv_formats_when_parquet_missing(
    tmp_path, extension
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim" / "2018"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.{extension}").write_text(stem, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_publishes_initial_warmstart(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    warmstart = router_dir / "init.linkstats.csv.gz"
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="r5/network/init.linkstats.csv.gz",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is not None
    assert isinstance(coupler.get("linkstats_warmstart"), str)


def test_seed_bootstrap_artifacts_to_coupler_uses_configured_warmstart_path(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = (
        tmp_path / "beam" / "input" / "sfbay" / "custom" / "warmstart.parquet"
    )
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="custom/warmstart.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_expands_router_directory_in_warmstart_path(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = (
        tmp_path
        / "beam"
        / "input"
        / "sfbay"
        / "r5"
        / "network"
        / "init.linkstats.parquet"
    )
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="{router_directory}/init.linkstats.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_does_not_probe_router_dir_when_warmstart_unset(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    (router_dir / "init.linkstats.csv.gz").write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is None

def test_seed_bootstrap_artifacts_to_coupler_publishes_activitysim_compile_artifacts(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    zarr_dir = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")

    sharrow_cache = tmp_path / "shared_cache" / "numba"
    sharrow_cache.mkdir(parents=True, exist_ok=True)
    (sharrow_cache / "cache.bin").write_text("cache", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand="activitysim", travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("zarr_skims") is not None
    assert coupler.get("asim_sharrow_cache_dir") is not None
    assert isinstance(coupler.get("zarr_skims"), str)
    assert isinstance(coupler.get("asim_sharrow_cache_dir"), str)


def test_bootstrap_output_invariant_accepts_valid_result():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap_probe"},
            "staged_artifact_summary": {"copied_records_total": 2},
        }
    )


def test_bootstrap_output_invariant_accepts_zero_copied_records_for_valid_summary():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap_probe"},
            "staged_artifact_summary": {"copied_records_total": 0},
        }
    )


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        {},
        {"staged_artifact_summary": {}},
    ],
)
def test_bootstrap_output_invariant_rejects_invalid_or_empty_result(invalid_result):
    with pytest.raises(RuntimeError, match="Bootstrap initialization invariant failed"):
        run_module._assert_bootstrap_output_invariant(invalid_result)


def _restart_settings():
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(
                activity_demand="activitysim",
                vehicle_ownership="atlas",
                traffic_assignment="beam",
            ),
        ),
        atlas=SimpleNamespace(scenario="baseline"),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        beam=SimpleNamespace(config="beam.conf"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
    )


def _resume_doctor_check_names(caplog) -> set:
    names = set()
    for record in caplog.records:
        if "[ResumeDoctor] check=" not in record.message:
            continue
        match = re.search(r"check=([^\s]+)", record.message)
        if match:
            names.add(match.group(1))
    return names


def test_resume_doctor_ready_summary_logs_expected_checks(tmp_path, caplog):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace(current_year=2018, current_inner_iter=0, data_initialized=True)

    archive_state_path = archive_run_dir / "run_state.yaml"
    local_state_path = local_run_dir / "run_state.yaml"
    archive_state_path.parent.mkdir(parents=True, exist_ok=True)
    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    archive_state_path.write_text("archive-state", encoding="utf-8")
    local_state_path.write_text("local-state", encoding="utf-8")

    local_consist_db_path = local_run_dir / ".consist" / "provenance.duckdb"
    local_consist_db_path.parent.mkdir(parents=True, exist_ok=True)
    local_consist_db_path.write_text("db", encoding="utf-8")

    snapshot_db_path = (
        snapshot_module.snapshot_latest_dir(str(archive_run_dir))
        / local_consist_db_path.name
    )
    snapshot_db_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_db_path.write_text("snapshot-db", encoding="utf-8")

    local_manifest = run_module.build_manifest_path(workspace=workspace, year=2018, iteration=0)
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text("manifest", encoding="utf-8")
    archive_manifest = (
        Path(archive_run_dir)
        / ".workflow"
        / local_manifest.name
    )
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text("archive-manifest", encoding="utf-8")

    with caplog.at_level("INFO"):
        run_module._run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
            archive_state_path=str(archive_state_path),
            local_state_path=str(local_state_path),
            local_consist_db_path=str(local_consist_db_path),
            restart_missing_artifacts_initial=[],
            restart_missing_artifacts_after_recovery=[],
            restart_reconstruction=None,
        )

    check_messages = [
        record.message
        for record in caplog.records
        if "[ResumeDoctor] check=" in record.message
    ]
    assert check_messages
    assert all(message.startswith("[ResumeDoctor] check=") for message in check_messages)
    assert _resume_doctor_check_names(caplog) >= {
        "archive_run_state",
        "local_run_state_mirror",
        "local_consist_db",
        "archive_latest_consist_db_snapshot",
        "required_restart_local_artifacts",
        "supply_demand_manifest_local",
        "supply_demand_manifest_archive",
    }
    assert "[ResumeDoctor] check=supply_demand_manifest_local status=ok" in caplog.text
    assert "[ResumeDoctor] check=supply_demand_manifest_archive status=ok" in caplog.text
    assert "[ResumeDoctor] summary status=ready reason=all_checks_ok" in caplog.text


def test_resume_doctor_degraded_summary_reports_missing_checks_and_manifest_checks(
    tmp_path, caplog
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace(current_year=2022, current_inner_iter=1, data_initialized=True)
    missing = [{"key": "activitysim_settings_yaml", "path": "missing", "reason": "test"}]

    with caplog.at_level("INFO"):
        run_module._run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
            archive_state_path=str(archive_run_dir / "run_state.yaml"),
            local_state_path=str(local_run_dir / "run_state.yaml"),
            local_consist_db_path=str(local_run_dir / ".consist" / "provenance.duckdb"),
            restart_missing_artifacts_initial=missing,
            restart_missing_artifacts_after_recovery=missing,
            restart_reconstruction=None,
        )

    check_names = _resume_doctor_check_names(caplog)
    assert check_names >= {
        "archive_run_state",
        "local_run_state_mirror",
        "local_consist_db",
        "archive_latest_consist_db_snapshot",
        "required_restart_local_artifacts",
        "supply_demand_manifest_local",
        "supply_demand_manifest_archive",
    }
    assert "[ResumeDoctor] check=supply_demand_manifest_local status=missing" in caplog.text
    assert "[ResumeDoctor] check=supply_demand_manifest_archive status=missing" in caplog.text
    assert "[ResumeDoctor] summary status=degraded reason=missing_checks:" in caplog.text


def test_format_restart_command_uses_config_and_archive_state():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "python run.py -c scenarios/settings-seattle.yaml -S '/tmp/pilates run/run_state.yaml'"
    )


def test_format_hpc_restart_command_requires_account_placeholder():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_hpc_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s '/tmp/pilates run/run_state.yaml'"
    )


def test_main_logs_restart_instructions_on_failure(tmp_path, monkeypatch, caplog):
    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    class SnapshotStub:
        def final_snapshot(self):
            return True

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(tmp_path / "archive-root"),
            local_workspace_root=str(tmp_path / "local-root"),
            enable_archive_copy=False,
            output_run_name="restart-hint-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file="scenarios/settings-seattle.yaml",
    )
    state = SimpleNamespace(
        run_info_path=None,
        data_initialized=False,
    )
    state.set_run_info_path = lambda path: setattr(state, "run_info_path", path)
    state.set_data_initialized = lambda initialized: setattr(
        state, "data_initialized", initialized
    )

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: 1)
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _fail_bootstrap(**_kwargs):
        raise RuntimeError("simulated bootstrap failure")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _fail_bootstrap)

    run_module._RUN_FAILURE_CONTEXT.clear()
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="simulated bootstrap failure"):
            run_module.main()
        run_module._log_restart_instructions_on_failure()

    assert "Run failed. Restart command:" in caplog.text
    assert "python run.py -c scenarios/settings-seattle.yaml -S " in caplog.text
    assert (
        "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s "
        in caplog.text
    )
    assert "run_state.yaml" in caplog.text


def test_restart_preflight_detects_missing_local_workspace_artifacts(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace()

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    assert {item["key"] for item in missing} == {
        "usim_datastore_base_h5",
        "omx_skims",
        "hh_size",
        "income_rates",
        "relmap",
        "schools",
        "school_districts",
        "activitysim_config_settings_yaml_configs",
        "activitysim_config_settings_yaml_configs_extended",
        "activitysim_config_settings_yaml_configs_mp",
        "activitysim_config_settings_yaml_configs_sh_compile",
    }


def test_restart_preflight_skips_activitysim_locals_outside_supply_demand_stage(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(current_major_stage=WorkflowState.Stage.vehicle_ownership_model)

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    assert "usim_datastore_base_h5" in keys
    assert "activitysim_config_settings_yaml_configs" not in keys
    assert "activitysim_config_settings_yaml_configs_mp" not in keys
    assert "zarr_skims" not in keys
    assert any(key.startswith("atlas_static::") for key in keys)


def test_restart_preflight_requires_atlas_static_inputs_in_vehicle_stage(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(current_major_stage=WorkflowState.Stage.vehicle_ownership_model)

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    paths = {item["path"] for item in missing}
    assert any(path.endswith("atlas/atlas_input/psid_names.Rdat") for path in paths)
    assert any(path.endswith("atlas/atlas_input/accessbility_2015.RData") for path in paths)


def test_restart_preflight_requires_zarr_skims_when_resuming_compiled_supply_demand(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        asim_compiled=True,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "zarr_skims" in keys
    assert any(path.endswith("activitysim/output/cache/skims.zarr") for path in paths)


def test_restart_preflight_requires_beam_region_dir_when_resuming_supply_demand(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        asim_compiled=False,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "beam_mutable_data_dir" in keys
    assert "beam_region_input_dir" in keys
    assert "beam_primary_config_file" in keys
    assert any(path.endswith("beam/input") for path in paths)
    assert any(path.endswith("beam/input/test") for path in paths)
    assert any(path.endswith("beam/input/test/beam.conf") for path in paths)


def test_repair_restart_beam_inputs_from_source_repopulates_missing_primary_config(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    settings = _restart_settings()
    state = SimpleNamespace()

    calls = []

    class _BeamPreprocessor:
        def copy_data_to_mutable_location(self, settings_arg, output_dir):
            calls.append((settings_arg, output_dir))
            target = (
                Path(output_dir)
                / settings_arg.run.region
                / settings_arg.beam.config
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("beam", encoding="utf-8")

    class _Factory:
        def get_preprocessor(self, model_name, state_arg):
            assert model_name == "beam"
            assert state_arg is state
            return _BeamPreprocessor()

    repaired = run_module._repair_restart_beam_inputs_from_source(
        settings=settings,
        state=state,
        workspace=workspace,
        model_factory_cls=lambda: _Factory(),
    )

    assert repaired is True
    assert calls == [(settings, workspace.get_beam_mutable_data_dir())]
    assert (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.config
    ).exists()


def test_restart_preflight_requires_activitysim_iteration_outputs_for_traffic_assignment(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.traffic_assignment,
        current_year=2017,
        current_inner_iter=0,
        asim_compiled=True,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "activitysim_iteration_beam_plans_parquet" in keys
    assert "activitysim_iteration_households_parquet" in keys
    assert "activitysim_iteration_persons_parquet" in keys
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/beam_plans.parquet")
        for path in paths
    )
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/households.parquet")
        for path in paths
    )
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/persons.parquet")
        for path in paths
    )


def test_reconstruct_restart_completed_run_outputs_materializes_manifest_run_ids(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    local_run_dir = tmp_path / "local-run"
    manifest_path = archive_run_dir / ".workflow" / "year_2018_iteration_0.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "activitysim_preprocess:",
                "  run_id: run-1",
                "activitysim_postprocess:",
                "  run_id: run-2",
            ]
        ),
        encoding="utf-8",
    )

    tracker = DummyTracker(
        responses=[],
        materialization_results=[
            MaterializationResult(materialized_from_filesystem={"run-1": "/tmp/a"}),
            MaterializationResult(materialized_from_filesystem={"run-2": "/tmp/b"}),
        ],
    )
    state = SimpleNamespace(
        current_year=2018,
        current_inner_iter=1,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.activity_demand,
        _settings={"supply_demand_iters": 2},
    )

    result = run_module._reconstruct_restart_completed_run_outputs(
        tracker=tracker,
        state=state,
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert result["run_ids"] == ["run-1", "run-2"]
    assert result["source_root"] == str(archive_run_dir.resolve())
    assert result["target_root"] == str(local_run_dir.resolve())
    assert result["materialization_result"].complete is True
    assert tracker.materialization_calls == [
        {
            "run_id": "run-1",
            "target_root": str(local_run_dir.resolve()),
            "source_root": str(archive_run_dir.resolve()),
            "preserve_existing": True,
        },
        {
            "run_id": "run-2",
            "target_root": str(local_run_dir.resolve()),
            "source_root": str(archive_run_dir.resolve()),
            "preserve_existing": True,
        },
    ]


def test_restart_repair_rewinds_stale_supply_demand_state_when_atlas_outputs_incomplete(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    atlas_output_dir = archive_run_dir / "atlas" / "atlas_output"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)
    (atlas_output_dir / "householdv_2017.csv").write_text("hh\n", encoding="utf-8")
    (atlas_output_dir / "vehicles_2017.csv").write_text("veh\n", encoding="utf-8")

    writes = []
    state = SimpleNamespace(
        Stage=WorkflowState.Stage,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.activity_demand,
        current_inner_iter=3,
        sub_stage_progress="postprocessor",
        current_year=2017,
        forecast_year=2023,
        write_state=lambda: writes.append("written"),
    )

    repaired = run_module._repair_restart_state_for_incomplete_atlas_outputs(
        settings=_restart_settings(),
        state=state,
        archive_run_dir=str(archive_run_dir),
    )

    assert repaired is True
    assert state.current_major_stage == WorkflowState.Stage.vehicle_ownership_model
    assert state.current_sub_stage is None
    assert state.current_inner_iter == 0
    assert state.sub_stage_progress is None
    assert writes == ["written"]


def test_restart_repair_keeps_supply_demand_when_all_atlas_outputs_complete(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    atlas_output_dir = archive_run_dir / "atlas" / "atlas_output"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)
    for atlas_year in (2017, 2019, 2021, 2023):
        (atlas_output_dir / f"householdv_{atlas_year}.csv").write_text(
            "hh\n", encoding="utf-8"
        )
        (atlas_output_dir / f"vehicles_{atlas_year}.csv").write_text(
            "veh\n", encoding="utf-8"
        )
        (atlas_output_dir / f"vehicles2_{atlas_year}.csv").write_text(
            "veh2\n", encoding="utf-8"
        )

    writes = []
    state = SimpleNamespace(
        Stage=WorkflowState.Stage,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.activity_demand,
        current_inner_iter=0,
        sub_stage_progress=None,
        current_year=2017,
        forecast_year=2023,
        write_state=lambda: writes.append("written"),
    )

    repaired = run_module._repair_restart_state_for_incomplete_atlas_outputs(
        settings=_restart_settings(),
        state=state,
        archive_run_dir=str(archive_run_dir),
    )

    assert repaired is False
    assert state.current_major_stage == WorkflowState.Stage.supply_demand_loop
    assert state.current_sub_stage == WorkflowState.Stage.activity_demand
    assert writes == []


def test_build_atlas_static_inputs_fallback_uses_atlas_static_key_scheme(tmp_path):
    settings = _restart_settings()
    workspace = DummyWorkspace(str(tmp_path / "local-run"), settings=settings)
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    (atlas_input_dir / "psid_names.Rdat").parent.mkdir(parents=True, exist_ok=True)
    (atlas_input_dir / "psid_names.Rdat").write_text("psid", encoding="utf-8")
    (
        atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv"
    ).write_text("used", encoding="utf-8")

    fallback = run_module.build_atlas_static_inputs_fallback(workspace)

    assert fallback["psid_names"] == str(atlas_input_dir / "psid_names.Rdat")
    assert (
        fallback["adopt/baseline/used_vehicles_2017"]
        == str(atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv")
    )
    assert "atlas_static_psid_names.Rdat" not in fallback


def test_resume_rewind_guardrail_blocks_by_default_and_allows_override(tmp_path):
    archive_state_path = tmp_path / "archive-run" / "run_state.yaml"
    archive_state_path.parent.mkdir(parents=True, exist_ok=True)
    archive_state_path.write_text(
        "year: 2022\nstage: null\niteration: 0\nasim_compiled: false\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(current_year=2019)

    with pytest.raises(RuntimeError, match="Refusing rewind resume"):
        run_module._enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=str(archive_state_path),
            allow_rewind_resume=False,
        )

    run_module._enforce_resume_rewind_guardrail(
        state=state,
        archive_state_path=str(archive_state_path),
        allow_rewind_resume=True,
    )


def test_main_enables_external_paths_for_archive_to_local_tracker_topology(
    tmp_path, monkeypatch
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class StateStub:
        def __init__(self):
            self.run_info_path = None
            self.data_initialized = False
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

        def set_data_initialized(self, initialized: bool) -> None:
            self.data_initialized = initialized

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="tracker-topology-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
    )
    state = StateStub()
    tracker_kwargs = {}

    def _capture_create_tracker(**kwargs):
        tracker_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _capture_create_tracker)
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _stop_after_bootstrap(**_kwargs):
        raise StopAfterBootstrap("stop after tracker init")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _stop_after_bootstrap)

    with pytest.raises(StopAfterBootstrap, match="stop after tracker init"):
        run_module.main()

    archive_run_dir = Path(tracker_kwargs["run_dir"])
    repo_root = Path(__file__).resolve().parents[1]
    inputs_mount = Path(tracker_kwargs["mounts"]["inputs"])
    workspace_mount = Path(tracker_kwargs["mounts"]["workspace"])
    project_root = Path(tracker_kwargs["project_root"])
    assert archive_run_dir.parent == archive_root
    assert archive_run_dir.name.startswith(settings.run.output_run_name)
    assert inputs_mount == repo_root.resolve()
    assert workspace_mount.parent == local_root
    assert workspace_mount.name == archive_run_dir.name
    assert project_root == repo_root.resolve()
    assert tracker_kwargs["allow_external_paths"] is True


def test_main_restart_strict_defers_missing_artifact_failure_until_after_bootstrap(
    tmp_path, monkeypatch
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class SnapshotStub:
        def final_snapshot(self):
            return True

    class StateStub:
        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.data_initialized = True
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

        def set_data_initialized(self, initialized: bool) -> None:
            self.data_initialized = initialized

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-strict-test"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
    )
    state = StateStub(str(run_state_path))

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(run_module, "_run_resume_doctor_diagnostics", lambda **_kwargs: None)
    monkeypatch.setattr(run_module, "_repair_restart_beam_inputs_from_source", lambda **_kwargs: False)
    monkeypatch.setattr(
        run_module,
        "_reconstruct_restart_completed_run_outputs",
        lambda **_kwargs: {
            "run_ids": ["run-1"],
            "source_root": str(archive_run_dir),
            "target_root": str(local_root / run_name),
            "materialization_result": MaterializationResult(
                materialized_from_filesystem={"run-1": "/restored/run-1"},
                skipped_missing_source=["households_asim_out_temp"],
            ),
        },
    )
    missing_before_bootstrap = [
        {
            "key": "activitysim_config_settings_yaml_configs",
            "path": "/missing/settings.yaml",
            "reason": "test",
        }
    ]
    missing_sequences = [
        list(missing_before_bootstrap),
        list(missing_before_bootstrap),
        [],
    ]

    def _missing_artifacts(**_kwargs):
        if not missing_sequences:
            return []
        return missing_sequences.pop(0)

    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        _missing_artifacts,
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap-run"},
            "staged_artifact_summary": {"copied_records_total": 0},
        },
    )
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StopAfterBootstrap("reached scenario")
        ),
    )

    with pytest.raises(StopAfterBootstrap, match="reached scenario"):
        run_module.main()

    assert missing_sequences == []


def test_main_restart_strict_still_fails_when_required_artifacts_remain_missing(
    tmp_path, monkeypatch
):
    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-strict-missing"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
    )
    state = SimpleNamespace(
        run_info_path=str(run_state_path),
        data_initialized=True,
        file_loc=None,
        mirror_file_loc=None,
    )
    state.set_run_info_path = lambda path: setattr(state, "run_info_path", path)
    state.set_data_initialized = lambda initialized: setattr(
        state, "data_initialized", initialized
    )

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(run_module, "_run_resume_doctor_diagnostics", lambda **_kwargs: None)
    monkeypatch.setattr(run_module, "_repair_restart_beam_inputs_from_source", lambda **_kwargs: False)
    monkeypatch.setattr(
        run_module,
        "_reconstruct_restart_completed_run_outputs",
        lambda **_kwargs: {
            "run_ids": ["run-1"],
            "source_root": str(archive_run_dir),
            "target_root": str(local_root / run_name),
            "materialization_result": MaterializationResult(
                materialized_from_filesystem={"run-1": "/restored/run-1"},
                skipped_missing_source=["households_asim_out_temp"],
            ),
        },
    )

    missing = [{"key": "usim_datastore_base_h5", "path": "/missing/base.h5", "reason": "test"}]

    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        lambda **_kwargs: list(missing),
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap-run"},
            "staged_artifact_summary": {"copied_records_total": 0},
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Strict restart preflight failed; required restart artifacts are still missing after restart bootstrap",
    ):
        run_module.main()


def test_resolve_consist_db_paths_uses_local_run_dir_by_default():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(
            database=SimpleNamespace(enabled=True, path="/global/scratch/provenance.duckdb")
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/local/job123/run-abc/.consist/provenance.duckdb"
    assert archive_path == "/global/scratch/run-abc/.consist/provenance.duckdb"


def test_resolve_consist_db_paths_uses_configured_path_when_local_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_local_run=False),
        shared=SimpleNamespace(
            database=SimpleNamespace(enabled=True, path="/global/scratch/provenance.duckdb")
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/global/scratch/provenance.duckdb"
    assert archive_path == "/global/scratch/provenance.duckdb"


def test_resolve_consist_db_paths_returns_none_when_db_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path="ignored")),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path is None
    assert archive_path is None


def test_mirror_consist_db_to_archive_copies_db_and_wal(tmp_path):
    local_db = tmp_path / "local" / "provenance.duckdb"
    local_db.parent.mkdir(parents=True, exist_ok=True)
    local_db.write_text("db", encoding="utf-8")
    local_wal = tmp_path / "local" / "provenance.duckdb.wal"
    local_wal.write_text("wal", encoding="utf-8")

    archive_db = tmp_path / "archive" / "provenance.duckdb"
    snapshot_module.mirror_consist_db_to_archive(str(local_db), str(archive_db))

    assert archive_db.read_text(encoding="utf-8") == "db"
    assert (tmp_path / "archive" / "provenance.duckdb.wal").read_text(
        encoding="utf-8"
    ) == "wal"


def test_restore_local_consist_db_from_snapshot_hydrates_missing_local_db(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    latest_dir = snapshot_module.snapshot_latest_dir(str(archive_run_dir))
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "provenance.duckdb").write_text("db", encoding="utf-8")
    (latest_dir / "provenance.duckdb.wal").write_text("wal", encoding="utf-8")
    (latest_dir / snapshot_module.snapshot_meta_filename("provenance.duckdb")).write_text(
        json.dumps({"snapshot_ts_utc": "2026-02-24T01:02:03Z"}),
        encoding="utf-8",
    )

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_restore_on_start=True,
            consist_db_restore_strict=False,
        )
    )

    restored = snapshot_module.restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=str(local_db),
        archive_run_dir=str(archive_run_dir),
    )

    assert restored is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"
    assert (
        local_db.parent / snapshot_module.snapshot_meta_filename(local_db.name)
    ).exists()


def test_seed_local_consist_db_from_shared_hydrates_missing_local_db(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    shared_wal = shared_db.with_suffix(".duckdb.wal")
    shared_wal.write_text("wal", encoding="utf-8")

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"


def test_seed_local_consist_db_from_shared_disabled_returns_false(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_seed_from_shared_on_start=False)
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is False
    assert not local_db.exists()


def test_seed_local_consist_db_from_shared_missing_source_returns_false(tmp_path):
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(tmp_path / "shared" / "missing.duckdb"),
    )

    assert seeded is False
    assert not local_db.exists()


def test_snapshot_manager_triggers_outer_iteration_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    local_db_path = tmp_path / "local" / ".consist" / "provenance.duckdb"
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(local_db_path),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    latest_db = (
        snapshot_module.snapshot_latest_dir(str(tmp_path / "archive-run"))
        / "provenance.duckdb"
    )
    assert latest_db.exists()


def test_snapshot_manager_interval_snapshot_safe_point_behavior(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=3600,
            consist_db_snapshot_on_outer_iteration=False,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    first = manager.maybe_snapshot_interval(reason="safe_point_1")
    second = manager.maybe_snapshot_interval(reason="safe_point_2")

    assert first is True
    assert second is False
    assert len(tracker.calls) == 1


def test_snapshot_manager_final_snapshot(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.final_snapshot()

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    assert "finalize" in Path(tracker.calls[0]["dest_path"]).parent.name


def test_snapshot_failures_are_non_fatal_and_logged(tmp_path, caplog):
    tracker = DummySnapshotTracker(fail=True)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    with caplog.at_level("WARNING"):
        did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is False
    assert "snapshot failed" in caplog.text.lower()


def test_snapshot_retention_keeps_expected_number_of_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=0,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=2,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    manager.snapshot(reason="snap_a", checkpoint=True)
    manager.snapshot(reason="snap_b", checkpoint=True)
    manager.snapshot(reason="snap_c", checkpoint=True)

    history_dir = snapshot_module.snapshot_history_dir(str(tmp_path / "archive-run"))
    history_entries = [path for path in history_dir.iterdir() if path.is_dir()]
    assert len(history_entries) == 2
