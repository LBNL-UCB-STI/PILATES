"""
Restart stage-boundary regression matrix.

These tests lock in the restart-without-bootstrap behavior at the stage entry
points we can cover cheaply with lightweight fakes:

1. ``land_use`` must preserve required UrbanSim datastore handles even when the
   restart-time preprocess output is thinner than a fresh-run output.
2. ``vehicle_ownership`` must be able to start from restored local ATLAS static
   files once restart recovery rebuilds the in-memory registry.
3. ``activity_demand`` must be able to start from restored ActivitySim compile
   artifacts without silently forcing recompilation.
4. ``traffic_assignment`` must be able to start directly from restored BEAM
   scenario inputs when ActivitySim is disabled/skipped.
5. A resumed mid-loop traffic-assignment iteration must retain promoted BEAM
   warmstart artifacts instead of silently dropping them.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import run as run_module
from pilates.atlas.outputs import AtlasRunOutputs
from pilates.config import load_config
from pilates.config.models import FullSkimsCreatorConfig
from pilates.generic.records import FileRecord, RecordStore
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_SHARROW_CACHE_DIR,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from workflow_state import WorkflowState


class CouplerStub:
    """Minimal in-memory coupler used by the stage boundary tests."""

    def __init__(self) -> None:
        self._values = {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def set(self, key, value) -> None:
        self._values[key] = value

    def set_from_artifact(self, key, value) -> None:
        self._values[key] = value

    def require(self, key):
        if key not in self._values:
            raise KeyError(f"Coupler missing key={key!r}")
        return self._values[key]


class FakeScenario:
    """
    Scenario stub that records calls and enforces required coupler inputs.

    This mirrors the contract we care about at restart boundaries: explicit
    inputs are materialized into the coupler and ``input_keys`` must already be
    resolvable by the time the step starts.
    """

    def __init__(self, coupler: CouplerStub) -> None:
        self.coupler = coupler
        self.calls = []

    def run(self, **kwargs):
        import inspect

        inputs = kwargs.get("inputs") or {}
        input_keys = kwargs.get("input_keys") or []
        fn = kwargs["fn"]
        model = kwargs.get("model")
        if model is None:
            step_meta = getattr(fn, "__consist_step__", None)
            model = getattr(step_meta, "model", None)
        self.calls.append(
            {
                "fn_name": getattr(fn, "__name__", "<unknown>"),
                "model": model,
                "inputs": dict(inputs),
                "input_keys": list(input_keys),
            }
        )

        for key, value in inputs.items():
            self.coupler.set(key, value)
        for key in input_keys:
            self.coupler.require(key)

        execution_options = kwargs.get("execution_options")
        runtime_kwargs = kwargs.get("runtime_kwargs") or getattr(
            execution_options, "runtime_kwargs", None
        )
        runtime_kwargs = dict(runtime_kwargs or {})
        fn_kwargs = dict(runtime_kwargs)

        sig = inspect.signature(fn)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if accepts_kwargs:
            fn_kwargs.update(inputs)
            for key in input_keys:
                fn_kwargs.setdefault(key, self.coupler.get(key))
        else:
            allowed = set(sig.parameters.keys())
            for key, value in inputs.items():
                if key in allowed:
                    fn_kwargs[key] = value
            for key in input_keys:
                if key in allowed:
                    fn_kwargs.setdefault(key, self.coupler.get(key))

        fn(**fn_kwargs)
        return {"status": "ok"}


class DummyPreprocessor:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def preprocess(self, workspace, previous_records=RecordStore()):
        return self._record_builder(self.model_name, "preprocess")


class DummyRunner:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def run(self, input_store, workspace):
        return self._record_builder(self.model_name, "run")


class DummyPostprocessor:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def postprocess(self, raw_outputs, workspace):
        return self._record_builder(self.model_name, "postprocess")


def _write_file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _build_settings(tmp_path: Path):
    config = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2017,
            "end_year": 2018,
            "travel_model_freq": 1,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "restart_boundary_test",
            "supply_demand_iters": 1,
            "models": {
                "land_use": "urbansim",
                "travel": "beam",
                "activity_demand": "activitysim",
                "vehicle_ownership": "atlas",
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["00001"]},
                "local_crs": "EPSG:4326",
            },
            "skims": {"fname": "skims.omx"},
            "database": {
                "enabled": False,
                "type": "duckdb",
                "path": str(tmp_path / "db.duckdb"),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        "urbansim": {
            "local_data_input_folder": str(tmp_path / "usim_input"),
            "local_mutable_data_folder": "urbansim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "usim_{region_id}.h5",
            "input_file_template_year": "usim_{region_id}_{year}.h5",
            "output_file_template": "usim_{year}.h5",
            "command_template": "run_usim",
            "region_mappings": {"region_to_region_id": {"test": "000"}},
        },
        "atlas": {
            "host_input_folder": "atlas/input",
            "warmstart_input_folder": "atlas/warmstart",
            "host_mutable_input_folder": "atlas/atlas_input",
            "host_output_folder": "atlas/atlas_output",
            "container_input_folder": "/atlas/input",
            "container_output_folder": "/atlas/output",
            "basedir": "/atlas",
            "codedir": "/atlas/code",
            "scenario": "baseline",
            "command_template": "atlas {0}",
        },
        "activitysim": {
            "local_input_folder": "activitysim/input",
            "local_mutable_data_folder": "activitysim/data",
            "local_output_folder": "activitysim/output",
            "local_configs_folder": "activitysim/configs",
            "local_mutable_configs_folder": "activitysim/configs_mutable",
            "validation_folder": "activitysim/validation",
            "command_template": "asim run",
            "main_configs_dir": "configs",
            "final_plans_folder": "activitysim/final_plans",
            "region_mappings": {"region_to_subdir": {"test": "test"}},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "beam/input",
            "local_mutable_data_folder": "beam/input",
            "local_output_folder": "beam/output",
            "scenario_folder": "beam/scenario",
            "router_directory": "router",
            "skims_shapefile": "beam/skims.shp",
            "skim_zone_source_id_col": "id",
            "skim_zone_geoid_col": "geoid",
        },
    }
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_config(str(config_path))


@pytest.fixture
def restart_stage_env(tmp_path, monkeypatch):
    from pilates.generic.model_factory import ModelFactory
    from pilates.utils import consist_runtime as cr

    cr.set_enabled(False)
    settings = _build_settings(tmp_path)
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_output_dir = Path(workspace.get_asim_output_dir())
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_output_dir = Path(workspace.get_beam_output_dir())
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    asim_configs_root = Path(workspace.get_asim_mutable_configs_dir())

    for path in (
        usim_dir,
        asim_dir,
        asim_output_dir,
        beam_dir,
        beam_output_dir,
        atlas_input_dir,
        atlas_output_dir,
        asim_configs_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for cfg_dir in (
        asim_configs_root / "configs",
        asim_configs_root / "configs_extended",
        asim_configs_root / "configs_mp",
        asim_configs_root / "configs_sh_compile",
    ):
        _write_file(cfg_dir / "settings.yaml")

    region_id = settings.urbansim.region_mappings["region_to_region_id"][
        settings.run.region
    ]
    usim_input_path = _write_file(
        usim_dir / settings.urbansim.input_file_template.format(region_id=region_id),
        "usim-input",
    )
    usim_output_path = _write_file(
        usim_dir / settings.urbansim.output_file_template.format(year=state.forecast_year),
        "usim-forecast",
    )
    usim_merged_path = _write_file(
        usim_dir / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5",
        "usim-merged",
    )

    land_use_path = _write_file(asim_dir / "land_use.csv")
    households_path = _write_file(asim_dir / "households.csv")
    persons_path = _write_file(asim_dir / "persons.csv")
    omx_path = _write_file(asim_dir / "skims.omx")
    zarr_path = _write_file(asim_output_dir / "cache" / "skims.zarr")
    numba_cache_path = _write_file(
        Path(workspace.full_path) / "shared_cache" / "numba" / "cache.bin"
    )

    beam_plans_path = _write_file(beam_dir / "plans.csv")
    beam_households_path = _write_file(beam_dir / "households.csv")
    beam_persons_path = _write_file(beam_dir / "persons.csv")
    beam_linkstats_path = _write_file(beam_dir / "linkstats.csv.gz")
    beam_full_skims_path = _write_file(beam_output_dir / "skimsODFull.csv.gz")

    def record_builder(model_name, phase):
        if phase == "preprocess":
            if model_name == "activitysim":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(land_use_path), short_name=ASIM_LAND_USE_IN),
                        FileRecord(file_path=str(households_path), short_name=ASIM_HOUSEHOLDS_IN),
                        FileRecord(file_path=str(persons_path), short_name=ASIM_PERSONS_IN),
                        FileRecord(file_path=str(omx_path), short_name=ASIM_OMX_SKIMS),
                    ]
                )
            if model_name == "beam":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(beam_plans_path), short_name=BEAM_PLANS_IN),
                        FileRecord(file_path=str(beam_households_path), short_name=BEAM_HOUSEHOLDS_IN),
                        FileRecord(file_path=str(beam_persons_path), short_name=BEAM_PERSONS_IN),
                        FileRecord(file_path=str(beam_linkstats_path), short_name=LINKSTATS_WARMSTART),
                    ]
                )
            return RecordStore()
        if phase == "run":
            if model_name == "urbansim":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(usim_output_path), short_name=USIM_FORECAST_OUTPUT)
                    ]
                )
            if model_name == "activitysim_compile":
                _write_file(zarr_path)
                _write_file(numba_cache_path)
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(zarr_path), short_name=ZARR_SKIMS),
                        FileRecord(
                            file_path=str(numba_cache_path.parent),
                            short_name=ASIM_SHARROW_CACHE_DIR,
                        ),
                    ]
                )
            if model_name == "beam_full_skim":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(beam_full_skims_path), short_name="skimsODFull")
                    ]
                )
            return RecordStore()
        if phase == "postprocess" and model_name == "urbansim":
            return RecordStore(
                recordList=[
                    FileRecord(
                        file_path=str(usim_merged_path),
                        short_name=f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}",
                    )
                ]
            )
        return RecordStore()

    monkeypatch.setattr(
        ModelFactory,
        "get_preprocessor",
        lambda self, model_name, state=None, major_stage=None: DummyPreprocessor(
            model_name, record_builder
        ),
    )
    monkeypatch.setattr(
        ModelFactory,
        "get_runner",
        lambda self, model_name, state=None, major_stage=None: DummyRunner(
            model_name, record_builder
        ),
    )
    monkeypatch.setattr(
        ModelFactory,
        "get_postprocessor",
        lambda self, model_name, state=None, major_stage=None: DummyPostprocessor(
            model_name, record_builder
        ),
    )

    coupler = CouplerStub()
    scenario = FakeScenario(coupler)
    env = {
        "settings": settings,
        "workspace": workspace,
        "state": state,
        "coupler": coupler,
        "scenario": scenario,
        "usim_input_path": str(usim_input_path),
        "zarr_path": str(zarr_path),
        "numba_cache_dir": str(numba_cache_path.parent),
        "atlas_input_dir": atlas_input_dir,
    }
    try:
        yield env
    finally:
        cr.set_enabled(None)


def test_restart_land_use_boundary_preserves_required_datastores(
    restart_stage_env, monkeypatch
):
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    geoid_to_zone_path = _write_file(
        Path(restart_stage_env["workspace"].get_usim_mutable_data_dir()) / "geoid_to_zone.csv"
    )
    captured = {}

    def _fake_run_workflow(*, steps, outputs_holder, **_kwargs):
        if any(step.name == "urbansim_preprocess" for step in steps):
            outputs_holder.urbansim_preprocess = SimpleNamespace(
                to_record_store=lambda: RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(geoid_to_zone_path),
                            short_name="geoid_to_zone",
                        )
                    ]
                )
            )
            return

        run_step = next(step for step in steps if step.name == "urbansim_run")
        captured["inputs"] = dict(run_step.inputs or {})
        outputs_holder.urbansim_run = SimpleNamespace(
            usim_datastore_h5=Path(restart_stage_env["usim_input_path"])
        )
        outputs_holder.urbansim_postprocess = None

    monkeypatch.setattr(land_use_stage, "run_workflow", _fake_run_workflow)
    monkeypatch.setattr(land_use_stage, "enqueue_archive_copy", lambda **_kwargs: None)
    monkeypatch.setattr(land_use_stage, "flush_archive_queue", lambda **_kwargs: None)

    outputs_holder = StepOutputsHolder()
    usim_inputs = run_land_use_stage(
        scenario=restart_stage_env["scenario"],
        state=restart_stage_env["state"],
        settings=restart_stage_env["settings"],
        workspace=restart_stage_env["workspace"],
        coupler=restart_stage_env["coupler"],
        year=restart_stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )

    assert captured["inputs"]["geoid_to_zone"] == str(geoid_to_zone_path)
    assert captured["inputs"][USIM_DATASTORE_CURRENT_H5] == restart_stage_env["usim_input_path"]
    assert captured["inputs"][USIM_DATASTORE_BASE_H5] == restart_stage_env["usim_input_path"]
    assert usim_inputs[USIM_DATASTORE_CURRENT_H5] == restart_stage_env["usim_input_path"]


def test_restart_vehicle_ownership_boundary_uses_rebuilt_atlas_registry(
    restart_stage_env, monkeypatch
):
    from pilates.workflows.stages import vehicle_ownership as vehicle_ownership_stage

    atlas_static_path = _write_file(
        restart_stage_env["atlas_input_dir"] / "psid_names.Rdat",
        "psid",
    )
    restored = run_module._restore_restart_workspace_atlas_registry(
        settings=restart_stage_env["settings"],
        workspace=restart_stage_env["workspace"],
    )
    assert restored >= 1

    restart_stage_env["state"].current_major_stage = (
        restart_stage_env["state"].Stage.vehicle_ownership_model
    )
    restart_stage_env["coupler"].set(
        USIM_DATASTORE_CURRENT_H5, restart_stage_env["usim_input_path"]
    )
    restart_stage_env["coupler"].set(
        USIM_DATASTORE_BASE_H5, restart_stage_env["usim_input_path"]
    )

    monkeypatch.setattr(
        vehicle_ownership_stage,
        "build_urbansim_inputs",
        lambda *_args, **_kwargs: (
            {
                USIM_DATASTORE_CURRENT_H5: restart_stage_env["usim_input_path"],
                USIM_DATASTORE_BASE_H5: restart_stage_env["usim_input_path"],
            },
            {},
        ),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "atlas_static_input_keys_for_interval",
        lambda *_args, **_kwargs: ("psid_names",),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "merge_model_expected_inputs",
        lambda _model_name, inputs, *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "enqueue_archive_copy",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "flush_archive_queue",
        lambda **_kwargs: None,
    )

    captured = {}

    def _fake_run_workflow(*, steps, state, workspace, outputs_holder, **_kwargs):
        atlas_run_step = next((step for step in steps if step.name == "atlas_run"), None)
        if atlas_run_step is not None:
            captured[state.year] = dict(atlas_run_step.inputs or {})
            raw_output = _write_file(
                Path(workspace.get_atlas_output_dir()) / f"households_{state.year}.csv"
            )
            outputs_holder.atlas_run = AtlasRunOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                raw_outputs={"atlas_households_csv": raw_output},
            )

    monkeypatch.setattr(vehicle_ownership_stage, "run_workflow", _fake_run_workflow)

    run_vehicle_ownership_stage(
        scenario=SimpleNamespace(),
        state=restart_stage_env["state"],
        settings=restart_stage_env["settings"],
        workspace=restart_stage_env["workspace"],
        coupler=restart_stage_env["coupler"],
        year=restart_stage_env["state"].forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {
            "psid_names": str(Path("fallback-should-not-win.Rdat"))
        },
    )

    atlas_run_inputs = captured[restart_stage_env["state"].year]
    assert atlas_run_inputs[USIM_DATASTORE_CURRENT_H5] == restart_stage_env["usim_input_path"]
    assert atlas_run_inputs[USIM_DATASTORE_BASE_H5] == restart_stage_env["usim_input_path"]
    assert atlas_run_inputs["psid_names"] == str(atlas_static_path)


def test_restart_activity_demand_boundary_reuses_restored_compile_artifacts(
    restart_stage_env, tmp_path
):
    settings = restart_stage_env["settings"]
    state = restart_stage_env["state"]
    workspace = restart_stage_env["workspace"]
    coupler = restart_stage_env["coupler"]
    scenario = restart_stage_env["scenario"]

    coupler.set(USIM_DATASTORE_CURRENT_H5, restart_stage_env["usim_input_path"])
    coupler.set(USIM_DATASTORE_BASE_H5, restart_stage_env["usim_input_path"])
    coupler.set(ZARR_SKIMS, restart_stage_env["zarr_path"])
    coupler.set(ASIM_SHARROW_CACHE_DIR, restart_stage_env["numba_cache_dir"])
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0
    state.compile_asim()

    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: restart_stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: restart_stage_env["usim_input_path"],
    }

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"restart_activity_demand_{year}_{iteration}.yaml",
    )

    assert not any(call.get("model") == "activitysim_compile" for call in scenario.calls)
    asim_run_calls = [
        call
        for call in scenario.calls
        if ZARR_SKIMS in (call.get("input_keys") or [])
        and ASIM_HOUSEHOLDS_IN in (call.get("input_keys") or [])
        and ASIM_PERSONS_IN in (call.get("input_keys") or [])
        and ASIM_LAND_USE_IN in (call.get("input_keys") or [])
    ]
    assert asim_run_calls, "Expected ActivitySim run to start from restored compile outputs."


def test_restart_traffic_assignment_boundary_uses_restored_default_beam_inputs(
    restart_stage_env, tmp_path
):
    settings = restart_stage_env["settings"]
    state = restart_stage_env["state"]
    workspace = restart_stage_env["workspace"]
    coupler = restart_stage_env["coupler"]
    scenario = restart_stage_env["scenario"]

    settings.run.models.activity_demand = None
    settings.activity_demand_enabled = False
    state._settings["activity_demand_enabled"] = False
    state.enabled_stages.discard(state.Stage.activity_demand)
    state.loop_substages = [state.Stage.traffic_assignment]
    settings.beam.full_skim = FullSkimsCreatorConfig(run_schedule="disabled")

    scenario_dir = (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.scenario_folder
    )
    default_plans = _write_file(scenario_dir / "plans.parquet")
    default_households = _write_file(scenario_dir / "households.parquet")
    default_persons = _write_file(scenario_dir / "persons.parquet")

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={
            USIM_DATASTORE_CURRENT_H5: restart_stage_env["usim_input_path"],
            USIM_DATASTORE_BASE_H5: restart_stage_env["usim_input_path"],
        },
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"restart_traffic_{year}_{iteration}.yaml",
    )

    beam_preprocess_calls = [
        call
        for call in scenario.calls
        if BEAM_PLANS_IN in call["inputs"]
        and BEAM_HOUSEHOLDS_IN in call["inputs"]
        and BEAM_PERSONS_IN in call["inputs"]
    ]
    assert beam_preprocess_calls, "Expected BEAM preprocess to start from restored defaults."
    beam_preprocess_inputs = beam_preprocess_calls[0]["inputs"]
    assert beam_preprocess_inputs[BEAM_PLANS_IN] == str(default_plans)
    assert beam_preprocess_inputs[BEAM_HOUSEHOLDS_IN] == str(default_households)
    assert beam_preprocess_inputs[BEAM_PERSONS_IN] == str(default_persons)


def test_restart_mid_iteration_traffic_assignment_preserves_promoted_warmstart(
    restart_stage_env, tmp_path
):
    settings = restart_stage_env["settings"]
    state = restart_stage_env["state"]
    workspace = restart_stage_env["workspace"]
    coupler = restart_stage_env["coupler"]
    scenario = restart_stage_env["scenario"]

    settings.run.models.activity_demand = None
    settings.activity_demand_enabled = False
    settings.run.supply_demand_iters = 2
    settings.beam.full_skim = FullSkimsCreatorConfig(run_schedule="disabled")
    state._settings["activity_demand_enabled"] = False
    state._settings["supply_demand_iters"] = 2
    state.enabled_stages.discard(state.Stage.activity_demand)
    state.loop_substages = [state.Stage.traffic_assignment]

    scenario_dir = (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.scenario_folder
    )
    _write_file(scenario_dir / "plans.parquet")
    _write_file(scenario_dir / "households.parquet")
    _write_file(scenario_dir / "persons.parquet")

    restored_linkstats = _write_file(tmp_path / "restored" / "linkstats.parquet")
    restored_plans = _write_file(tmp_path / "restored" / "beam_plans.parquet")
    coupler.set(LINKSTATS, str(restored_linkstats))
    coupler.set(BEAM_PLANS_OUT, str(restored_plans))

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 1

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={
            USIM_DATASTORE_CURRENT_H5: restart_stage_env["usim_input_path"],
            USIM_DATASTORE_BASE_H5: restart_stage_env["usim_input_path"],
        },
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"restart_midloop_{year}_{iteration}.yaml",
    )

    beam_preprocess_calls = [
        call for call in scenario.calls if LINKSTATS in call.get("inputs", {})
    ]
    assert beam_preprocess_calls, "Expected resumed BEAM preprocess to receive prior linkstats."
    assert beam_preprocess_calls[0]["inputs"][LINKSTATS] == str(restored_linkstats)
    beam_run_calls = [
        call
        for call in scenario.calls
        if BEAM_PLANS_IN in (call.get("input_keys") or [])
        and BEAM_HOUSEHOLDS_IN in (call.get("input_keys") or [])
        and BEAM_PERSONS_IN in (call.get("input_keys") or [])
    ]
    assert beam_run_calls, "Expected resumed mid-loop traffic assignment to reach BEAM run."
