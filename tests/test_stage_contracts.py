"""
Narrative contract tests for stage orchestration wiring.

This module documents how major workflow stages are expected to assemble and
exchange inputs/outputs through the scenario coupler:

1. Land use stage must publish UrbanSim datastore handles.
2. Vehicle ownership stage must consume datastore inputs and preserve coupler
   continuity for downstream stages.
3. Supply-demand stage must wire ActivitySim/BEAM inputs with the canonical
   input resolution rules and avoid over-requiring optional warmstart artifacts.
4. BEAM postprocess key selection must include only the artifacts required by
   the postprocess phase and maintain legacy fallback behavior.

The tests use lightweight fakes (``FakeScenario`` and ``CouplerStub``) to make
contract expectations explicit without running heavy model containers.
"""

from pathlib import Path

import pytest
import yaml

from pilates.config import load_config
from pilates.config.models import FullSkimsCreatorConfig
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_FULL_SKIMS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.supply_demand import (
    _build_beam_postprocess_input_keys,
    _run_traffic_assignment_phase,
    TrafficAssignmentPhaseInputs,
)
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from workflow_state import WorkflowState


class CouplerStub:
    """Minimal in-memory coupler implementation for stage contract tests."""

    def __init__(self) -> None:
        self._values = {}

    def set(self, key, value) -> None:
        self._values[key] = value

    def set_from_artifact(self, key, value) -> None:
        self._values[key] = value

    def get(self, key, default=None):
        return self._values.get(key, default)

    def pop(self, key, default=None):
        return self._values.pop(key, default)

    def keys(self):
        return list(self._values.keys())

    def require(self, key):
        if key not in self._values:
            raise KeyError(f"Coupler missing key={key!r}")
        return self._values[key]


class FakeScenario:
    """
    Scenario stub that enforces coupler key requirements and records each call.

    This mirrors the contract-level behavior we depend on in production:
    ``inputs`` materialize concrete values, while ``input_keys`` must already
    exist in the coupler.
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

        runtime_kwargs = kwargs.get("runtime_kwargs") or {}
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
    """Deterministic preprocessor stub backed by ``record_builder``."""

    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def preprocess(self, workspace, previous_records=RecordStore()):
        return self._record_builder(self.model_name, "preprocess")


class DummyRunner:
    """Deterministic runner stub backed by ``record_builder``."""

    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def run(self, input_store, workspace):
        return self._record_builder(self.model_name, "run")


class DummyPostprocessor:
    """Deterministic postprocessor stub backed by ``record_builder``."""

    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def postprocess(self, raw_outputs, workspace):
        return self._record_builder(self.model_name, "postprocess")


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_settings(tmp_path: Path):
    """Build a compact workflow config that exercises all stage contracts."""

    config = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2017,
            "end_year": 2018,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "test_run",
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
            "database": {"enabled": False, "type": "duckdb", "path": str(tmp_path / "db.duckdb")},
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
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle)
    return load_config(str(config_path))


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    """
    Shared stage test harness with fake models, workspace, and scenario.

    The fixture seeds files/artifacts that each stage expects so individual
    tests can focus on wiring contracts rather than file setup.
    """

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
    asim_configs_dir = Path(workspace.get_asim_mutable_configs_dir())
    asim_out_dir = Path(workspace.get_asim_output_dir())
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_out_dir = Path(workspace.get_beam_output_dir())
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    atlas_output_dir = Path(workspace.get_atlas_output_dir())

    for path in (
        usim_dir,
        asim_dir,
        asim_configs_dir,
        asim_out_dir,
        beam_dir,
        beam_out_dir,
        atlas_input_dir,
        atlas_output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    region_id = settings.urbansim.region_mappings["region_to_region_id"][
        settings.run.region
    ]
    usim_input_path = usim_dir / settings.urbansim.input_file_template.format(
        region_id=region_id
    )
    usim_output_path = usim_dir / settings.urbansim.output_file_template.format(
        year=state.forecast_year
    )
    usim_merged_path = usim_dir / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    _write_file(usim_input_path)
    _write_file(usim_output_path)
    _write_file(usim_merged_path)

    land_use_path = asim_dir / "land_use.csv"
    households_path = asim_dir / "households.csv"
    persons_path = asim_dir / "persons.csv"
    omx_path = asim_dir / "skims.omx"
    _write_file(land_use_path)
    _write_file(households_path)
    _write_file(persons_path)
    _write_file(omx_path)

    zarr_path = asim_out_dir / "cache" / "skims.zarr"
    _write_file(zarr_path)

    beam_plans_path = beam_dir / "plans.csv"
    beam_households_path = beam_dir / "households.csv"
    beam_persons_path = beam_dir / "persons.csv"
    beam_linkstats_path = beam_dir / "linkstats.csv.gz"
    beam_full_skims_path = beam_out_dir / "skimsODFull.csv.gz"
    _write_file(beam_plans_path)
    _write_file(beam_households_path)
    _write_file(beam_persons_path)
    _write_file(beam_linkstats_path)
    _write_file(beam_full_skims_path)

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
                return RecordStore(
                    recordList=[FileRecord(file_path=str(zarr_path), short_name=ZARR_SKIMS)]
                )
            if model_name == "beam_full_skim":
                return RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(beam_full_skims_path),
                            short_name=BEAM_FULL_SKIMS,
                        )
                    ]
                )
            return RecordStore()
        if phase == "postprocess":
            if model_name == "urbansim":
                return RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(usim_merged_path),
                            short_name=f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}",
                        )
                    ]
                )
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory

    def _make_preprocessor(self, model_name, state=None, major_stage=None):
        return DummyPreprocessor(model_name, record_builder)

    def _make_runner(self, model_name, state=None, major_stage=None):
        return DummyRunner(model_name, record_builder)

    def _make_postprocessor(self, model_name, state=None, major_stage=None):
        return DummyPostprocessor(model_name, record_builder)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _make_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _make_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _make_postprocessor)

    coupler = CouplerStub()
    scenario = FakeScenario(coupler)

    env = {
        "settings": settings,
        "workspace": workspace,
        "state": state,
        "coupler": coupler,
        "scenario": scenario,
        "usim_input_path": str(usim_input_path),
    }
    try:
        yield env
    finally:
        cr.set_enabled(None)


def test_land_use_stage_contract(stage_env):
    """Land use must publish datastore handles needed by later stages."""
    from pilates.workflows.steps import StepOutputsHolder

    outputs_holder = StepOutputsHolder()

    usim_inputs = run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )
    assert USIM_DATASTORE_BASE_H5 in usim_inputs
    assert USIM_DATASTORE_CURRENT_H5 in usim_inputs
    assert usim_inputs[USIM_DATASTORE_BASE_H5].endswith("usim_000.h5")
    assert stage_env["coupler"].get(USIM_DATASTORE_H5) is not None
    assert stage_env["coupler"].get(USIM_DATASTORE_BASE_H5) is not None


def test_vehicle_ownership_stage_contract(stage_env):
    """Vehicle ownership should consume UrbanSim inputs and keep datastore state."""
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    run_vehicle_ownership_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        build_atlas_static_inputs_fallback=lambda workspace: {},
    )
    assert stage_env["coupler"].get(USIM_DATASTORE_H5) is not None


def test_supply_demand_stage_contract(stage_env, tmp_path):
    """Supply-demand should run ActivitySim with resolved required input keys."""
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )
    assert stage_env["coupler"].get(ZARR_SKIMS) is not None
    asim_run_calls = [
        call
        for call in stage_env["scenario"].calls
        if ZARR_SKIMS in (call.get("input_keys") or [])
        and ASIM_HOUSEHOLDS_IN in (call.get("input_keys") or [])
        and ASIM_PERSONS_IN in (call.get("input_keys") or [])
        and ASIM_LAND_USE_IN in (call.get("input_keys") or [])
    ]
    assert asim_run_calls, "Expected an ActivitySim run step call."
    assert ASIM_OMX_SKIMS not in asim_run_calls[0]["input_keys"]


def test_supply_demand_stage_beam_only_uses_default_scenario_inputs(stage_env, tmp_path):
    """
    Beam-only mode should source default scenario plans/households/persons.

    This documents behavior when ActivitySim is disabled and BEAM starts from
    static scenario files instead of coupler-provided activity-demand outputs.
    """
    settings = stage_env["settings"]
    state = stage_env["state"]
    workspace = stage_env["workspace"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    settings.run.models.activity_demand = None
    settings.activity_demand_enabled = False
    state._settings["activity_demand_enabled"] = False
    state.enabled_stages.discard(state.Stage.activity_demand)
    state.loop_substages = [state.Stage.traffic_assignment]

    scenario_dir = (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.scenario_folder
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    default_plans = scenario_dir / "plans.parquet"
    default_households = scenario_dir / "households.parquet"
    default_persons = scenario_dir / "persons.parquet"
    _write_file(default_plans)
    _write_file(default_households)
    _write_file(default_persons)

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_beam_only_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    beam_preprocess_calls = [
        call
        for call in scenario.calls
        if BEAM_PLANS_IN in call["inputs"]
        and BEAM_HOUSEHOLDS_IN in call["inputs"]
        and BEAM_PERSONS_IN in call["inputs"]
    ]
    assert beam_preprocess_calls, "Expected a BEAM preprocess step call with default inputs."
    beam_preprocess_inputs = beam_preprocess_calls[0]["inputs"]
    assert beam_preprocess_inputs[BEAM_PLANS_IN] == str(default_plans)
    assert beam_preprocess_inputs[BEAM_HOUSEHOLDS_IN] == str(default_households)
    assert beam_preprocess_inputs[BEAM_PERSONS_IN] == str(default_persons)

    beam_run_calls = [
        call
        for call in scenario.calls
        if BEAM_PLANS_IN in (call.get("input_keys") or [])
        and BEAM_HOUSEHOLDS_IN in (call.get("input_keys") or [])
        and BEAM_PERSONS_IN in (call.get("input_keys") or [])
    ]
    assert beam_run_calls, "Expected BEAM run step call."
    run_input_keys = beam_run_calls[0].get("input_keys") or []
    assert BEAM_PLANS_IN in run_input_keys
    assert BEAM_HOUSEHOLDS_IN in run_input_keys
    assert BEAM_PERSONS_IN in run_input_keys


def test_supply_demand_stage_runs_full_skim_after_each_iteration(stage_env, tmp_path):
    """
    Full-skim schedule after_each_iteration should run dedicated full-skim step.
    """
    settings = stage_env["settings"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    settings.beam.full_skim = FullSkimsCreatorConfig(
        run_schedule="after_each_iteration"
    )
    coupler.set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_full_skim_each_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=stage_env["workspace"],
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    full_skim_calls = [call for call in scenario.calls if call.get("model") == "beam_full_skim"]
    assert full_skim_calls, "Expected BEAM full-skim step call."
    assert coupler.get(BEAM_FULL_SKIMS) is not None


def test_supply_demand_stage_standalone_full_skim_skips_beam_run(stage_env, tmp_path):
    """
    Standalone full-skim schedule should bypass normal BEAM run/postprocess steps.
    """
    settings = stage_env["settings"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    settings.beam.full_skim = FullSkimsCreatorConfig(run_schedule="standalone")
    coupler.set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_full_skim_standalone_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=stage_env["workspace"],
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    assert not any(call.get("model") == "beam_run" for call in scenario.calls)
    assert not any(call.get("model") == "beam_postprocess" for call in scenario.calls)
    assert any(call.get("model") == "beam_full_skim" for call in scenario.calls)
    assert coupler.get(BEAM_FULL_SKIMS) is not None


def test_traffic_assignment_does_not_require_missing_linkstats_warmstart(
    stage_env, monkeypatch, tmp_path
):
    """
    Regression: do not require LINKSTATS_WARMSTART unless that exact key is
    present in beam_preprocess_inputs.
    """
    from pilates.generic.model_factory import ModelFactory
    from pilates.workflows.steps import StepOutputsHolder
    from pilates.activitysim.outputs import ActivitySimPostprocessOutputs

    settings = stage_env["settings"]
    state = stage_env["state"]
    workspace = stage_env["workspace"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    asim_outputs = RecordStore(
        recordList=[
            FileRecord(file_path=str(tmp_path / "beam_plans_out.parquet"), short_name="beam_plans_out"),
            FileRecord(file_path=str(tmp_path / "households_asim_out.parquet"), short_name="households_asim_out"),
            FileRecord(file_path=str(tmp_path / "persons_asim_out.parquet"), short_name="persons_asim_out"),
        ]
    )
    zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    coupler.set(ZARR_SKIMS, str(zarr_path))
    linkstats_history_path = tmp_path / "linkstats_parquet_2018_0.parquet"
    _write_file(linkstats_history_path)
    previous_beam_outputs = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(linkstats_history_path),
                short_name="linkstats_parquet_2018_0",
            )
        ]
    )

    class _BeamPreprocessorNoWarmstart:
        def preprocess(self, workspace, previous_records=RecordStore()):
            beam_dir = Path(workspace.get_beam_mutable_data_dir())
            plans = beam_dir / "plans.csv"
            households = beam_dir / "households.csv"
            persons = beam_dir / "persons.csv"
            for path in (plans, households, persons):
                _write_file(path)
            return RecordStore(
                recordList=[
                    FileRecord(file_path=str(plans), short_name=BEAM_PLANS_IN),
                    FileRecord(file_path=str(households), short_name=BEAM_HOUSEHOLDS_IN),
                    FileRecord(file_path=str(persons), short_name=BEAM_PERSONS_IN),
                ]
            )

    original_get_preprocessor = ModelFactory.get_preprocessor

    def _patched_get_preprocessor(self, model_name, state=None, major_stage=None):
        if model_name == "beam":
            return _BeamPreprocessorNoWarmstart()
        return original_get_preprocessor(self, model_name, state, major_stage)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _patched_get_preprocessor)

    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=Path(workspace.get_asim_output_dir()),
    )

    _run_traffic_assignment_phase(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        inputs=TrafficAssignmentPhaseInputs(
            year=state.forecast_year,
            iteration=0,
            activity_demand_outputs=asim_outputs,
            previous_beam_outputs=previous_beam_outputs,
        ),
        outputs_holder=outputs_holder,
    )

    beam_run_calls = [
        call
        for call in scenario.calls
        if BEAM_PLANS_IN in (call.get("input_keys") or [])
        and BEAM_HOUSEHOLDS_IN in (call.get("input_keys") or [])
        and BEAM_PERSONS_IN in (call.get("input_keys") or [])
    ]
    assert beam_run_calls, "Expected BEAM run step to execute."
    assert LINKSTATS_WARMSTART not in beam_run_calls[0].get("input_keys", [])


def test_build_beam_postprocess_input_keys_filters_to_used_artifacts():
    """Postprocess key builder should include only artifacts consumed downstream."""
    keys = [
        "beam_output_counts_xml_2018_0",
        "linkstats_parquet_2018_0",
        "raw_od_skims_2018_0",
        "raw_od_skims_2018_0_sub0",
        "raw_od_skims_zarr_2018_0",
        "events_parquet_2018_0",
        "events_parquet_2018_0_sub0",
    ]
    selected = _build_beam_postprocess_input_keys(
        upstream_keys=keys,
        year=2018,
        iteration=0,
        include_zarr_skims=True,
    )
    assert selected is not None
    assert "raw_od_skims_2018_0" in selected
    assert "raw_od_skims_zarr_2018_0" in selected
    assert "events_parquet_2018_0" in selected
    assert ZARR_SKIMS in selected
    assert "beam_output_counts_xml_2018_0" not in selected
    assert "linkstats_parquet_2018_0" not in selected


def test_build_beam_postprocess_input_keys_falls_back_for_legacy_names():
    """Legacy naming fallbacks remain supported for compatibility."""
    keys = [
        "raw_od_skims_legacy",
        "events_parquet_legacy",
        "beam_output_network_xml_2018_0",
    ]
    selected = _build_beam_postprocess_input_keys(
        upstream_keys=keys,
        year=2018,
        iteration=0,
        include_zarr_skims=False,
    )
    assert selected == ["raw_od_skims_legacy", "events_parquet_legacy"]
