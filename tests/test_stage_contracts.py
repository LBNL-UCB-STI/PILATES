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
import shutil

import pytest
import yaml

from pilates.config import load_config
from pilates.config.models import FullSkimsCreatorConfig
from pilates.generic.records import RecordStore
from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
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
from tests.workflow_contract_harness import (
    CouplerStub,
    DummyPostprocessor,
    DummyPreprocessor,
    DummyRunner,
    FakeScenario,
)
from workflow_state import WorkflowState


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
    numba_cache_path = Path(workspace.full_path) / "shared_cache" / "numba"
    _write_file(numba_cache_path / "cache.bin")

    beam_plans_path = beam_dir / "plans.csv"
    beam_households_path = beam_dir / "households.csv"
    beam_persons_path = beam_dir / "persons.csv"
    beam_linkstats_path = beam_dir / "linkstats.csv.gz"
    beam_full_skims_path = beam_out_dir / "skimsODFull.csv.gz"
    atlas_householdv_path = atlas_output_dir / f"householdv_{state.forecast_year}.csv"
    atlas_vehicles_path = atlas_output_dir / f"vehicles_{state.forecast_year}.csv"
    atlas_vehicles2_path = atlas_output_dir / f"vehicles2_{state.forecast_year}.csv"
    _write_file(beam_plans_path)
    _write_file(beam_households_path)
    _write_file(beam_persons_path)
    _write_file(beam_linkstats_path)
    _write_file(beam_full_skims_path)
    _write_file(atlas_householdv_path)
    _write_file(atlas_vehicles_path)
    _write_file(atlas_vehicles2_path)

    def record_builder(model_name, phase):
        if phase == "preprocess":
            if model_name == "urbansim":
                return UrbanSimPreprocessOutputs(
                    usim_mutable_data_dir=usim_dir,
                    prepared_inputs={},
                )
            if model_name == "atlas":
                return AtlasPreprocessOutputs(
                    atlas_mutable_input_dir=Path(workspace.get_atlas_mutable_input_dir()),
                    prepared_inputs={},
                )
            if model_name == "activitysim":
                return ActivitySimPreprocessOutputs(
                    mutable_data_dir=asim_dir,
                    land_use_table=land_use_path,
                    households_table=households_path,
                    persons_table=persons_path,
                    omx_skims=omx_path,
                )
            if model_name == "beam":
                return BeamPreprocessOutputs(
                    beam_mutable_data_dir=beam_dir,
                    prepared_inputs={
                        BEAM_PLANS_IN: beam_plans_path,
                        BEAM_HOUSEHOLDS_IN: beam_households_path,
                        BEAM_PERSONS_IN: beam_persons_path,
                        LINKSTATS_WARMSTART: beam_linkstats_path,
                    },
                )
            return RecordStore()
        if phase == "run":
            if model_name == "urbansim":
                return UrbanSimRunOutputs(
                    usim_datastore_h5=usim_output_path,
                    raw_outputs={
                        USIM_FORECAST_OUTPUT: usim_output_path,
                    },
                )
            if model_name == "atlas":
                return AtlasRunOutputs(
                    atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                    raw_outputs={
                        f"householdv_{state.forecast_year}": atlas_householdv_path,
                        f"vehicles_{state.forecast_year}": atlas_vehicles_path,
                    },
                )
            if model_name == "activitysim":
                return ActivitySimRunOutputs(
                    output_dir=asim_out_dir,
                    raw_outputs={},
                )
            if model_name == "activitysim_compile":
                _write_file(zarr_path)
                _write_file(numba_cache_path / "cache.bin")
                return ActivitySimCompileOutputs(
                    zarr_skims=zarr_path,
                    sharrow_cache_dir=numba_cache_path,
                )
            if model_name == "beam":
                return BeamRunOutputs(
                    beam_output_dir=beam_out_dir,
                    raw_outputs={
                        "linkstats": beam_linkstats_path,
                        "beam_plans_out": beam_plans_path,
                    },
                )
            if model_name == "beam_full_skim":
                return BeamFullSkimOutputs(
                    full_skims=beam_full_skims_path,
                )
            return RecordStore()
        if phase == "postprocess":
            if model_name == "urbansim":
                return UrbanSimPostprocessOutputs(
                    usim_datastore_h5=usim_merged_path,
                    processed_outputs={
                        f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}": usim_merged_path,
                    },
                )
            if model_name == "atlas":
                return AtlasPostprocessOutputs(
                    atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                    usim_datastore_h5=Path(usim_input_path),
                    processed_outputs={
                        "atlas_vehicles2_output": atlas_vehicles2_path,
                    },
                )
            if model_name == "activitysim":
                return ActivitySimPostprocessOutputs(
                    usim_datastore_h5=None,
                    asim_output_dir=asim_out_dir,
                    processed_outputs={
                        "beam_plans_out": beam_plans_path,
                        "households_asim_out": households_path,
                        "persons_asim_out": persons_path,
                    },
                )
            if model_name == "beam":
                return BeamPostprocessOutputs(zarr_skims=zarr_path)
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory

    def _make_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPreprocessor(model_name, record_builder)

    def _make_runner(self, model_name, state=None, *_args, **_kwargs):
        return DummyRunner(model_name, record_builder)

    def _make_postprocessor(self, model_name, state=None, *_args, **_kwargs):
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


def test_land_use_stage_flushes_archive_queue_at_boundary(stage_env, monkeypatch):
    """Land use boundary should enqueue restart H5s and flush archive queue."""
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    enqueue_calls = []
    flush_calls = []
    monkeypatch.setattr(
        land_use_stage,
        "enqueue_archive_copy",
        lambda **kwargs: enqueue_calls.append(kwargs),
    )
    monkeypatch.setattr(
        land_use_stage,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: flush_calls.append(timeout),
    )

    outputs_holder = StepOutputsHolder()
    run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )

    keys = {call["key"] for call in enqueue_calls}
    assert USIM_DATASTORE_BASE_H5 in keys
    assert USIM_DATASTORE_CURRENT_H5 in keys
    assert any(key.startswith("usim_year_output_h5_") for key in keys)
    assert flush_calls == [300]


def test_land_use_stage_archives_forecast_output_using_state_forecast_year(
    stage_env, monkeypatch
):
    """Archive keys/paths should follow the produced forecast-year datastore."""
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    enqueue_calls = []
    monkeypatch.setattr(
        land_use_stage,
        "enqueue_archive_copy",
        lambda **kwargs: enqueue_calls.append(kwargs),
    )
    monkeypatch.setattr(
        land_use_stage,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: None,
    )

    stage_env["state"].forecast_year = stage_env["state"].year + 2
    forecast_path = (
        Path(stage_env["workspace"].get_usim_mutable_data_dir())
        / stage_env["settings"].urbansim.output_file_template.format(
            year=stage_env["state"].forecast_year
        )
    )
    _write_file(forecast_path)

    outputs_holder = StepOutputsHolder()
    run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].year,
        outputs_holder_year=outputs_holder,
    )

    forecast_archive = next(
        call
        for call in enqueue_calls
        if call["key"] == f"usim_year_output_h5_{stage_env['state'].forecast_year}"
    )
    assert forecast_archive["path"] == str(forecast_path)


def test_land_use_stage_merges_declared_datastore_when_preprocess_outputs_are_partial(
    stage_env, monkeypatch
):
    """Restart-thin preprocess outputs must not drop the required UrbanSim H5."""
    from types import SimpleNamespace
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    geoid_to_zone_path = (
        Path(stage_env["workspace"].get_usim_mutable_data_dir()) / "geoid_to_zone.csv"
    )
    _write_file(geoid_to_zone_path)

    captured = {}

    def _fake_run_workflow(
        *,
        steps,
        outputs_holder,
        workspace,
        state,
        **_kwargs,
    ):
        if any(step.name == "urbansim_preprocess" for step in steps):
            outputs_holder.urbansim_preprocess = SimpleNamespace(
                _iter_record_items=lambda: iter(
                    [
                        (
                            "geoid_to_zone",
                            geoid_to_zone_path,
                            "UrbanSim preprocess output: geoid_to_zone",
                        )
                    ]
                )
            )
            return

        run_step = next(step for step in steps if step.name == "urbansim_run")
        captured["inputs"] = dict(run_step.inputs or {})
        captured["input_keys"] = list(run_step.input_keys or [])
        outputs_holder.urbansim_run = SimpleNamespace(
            usim_datastore_h5=Path(stage_env["usim_input_path"])
        )
        outputs_holder.urbansim_postprocess = None

    monkeypatch.setattr(land_use_stage, "run_workflow", _fake_run_workflow)
    monkeypatch.setattr(
        land_use_stage,
        "enqueue_archive_copy",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        land_use_stage,
        "flush_archive_queue",
        lambda **_kwargs: None,
    )

    outputs_holder = StepOutputsHolder()
    run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )

    assert captured["inputs"]["geoid_to_zone"] == str(geoid_to_zone_path)
    assert captured["inputs"][USIM_DATASTORE_CURRENT_H5] == stage_env["usim_input_path"]
    assert captured["inputs"][USIM_DATASTORE_BASE_H5] == stage_env["usim_input_path"]


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


def test_vehicle_ownership_stage_flushes_per_subyear(stage_env, monkeypatch):
    """ATLAS subyear boundaries should enqueue restart artifacts and flush."""
    from pilates.workflows.stages import vehicle_ownership as vo_stage

    state = stage_env["state"]
    settings = stage_env["settings"]
    workspace = stage_env["workspace"]
    scenario = stage_env["scenario"]
    coupler = stage_env["coupler"]

    state.forecast_year = state.year + 4  # 2017, 2019, 2021
    coupler.set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])

    atlas_years = [state.year, state.year + 2, state.year + 4]
    for atlas_year in atlas_years:
        year_dir = (
            Path(workspace.get_atlas_mutable_input_dir()) / f"year{atlas_year}"
        )
        year_dir.mkdir(parents=True, exist_ok=True)
        _write_file(year_dir / "vehicles_output.RData")

    enqueue_calls = []
    flush_calls = []
    monkeypatch.setattr(
        vo_stage,
        "enqueue_archive_copy",
        lambda **kwargs: enqueue_calls.append(kwargs),
    )
    monkeypatch.setattr(
        vo_stage,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: flush_calls.append(timeout),
    )

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )

    assert flush_calls == [300, 300, 300]
    for atlas_year in atlas_years:
        assert any(
            call["key"] == f"atlas_input_year_dir_{atlas_year}"
            and str(call["path"]).endswith(f"year{atlas_year}")
            for call in enqueue_calls
        )
        assert any(
            call["key"] == f"atlas_rdata_{atlas_year}"
            and str(call["path"]).endswith("vehicles_output.RData")
            for call in enqueue_calls
        )


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


def test_supply_demand_forces_compile_when_numba_cache_missing_for_multiprocess(
    stage_env, tmp_path
):
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

    # Simulate restart state claiming ActivitySim is already compiled.
    state.compile_asim()

    # Keep zarr skims present but delete numba cache to trigger forced compile.
    zarr_path = Path(stage_env["workspace"].get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    stage_env["coupler"].set(ZARR_SKIMS, str(zarr_path))

    numba_cache_dir = Path(stage_env["workspace"].full_path) / "shared_cache" / "numba"
    if numba_cache_dir.exists():
        shutil.rmtree(numba_cache_dir)

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_force_compile_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    compile_calls = [
        call for call in stage_env["scenario"].calls if call.get("model") == "activitysim_compile"
    ]
    assert compile_calls, "Expected forced ActivitySim compile when numba cache is missing."


def test_supply_demand_stage_flushes_and_enqueues_manifest(stage_env, monkeypatch, tmp_path):
    """Supply-demand iteration boundary should enqueue/flush manifest artifacts."""
    from pilates.workflows.stages import supply_demand as sd_stage

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

    manifest_path = tmp_path / "manifest_boundary.yaml"

    enqueue_calls = []
    flush_calls = []
    monkeypatch.setattr(
        sd_stage,
        "enqueue_archive_copy",
        lambda **kwargs: enqueue_calls.append(kwargs),
    )
    monkeypatch.setattr(
        sd_stage,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: flush_calls.append(timeout),
    )

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, _year, _iteration: manifest_path,
    )

    assert manifest_path.exists()
    assert any(
        call["key"] == "workflow_manifest" and Path(call["path"]) == manifest_path
        for call in enqueue_calls
    )
    assert flush_calls == [300]


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


def test_supply_demand_stage_does_not_skip_next_year_land_use_after_final_substage(
    stage_env, tmp_path
):
    """
    Completing the final traffic-assignment substage should advance into the
    next year's land-use stage exactly once.
    """
    settings = stage_env["settings"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    coupler.set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    start_year = state.current_year

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_no_skip_{year}_{iteration}.json"

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

    assert state.current_year == start_year + 1
    assert state.current_major_stage == state.Stage.land_use
    assert state.current_sub_stage is None


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

    asim_outputs = {
        "beam_plans_out": str(tmp_path / "beam_plans_out.parquet"),
        "households_asim_out": str(tmp_path / "households_asim_out.parquet"),
        "persons_asim_out": str(tmp_path / "persons_asim_out.parquet"),
    }
    zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    coupler.set(ZARR_SKIMS, str(zarr_path))
    linkstats_history_path = tmp_path / "linkstats_parquet_2018_0.parquet"
    _write_file(linkstats_history_path)
    previous_beam_outputs = {
        "linkstats_parquet_2018_0": str(linkstats_history_path)
    }

    class _BeamPreprocessorNoWarmstart:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ):
            beam_dir = Path(workspace.get_beam_mutable_data_dir())
            plans = beam_dir / "plans.csv"
            households = beam_dir / "households.csv"
            persons = beam_dir / "persons.csv"
            for path in (plans, households, persons):
                _write_file(path)
            return BeamPreprocessOutputs(
                beam_mutable_data_dir=beam_dir,
                prepared_inputs={
                    BEAM_PLANS_IN: plans,
                    BEAM_HOUSEHOLDS_IN: households,
                    BEAM_PERSONS_IN: persons,
                },
            )

    original_get_preprocessor = ModelFactory.get_preprocessor

    def _patched_get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamPreprocessorNoWarmstart()
        return original_get_preprocessor(self, model_name, state)

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


def test_beam_postprocess_uses_explicit_sub_iteration_run_artifacts(
    stage_env, monkeypatch, tmp_path
):
    """
    Regression: BEAM postprocess must consume sub-iteration run artifacts via
    explicit inputs, not coupler-required input_keys.
    """
    from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
    from pilates.generic.model_factory import ModelFactory
    from pilates.workflows.steps import StepOutputsHolder

    settings = stage_env["settings"]
    state = stage_env["state"]
    workspace = stage_env["workspace"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    coupler.set(ZARR_SKIMS, str(zarr_path))

    class _BeamPreprocessorNoWarmstart:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ):
            beam_dir = Path(workspace.get_beam_mutable_data_dir())
            plans = beam_dir / "plans.csv"
            households = beam_dir / "households.csv"
            persons = beam_dir / "persons.csv"
            for path in (plans, households, persons):
                _write_file(path)
            return BeamPreprocessOutputs(
                beam_mutable_data_dir=beam_dir,
                prepared_inputs={
                    BEAM_PLANS_IN: plans,
                    BEAM_HOUSEHOLDS_IN: households,
                    BEAM_PERSONS_IN: persons,
                },
            )

    class _BeamRunnerWithSubIterationOutputs:
        def run(self, inputs, workspace, **_kwargs):
            beam_out = Path(workspace.get_beam_output_dir())
            skim_sub0 = beam_out / "0.skimsActivitySimOD_current.zarr"
            events_sub0 = beam_out / "0.events.parquet"
            _write_file(skim_sub0)
            _write_file(events_sub0)
            return BeamRunOutputs(
                beam_output_dir=beam_out,
                raw_outputs={
                    f"raw_od_skims_zarr_{state.forecast_year}_0_sub0": skim_sub0,
                    f"events_parquet_{state.forecast_year}_0_sub0": events_sub0,
                },
            )

    original_get_preprocessor = ModelFactory.get_preprocessor
    original_get_runner = ModelFactory.get_runner

    def _patched_get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamPreprocessorNoWarmstart()
        return original_get_preprocessor(self, model_name, state)

    def _patched_get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamRunnerWithSubIterationOutputs()
        return original_get_runner(self, model_name, state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _patched_get_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _patched_get_runner)

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
            activity_demand_outputs={},
            previous_beam_outputs=None,
        ),
        outputs_holder=outputs_holder,
    )

    beam_postprocess_calls = [
        call for call in scenario.calls if call.get("model") == "beam_postprocess"
    ]
    assert beam_postprocess_calls, "Expected BEAM postprocess step to execute."
    postprocess_call = beam_postprocess_calls[0]
    assert f"events_parquet_{state.forecast_year}_0_sub0" in postprocess_call["inputs"]
    assert (
        f"raw_od_skims_zarr_{state.forecast_year}_0_sub0"
        in postprocess_call["inputs"]
    )
    assert f"events_parquet_{state.forecast_year}_0_sub0" not in (
        postprocess_call["input_keys"] or []
    )
    assert f"raw_od_skims_zarr_{state.forecast_year}_0_sub0" not in (
        postprocess_call["input_keys"] or []
    )


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
