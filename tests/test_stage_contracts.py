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

import shutil
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from consist.types import BindingResult

from pilates.config import load_config
from pilates.config.models import FullSkimsCreatorConfig
from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
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
    ASIM_SHARROW_CACHE_DIR,
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_FULL_SKIMS,
    FINAL_SKIMS_OMX,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from pilates.workflows.orchestration import ManifestConfig, StageRunner, StepRef
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.stages.land_use import run_land_use_stage as _run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage as _run_supply_demand_stage
from pilates.workflows.stages import supply_demand as supply_demand_stage
from pilates.workflows.stages.handoffs import LandUseToSupplyDemandHandoff
from pilates.workflows.stages.supply_demand import (
    _run_traffic_assignment_phase,
    TrafficAssignmentPhaseInputs,
)
from pilates.workflows.stages.supply_demand_beam import _build_beam_postprocess_input_keys
from pilates.workflows.stages.supply_demand_resume import (
    _restore_activity_demand_outputs_for_resume,
    _restore_supply_demand_usim_inputs_for_resume,
    seed_supply_demand_parent_run_ids_for_resume,
)
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage as _run_vehicle_ownership_stage
import h5py

from tests.workflow_contract_harness import (
    CouplerStub,
    DummyPostprocessor,
    DummyPreprocessor,
    DummyRunner,
    FakeScenario,
    build_runtime_context,
)
from workflow_state import WorkflowState
from pilates.runtime.context import WorkflowRuntimeContext


def run_land_use_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or build_runtime_context(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if surface is not None:
        context = WorkflowRuntimeContext.from_parts(
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            surface=surface,
        )
    return _run_land_use_stage(context=context, **kwargs)


def run_vehicle_ownership_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or build_runtime_context(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if surface is not None:
        context = WorkflowRuntimeContext.from_parts(
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            surface=surface,
        )
    return _run_vehicle_ownership_stage(context=context, **kwargs)


def run_supply_demand_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or build_runtime_context(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if surface is not None:
        context = WorkflowRuntimeContext.from_parts(
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            surface=surface,
        )
    return _run_supply_demand_stage(context=context, **kwargs)


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".h5":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("dummy", data=[1])
        return
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


def test_stage_runner_run_step_forwards_single_step_context():
    captured = {}
    step = StepRef(name="unit_step", step_func=lambda **_kwargs: None)
    manifest_config = ManifestConfig(path=Path("/tmp/test-stage-runner.yaml"))

    def _fake_run_workflow(**kwargs):
        captured.update(kwargs)

    runner = StageRunner(
        stage_name="unit_stage",
        scenario=object(),
        state=object(),
        settings=object(),
        workspace=object(),
        coupler=object(),
        outputs_holder=StepOutputsHolder(),
        name_suffix="phase3",
        iteration=7,
        manifest_config=manifest_config,
        runtime_kwargs_extra={"base": "value"},
        run_workflow_fn=_fake_run_workflow,
    )

    runner.run_step(
        step=step,
        stage_name="override_stage",
        runtime_kwargs_extra={"extra": "value"},
    )

    assert captured["stage_name"] == "override_stage"
    assert captured["steps"] == [step]
    assert captured["name_suffix"] == "phase3"
    assert captured["iteration"] == 7
    assert captured["manifest_config"] == manifest_config
    assert captured["runtime_kwargs_extra"] == {"base": "value", "extra": "value"}


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

    beam_config_path = beam_dir / settings.run.region / settings.beam.config
    _write_file(beam_config_path)

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
    beam_final_omx_path = beam_dir / settings.run.region / settings.shared.skims.fname
    atlas_householdv_path = atlas_output_dir / f"householdv_{state.forecast_year}.csv"
    atlas_vehicles_path = atlas_output_dir / f"vehicles_{state.forecast_year}.csv"
    atlas_vehicles2_path = atlas_output_dir / f"vehicles2_{state.forecast_year}.csv"
    _write_file(beam_plans_path)
    _write_file(beam_households_path)
    _write_file(beam_persons_path)
    _write_file(beam_linkstats_path)
    _write_file(beam_full_skims_path)
    _write_file(beam_final_omx_path)
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
                prepared_inputs = {}
                grave_candidates = sorted(
                    Path(workspace.get_atlas_mutable_input_dir()).glob("year*/grave.csv")
                )
                if grave_candidates:
                    prepared_inputs["atlas_grave_csv"] = grave_candidates[-1]
                return AtlasPreprocessOutputs(
                    atlas_mutable_input_dir=Path(workspace.get_atlas_mutable_input_dir()),
                    prepared_inputs=prepared_inputs,
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
                _write_file(usim_input_path)
                return ActivitySimPostprocessOutputs(
                    usim_datastore_h5=Path(usim_input_path),
                    asim_output_dir=asim_out_dir,
                    processed_outputs={
                        "beam_plans_out": beam_plans_path,
                        "households_asim_out": households_path,
                        "persons_asim_out": persons_path,
                    },
                )
            if model_name == "beam":
                return BeamPostprocessOutputs(
                    zarr_skims=zarr_path,
                    final_skims_omx=beam_final_omx_path,
                )
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
    coupler.set(USIM_FORECAST_OUTPUT, str(usim_output_path))
    coupler.set(USIM_POPULATION_SOURCE_H5, str(usim_output_path))
    scenario = FakeScenario(coupler)

    env = {
        "settings": settings,
        "workspace": workspace,
        "state": state,
        "context": build_runtime_context(
            settings=settings,
            state=state,
            workspace=workspace,
        ),
        "coupler": coupler,
        "scenario": scenario,
        "usim_input_path": str(usim_input_path),
        "usim_output_path": str(usim_output_path),
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


def test_land_use_stage_returns_typed_supply_demand_handoff(stage_env):
    from pilates.workflows.steps import StepOutputsHolder

    outputs_holder = StepOutputsHolder()

    handoff = run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )

    assert isinstance(handoff, LandUseToSupplyDemandHandoff)
    assert handoff.usim_datastore_base_h5 is not None
    assert handoff.usim_datastore_current_h5 is not None
    assert handoff.to_input_mapping()[USIM_DATASTORE_BASE_H5].endswith("usim_000.h5")


def test_land_use_stage_prefers_explicit_beam_skims_artifact(stage_env):
    """UrbanSim preprocess should consume BEAM skims via the coupler when available."""
    from pilates.workflows.steps import StepOutputsHolder

    beam_final_omx = (
        Path(stage_env["workspace"].get_beam_output_dir()) / "final-skims.omx"
    )
    _write_file(beam_final_omx)
    stage_env["coupler"].set(FINAL_SKIMS_OMX, str(beam_final_omx))

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

    preprocess_call = next(
        call
        for call in stage_env["scenario"].calls
        if call["model"] == "urbansim_preprocess"
    )
    preprocess_binding = preprocess_call["binding"]
    assert isinstance(preprocess_binding, BindingResult)
    assert FINAL_SKIMS_OMX in (preprocess_binding.optional_input_keys or [])
    assert FINAL_SKIMS_OMX not in (preprocess_binding.inputs or {})


def test_land_use_stage_keeps_urbansim_preprocess_inputs_file_scoped(stage_env, caplog):
    """UrbanSim preprocess should not inherit model-wide scratch/runtime inputs."""
    from pilates.workflows.steps import StepOutputsHolder

    outputs_holder = StepOutputsHolder()
    with caplog.at_level("WARNING"):
        run_land_use_stage(
            scenario=stage_env["scenario"],
            state=stage_env["state"],
            settings=stage_env["settings"],
            workspace=stage_env["workspace"],
            coupler=stage_env["coupler"],
            year=stage_env["state"].forecast_year,
            outputs_holder_year=outputs_holder,
        )

    preprocess_call = next(
        call
        for call in stage_env["scenario"].calls
        if call["model"] == "urbansim_preprocess"
    )
    preprocess_binding = preprocess_call["binding"]
    assert isinstance(preprocess_binding, BindingResult)
    preprocess_inputs = dict(preprocess_binding.inputs or {})

    assert "usim_mutable_data_dir" not in preprocess_inputs
    assert "usim_source_data_dir" not in preprocess_inputs
    assert "usim_output_h5" not in preprocess_inputs
    assert USIM_POPULATION_SOURCE_H5 not in preprocess_inputs
    assert not any(
        "undeclared input key 'usim_output_h5'" in record.message
        for record in caplog.records
    )


def test_land_use_stage_flushes_archive_queue_at_boundary(stage_env, monkeypatch):
    """Land use boundary should enqueue restart H5s and flush archive queue."""
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    archive_now_calls = []
    flush_calls = []
    monkeypatch.setattr(
        land_use_stage,
        "archive_copy_now",
        lambda **kwargs: archive_now_calls.append(kwargs),
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

    keys = {call["key"] for call in archive_now_calls}
    assert USIM_DATASTORE_BASE_H5 in keys
    assert USIM_DATASTORE_CURRENT_H5 in keys
    assert any(key.startswith("usim_year_output_h5_") for key in keys)
    assert "workflow_manifest" not in keys
    assert flush_calls == [300]


def test_land_use_stage_archives_forecast_output_using_state_forecast_year(
    stage_env, monkeypatch
):
    """Archive keys/paths should follow the produced forecast-year datastore."""
    from pilates.workflows.stages import land_use as land_use_stage
    from pilates.workflows.steps import StepOutputsHolder

    archive_now_calls = []
    monkeypatch.setattr(
        land_use_stage,
        "archive_copy_now",
        lambda **kwargs: archive_now_calls.append(kwargs),
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
        for call in archive_now_calls
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
        captured["inputs"] = dict(getattr(run_step.binding, "inputs", {}) or {})
        captured["input_keys"] = list(
            getattr(run_step.binding, "input_keys", None) or []
        )
        outputs_holder.urbansim_run = SimpleNamespace(
            usim_datastore_h5=Path(stage_env["usim_input_path"])
        )
        outputs_holder.urbansim_postprocess = None

    monkeypatch.setattr(land_use_stage, "run_workflow", _fake_run_workflow)
    monkeypatch.setattr(
        land_use_stage,
        "archive_copy_now",
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
    assert "usim_source_data_dir" not in captured["inputs"]


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


def test_vehicle_ownership_stage_prefers_explicit_beam_skims_artifact(stage_env):
    explicit_final_omx = (
        Path(stage_env["workspace"].get_beam_output_dir()) / "atlas-final-skims.omx"
    )
    _write_file(explicit_final_omx)
    stage_env["coupler"].set(FINAL_SKIMS_OMX, str(explicit_final_omx))
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

    preprocess_calls = [
        call for call in stage_env["scenario"].calls if call.get("model") == "atlas_preprocess"
    ]
    assert preprocess_calls, "Expected an ATLAS preprocess step call."
    assert FINAL_SKIMS_OMX in (preprocess_calls[0].get("optional_input_keys") or [])
    assert FINAL_SKIMS_OMX not in (preprocess_calls[0].get("inputs") or {})


def test_vehicle_ownership_stage_flushes_per_subyear(stage_env, monkeypatch):
    """ATLAS subyear boundaries should enqueue restart artifacts and flush."""
    from pilates.workflows.stages import vehicle_ownership as vo_stage

    state = stage_env["state"]
    settings = stage_env["settings"]
    workspace = stage_env["workspace"]
    scenario = stage_env["scenario"]
    coupler = stage_env["coupler"]

    state.forecast_year = state.year + 4  # 2017, 2019, 2021
    forecast_usim_path = (
        Path(workspace.get_usim_mutable_data_dir())
        / settings.urbansim.output_file_template.format(year=state.forecast_year)
    )
    _write_file(forecast_usim_path)
    coupler.set(USIM_DATASTORE_CURRENT_H5, str(forecast_usim_path))
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])

    atlas_years = [state.year, state.year + 2, state.year + 4]
    for atlas_year in atlas_years:
        year_dir = (
            Path(workspace.get_atlas_mutable_input_dir()) / f"year{atlas_year}"
        )
        year_dir.mkdir(parents=True, exist_ok=True)
        _write_file(year_dir / "vehicles_output.RData")
        if atlas_year > state.start_year:
            _write_file(year_dir / "grave.csv")

    archive_now_calls = []
    flush_calls = []
    monkeypatch.setattr(
        vo_stage,
        "archive_copy_now",
        lambda **kwargs: archive_now_calls.append(kwargs),
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
            for call in archive_now_calls
        )
        assert any(
            call["key"] == f"atlas_rdata_{atlas_year}"
            and str(call["path"]).endswith("vehicles_output.RData")
            for call in archive_now_calls
        )


def test_atlas_sub_years_cover_each_two_year_increment(stage_env):
    from pilates.workflows.stages import vehicle_ownership as vo_stage

    state = stage_env["state"]
    state.forecast_year = state.year + 6

    assert vo_stage._atlas_sub_years(state) == [
        state.year,
        state.year + 2,
        state.year + 4,
        state.year + 6,
    ]


def test_vehicle_ownership_stage_persists_subyear_manifest_run_ids_and_restores(
    stage_env, monkeypatch
):
    """ATLAS should persist/run-id checkpoint per sub-year and skip on replay."""
    from pilates.workflows.stages import vehicle_ownership as vo_stage

    class _RunIdScenario(FakeScenario):
        def __init__(self, coupler):
            super().__init__(coupler)
            self.restored_run_ids = []

        def remember_restored_run_id(
            self, *, model_name, year, iteration, run_id
        ) -> None:
            self.restored_run_ids.append(
                {
                    "model_name": model_name,
                    "year": year,
                    "iteration": iteration,
                    "run_id": run_id,
                }
            )

        def run(self, **kwargs):
            super().run(**kwargs)
            fn = kwargs["fn"]
            model = kwargs.get("model")
            if model is None:
                step_meta = getattr(fn, "__consist_step__", None)
                model = getattr(step_meta, "model", None)
            year = kwargs.get("year")
            iteration = kwargs.get("iteration", 0)
            phase = kwargs.get("phase")
            run_id = f"{model}_y{year}_i{iteration}_p{phase}"
            return SimpleNamespace(
                cache_hit=False,
                run=SimpleNamespace(id=run_id),
            )

    state = stage_env["state"]
    settings = stage_env["settings"]
    workspace = stage_env["workspace"]
    coupler = stage_env["coupler"]
    scenario = _RunIdScenario(coupler)

    state.forecast_year = state.year + 4  # 2017, 2019, 2021
    forecast_usim_path = (
        Path(workspace.get_usim_mutable_data_dir())
        / settings.urbansim.output_file_template.format(year=state.forecast_year)
    )
    _write_file(forecast_usim_path)
    coupler.set(USIM_DATASTORE_CURRENT_H5, str(forecast_usim_path))
    coupler.set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    monkeypatch.setattr(vo_stage, "archive_copy_now", lambda **_kwargs: None)
    monkeypatch.setattr(
        vo_stage,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: None,
    )

    atlas_years = [state.year, state.year + 2, state.year + 4]
    run_id_expectations = {}
    for atlas_year in atlas_years:
        year_dir = Path(workspace.get_atlas_mutable_input_dir()) / f"year{atlas_year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        if atlas_year > state.start_year:
            _write_file(year_dir / "grave.csv")
        run_id_expectations[atlas_year] = {
            "atlas_preprocess": f"atlas_preprocess_y{atlas_year}_i0_ppreprocess",
            "atlas_run": f"atlas_run_y{atlas_year}_i0_prun",
            "atlas_postprocess": f"atlas_postprocess_y{atlas_year}_i0_ppostprocess",
        }

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )
    first_run_call_count = len(scenario.calls)
    assert first_run_call_count == len(atlas_years) * 3

    for atlas_year in atlas_years:
        manifest_path = vo_stage._atlas_subyear_manifest_path(
            workspace=workspace,
            forecast_year=state.forecast_year,
            atlas_year=atlas_year,
        )
        assert manifest_path.exists()
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest_data.keys()) == {
            "atlas_preprocess",
            "atlas_run",
            "atlas_postprocess",
        }
        for step_name, expected_run_id in run_id_expectations[atlas_year].items():
            assert manifest_data[step_name]["run_id"] == expected_run_id

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )
    assert len(scenario.calls) == first_run_call_count

    restored = {
        (
            item["model_name"],
            item["year"],
            item["iteration"],
            item["run_id"],
        )
        for item in scenario.restored_run_ids
    }
    for atlas_year in atlas_years:
        expected = run_id_expectations[atlas_year]
        assert (
            "atlas_preprocess",
            atlas_year,
            0,
            expected["atlas_preprocess"],
        ) in restored
        assert (
            "atlas_run",
            atlas_year,
            0,
            expected["atlas_run"],
        ) in restored
        assert (
            "atlas_postprocess",
            atlas_year,
            0,
            expected["atlas_postprocess"],
        ) in restored


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
    beam_config_path = (
        Path(stage_env["workspace"].get_beam_mutable_data_dir())
        / stage_env["settings"].run.region
        / stage_env["settings"].beam.config
    )
    _write_file(beam_config_path)

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


def test_supply_demand_republishes_existing_zarr_skims_on_compiled_restart(
    stage_env, tmp_path
):
    beam_config_path = (
        Path(stage_env["workspace"].get_beam_mutable_data_dir())
        / stage_env["settings"].run.region
        / stage_env["settings"].beam.config
    )
    _write_file(beam_config_path)

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
    state.compile_asim()

    zarr_path = Path(stage_env["workspace"].get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    stage_env["coupler"]._values.pop(ZARR_SKIMS, None)

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_republish_zarr_{year}_{iteration}.json"

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
    assert not compile_calls, "Did not expect ActivitySim compile when local artifacts already exist."
    assert stage_env["coupler"].get(ZARR_SKIMS) is not None


def test_supply_demand_republishes_compile_artifacts_from_archive_on_compiled_restart(
    stage_env, tmp_path, monkeypatch
):
    beam_config_path = (
        Path(stage_env["workspace"].get_beam_mutable_data_dir())
        / stage_env["settings"].run.region
        / stage_env["settings"].beam.config
    )
    _write_file(beam_config_path)

    stage_env["settings"].activitysim.num_processes = 25
    stage_env["settings"].activitysim.persist_sharrow_cache = True

    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    workspace = stage_env["workspace"]
    local_root = Path(workspace.full_path)
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0
    state.compile_asim()

    zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    if zarr_path.exists() and not zarr_path.is_dir():
        zarr_path.unlink()
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / ".zattrs").write_text("{}")

    numba_cache_dir = Path(workspace.full_path) / "shared_cache" / "numba"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    (numba_cache_dir / "entry.bin").write_text("cache")

    archive_zarr_path = archive_root / zarr_path.relative_to(local_root)
    archive_numba_dir = archive_root / numba_cache_dir.relative_to(local_root)
    archive_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(zarr_path, archive_zarr_path, dirs_exist_ok=True)
    shutil.copytree(numba_cache_dir, archive_numba_dir, dirs_exist_ok=True)

    shutil.rmtree(zarr_path)
    shutil.rmtree(numba_cache_dir)
    stage_env["coupler"]._values.pop(ZARR_SKIMS, None)
    stage_env["coupler"]._values.pop(ASIM_SHARROW_CACHE_DIR, None)

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_archive_republish_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=workspace,
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    compile_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_compile"
    ]
    assert not compile_calls, "Did not expect ActivitySim compile when archive artifacts exist."
    assert zarr_path.exists()
    assert numba_cache_dir.exists()
    assert stage_env["coupler"].get(ZARR_SKIMS) is not None
    assert stage_env["coupler"].get(ASIM_SHARROW_CACHE_DIR) is not None


def test_supply_demand_activitysim_preprocess_prefers_explicit_beam_omx(
    stage_env, tmp_path
):
    beam_config_path = (
        Path(stage_env["workspace"].get_beam_mutable_data_dir())
        / stage_env["settings"].run.region
        / stage_env["settings"].beam.config
    )
    _write_file(beam_config_path)

    explicit_final_omx = (
        Path(stage_env["workspace"].get_beam_output_dir()) / "explicit-final-skims.omx"
    )
    _write_file(explicit_final_omx)
    stage_env["coupler"].set(FINAL_SKIMS_OMX, str(explicit_final_omx))
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

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    preprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_preprocess"
    ]
    assert preprocess_calls, "Expected an ActivitySim preprocess step call."
    binding = preprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert FINAL_SKIMS_OMX in (binding.optional_input_keys or [])
    assert FINAL_SKIMS_OMX not in (binding.inputs or {})


def test_supply_demand_activitysim_preprocess_uses_base_population_source_when_land_use_disabled(
    stage_env, tmp_path
):
    stale_current = Path(stage_env["workspace"].get_usim_mutable_data_dir()) / "stale_2023.h5"
    _write_file(stale_current)

    stage_env["settings"].land_use_enabled = False
    stage_env["state"].enabled_stages.discard(stage_env["state"].Stage.land_use)
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, str(stale_current))
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    usim_inputs = {
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
    }

    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    preprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_preprocess"
    ]
    assert preprocess_calls, "Expected an ActivitySim preprocess step call."
    binding = preprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert (
        USIM_POPULATION_SOURCE_H5 in (binding.input_keys or [])
        or (binding.inputs or {}).get(USIM_POPULATION_SOURCE_H5)
        == stage_env["usim_input_path"]
    )


def test_supply_demand_activitysim_restart_requires_explicit_population_roles(
    stage_env, tmp_path
):
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0
    state.is_restart_run = True
    stage_env["coupler"].pop(USIM_POPULATION_SOURCE_H5, None)
    stage_env["coupler"].pop(USIM_DATASTORE_CURRENT_H5, None)

    with pytest.raises(RuntimeError, match="role split"):
        run_supply_demand_stage(
            scenario=stage_env["scenario"],
            state=state,
            settings=stage_env["settings"],
            workspace=stage_env["workspace"],
            coupler=stage_env["coupler"],
            year=state.forecast_year,
            usim_inputs={
                USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
            },
            build_manifest_path=lambda _workspace, year, iteration: tmp_path
            / f"manifest_{year}_{iteration}.json",
        )


def test_supply_demand_activitysim_restart_with_empty_inputs_still_requires_explicit_population_roles(
    stage_env, tmp_path
):
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0
    state.is_restart_run = True
    stage_env["coupler"].pop(USIM_POPULATION_SOURCE_H5, None)
    stage_env["coupler"].pop(USIM_DATASTORE_CURRENT_H5, None)

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    assert stage_env["coupler"].get(USIM_DATASTORE_CURRENT_H5) is not None
    assert stage_env["coupler"].get(USIM_POPULATION_SOURCE_H5) is not None


def test_supply_demand_stage_accepts_typed_land_use_handoff(stage_env, tmp_path):
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    handoff = LandUseToSupplyDemandHandoff(
        usim_datastore_base_h5=stage_env["usim_input_path"],
        usim_datastore_current_h5=stage_env["usim_input_path"],
        usim_population_source_h5=stage_env["usim_input_path"],
    )

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        handoff=handoff,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    preprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call["model"] == "activitysim_preprocess"
    ]
    assert preprocess_calls, "Expected an ActivitySim preprocess step call."
    binding = preprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert (
        USIM_POPULATION_SOURCE_H5 in (binding.input_keys or [])
        or (binding.inputs or {}).get(USIM_POPULATION_SOURCE_H5)
        == stage_env["usim_input_path"]
    )


def test_supply_demand_activitysim_postprocess_preserves_explicit_usim_base_input(
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

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    postprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_postprocess"
    ]
    assert postprocess_calls, "Expected an ActivitySim postprocess step call."
    binding = postprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert (
        (binding.inputs or {}).get(USIM_DATASTORE_BASE_H5) == stage_env["usim_input_path"]
        or USIM_DATASTORE_BASE_H5 in (binding.input_keys or [])
        or USIM_DATASTORE_BASE_H5 in (binding.optional_input_keys or [])
    )


def test_supply_demand_activitysim_postprocess_uses_local_usim_base_fallback(
    stage_env, tmp_path
):
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].pop(USIM_DATASTORE_BASE_H5, None)
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
    }

    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    postprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_postprocess"
    ]
    assert postprocess_calls, "Expected an ActivitySim postprocess step call."
    binding = postprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert USIM_DATASTORE_BASE_H5 not in (binding.input_keys or [])


@pytest.mark.parametrize(
    ("current_year", "forecast_year"),
    [
        (2019, 2021),
        (2021, 2023),
    ],
)
def test_supply_demand_activitysim_postprocess_binds_population_source_to_forecast_year(
    stage_env,
    tmp_path,
    current_year,
    forecast_year,
):
    usim_dir = Path(stage_env["workspace"].get_usim_mutable_data_dir())
    current_h5 = usim_dir / stage_env["settings"].urbansim.output_file_template.format(
        year=current_year
    )
    forecast_h5 = usim_dir / stage_env["settings"].urbansim.output_file_template.format(
        year=forecast_year
    )
    _write_file(current_h5)
    _write_file(forecast_h5)

    state = stage_env["state"]
    state.current_year = current_year
    state.forecast_year = forecast_year
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    stage_env["coupler"].set(USIM_POPULATION_SOURCE_H5, str(current_h5))
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, str(current_h5))
    usim_inputs = {
        USIM_POPULATION_SOURCE_H5: str(current_h5),
        USIM_DATASTORE_CURRENT_H5: str(current_h5),
    }

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    postprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_postprocess"
    ]
    assert postprocess_calls, "Expected an ActivitySim postprocess step call."
    binding = postprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert (binding.inputs or {}).get(USIM_POPULATION_SOURCE_H5) == str(forecast_h5)
    assert (binding.inputs or {}).get(USIM_DATASTORE_CURRENT_H5) == str(current_h5)


def test_supply_demand_activitysim_postprocess_skips_h5_bindings_without_land_use(
    stage_env, tmp_path
):
    stage_env["settings"].land_use_enabled = False
    stage_env["state"].enabled_stages.discard(stage_env["state"].Stage.land_use)
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_POPULATION_SOURCE_H5, stage_env["usim_output_path"])

    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs={USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"]},
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    postprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_postprocess"
    ]
    assert postprocess_calls, "Expected an ActivitySim postprocess step call."
    binding = postprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    binding_inputs = binding.inputs or {}
    binding_input_keys = set(binding.input_keys or [])
    binding_optional_keys = set(binding.optional_input_keys or [])
    assert USIM_POPULATION_SOURCE_H5 not in binding_inputs
    assert USIM_DATASTORE_CURRENT_H5 not in binding_inputs
    assert USIM_POPULATION_SOURCE_H5 not in binding_input_keys
    assert USIM_DATASTORE_CURRENT_H5 not in binding_input_keys
    assert USIM_POPULATION_SOURCE_H5 not in binding_optional_keys
    assert USIM_DATASTORE_CURRENT_H5 not in binding_optional_keys


def test_supply_demand_activitysim_postprocess_surface_policy_is_authoritative(
    stage_env, tmp_path
):
    from pilates.workflows.surface import build_enabled_workflow_surface
    from dataclasses import replace

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

    surface = build_enabled_workflow_surface(stage_env["settings"], state=state)
    postprocess = surface.step_surface("activitysim_postprocess")
    assert postprocess is not None
    shadow_surface = replace(
        surface,
        step_surfaces={
            **surface.step_surfaces,
            "activitysim_postprocess": replace(
                postprocess,
                required_input_keys=tuple(
                    key
                    for key in postprocess.required_input_keys
                    if key != USIM_DATASTORE_CURRENT_H5
                ),
            ),
        },
    )

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
        surface=shadow_surface,
    )

    postprocess_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_postprocess"
    ]
    assert postprocess_calls, "Expected an ActivitySim postprocess step call."
    binding = postprocess_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert USIM_DATASTORE_CURRENT_H5 not in (binding.input_keys or [])


def test_supply_demand_stage_forwards_surface_to_traffic_assignment(
    stage_env, tmp_path, monkeypatch
):
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    captured = {}

    monkeypatch.setattr(
        supply_demand_stage,
        "_run_activity_demand_phase",
        lambda **_kwargs: SimpleNamespace(activity_demand_outputs={}),
    )

    def _fake_traffic_assignment_phase(**kwargs):
        captured["surface"] = kwargs.get("surface")
        return SimpleNamespace(previous_beam_outputs=None)

    monkeypatch.setattr(
        supply_demand_stage,
        "_run_traffic_assignment_phase",
        _fake_traffic_assignment_phase,
    )

    surface = object()
    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
        surface=surface,
    )

    assert captured["surface"] is surface


def test_supply_demand_activitysim_run_keeps_numba_cache_optional(
    stage_env, tmp_path
):
    stage_env["settings"].activitysim.num_processes = 25
    stage_env["settings"].activitysim.persist_sharrow_cache = True
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

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: tmp_path
        / f"manifest_{year}_{iteration}.json",
    )

    run_calls = [
        call
        for call in stage_env["scenario"].calls
        if call.get("model") == "activitysim_run"
    ]
    assert run_calls, "Expected an ActivitySim run step call."
    binding = run_calls[0].get("binding")
    assert isinstance(binding, BindingResult)
    assert ASIM_SHARROW_CACHE_DIR not in (binding.input_keys or [])
    assert ASIM_SHARROW_CACHE_DIR in (binding.optional_input_keys or [])


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

    archive_now_calls = []
    flush_calls = []
    monkeypatch.setattr(
        sd_stage,
        "archive_copy_now",
        lambda **kwargs: archive_now_calls.append(kwargs),
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
        for call in archive_now_calls
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
    coupler.set(BEAM_PLANS_IN, str(default_plans))
    coupler.set(BEAM_HOUSEHOLDS_IN, str(default_households))
    coupler.set(BEAM_PERSONS_IN, str(default_persons))

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
        call for call in scenario.calls if call.get("model") == "beam_preprocess"
    ]
    assert beam_preprocess_calls, "Expected a BEAM preprocess step call."
    beam_preprocess_binding = beam_preprocess_calls[0]["binding"]
    assert isinstance(beam_preprocess_binding, BindingResult)
    beam_preprocess_inputs = beam_preprocess_binding.inputs or {}
    assert beam_preprocess_inputs[BEAM_PLANS_IN] == str(default_plans)
    assert beam_preprocess_inputs[BEAM_HOUSEHOLDS_IN] == str(default_households)
    assert beam_preprocess_inputs[BEAM_PERSONS_IN] == str(default_persons)

    beam_run_calls = [call for call in scenario.calls if call.get("model") == "beam_run"]
    assert beam_run_calls, "Expected BEAM run step call."
    run_input_keys = beam_run_calls[0].get("input_keys") or []
    assert BEAM_PLANS_IN in run_input_keys
    assert BEAM_HOUSEHOLDS_IN in run_input_keys
    assert BEAM_PERSONS_IN in run_input_keys


def test_supply_demand_stage_beam_only_clamps_outer_iterations(
    stage_env, tmp_path, caplog
):
    settings = stage_env["settings"]
    state = stage_env["state"]
    workspace = stage_env["workspace"]
    coupler = stage_env["coupler"]
    scenario = stage_env["scenario"]

    settings.run.models.activity_demand = None
    settings.activity_demand_enabled = False
    settings.run.supply_demand_iters = 3
    state._settings["activity_demand_enabled"] = False
    state._settings["supply_demand_iters"] = 3
    state.enabled_stages.discard(state.Stage.activity_demand)
    state.loop_substages = [state.Stage.traffic_assignment]

    scenario_dir = (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.scenario_folder
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plans_path = scenario_dir / "plans.parquet"
    households_path = scenario_dir / "households.parquet"
    persons_path = scenario_dir / "persons.parquet"
    _write_file(plans_path)
    _write_file(households_path)
    _write_file(persons_path)
    coupler.set(BEAM_PLANS_IN, str(plans_path))
    coupler.set(BEAM_HOUSEHOLDS_IN, str(households_path))
    coupler.set(BEAM_PERSONS_IN, str(persons_path))

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    caplog.set_level(logging.WARNING)

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: (
            tmp_path / f"manifest_beam_only_clamped_{year}_{iteration}.json"
        ),
    )

    beam_preprocess_calls = [
        call for call in scenario.calls if call.get("model") == "beam_preprocess"
    ]
    assert len(beam_preprocess_calls) == 1
    beam_preprocess_binding = beam_preprocess_calls[0]["binding"]
    assert isinstance(beam_preprocess_binding, BindingResult)
    assert beam_preprocess_binding.inputs[BEAM_PLANS_IN] == str(plans_path)
    assert beam_preprocess_binding.inputs[BEAM_HOUSEHOLDS_IN] == str(
        households_path
    )
    assert beam_preprocess_binding.inputs[BEAM_PERSONS_IN] == str(persons_path)
    assert "Clamping outer supply-demand iterations to 1" in caplog.text


def test_restore_activity_demand_outputs_for_resume_reuses_coupler_artifacts(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    zarr = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(zarr)

    coupler.set("beam_plans_asim_out", str(beam_plans))
    coupler.set("households_asim_out", str(households))
    coupler.set("persons_asim_out", str(persons))
    coupler.set(ZARR_SKIMS, str(zarr))

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=stage_env["settings"],
    )

    assert restored == {
        "beam_plans_asim_out": str(beam_plans),
        "households_asim_out": str(households),
        "persons_asim_out": str(persons),
        ZARR_SKIMS: str(zarr),
    }
    assert holder.activitysim_postprocess is not None
    assert holder.activitysim_postprocess.asim_output_dir is None
    assert holder.activitysim_postprocess.processed_outputs == {
        "beam_plans_asim_out": beam_plans,
        "households_asim_out": households,
        "persons_asim_out": persons,
        ZARR_SKIMS: zarr,
    }


def test_restore_activity_demand_outputs_for_resume_republishes_zarr_skims(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs" / "skims.zarr"
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)

    coupler.set("beam_plans_asim_out", str(beam_plans))
    coupler.set("households_asim_out", str(households))
    coupler.set("persons_asim_out", str(persons))

    holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=None,
        processed_outputs={
            "beam_plans_asim_out": beam_plans,
            "households_asim_out": households,
            "persons_asim_out": persons,
            ZARR_SKIMS: archived_zarr,
        },
    )

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=stage_env["settings"],
    )

    assert restored is not None
    assert coupler.get(ZARR_SKIMS) == str(archived_zarr)


def test_restore_activity_demand_outputs_for_resume_promotes_archived_zarr_skims(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)

    holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=None,
        processed_outputs={
            "beam_plans_asim_out": beam_plans,
            "households_asim_out": households,
            "persons_asim_out": persons,
            "asim_input_skims_zarr_archived": archived_zarr,
        },
    )

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=stage_env["settings"],
    )

    assert restored is not None
    assert Path(restored[ZARR_SKIMS]) == archived_zarr
    assert coupler.get(ZARR_SKIMS) == str(archived_zarr)


def test_restore_activity_demand_outputs_for_resume_manifest_restore_rehydrates_coupler(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)

    manifest_path = tmp_path / "activitysim_postprocess_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=usim_datastore,
                            asim_output_dir=Path(workspace.get_asim_output_dir()),
                            processed_outputs={
                                "beam_plans_asim_out": beam_plans,
                                "households_asim_out": households,
                                "persons_asim_out": persons,
                                "asim_input_skims_zarr_archived": archived_zarr,
                            },
                        )
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert Path(restored[ZARR_SKIMS]) == archived_zarr
    assert coupler.get("beam_plans_asim_out") == str(beam_plans)
    assert coupler.get("households_asim_out") == str(households)
    assert coupler.get("persons_asim_out") == str(persons)
    assert artifact_to_path(coupler.get(ZARR_SKIMS), workspace) == str(archived_zarr)


def test_restore_activity_demand_outputs_for_resume_manifest_workspace_uris_rehydrate_coupler(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)

    workspace_root = Path(workspace.full_path)

    def _workspace_uri(path: Path) -> str:
        return f"workspace://{path.relative_to(workspace_root).as_posix()}"

    manifest_path = tmp_path / "activitysim_postprocess_manifest_workspace_uris.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": {
                        "usim_datastore_h5": _workspace_uri(usim_datastore),
                        "asim_output_dir": _workspace_uri(
                            Path(workspace.get_asim_output_dir())
                        ),
                        "processed_outputs": {
                            "beam_plans_asim_out": _workspace_uri(beam_plans),
                            "households_asim_out": _workspace_uri(households),
                            "persons_asim_out": _workspace_uri(persons),
                            "asim_input_skims_zarr_archived": _workspace_uri(
                                archived_zarr
                            ),
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert Path(restored["beam_plans_asim_out"]) == beam_plans
    assert Path(restored["households_asim_out"]) == households
    assert Path(restored["persons_asim_out"]) == persons
    assert Path(restored[ZARR_SKIMS]) == archived_zarr
    assert coupler.get("beam_plans_asim_out") == str(beam_plans)
    assert coupler.get("households_asim_out") == str(households)
    assert coupler.get("persons_asim_out") == str(persons)
    assert artifact_to_path(coupler.get(ZARR_SKIMS), workspace) == str(archived_zarr)


def test_restore_activity_demand_outputs_for_resume_seeds_activitysim_run_parent_link(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()
    remembered = []

    class _Scenario:
        def remember_restored_run_id(self, **kwargs):
            remembered.append(kwargs)

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)

    manifest_path = tmp_path / "activitysim_postprocess_manifest_with_run_id.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_run": {
                    "run_id": "activitysim_run_y2018_i0_prun",
                },
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=usim_datastore,
                            asim_output_dir=Path(workspace.get_asim_output_dir()),
                            processed_outputs={
                                "beam_plans_asim_out": beam_plans,
                                "households_asim_out": households,
                                "persons_asim_out": persons,
                                "asim_input_skims_zarr_archived": archived_zarr,
                            },
                        )
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    restored = _restore_activity_demand_outputs_for_resume(
        scenario=_Scenario(),
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert remembered == [
        {
            "model_name": "activitysim_run",
            "year": state.forecast_year,
            "iteration": state.current_inner_iter,
            "run_id": "activitysim_run_y2018_i0_prun",
        }
    ]


def test_restore_activity_demand_outputs_for_resume_seeds_parent_link_from_manifest_run_epoch(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()
    remembered = []

    class _Scenario:
        def remember_restored_run_id(self, **kwargs):
            remembered.append(kwargs)

    # Simulate a year-boundary resume where the current state has already moved on,
    # but the manifest still restores the just-finished ActivitySim run.
    state.current_year = 2018
    state.forecast_year = 2019
    state.current_inner_iter = 1

    iter_dir = Path(workspace.get_asim_output_dir()) / "year-2017-iteration-1"
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = (
        Path(workspace.get_asim_output_dir())
        / "inputs-year-2017-iteration-1"
        / "skims.zarr"
    )
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}2018.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)

    manifest_path = tmp_path / "activitysim_postprocess_manifest_with_epoch_run_id.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_run": {
                    "run_id": (
                        "archive_after_year_complete__step_func__y2018__i1__phase_run_fa2556e2"
                    ),
                },
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=usim_datastore,
                            asim_output_dir=Path(workspace.get_asim_output_dir()),
                            processed_outputs={
                                "beam_plans_asim_out": beam_plans,
                                "households_asim_out": households,
                                "persons_asim_out": persons,
                                "asim_input_skims_zarr_archived": archived_zarr,
                            },
                        )
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    restored = _restore_activity_demand_outputs_for_resume(
        scenario=_Scenario(),
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert remembered == [
        {
            "model_name": "activitysim_run",
            "year": 2018,
            "iteration": 1,
            "run_id": "archive_after_year_complete__step_func__y2018__i1__phase_run_fa2556e2",
        }
    ]


def test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    remembered = []

    class _Scenario:
        def remember_restored_run_id(self, **kwargs):
            remembered.append(kwargs)

    state.file_loc = str(tmp_path / "archive" / "state.yaml")
    archive_workflow_dir = tmp_path / "archive" / "run" / ".workflow"
    archive_workflow_dir.mkdir(parents=True, exist_ok=True)
    Path(state.file_loc).parent.mkdir(parents=True, exist_ok=True)
    Path(state.file_loc).write_text("year: 2018\n", encoding="utf-8")
    (archive_workflow_dir / "year_2017_iteration_1.yaml").write_text(
        yaml.safe_dump(
            {
                "activitysim_run": {
                    "run_id": "archive_after_year_complete__step_func__y2018__i1__phase_run_fa2556e2",
                },
                "beam_run": {
                    "run_id": "archive_after_year_complete__step_func__y2017__i1__phase_run_4a11b8f0",
                },
            }
        ),
        encoding="utf-8",
    )

    seed_supply_demand_parent_run_ids_for_resume(
        scenario=_Scenario(),
        workspace=workspace,
        state=state,
    )

    assert remembered == [
        {
            "model_name": "activitysim_run",
            "year": 2018,
            "iteration": 1,
            "run_id": "archive_after_year_complete__step_func__y2018__i1__phase_run_fa2556e2",
        },
        {
            "model_name": "beam_run",
            "year": 2017,
            "iteration": 1,
            "run_id": "archive_after_year_complete__step_func__y2017__i1__phase_run_4a11b8f0",
        },
    ]


def test_restore_activity_demand_outputs_for_resume_finds_archive_side_manifest(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)

    local_manifest_path = (
        Path(workspace.full_path) / ".workflow" / "year_2017_iteration_0.yaml"
    )
    archive_root = tmp_path / "archive_run"
    archive_manifest_path = archive_root / "run" / ".workflow" / "year_2017_iteration_0.yaml"
    archive_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=usim_datastore,
                            asim_output_dir=Path(workspace.get_asim_output_dir()),
                            processed_outputs={
                                "beam_plans_asim_out": beam_plans,
                                "households_asim_out": households,
                                "persons_asim_out": persons,
                                "asim_input_skims_zarr_archived": archived_zarr,
                            },
                        )
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    state.file_loc = str(archive_root / "state.yaml")
    _write_file(Path(state.file_loc))

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=local_manifest_path,
    )

    assert restored is not None
    assert Path(restored["beam_plans_asim_out"]) == beam_plans
    assert Path(restored["households_asim_out"]) == households
    assert Path(restored["persons_asim_out"]) == persons
    assert Path(restored[ZARR_SKIMS]) == archived_zarr


def test_restore_activity_demand_outputs_for_resume_manifest_reuses_coupler_zarr(
    stage_env,
    tmp_path,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]
    holder = StepOutputsHolder()

    iter_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    beam_plans = iter_dir / "beam_plans.parquet"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    archived_zarr = iter_dir / "inputs-year-2017-iteration-0" / "skims.zarr"
    usim_datastore = (
        Path(workspace.get_usim_mutable_data_dir())
        / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    _write_file(beam_plans)
    _write_file(households)
    _write_file(persons)
    _write_file(archived_zarr)
    _write_file(usim_datastore)
    coupler.set(ZARR_SKIMS, str(archived_zarr))

    manifest_path = tmp_path / "activitysim_postprocess_manifest_without_zarr.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=usim_datastore,
                            asim_output_dir=Path(workspace.get_asim_output_dir()),
                            processed_outputs={
                                "beam_plans_asim_out": beam_plans,
                                "households_asim_out": households,
                                "persons_asim_out": persons,
                            },
                        )
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    restored = _restore_activity_demand_outputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        outputs_holder=holder,
        state=state,
        settings=settings,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert Path(restored["beam_plans_asim_out"]) == beam_plans
    assert Path(restored["households_asim_out"]) == households
    assert Path(restored["persons_asim_out"]) == persons
    assert Path(restored[ZARR_SKIMS]) == archived_zarr
    assert artifact_to_path(coupler.get(ZARR_SKIMS), workspace) == str(archived_zarr)


def test_restore_supply_demand_usim_inputs_for_resume_republishes_year_scoped_roles(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    base_h5 = usim_dir / get_usim_datastore_fname(settings, io="input")
    current_h5 = usim_dir / get_usim_datastore_fname(
        settings,
        io="output",
        year=state.forecast_year,
    )
    _write_file(base_h5)
    _write_file(current_h5)

    restored = _restore_supply_demand_usim_inputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        state=state,
        settings=settings,
    )

    assert restored[USIM_DATASTORE_BASE_H5] == str(base_h5)
    assert restored[USIM_DATASTORE_CURRENT_H5] == str(current_h5)
    assert restored[USIM_FORECAST_OUTPUT] == str(current_h5)
    assert restored[USIM_POPULATION_SOURCE_H5] == str(current_h5)
    assert coupler.get(USIM_DATASTORE_CURRENT_H5) == str(current_h5)
    assert coupler.get(USIM_POPULATION_SOURCE_H5) == str(current_h5)


def test_restore_supply_demand_usim_inputs_for_resume_falls_back_to_base_when_output_missing(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    base_h5 = usim_dir / get_usim_datastore_fname(settings, io="input")
    current_h5 = usim_dir / get_usim_datastore_fname(
        settings,
        io="output",
        year=state.forecast_year,
    )
    _write_file(base_h5)
    if current_h5.exists():
        current_h5.unlink()

    restored = _restore_supply_demand_usim_inputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        state=state,
        settings=settings,
    )

    assert restored[USIM_DATASTORE_BASE_H5] == str(base_h5)
    assert restored[USIM_DATASTORE_CURRENT_H5] == str(base_h5)
    assert restored[USIM_POPULATION_SOURCE_H5] == str(base_h5)
    assert USIM_FORECAST_OUTPUT not in restored
    assert coupler.get(USIM_DATASTORE_CURRENT_H5) == str(base_h5)


def test_restore_supply_demand_usim_inputs_for_resume_promotes_population_source_to_current(
    stage_env,
):
    workspace = stage_env["workspace"]
    state = stage_env["state"]
    coupler = stage_env["coupler"]
    settings = stage_env["settings"]

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    base_h5 = usim_dir / get_usim_datastore_fname(settings, io="input")
    _write_file(base_h5)
    coupler.set(USIM_POPULATION_SOURCE_H5, str(base_h5))

    restored = _restore_supply_demand_usim_inputs_for_resume(
        coupler=coupler,
        workspace=workspace,
        state=state,
        settings=settings,
    )

    assert restored[USIM_DATASTORE_CURRENT_H5] == str(base_h5)
    assert restored[USIM_POPULATION_SOURCE_H5] == str(base_h5)
    assert coupler.get(USIM_DATASTORE_CURRENT_H5) == str(base_h5)


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
    usim_next = Path(workspace.get_usim_mutable_data_dir()) / "activitysim_next.h5"
    _write_file(usim_next)
    outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_next,
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
    assert LINKSTATS_WARMSTART not in beam_run_calls[0].get("optional_input_keys", [])


def test_traffic_assignment_publishes_present_linkstats_warmstart(
    stage_env, monkeypatch, tmp_path
):
    """
    Regression: if BEAM preprocess returns LINKSTATS_WARMSTART, the workflow
    must publish it for the subsequent beam_run step.
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
    warmstart_path = tmp_path / "init.linkstats.csv.gz"
    _write_file(warmstart_path)
    previous_beam_outputs = {
        "linkstats_parquet_2018_0": str(warmstart_path)
    }

    class _BeamPreprocessorWithWarmstart:
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
                    LINKSTATS_WARMSTART: warmstart_path,
                },
            )

    original_get_preprocessor = ModelFactory.get_preprocessor

    def _patched_get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamPreprocessorWithWarmstart()
        return original_get_preprocessor(self, model_name, state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _patched_get_preprocessor)

    outputs_holder = StepOutputsHolder()
    usim_next = Path(workspace.get_usim_mutable_data_dir()) / "activitysim_next.h5"
    _write_file(usim_next)
    outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_next,
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
    assert LINKSTATS_WARMSTART in beam_run_calls[0].get("optional_input_keys", [])


def test_traffic_assignment_prefers_coupler_warmstart_artifact(
    stage_env, tmp_path
):
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
    plans_path = scenario_dir / "plans.parquet"
    households_path = scenario_dir / "households.parquet"
    persons_path = scenario_dir / "persons.parquet"
    _write_file(plans_path)
    _write_file(households_path)
    _write_file(persons_path)
    coupler.set(BEAM_PLANS_IN, str(plans_path))
    coupler.set(BEAM_HOUSEHOLDS_IN, str(households_path))
    coupler.set(BEAM_PERSONS_IN, str(persons_path))

    warmstart_path = tmp_path / "init.linkstats.csv.gz"
    _write_file(warmstart_path)
    coupler.set(LINKSTATS_WARMSTART, str(warmstart_path))

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
        USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
    }

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=lambda _workspace, year, iteration: (
            tmp_path / f"manifest_coupler_warmstart_{year}_{iteration}.json"
        ),
    )

    beam_preprocess_calls = [
        call for call in scenario.calls if call.get("model") == "beam_preprocess"
    ]
    assert beam_preprocess_calls, "Expected a BEAM preprocess step call."
    assert (
        beam_preprocess_calls[0]["binding"].inputs[LINKSTATS_WARMSTART]
        == str(warmstart_path)
    )


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
    usim_next = Path(workspace.get_usim_mutable_data_dir()) / "activitysim_next.h5"
    _write_file(usim_next)
    outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_next,
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
    assert (
        f"events_parquet_{state.forecast_year}_0_sub0"
        in postprocess_call["binding"].inputs
    )
    assert (
        f"raw_od_skims_zarr_{state.forecast_year}_0_sub0"
        in postprocess_call["binding"].inputs
    )
    assert f"events_parquet_{state.forecast_year}_0_sub0" not in (
        postprocess_call["binding"].input_keys or []
    )
    assert f"raw_od_skims_zarr_{state.forecast_year}_0_sub0" not in (
        postprocess_call["binding"].input_keys or []
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
