"""
Stubbed workflow matrix tests built on the golden Consist harness.

These tests complement the broad golden workflow by exercising smaller run
shapes that have broken in production:

1. BEAM-only supply-demand with no ActivitySim zarr handoff.
2. ActivitySim+BEAM supply-demand where OMX archive output is legitimately absent.
3. Land use + ATLAS stage wiring with stage-local output ownership checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    normalize_asim_output_key,
)
from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.beam.runner import BeamRunner
from pilates.generic.records import FileRecord, RecordStore
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    USIM_DATASTORE_H5,
)
from pilates.workflows.stages.land_use import run_land_use_stage as _run_land_use_stage
from pilates.workflows.stages.supply_demand import (
    run_supply_demand_stage as _run_supply_demand_stage,
)
from pilates.workflows.stages.vehicle_ownership import (
    run_vehicle_ownership_stage as _run_vehicle_ownership_stage,
)
from pilates.workflows.steps import StepOutputsHolder
from tests.test_golden_stub_workflow import (
    DummyPostprocessor,
    DummyPreprocessor,
    DummyRunner,
    _write_file,
    _write_parquet,
)
from workflow_state import WorkflowState


def run_land_use_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_land_use_stage(context=context, **kwargs)


def run_vehicle_ownership_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_vehicle_ownership_stage(context=context, **kwargs)


def run_supply_demand_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_supply_demand_stage(context=context, **kwargs)


def _reconfigure_models(
    settings,
    *,
    tmp_path: Path,
    state_file_name: str,
    land_use: bool,
    vehicle_ownership: bool,
    activity_demand: bool,
    traffic_assignment: bool,
) -> WorkflowState:
    settings.run.models.land_use = "urbansim" if land_use else None
    settings.run.models.vehicle_ownership = "atlas" if vehicle_ownership else None
    settings.run.models.activity_demand = "activitysim" if activity_demand else None
    settings.run.models.travel = "beam" if traffic_assignment else None

    settings.land_use_enabled = land_use
    settings.vehicle_ownership_model_enabled = vehicle_ownership
    settings.activity_demand_enabled = activity_demand
    settings.traffic_assignment_enabled = traffic_assignment

    runtime = getattr(settings, "runtime", None)
    flags = getattr(runtime, "flags", None)
    if flags is not None:
        flags.land_use_enabled = land_use
        flags.vehicle_ownership_model_enabled = vehicle_ownership
        flags.activity_demand_enabled = activity_demand
        flags.traffic_assignment_enabled = traffic_assignment

    state_file = tmp_path / state_file_name
    settings.state_file_loc = str(state_file)
    runtime_options = getattr(runtime, "options", None)
    if runtime_options is not None:
        runtime_options.state_file_loc = str(state_file)

    return WorkflowState.from_settings(settings)


def _mark_initialized(env: dict, state: WorkflowState, *, marker_name: str) -> None:
    workspace = env["workspace"]
    scenario = env["scenario"]
    source_path = Path(env["usim_input_path"])
    marker = Path(workspace.full_path) / marker_name
    marker_key = marker_name.lstrip(".").replace(".", "_")
    with scenario.trace(
        "initialization",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init", "stub-matrix"],
    ):
        cr.log_input(source_path, key=f"{marker_key}_source")
        _write_file(marker, "initialized")
        cr.log_output(marker, key=f"{marker_key}_marker")
        state.set_data_initialized(True)


def _manifest_builder(root: Path, prefix: str):
    def _build(_workspace, year: int, iteration: int) -> Path:
        manifest_dir = root / prefix
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"{year}_{iteration}.yaml"

    return _build


def _scenario_run(env: dict):
    tracker = env["tracker"]
    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs
    return runs[0]


def _steps_by_model(env: dict) -> dict[str, dict]:
    scenario_run = _scenario_run(env)
    return {step["model"]: step for step in scenario_run.meta["steps"]}


def _scenario_output_keys(env: dict) -> set[str]:
    tracker = env["tracker"]
    scenario_run = _scenario_run(env)
    outputs = tracker.get_artifacts_for_run(scenario_run.id).outputs
    return {
        getattr(artifact, "key", None)
        for artifact in outputs or []
        if getattr(artifact, "key", None)
    }


def test_stubbed_beam_only_supply_demand_runs_without_activitysim_zarr_inputs(
    golden_stub_env,
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="beam_only_state.yaml",
        land_use=False,
        vehicle_ownership=False,
        activity_demand=False,
        traffic_assignment=True,
    )
    _mark_initialized(env, state, marker_name=".beam_only_initialized.txt")

    zarr_path = Path(env["zarr_path"])
    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_scenario_dir = (
        beam_input_dir / settings.run.region / settings.beam.scenario_folder
    )
    beam_output_dir = Path(workspace.get_beam_output_dir())
    beam_plans = beam_input_dir / "plans.csv"
    beam_households = beam_input_dir / "households.csv"
    beam_persons = beam_input_dir / "persons.csv"
    for path in (beam_plans, beam_households, beam_persons):
        _write_file(path, "stub")
    for path in (
        beam_scenario_dir / "plans.csv",
        beam_scenario_dir / "households.csv",
        beam_scenario_dir / "persons.csv",
    ):
        _write_file(path, "stub")

    def _record_builder(model_name, phase, *, workspace=None, **_kwargs):
        if phase == "preprocess" and model_name == "beam":
            return BeamPreprocessOutputs(
                beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                prepared_inputs={
                    BEAM_PLANS_IN: beam_plans,
                    BEAM_HOUSEHOLDS_IN: beam_households,
                    BEAM_PERSONS_IN: beam_persons,
                },
            )
        if phase == "postprocess" and model_name == "beam":
            split_event = beam_output_dir / "split.PathTraversal.parquet"
            split_links = beam_output_dir / "split.PathTraversal.links.parquet"
            _write_file(split_event, "events")
            _write_file(split_links, "links")
            return BeamPostprocessOutputs(
                split_events={
                    f"events_parquet_{state.forecast_year}_{state.iteration}_type_PathTraversal": split_event
                },
                split_event_links={
                    f"path_traversal_links_{state.forecast_year}_{state.iteration}": split_links
                },
            )
        return RecordStore()

    def _get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPreprocessor(model_name, _record_builder, state=state)

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name != "beam":
            raise AssertionError(f"Unexpected runner request: {model_name}")
        return BeamRunner(model_name, state)

    def _get_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPostprocessor(model_name, _record_builder, state=state)

    def _fake_beam_run(self, input_store, _workspace):
        assert "zarr_skims" not in input_store.to_mapping()
        events = beam_output_dir / "events.parquet"
        linkstats = beam_output_dir / "linkstats.csv.gz"
        plans_out = beam_output_dir / "plans.parquet"
        for path in (events, linkstats, plans_out):
            _write_file(path, "stub")
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(events),
                    short_name=f"events_parquet_{state.forecast_year}_{state.iteration}_sub0",
                ),
                FileRecord(file_path=str(linkstats), short_name=LINKSTATS),
                FileRecord(file_path=str(plans_out), short_name=BEAM_PLANS_OUT),
            ]
        )

    from pilates.generic.model_factory import ModelFactory

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _get_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _get_postprocessor)
    monkeypatch.setattr(BeamRunner, "_run", _fake_beam_run)

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=_manifest_builder(tmp_path, "beam_only_manifests"),
    )

    steps = _steps_by_model(env)
    assert "beam_preprocess" in steps
    assert "beam_run" in steps
    assert "beam_postprocess" in steps
    assert "activitysim_postprocess" not in steps
    beam_postprocess_outputs = set(
        (steps["beam_postprocess"].get("outputs") or {}).values()
    )
    assert "zarr_skims" not in beam_postprocess_outputs
    assert (
        f"events_parquet_{state.forecast_year}_{state.iteration}_type_PathTraversal"
        in beam_postprocess_outputs
    )


def test_stubbed_activitysim_beam_supply_demand_allows_missing_optional_omx_archive(
    golden_stub_env,
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="asim_beam_state.yaml",
        land_use=False,
        vehicle_ownership=False,
        activity_demand=True,
        traffic_assignment=True,
    )
    _mark_initialized(env, state, marker_name=".asim_beam_initialized.txt")

    asim_input_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_output_dir = Path(workspace.get_asim_output_dir())
    beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_output_dir = Path(workspace.get_beam_output_dir())
    sharrow_cache_dir = Path(env["sharrow_cache_dir"])
    zarr_path = Path(env["zarr_path"])

    temp_names = (
        "accessibility",
        "disaggregate_accessibility",
        "joint_tour_participants",
        "land_use",
        "non_mandatory_tour_destination_accessibility",
        "households",
        "persons",
        "tours",
        "trips",
        "beam_plans",
    )
    raw_outputs = {}
    processed_outputs = {}
    for name in temp_names:
        temp_path = asim_output_dir / "final_pipeline" / name / "final.parquet"
        _write_parquet(temp_path, pd.DataFrame({"id": [1]}))
        raw_outputs[f"{name}_asim_out_temp"] = temp_path
        processed_outputs[normalize_asim_output_key(name)] = temp_path

    processed_outputs["asim_input_households_csv_archived"] = (
        asim_input_dir / "households.csv"
    )
    processed_outputs["asim_input_persons_csv_archived"] = (
        asim_input_dir / "persons.csv"
    )
    processed_outputs["asim_input_land_use_csv_archived"] = (
        asim_input_dir / "land_use.csv"
    )
    processed_outputs["asim_input_skims_zarr_archived"] = zarr_path

    beam_plans = beam_input_dir / "plans.csv"
    beam_households = beam_input_dir / "households.csv"
    beam_persons = beam_input_dir / "persons.csv"
    for path in (beam_plans, beam_households, beam_persons):
        _write_file(path, "stub")

    def _record_builder(model_name, phase, *, workspace=None, **_kwargs):
        if phase == "preprocess":
            if model_name == "activitysim":
                return ActivitySimPreprocessOutputs(
                    mutable_data_dir=Path(workspace.get_asim_mutable_data_dir()),
                    land_use_table=asim_input_dir / "land_use.csv",
                    households_table=asim_input_dir / "households.csv",
                    persons_table=asim_input_dir / "persons.csv",
                    omx_skims=asim_input_dir / "skims.omx",
                )
            if model_name == "beam":
                return BeamPreprocessOutputs(
                    beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                    prepared_inputs={
                        BEAM_PLANS_IN: beam_plans,
                        BEAM_HOUSEHOLDS_IN: beam_households,
                        BEAM_PERSONS_IN: beam_persons,
                    },
                )
        if phase == "run":
            if model_name == "activitysim_compile":
                return ActivitySimCompileOutputs(
                    zarr_skims=zarr_path,
                    sharrow_cache_dir=sharrow_cache_dir,
                )
            if model_name == "activitysim":
                return ActivitySimRunOutputs(
                    output_dir=Path(workspace.get_asim_output_dir()),
                    raw_outputs=raw_outputs,
                )
            if model_name == "beam":
                linkstats = beam_output_dir / "linkstats.csv.gz"
                plans_out = beam_output_dir / "plans.parquet"
                _write_file(linkstats, "stub")
                _write_file(plans_out, "stub")
                return BeamRunOutputs(
                    beam_output_dir=Path(workspace.get_beam_output_dir()),
                    raw_outputs={
                        LINKSTATS: linkstats,
                        BEAM_PLANS_OUT: plans_out,
                    },
                )
        if phase == "postprocess":
            if model_name == "activitysim":
                return ActivitySimPostprocessOutputs(
                    usim_datastore_h5=None,
                    asim_output_dir=Path(workspace.get_asim_output_dir()),
                    processed_outputs=processed_outputs,
                )
            if model_name == "beam":
                return BeamPostprocessOutputs(
                    zarr_skims=zarr_path,
                )
        return RecordStore()

    def _get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPreprocessor(model_name, _record_builder, state=state)

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        return DummyRunner(model_name, _record_builder, state=state)

    def _get_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPostprocessor(model_name, _record_builder, state=state)

    from pilates.generic.model_factory import ModelFactory

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _get_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _get_postprocessor)

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=_manifest_builder(tmp_path, "asim_beam_manifests"),
    )

    steps = _steps_by_model(env)
    assert "activitysim_postprocess" in steps
    assert "beam_run" in steps
    postprocess_outputs = set(
        (steps["activitysim_postprocess"].get("outputs") or {}).values()
    )
    assert "asim_input_skims_zarr_archived" in postprocess_outputs
    assert "asim_input_skims_omx_archived" not in postprocess_outputs


def test_stubbed_land_use_atlas_stage_keeps_usim_datastore_out_of_atlas_run_outputs(
    golden_stub_env,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="atlas_stage_state.yaml",
        land_use=True,
        vehicle_ownership=True,
        activity_demand=False,
        traffic_assignment=False,
    )
    _mark_initialized(env, state, marker_name=".atlas_stage_initialized.txt")

    outputs_holder_year = StepOutputsHolder()
    usim_inputs = run_land_use_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        outputs_holder_year=outputs_holder_year,
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

    assert USIM_DATASTORE_H5 in usim_inputs
    assert coupler.get(USIM_DATASTORE_H5) is not None

    steps = _steps_by_model(env)
    assert "atlas_run" in steps
    atlas_run_outputs = set((steps["atlas_run"].get("outputs") or {}).values())
    assert "atlas_output_dir" in atlas_run_outputs
    assert USIM_DATASTORE_H5 not in atlas_run_outputs
