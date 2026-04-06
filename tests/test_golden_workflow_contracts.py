"""
Focused golden-workflow contract tests.

These tests are smaller companions to ``test_golden_stub_workflow``. They keep
the end-to-end narrative value of the golden harness while locking in a few
stable surfaces that are easy to regress during orchestration refactors:

1. which step publishes raw vs finalized outputs
2. which artifacts are promoted to the scenario surface
3. which temporary artifacts must remain step-local
"""

from __future__ import annotations

from pathlib import Path

from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    BEAM_PLANS_OUT,
    LINKSTATS,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from pilates.workflows.steps import StepOutputsHolder
from tests.test_golden_stub_workflow import (
    EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS,
    EXPECTED_STAGE_MODELS,
    _artifact_map,
    _write_file,
    golden_stub_env,
)


def _initialize(env) -> None:
    state = env["state"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    source_path = Path(env["usim_input_path"])
    init_marker = Path(workspace.full_path) / ".golden_surface_init_marker.txt"
    with scenario.trace(
        "initialization",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init", "surface-contract"],
    ):
        cr.log_input(source_path, key="golden_surface_init_source")
        _write_file(init_marker, "initialized")
        cr.log_output(init_marker, key="golden_surface_init_marker")
        state.set_data_initialized(True)


def test_golden_workflow_preserves_current_stage_surfaces_on_scenario_outputs(
    golden_stub_env, tmp_path
):
    settings = golden_stub_env["settings"]
    workspace = golden_stub_env["workspace"]
    state = golden_stub_env["state"]
    scenario = golden_stub_env["scenario"]
    coupler = golden_stub_env["coupler"]
    tracker = golden_stub_env["tracker"]

    _initialize(golden_stub_env)

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
    land_use_datastore = artifact_to_path(coupler.get(USIM_DATASTORE_H5), workspace)
    assert land_use_datastore is not None
    assert Path(land_use_datastore).resolve() == Path(
        usim_inputs[USIM_DATASTORE_CURRENT_H5]
    ).resolve()

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )
    after_vehicle_ownership = artifact_to_path(coupler.get(USIM_DATASTORE_H5), workspace)
    assert after_vehicle_ownership is not None
    assert Path(after_vehicle_ownership).exists()
    assert Path(after_vehicle_ownership).parent == Path(land_use_datastore).parent

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    manifest_dir = tmp_path / "golden_surface_manifests"

    def _build_manifest_path(_workspace, year, iteration):
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"manifest_{year}_{iteration}.yaml"

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

    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs
    scenario_run = runs[0]
    steps = scenario_run.meta["steps"]
    assert [step["model"] for step in steps] == list(EXPECTED_STAGE_MODELS)
    steps_by_model = {step["model"]: step for step in steps}

    activitysim_run_outputs = set(
        (steps_by_model["activitysim_run"].get("outputs") or {}).values()
    )
    activitysim_postprocess_outputs = set(
        (steps_by_model["activitysim_postprocess"].get("outputs") or {}).values()
    )
    beam_run_outputs = set((steps_by_model["beam_run"].get("outputs") or {}).values())

    assert EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS <= activitysim_run_outputs
    assert EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS <= activitysim_postprocess_outputs
    assert {LINKSTATS, BEAM_PLANS_OUT} <= beam_run_outputs

    scenario_outputs = _artifact_map(tracker.get_artifacts_for_run(scenario_run.id).outputs)
    scenario_output_keys = set(scenario_outputs)

    final_usim_datastore = artifact_to_path(coupler.get(USIM_DATASTORE_H5), workspace)
    assert final_usim_datastore is not None
    assert Path(final_usim_datastore).name == (
        f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    assert Path(final_usim_datastore).parent == Path(after_vehicle_ownership).parent
    assert Path(final_usim_datastore).resolve() == Path(
        scenario_outputs[USIM_DATASTORE_H5].path
    ).resolve()

    assert EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS <= scenario_output_keys
    assert {ZARR_SKIMS, LINKSTATS, BEAM_PLANS_OUT} <= scenario_output_keys
    assert {"householdv_2017", "vehicles_2017", "atlas_vehicles2_output"} <= scenario_output_keys
    assert activitysim_run_outputs <= scenario_output_keys
    assert activitysim_postprocess_outputs <= scenario_output_keys
    assert beam_run_outputs <= scenario_output_keys

    for key in EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS:
        artifact_path = getattr(scenario_outputs[key], "path", None)
        assert artifact_path is not None
        assert "year-2017-iteration-0" in str(artifact_path)
        assert Path(artifact_path).exists()
