from __future__ import annotations

import os
from typing import Callable, Dict, Mapping, Optional, Union

from pilates.generic.records import RecordStore
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.utils.consist_config import build_step_consist_kwargs
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    clean_expected_outputs,
    resolve_artifact_from_value,
    update_coupler_from_beam_outputs,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    WorkflowStage,
    WorkflowStepSpec,
    run_manifested_steps,
)
from pilates.workflows.step_io import build_outputs
from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_constants import (
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_INPUT,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    USIM_DATASTORE_H5,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
)
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState


def run_supply_demand_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: object,
    year: int,
    usim_inputs: Mapping[str, Union[str, os.PathLike]],
    build_manifest_path: Callable[[Workspace, int, int], os.PathLike],
) -> None:
    """
    Run the supply-demand loop (ActivitySim + BEAM) for the year.

    This stage iterates the activity-demand and traffic-assignment sub-stages:
    ActivitySim preprocess -> compile (once per year) -> run/postprocess produces
    household/person/activity outputs, which feed BEAM. BEAM then runs traffic
    assignment and postprocessing, producing skims and other artifacts that can
    be fed into the next iteration. Manifest checkpointing is used around
    ActivitySim preprocess and run/postprocess to support restart/resume without
    re-running completed steps.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : object
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Input mapping (including any UrbanSim datastore paths) used to seed
        ActivitySim preprocessing when land use was not run.
    build_manifest_path : Callable[[Workspace, int, int], os.PathLike]
        Factory for per-year/per-iteration manifest file locations.
    """
    total_iters = settings.run.supply_demand_iters
    previous_beam_outputs: Optional[RecordStore] = None

    for i in range(state.iteration, total_iters):
        state.iteration = i
        formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
        activity_demand_outputs = None
        outputs_holder = StepOutputsHolder()
        manifest_path = build_manifest_path(workspace, year, i)
        manifest_config = ManifestConfig(path=manifest_path)

        # C1. ACTIVITY DEMAND
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.activity_demand,
        ):
            formatted_print("ACTIVITY DEMAND MODEL")

            # ActivitySim runs in two manifest-checkpointed phases:
            # 1) Preprocess (per-iteration) to prepare compile inputs.
            # 2) Compile (per-year) outside manifest checkpointing.
            # 3) Run/Postprocess (per-iteration) for demand outputs.
            preprocess_inputs = None
            preprocess_input_keys = [USIM_DATASTORE_H5]
            if USIM_DATASTORE_H5 in usim_inputs:
                preprocess_inputs = {USIM_DATASTORE_H5: usim_inputs[USIM_DATASTORE_H5]}
                preprocess_input_keys = None
            else:
                get_value = getattr(coupler, "get", None)
                if callable(get_value):
                    if get_value(USIM_DATASTORE_H5) is not None:
                        preprocess_input_keys = [USIM_DATASTORE_H5]
                    else:
                        fallback_inputs, _ = build_urbansim_inputs(
                            settings, state, workspace, year
                        )
                        if USIM_DATASTORE_H5 in fallback_inputs:
                            preprocess_inputs = {
                                USIM_DATASTORE_H5: fallback_inputs[USIM_DATASTORE_H5]
                            }
                            preprocess_input_keys = None
                else:
                    fallback_inputs, _ = build_urbansim_inputs(
                        settings, state, workspace, year
                    )
                    if USIM_DATASTORE_H5 in fallback_inputs:
                        preprocess_inputs = {
                            USIM_DATASTORE_H5: fallback_inputs[USIM_DATASTORE_H5]
                        }
                        preprocess_input_keys = None

            preprocess_specs = [
                WorkflowStepSpec(
                    name="activitysim_preprocess",
                    step_func=make_activitysim_preprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=preprocess_input_keys,
                    inputs=preprocess_inputs,
                )
            ]
            run_manifested_steps(
                stage_name="activity_demand_preprocess",
                steps=preprocess_specs,
                outputs_holder=outputs_holder,
                manifest_config=manifest_config,
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                name_suffix=f"{year}_iter{i}",
                iteration=i,
            )

            # ActivitySim Compilation: run once per year after preprocess.
            if not state.asim_compiled:
                upstream = outputs_holder.activitysim_preprocess
                if upstream is None:
                    raise RuntimeError(
                        "ActivitySim compile requires preprocess outputs."
                    )
                compile_inputs = upstream.to_record_store().to_mapping()
                expected_compile_outputs = clean_expected_outputs(
                    build_outputs(
                        "activitysim_compile",
                        settings,
                        state,
                        workspace,
                        components=("runner",),
                    )
                )
                activitysim_compile_step = make_activitysim_compile_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                )
                compile_config = build_step_config(
                    fn=activitysim_compile_step,
                    name=f"activitysim_compile_{year}",
                    model="activitysim_compile",
                    state=state,
                    iteration=-1,
                    inputs=compile_inputs or None,
                    output_paths=expected_compile_outputs or None,
                    cache_mode="overwrite",
                    load_inputs=False,
                    runtime_kwargs=common_runtime_kwargs(
                        settings=settings,
                        state=state,
                        workspace=workspace,
                        expected_outputs=expected_compile_outputs,
                    ),
                    consist_kwargs=build_step_consist_kwargs(
                        "activitysim_compile",
                        settings,
                        workspace_path=workspace.full_path,
                    ),
                )
                scenario.run(**compile_config.to_kwargs())
            else:
                zarr_value = None
                get_value = getattr(coupler, "get", None)
                if callable(get_value):
                    zarr_value = resolve_artifact_from_value(
                        get_value(ZARR_SKIMS),
                        key=ZARR_SKIMS,
                        workspace=workspace,
                    )
                zarr_path = artifact_to_path(zarr_value, workspace)
                if not zarr_path:
                    candidate = os.path.join(
                        workspace.get_asim_output_dir(),
                        "cache",
                        "skims.zarr",
                    )
                    if os.path.exists(candidate):
                        zarr_path = candidate
                if not zarr_path or not os.path.exists(zarr_path):
                    raise RuntimeError(
                        "ActivitySim run requires zarr_skims, "
                        "but none were found while asim_compiled=True."
                    )

            activitysim_specs = [
                WorkflowStepSpec(
                    name="activitysim_run",
                    step_func=make_activitysim_run_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=[ASIM_MUTABLE_DATA_DIR, ZARR_SKIMS],
                ),
                WorkflowStepSpec(
                    name="activitysim_postprocess",
                    step_func=make_activitysim_postprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=[ASIM_OUTPUT_DIR],
                    inputs={
                        USIM_MUTABLE_DATA_DIR: workspace.get_usim_mutable_data_dir()
                    },
                ),
            ]
            run_manifested_steps(
                stage_name="activity_demand_run_postprocess",
                steps=activitysim_specs,
                outputs_holder=outputs_holder,
                manifest_config=manifest_config,
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                name_suffix=f"{year}_iter{i}",
                iteration=i,
            )

            state.complete_step(
                state.Stage.supply_demand_loop,
                i,
                state.Stage.activity_demand,
            )

        postprocess_outputs = outputs_holder.activitysim_postprocess
        activity_demand_outputs = (
            postprocess_outputs.to_record_store()
            if postprocess_outputs is not None
            else None
        )

        # C2. TRAFFIC ASSIGNMENT
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.traffic_assignment,
        ):
            formatted_print("TRAFFIC ASSIGNMENT MODEL")
            beam_preprocess_inputs: Dict[str, Any] = {}
            if activity_demand_outputs is not None:
                beam_preprocess_inputs.update(activity_demand_outputs.to_mapping())
            if previous_beam_outputs is not None:
                beam_preprocess_inputs.update(previous_beam_outputs.to_mapping())
            if getattr(settings, "vehicle_ownership_model_enabled", False) and i == 0:
                if state.run_info_path and os.path.exists(state.run_info_path):
                    previous_run_dir = os.path.dirname(state.run_info_path)
                    atlas_output_dir = os.path.join(
                        previous_run_dir, "atlas", "atlas_output"
                    )
                else:
                    atlas_output_dir = workspace.get_atlas_output_dir()
                atlas_vehicle_path = os.path.join(
                    atlas_output_dir,
                    f"vehicles2_{state.forecast_year}.csv",
                )
                if not os.path.exists(atlas_vehicle_path):
                    atlas_vehicle_path = os.path.join(
                        atlas_output_dir,
                        f"vehicles2_{state.forecast_year - 1}.csv",
                    )
                if os.path.exists(atlas_vehicle_path):
                    beam_preprocess_inputs.setdefault(
                        ATLAS_VEHICLES2_INPUT, atlas_vehicle_path
                    )

            beam_preprocess_input_keys = []
            if getattr(settings, "activity_demand_enabled", False):
                beam_preprocess_input_keys.append(ASIM_OUTPUT_DIR)
            if getattr(settings, "vehicle_ownership_model_enabled", False):
                beam_preprocess_input_keys.append(ATLAS_OUTPUT_DIR)

            beam_postprocess_input_keys = [BEAM_OUTPUT_DIR]
            if getattr(settings, "activity_demand_enabled", False):
                beam_postprocess_input_keys.append(ASIM_OUTPUT_DIR)

            beam_steps = [
                WorkflowStepSpec(
                    name="beam_preprocess",
                    step_func=make_beam_preprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=beam_preprocess_input_keys or None,
                    inputs=beam_preprocess_inputs or None,
                ),
                WorkflowStepSpec(
                    name="beam_run",
                    step_func=make_beam_run_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=[BEAM_MUTABLE_DATA_DIR, ZARR_SKIMS],
                ),
                WorkflowStepSpec(
                    name="beam_postprocess",
                    step_func=make_beam_postprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                    ),
                    input_keys=beam_postprocess_input_keys,
                ),
            ]

            WorkflowStage(
                name="beam",
                stage_type=state.Stage.traffic_assignment,
                steps=beam_steps,
            ).run(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                outputs_holder=outputs_holder,
                name_suffix=f"{year}_iter{i}",
                iteration=i,
                runtime_kwargs_extra={
                    "activity_demand_outputs": activity_demand_outputs,
                    "previous_beam_outputs": previous_beam_outputs,
                },
            )

            beam_run_outputs = outputs_holder.beam_run
            beam_post_outputs = outputs_holder.beam_postprocess
            combined_beam_outputs = None
            if beam_run_outputs is not None or beam_post_outputs is not None:
                combined_beam_outputs = RecordStore()
                if beam_run_outputs is not None:
                    combined_beam_outputs += beam_run_outputs.to_record_store()
                if beam_post_outputs is not None:
                    combined_beam_outputs += beam_post_outputs.to_record_store()
            previous_beam_outputs = combined_beam_outputs
            if combined_beam_outputs is not None:
                update_coupler_from_beam_outputs(
                    combined_beam_outputs, coupler, workspace
                )

            state.complete_step(
                state.Stage.supply_demand_loop,
                i,
                state.Stage.traffic_assignment,
            )

    state.complete_step(state.Stage.supply_demand_loop)
