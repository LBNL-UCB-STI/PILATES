from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

from pilates.generic.records import RecordStore
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
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
    ASIM_OMX_SKIMS,
    ATLAS_VEHICLES2_INPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    ZARR_SKIMS,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState


@dataclass
class ActivityDemandPhaseInputs:
    """
    Inputs for one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Pre-resolved UrbanSim datastore inputs, if land use already ran or
        fallback inputs were provided.
    """
    year: int
    iteration: int
    usim_inputs: Mapping[str, Union[str, os.PathLike]]


@dataclass
class ActivityDemandPhaseOutputs:
    """
    Outputs from one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    activity_demand_outputs : Optional[RecordStore]
        RecordStore containing ActivitySim outputs needed downstream
        (e.g., households, persons, plans). None if not produced.
    """
    activity_demand_outputs: Optional[RecordStore]


@dataclass
class TrafficAssignmentPhaseInputs:
    """
    Inputs for one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    activity_demand_outputs : Optional[RecordStore]
        ActivitySim outputs used to seed BEAM inputs for this iteration.
    previous_beam_outputs : Optional[RecordStore]
        Prior BEAM outputs (e.g., linkstats) used for warm-starting.
    """
    year: int
    iteration: int
    activity_demand_outputs: Optional[RecordStore]
    previous_beam_outputs: Optional[RecordStore]


@dataclass
class TrafficAssignmentPhaseOutputs:
    """
    Outputs from one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    previous_beam_outputs : Optional[RecordStore]
        Combined BEAM run + postprocess outputs for warm-starting the
        next iteration, if available.
    """
    previous_beam_outputs: Optional[RecordStore]


def _find_initial_linkstats_warmstart(
    settings: PilatesConfig, workspace: Workspace
) -> Optional[str]:
    base_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        settings.beam.router_directory,
    )
    candidates = [
        os.path.join(base_dir, "init.linkstats.parquet"),
        os.path.join(base_dir, "init.linkstats.csv.gz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    inputs: ActivityDemandPhaseInputs,
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
) -> ActivityDemandPhaseOutputs:
    """
    Run ActivitySim for a single supply-demand iteration.

    This executes the ActivitySim preprocess, compile (once per year),
    and run/postprocess steps. It also assembles the required inputs
    from UrbanSim outputs or fallbacks and ensures skims are available
    when resuming after compilation.

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
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    inputs : ActivityDemandPhaseInputs
        Inputs required for this iteration.
    outputs_holder : StepOutputsHolder
        Accumulator for step outputs within the iteration.
    manifest_config : ManifestConfig
        Manifest checkpointing configuration for ActivitySim steps.

    Returns
    -------
    ActivityDemandPhaseOutputs
        RecordStore of ActivitySim outputs for downstream BEAM inputs.
    """
    formatted_print("ACTIVITY DEMAND MODEL")

    # ActivitySim runs in two manifest-checkpointed phases:
    # 1) Preprocess (per-iteration) to prepare compile inputs.
    # 2) Compile (per-year) outside manifest checkpointing.
    # 3) Run/Postprocess (per-iteration) for demand outputs.
    preprocess_inputs = None
    preprocess_input_keys = [USIM_DATASTORE_H5]
    if USIM_DATASTORE_H5 in inputs.usim_inputs:
        preprocess_inputs = {USIM_DATASTORE_H5: inputs.usim_inputs[USIM_DATASTORE_H5]}
        preprocess_input_keys = None
    else:
        get_value = getattr(coupler, "get", None)
        if callable(get_value) and get_value(USIM_DATASTORE_H5) is not None:
            preprocess_input_keys = [USIM_DATASTORE_H5]
        else:
            fallback_inputs, _ = build_urbansim_inputs(
                settings, state, workspace, inputs.year
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
        name_suffix=f"{inputs.year}_iter{inputs.iteration}",
        iteration=inputs.iteration,
    )

    # ActivitySim Compilation: run once per year after preprocess.
    if not state.asim_compiled:
        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim compile requires preprocess outputs.")
        compile_inputs = upstream.to_record_store().to_mapping()
        if ASIM_OMX_SKIMS in compile_inputs:
            compile_inputs = {ASIM_OMX_SKIMS: compile_inputs[ASIM_OMX_SKIMS]}
        else:
            compile_inputs = {}
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
            name=f"activitysim_compile_{inputs.year}",
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

    upstream_preprocess = outputs_holder.activitysim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    asim_run_input_keys = [
        short_name for short_name, _, _ in upstream_preprocess._iter_record_items()
    ]
    asim_run_input_keys.append(ZARR_SKIMS)

    activitysim_postprocess_inputs: Dict[str, str] = {}
    usim_input_fname = get_usim_datastore_fname(settings, io="input")
    usim_input_path = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_input_fname
    )
    if os.path.exists(usim_input_path):
        activitysim_postprocess_inputs[USIM_DATASTORE_H5] = usim_input_path

    activitysim_specs = [
        WorkflowStepSpec(
            name="activitysim_run",
            step_func=make_activitysim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=asim_run_input_keys or None,
        ),
        WorkflowStepSpec(
            name="activitysim_postprocess",
            step_func=make_activitysim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=None,
            inputs=activitysim_postprocess_inputs or None,
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
        name_suffix=f"{inputs.year}_iter{inputs.iteration}",
        iteration=inputs.iteration,
    )

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.activity_demand,
    )

    postprocess_outputs = outputs_holder.activitysim_postprocess
    activity_demand_outputs = (
        postprocess_outputs.to_record_store()
        if postprocess_outputs is not None
        else None
    )

    return ActivityDemandPhaseOutputs(
        activity_demand_outputs=activity_demand_outputs
    )


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    inputs: TrafficAssignmentPhaseInputs,
    outputs_holder: StepOutputsHolder,
) -> TrafficAssignmentPhaseOutputs:
    """
    Run BEAM for a single supply-demand iteration.

    This prepares BEAM inputs from ActivitySim outputs, warm-starts
    linkstats when available, executes preprocess/run/postprocess,
    and updates the coupler with BEAM artifacts for subsequent steps.

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
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    inputs : TrafficAssignmentPhaseInputs
        Inputs required for this iteration.
    outputs_holder : StepOutputsHolder
        Accumulator for step outputs within the iteration.

    Returns
    -------
    TrafficAssignmentPhaseOutputs
        Combined BEAM outputs for warm-starting the next iteration.
    """
    formatted_print("TRAFFIC ASSIGNMENT MODEL")
    if (
        inputs.activity_demand_outputs is None
        and inputs.iteration == 0
        and inputs.previous_beam_outputs is None
    ):
        raise RuntimeError(
            "TrafficAssignment iteration 0 requires activity_demand_outputs "
            "or previous_beam_outputs. Ensure ActivityDemand completed or "
            "provide warm-start outputs before running BEAM."
        )
    beam_preprocess_inputs: Dict[str, Any] = {}
    if inputs.activity_demand_outputs is not None:
        asim_input_keys = {
            "beam_plans",
            "beam_plans_out",
            "households",
            "linkstats",
            "persons",
        }
        for key, value in inputs.activity_demand_outputs.to_mapping().items():
            if key in asim_input_keys:
                beam_preprocess_inputs[key] = value
    if inputs.previous_beam_outputs is not None:
        for key, value in inputs.previous_beam_outputs.to_mapping().items():
            if key.startswith("linkstats"):
                beam_preprocess_inputs[key] = value
    if (
        inputs.previous_beam_outputs is None
        or not any(
            key.startswith("linkstats")
            for key in inputs.previous_beam_outputs.to_mapping().keys()
        )
    ):
        warmstart_path = _find_initial_linkstats_warmstart(settings, workspace)
        if warmstart_path:
            beam_preprocess_inputs.setdefault(LINKSTATS_WARMSTART, warmstart_path)
    if getattr(settings, "vehicle_ownership_model_enabled", False) and inputs.iteration == 0:
        if state.run_info_path and os.path.exists(state.run_info_path):
            previous_run_dir = os.path.dirname(state.run_info_path)
            atlas_output_dir = os.path.join(previous_run_dir, "atlas", "atlas_output")
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
            beam_preprocess_inputs.setdefault(ATLAS_VEHICLES2_INPUT, atlas_vehicle_path)

    zarr_input_keys = None
    beam_prepared_input_keys = None
    if inputs.activity_demand_outputs is not None:
        zarr_input_keys = [ZARR_SKIMS]
        beam_prepared_input_keys = [
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            LINKSTATS_WARMSTART,
        ]

    beam_run_input_keys = []
    if zarr_input_keys:
        beam_run_input_keys.extend(zarr_input_keys)
    if beam_prepared_input_keys:
        beam_run_input_keys.extend(beam_prepared_input_keys)
    if not beam_run_input_keys:
        beam_run_input_keys = None

    beam_steps = [
        WorkflowStepSpec(
            name="beam_preprocess",
            step_func=make_beam_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=None,
            inputs=beam_preprocess_inputs or None,
        ),
        WorkflowStepSpec(
            name="beam_run",
            step_func=make_beam_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=beam_run_input_keys,
        ),
        WorkflowStepSpec(
            name="beam_postprocess",
            step_func=make_beam_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=zarr_input_keys,
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
        name_suffix=f"{inputs.year}_iter{inputs.iteration}",
        iteration=inputs.iteration,
        runtime_kwargs_extra={
            "activity_demand_outputs": inputs.activity_demand_outputs,
            "previous_beam_outputs": inputs.previous_beam_outputs,
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

    if combined_beam_outputs is not None:
        update_coupler_from_beam_outputs(combined_beam_outputs, coupler, workspace)

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return TrafficAssignmentPhaseOutputs(previous_beam_outputs=combined_beam_outputs)


def run_supply_demand_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
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
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Input mapping (including any UrbanSim datastore paths) used to seed
        ActivitySim preprocessing when land use was not run.
    build_manifest_path : Callable[[Workspace, int, int], os.PathLike]
        Factory for per-year/per-iteration manifest file locations.

    State Machine & Resume Behavior
    -------------------------------
    The stage maintains iteration state via ``state.iteration`` and records
    per-step completion in a manifest under ``.workflow/``. On resume:

    1. **ActivityDemand Phase**:
       - If ``should_run(...)`` returns True, preprocess → compile (once per year)
         → run/postprocess executes and updates coupler outputs.
       - If it returns False, ActivitySim steps are skipped and downstream
         phases must rely on previously-produced artifacts.

    2. **TrafficAssignment Phase**:
       - Requires ActivitySim outputs or prior BEAM outputs to seed inputs.
         If neither is available on iteration 0, the stage raises an error.
       - For later iterations, warm-start linkstats may be pulled from prior
         BEAM outputs or from an initial warm-start file.

    3. **Convergence Check**:
       - Checked at each iteration boundary. If convergence is detected, the
         loop exits early and the stage marks completion.

    Warnings
    --------
    - Removing ``.workflow/`` manifests forces all iterations to re-run.
    - If resuming mid-iteration, ensure coupler artifacts for ActivitySim
      outputs are available before running BEAM.
    - Do not mutate coupler keys between iterations; they carry warm-start
      state to the next iteration.
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
            activity_demand_inputs = ActivityDemandPhaseInputs(
                year=year,
                iteration=i,
                usim_inputs=usim_inputs,
            )
            activity_demand_outputs = _run_activity_demand_phase(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                inputs=activity_demand_inputs,
                outputs_holder=outputs_holder,
                manifest_config=manifest_config,
            ).activity_demand_outputs

        # C2. TRAFFIC ASSIGNMENT
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.traffic_assignment,
        ):
            traffic_inputs = TrafficAssignmentPhaseInputs(
                year=year,
                iteration=i,
                activity_demand_outputs=activity_demand_outputs,
                previous_beam_outputs=previous_beam_outputs,
            )
            previous_beam_outputs = _run_traffic_assignment_phase(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                inputs=traffic_inputs,
                outputs_holder=outputs_holder,
            ).previous_beam_outputs

    state.complete_step(state.Stage.supply_demand_loop)
