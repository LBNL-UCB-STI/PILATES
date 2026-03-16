from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.formatting import formatted_print
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.workflows.input_resolution import resolve_step_inputs
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_INPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


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
    activity_demand_outputs : Optional[dict[str, Any]]
        ActivitySim outputs used to seed BEAM inputs for this iteration.
    previous_beam_outputs : Optional[dict[str, Any]]
        Prior BEAM outputs (e.g., linkstats) used for warm-starting.
    """

    year: int
    iteration: int
    activity_demand_outputs: Optional[Dict[str, Any]]
    previous_beam_outputs: Optional[Dict[str, Any]]


@dataclass
class TrafficAssignmentPhaseOutputs:
    """
    Outputs from one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    previous_beam_outputs : Optional[dict[str, Any]]
        Combined BEAM run + postprocess outputs for warm-starting the
        next iteration, if available.
    """

    previous_beam_outputs: Optional[Dict[str, Any]]


def _run_supply_demand_workflow(
    *,
    stage_name: str,
    steps: list[StepRef],
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    runtime_kwargs_extra: Optional[dict[str, Any]] = None,
) -> None:
    """Run workflow steps with shared supply-demand stage context."""
    run_workflow(
        stage_name=stage_name,
        steps=steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
    )


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = getattr(settings, "beam", None)
    skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg else None
    if skim_cfg is None:
        return "disabled"
    return getattr(skim_cfg, "run_schedule", "standalone")


def _should_run_full_skim(settings: PilatesConfig, iteration: int) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        return True
    if schedule == "after_each_iteration":
        return True
    if schedule == "after_final_iteration":
        total_iters = settings.run.supply_demand_iters
        return iteration == total_iters - 1
    return False


def _is_iteration_scoped_artifact_key(
    key: str, *, prefix: str, year: int, iteration: int
) -> bool:
    base = f"{prefix}_{year}_{iteration}"
    return key == base or key.startswith(f"{base}_sub")


def _build_beam_postprocess_input_keys(
    *,
    upstream_keys: Iterable[str],
    year: int,
    iteration: int,
    include_zarr_skims: bool,
) -> Optional[list[str]]:
    """
    Select BEAM postprocess coupler inputs from BEAM run outputs.

    BEAM postprocess only consumes BEAM events parquet and OD skims artifacts
    from the run output store, plus upstream ActivitySim ``zarr_skims`` when
    available. Trimming input keys to this set keeps run identity aligned with
    actual behavior while avoiding unnecessary cache invalidation from unrelated
    BEAM outputs.
    """
    selected: list[str] = []
    keys = list(upstream_keys)

    for key in keys:
        if _is_iteration_scoped_artifact_key(
            key, prefix="events_parquet", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims_zarr", year=year, iteration=iteration
        ):
            selected.append(key)

    # Conservative fallback for naming drift: keep skim/event dependencies if
    # exact iteration-scoped keys are absent.
    if not any(key.startswith("raw_od_skims") for key in selected):
        selected.extend(key for key in keys if key.startswith("raw_od_skims"))
    if not any(key.startswith("events_parquet_") for key in selected):
        selected.extend(key for key in keys if key.startswith("events_parquet_"))

    if include_zarr_skims:
        selected.append(ZARR_SKIMS)

    deduped = list(dict.fromkeys(selected))
    return deduped or None


def _collect_previous_beam_outputs(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    state: WorkflowState,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Resolve previous BEAM outputs for warm-starting.

    When explicit previous outputs are unavailable, this attempts to hydrate
    a minimal promoted store from coupler keys written by BEAM postprocess.
    """
    _ = state, iteration
    if previous_beam_outputs is not None:
        return previous_beam_outputs

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None

    promoted_outputs: Dict[str, Any] = {}
    for key in (LINKSTATS_WARMSTART, LINKSTATS, BEAM_PLANS_OUT):
        value = get_value(key)
        if value is None:
            continue
        path = artifact_to_path(value, workspace)
        if path and os.path.exists(path):
            promoted_outputs[key] = value
    return promoted_outputs or None


def _collect_beam_preprocess_inputs(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    state: WorkflowState,
    iteration: int,
    activity_demand_outputs: Optional[Dict[str, Any]],
    previous_beam_outputs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build preprocess inputs for BEAM from available upstream sources.

    Source precedence:
    1) ActivitySim outputs (when enabled and available)
    2) Default BEAM scenario files (when ActivitySim is disabled)
    3) Prior BEAM outputs and warm-start linkstats
    4) ATLAS vehicles2 (iteration 0 only, when vehicle ownership is enabled)
    """
    beam_preprocess_inputs: Dict[str, Any] = {}
    forecast_year = state.forecast_year
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before building BEAM inputs."
        )

    if activity_demand_outputs is not None:
        asim_input_keys = {
            "beam_plans_asim_out",
            "beam_plans_out",
            "households_asim_out",
            "linkstats",
            "persons_asim_out",
        }
        for key, value in activity_demand_outputs.items():
            if key in asim_input_keys:
                beam_preprocess_inputs[key] = value
    elif settings.run.models.activity_demand is None:
        logger.info("Falling back on default inputs to BEAM")
        # In BEAM-only mode the copied mutable scenario already contains the
        # canonical plans/households/persons files. Let beam_preprocess resolve
        # and publish those inputs from the staged scenario directory instead of
        # passing filesystem paths through Consist ahead of time.
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
                value = get_value(key)
                if value is not None:
                    beam_preprocess_inputs[key] = value
    elif previous_beam_outputs is None:
        raise RuntimeError(
            "TrafficAssignment iteration 0 requires activity_demand_outputs "
            "or previous_beam_outputs. Ensure ActivityDemand completed or "
            "provide warm-start outputs before running BEAM."
        )

    if previous_beam_outputs is not None:
        for key, value in previous_beam_outputs.items():
            if key.startswith("linkstats"):
                beam_preprocess_inputs[key] = value

    if previous_beam_outputs is None or not any(
        key.startswith("linkstats") for key in previous_beam_outputs.keys()
    ):
        get_value = getattr(coupler, "get", None)
        coupler_warmstart = None
        if callable(get_value):
            value = get_value(LINKSTATS_WARMSTART)
            coupler_warmstart = (
                artifact_to_path(value, workspace) if value is not None else None
            )
        if coupler_warmstart and os.path.exists(coupler_warmstart):
            beam_preprocess_inputs.setdefault(LINKSTATS_WARMSTART, coupler_warmstart)
        else:
            warmstart_path = resolve_initial_linkstats_path(settings, workspace)
            if warmstart_path:
                beam_preprocess_inputs.setdefault(LINKSTATS_WARMSTART, warmstart_path)

    if getattr(settings, "vehicle_ownership_model_enabled", False) and iteration == 0:
        if state.run_info_path and os.path.exists(state.run_info_path):
            previous_run_dir = os.path.dirname(state.run_info_path)
            atlas_output_dir = os.path.join(previous_run_dir, "atlas", "atlas_output")
        else:
            atlas_output_dir = workspace.get_atlas_output_dir()
        atlas_vehicle_path = os.path.join(
            atlas_output_dir,
            f"vehicles2_{forecast_year}.csv",
        )
        if not os.path.exists(atlas_vehicle_path):
            atlas_vehicle_path = os.path.join(
                atlas_output_dir,
                f"vehicles2_{forecast_year - 1}.csv",
            )
        if os.path.exists(atlas_vehicle_path):
            beam_preprocess_inputs.setdefault(ATLAS_VEHICLES2_INPUT, atlas_vehicle_path)

    return beam_preprocess_inputs


def _derive_beam_run_input_keys(
    *,
    beam_preprocess_inputs: Mapping[str, Any],
    activity_demand_outputs: Optional[Dict[str, Any]],
) -> list[str]:
    """
    Derive BEAM run input keys from preprocess outputs and warm-start signals.

    beam_preprocess always publishes the canonical plans/households/persons trio,
    regardless of whether they came from ActivitySim outputs or from existing
    default files in the copied BEAM scenario directory.
    """
    _ = activity_demand_outputs
    run_input_keys = [
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    ]

    # Only require LINKSTATS_WARMSTART at BEAM run time when that explicit key
    # is provided to preprocess. Other linkstats* artifacts may exist for
    # bookkeeping/history but do not guarantee a warm-start input artifact.
    if LINKSTATS_WARMSTART in beam_preprocess_inputs:
        run_input_keys.append(LINKSTATS_WARMSTART)
    else:
        logger.debug(
            "[BEAM] linkstats warmstart not available; omitting %s from inputs",
            LINKSTATS_WARMSTART,
        )

    return run_input_keys


def _finalize_beam_run_input_keys(
    *,
    beam_run_input_keys: Optional[list[str]],
    outputs_holder: StepOutputsHolder,
) -> list[str]:
    """
    Reconcile BEAM run inputs with the artifacts actually published by preprocess.

    The pre-run key derivation happens before BEAM preprocess executes, but
    preprocess may decide to publish ``linkstats_warmstart`` after resolving
    previous outputs. Use the realized preprocess outputs as the final contract.
    """
    finalized_keys = list(
        beam_run_input_keys
        or [
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ]
    )
    preprocess_outputs = outputs_holder.beam_preprocess
    prepared_inputs = (
        preprocess_outputs.prepared_inputs if preprocess_outputs is not None else {}
    )
    has_warmstart = LINKSTATS_WARMSTART in prepared_inputs
    if has_warmstart and LINKSTATS_WARMSTART not in finalized_keys:
        finalized_keys.append(LINKSTATS_WARMSTART)
    if not has_warmstart and LINKSTATS_WARMSTART in finalized_keys:
        finalized_keys = [
            key for key in finalized_keys if key != LINKSTATS_WARMSTART
        ]
    return finalized_keys


def _run_beam_preprocess_step(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_inputs: Mapping[str, Any],
    runtime_kwargs_extra: Mapping[str, Any],
) -> None:
    """
    Execute BEAM preprocess with explicit resolved inputs.
    """
    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_preprocess",
                step_func=make_beam_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=None,
                inputs=resolve_step_inputs(
                    keys=beam_preprocess_inputs.keys(),
                    explicit_inputs=beam_preprocess_inputs,
                ).stepref_inputs(),
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_inputs: Mapping[str, Any],
    beam_run_input_keys: Optional[list[str]],
    include_zarr_skims: bool,
    runtime_kwargs_extra: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Execute BEAM preprocess/run/postprocess and return combined outputs.
    """
    _run_beam_preprocess_step(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        beam_preprocess_inputs=beam_preprocess_inputs,
        runtime_kwargs_extra=runtime_kwargs_extra,
    )
    beam_run_input_keys = _finalize_beam_run_input_keys(
        beam_run_input_keys=beam_run_input_keys,
        outputs_holder=outputs_holder,
    )

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_run",
                step_func=make_beam_run_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=beam_run_input_keys,
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )

    upstream_run = outputs_holder.beam_run
    if upstream_run is None:
        raise RuntimeError("BEAM run must complete first")
    beam_postprocess_input_keys = _build_beam_postprocess_input_keys(
        upstream_keys=[
            short_name for short_name, _, _ in upstream_run._iter_record_items()
        ],
        year=year,
        iteration=iteration,
        include_zarr_skims=include_zarr_skims,
    )
    beam_postprocess_resolution = None
    if beam_postprocess_input_keys:
        beam_postprocess_resolution = resolve_step_inputs(
            keys=beam_postprocess_input_keys,
            coupler=coupler,
            explicit_inputs=step_output_handoff_mapping(upstream_run, coupler=coupler),
        )

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_postprocess",
                step_func=make_beam_postprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=(
                    beam_postprocess_resolution.stepref_input_keys()
                    if beam_postprocess_resolution is not None
                    else None
                ),
                inputs=(
                    beam_postprocess_resolution.stepref_inputs()
                    if beam_postprocess_resolution is not None
                    else None
                ),
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )

    if outputs_holder.beam_run is None and outputs_holder.beam_postprocess is None:
        return None

    combined_beam_outputs: Dict[str, Any] = {}
    if outputs_holder.beam_run is not None:
        combined_beam_outputs.update(
            step_output_handoff_mapping(outputs_holder.beam_run, coupler=coupler)
        )
    if outputs_holder.beam_postprocess is not None:
        combined_beam_outputs.update(
            step_output_handoff_mapping(
                outputs_holder.beam_postprocess,
                coupler=coupler,
            )
        )
    return combined_beam_outputs


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
    runtime_kwargs_extra: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Execute dedicated BEAM full-skim step and return its outputs.
    """
    runtime_kwargs = dict(runtime_kwargs_extra)
    runtime_kwargs["previous_beam_outputs"] = previous_beam_outputs

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_full_skim",
                step_func=make_beam_full_skim_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=None,
                year=state.forecast_year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs,
    )

    if outputs_holder.beam_full_skim is None:
        return None
    return step_output_handoff_mapping(outputs_holder.beam_full_skim, coupler=coupler)


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

    previous_beam_outputs = _collect_previous_beam_outputs(
        coupler=coupler,
        workspace=workspace,
        state=state,
        iteration=inputs.iteration,
        previous_beam_outputs=inputs.previous_beam_outputs,
    )
    beam_preprocess_inputs = _collect_beam_preprocess_inputs(
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        state=state,
        iteration=inputs.iteration,
        activity_demand_outputs=inputs.activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
    )
    beam_run_input_keys = _derive_beam_run_input_keys(
        beam_preprocess_inputs=beam_preprocess_inputs,
        activity_demand_outputs=inputs.activity_demand_outputs,
    )

    traffic_runtime_kwargs = {
        "activity_demand_outputs": inputs.activity_demand_outputs,
        "previous_beam_outputs": previous_beam_outputs,
        "beam_preprocess_inputs": beam_preprocess_inputs,
    }
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        _run_beam_preprocess_step(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_inputs=beam_preprocess_inputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            previous_beam_outputs=previous_beam_outputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_inputs=beam_preprocess_inputs,
            beam_run_input_keys=beam_run_input_keys,
            include_zarr_skims=bool(inputs.activity_demand_outputs),
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
        if _should_run_full_skim(settings, inputs.iteration):
            full_skim_outputs = _run_beam_full_skim_step(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                outputs_holder=outputs_holder,
                year=inputs.year,
                iteration=inputs.iteration,
                previous_beam_outputs=combined_beam_outputs,
                runtime_kwargs_extra=traffic_runtime_kwargs,
            )
            if full_skim_outputs is not None:
                if combined_beam_outputs is None:
                    combined_beam_outputs = {}
                combined_beam_outputs.update(full_skim_outputs)

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return TrafficAssignmentPhaseOutputs(previous_beam_outputs=combined_beam_outputs)
