from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.utils.formatting import formatted_print
from pilates.workflows.binding import (
    BindingPlan,
    beam_preprocess_binding_plan,
    build_binding_plan,
    build_key_only_binding_plan,
)
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.orchestration import StageRunner
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_keys import (
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

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface


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
        if artifact_to_existing_path(value, workspace):
            promoted_outputs[key] = value
    return promoted_outputs or None


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


def _make_beam_stage_runner(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    runtime_kwargs_extra: Optional[Mapping[str, Any]] = None,
    context: WorkflowRuntimeContext,
) -> StageRunner:
    """Build the execution context shared by BEAM stage slices."""
    return StageRunner(
        stage_name="beam",
        scenario=scenario,
        state=context.state,
        settings=context.settings,
        workspace=context.workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
        run_workflow_fn=run_workflow,
    )


def _run_beam_preprocess_step(
    *,
    stage_runner: StageRunner,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    beam_preprocess_binding: BindingPlan,
) -> None:
    """
    Execute BEAM preprocess with explicit resolved inputs.
    """
    stage_runner.run_step(
        step=StepRef(
            name="beam_preprocess",
            step_func=make_beam_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=beam_preprocess_binding,
            year=year,
        )
    )


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_binding: BindingPlan,
    beam_run_input_keys: Optional[list[str]],
    include_zarr_skims: bool,
    runtime_kwargs_extra: Mapping[str, Any],
    context: WorkflowRuntimeContext,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute BEAM preprocess/run/postprocess and return combined outputs.
    """
    surface = surface or context.surface
    stage_runner = _make_beam_stage_runner(
        scenario=scenario,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
        context=context,
    )
    _run_beam_preprocess_step(
        stage_runner=stage_runner,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        beam_preprocess_binding=beam_preprocess_binding,
    )
    beam_run_input_keys = _finalize_beam_run_input_keys(
        beam_run_input_keys=beam_run_input_keys,
        outputs_holder=outputs_holder,
    )

    stage_runner.run_step(
        step=StepRef(
            name="beam_run",
            step_func=make_beam_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=build_key_only_binding_plan(
                step_name="beam_run",
                input_keys=beam_run_input_keys,
                optional_input_keys=[LINKSTATS_WARMSTART],
                coupler=coupler,
                settings=context.settings,
                state=context.state,
                workspace=context.workspace,
                year=year,
            ),
            year=year,
        )
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
    beam_postprocess_binding = None
    if beam_postprocess_input_keys:
        optional_keys = [ZARR_SKIMS] if ZARR_SKIMS in beam_postprocess_input_keys else []
        required_keys = [
            key for key in beam_postprocess_input_keys if key not in optional_keys
        ]
        beam_postprocess_binding = build_binding_plan(
            step_name="beam_postprocess",
            coupler=coupler,
            explicit_inputs=step_output_handoff_mapping(upstream_run, coupler=coupler),
            required_keys=required_keys,
            optional_keys=optional_keys,
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            year=year,
            surface=surface,
        )

    stage_runner.run_step(
        step=StepRef(
            name="beam_postprocess",
            step_func=make_beam_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=beam_postprocess_binding,
            year=year,
        )
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
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
    runtime_kwargs_extra: Mapping[str, Any],
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """
    Execute dedicated BEAM full-skim step and return its outputs.
    """
    runtime_kwargs = dict(runtime_kwargs_extra)
    runtime_kwargs["previous_beam_outputs"] = previous_beam_outputs
    stage_runner = _make_beam_stage_runner(
        scenario=scenario,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs,
        context=context,
    )
    stage_runner.run_step(
        step=StepRef(
            name="beam_full_skim",
            step_func=make_beam_full_skim_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=BindingPlan(),
            year=context.state.forecast_year,
        )
    )

    if outputs_holder.beam_full_skim is None:
        return None
    return step_output_handoff_mapping(outputs_holder.beam_full_skim, coupler=coupler)


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    inputs: TrafficAssignmentPhaseInputs,
    outputs_holder: StepOutputsHolder,
    context: Optional[WorkflowRuntimeContext] = None,
    state: Optional[WorkflowState] = None,
    settings: Optional[PilatesConfig] = None,
    workspace: Optional[Workspace] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
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
    runtime_context = ensure_workflow_runtime_context(
        context=context,
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    settings = runtime_context.settings
    state = runtime_context.state
    workspace = runtime_context.workspace
    surface = runtime_context.surface

    formatted_print("TRAFFIC ASSIGNMENT MODEL")

    previous_beam_outputs = _collect_previous_beam_outputs(
        coupler=coupler,
        workspace=workspace,
        state=state,
        iteration=inputs.iteration,
        previous_beam_outputs=inputs.previous_beam_outputs,
    )
    beam_preprocess_binding = beam_preprocess_binding_plan(
        coupler=coupler,
        settings=settings,
        state=state,
        workspace=workspace,
        year=inputs.year,
        activity_demand_outputs=inputs.activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
        surface=surface,
    )
    beam_preprocess_inputs = dict(beam_preprocess_binding.inputs or {})
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
        standalone_runner = _make_beam_stage_runner(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
        )
        _run_beam_preprocess_step(
            stage_runner=standalone_runner,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            beam_preprocess_binding=beam_preprocess_binding,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            previous_beam_outputs=previous_beam_outputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_binding=beam_preprocess_binding,
            beam_run_input_keys=beam_run_input_keys,
            include_zarr_skims=bool(inputs.activity_demand_outputs),
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
            surface=surface,
        )
        if _should_run_full_skim(settings, inputs.iteration):
            full_skim_outputs = _run_beam_full_skim_step(
                scenario=scenario,
                coupler=coupler,
                outputs_holder=outputs_holder,
                year=inputs.year,
                iteration=inputs.iteration,
                previous_beam_outputs=combined_beam_outputs,
                runtime_kwargs_extra=traffic_runtime_kwargs,
                context=runtime_context,
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
