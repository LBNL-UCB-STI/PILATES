from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from pilates.config.models import PilatesConfig
from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import (
    activitysim_postprocess,
    activitysim_preprocess,
    activitysim_run,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityDemandPhaseInputs:
    """Stage policy inputs for one ActivitySim supply-demand iteration."""

    year: int
    iteration: int
    usim_inputs: Mapping[str, Union[str, bytes]]


@dataclass(frozen=True)
class ActivityDemandPhaseOutputs:
    """Typed ActivitySim projection expressed as the next stage's path policy."""

    activity_demand_outputs: Optional[Dict[str, Any]]


def _activitysim_population_year(state: WorkflowState) -> int:
    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is None:
        forecast_year = getattr(state, "year", None)
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year or WorkflowState.year must be set before "
            "ActivitySim execution."
        )
    return int(forecast_year)


def _typed_output_paths(outputs: Any) -> Dict[str, Any]:
    """Expose a typed projection to stage policy without mutating the coupler."""

    return {key: path for key, path, _description in outputs._iter_record_items()}


def _run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    inputs: ActivityDemandPhaseInputs,
    outputs_holder: object | None = None,
    manifest_config: object | None = None,
    context: Optional[WorkflowRuntimeContext] = None,
    state: Optional[WorkflowState] = None,
    settings: Optional[PilatesConfig] = None,
    workspace: Optional[Workspace] = None,
    surface: object | None = None,
) -> ActivityDemandPhaseOutputs:
    """Run the three native ActivitySim definitions for one iteration.

    Semantic input selection, binding, output materialization and cache admission
    belong to the definitions and Consist.  The stage retains just the ordered
    policy invocation and the typed return used by the next stage.
    """

    del coupler, outputs_holder, manifest_config
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
    population_year = _activitysim_population_year(state)

    formatted_print("ACTIVITY DEMAND MODEL")
    logger.info("[activity_demand] year=%s iteration=%s", inputs.year, inputs.iteration)
    # ``inputs.usim_inputs`` is a typed stage handoff for durable/archive policy;
    # the resolver intentionally reads the scenario coupler as its sole semantic
    # selection source.
    _, preprocess_outputs = execute_step(
        scenario=scenario,
        definition=activitysim_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=inputs.iteration,
        phase="preprocess",
    )
    del preprocess_outputs
    _, run_outputs = execute_step(
        scenario=scenario,
        definition=activitysim_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=inputs.iteration,
        phase="run",
    )
    del run_outputs
    _, postprocess_outputs = execute_step(
        scenario=scenario,
        definition=activitysim_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=inputs.iteration,
        phase="postprocess",
    )
    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.activity_demand,
    )
    return ActivityDemandPhaseOutputs(
        activity_demand_outputs=_typed_output_paths(postprocess_outputs)
    )
