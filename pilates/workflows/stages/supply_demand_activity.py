from __future__ import annotations

import logging
from dataclasses import dataclass

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import (
    activitysim_postprocess,
    activitysim_preprocess,
    activitysim_run,
)
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityDemandPhaseInputs:
    """Stage policy inputs for one ActivitySim supply-demand iteration."""

    year: int
    iteration: int


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


def _run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    inputs: ActivityDemandPhaseInputs,
    context: WorkflowRuntimeContext,
) -> None:
    """Run the three native ActivitySim definitions for one iteration.

    Semantic input selection, binding, output materialization and cache admission
    belong to the definitions and Consist.  The stage retains just the ordered
    policy invocation and the typed return used by the next stage.
    """

    settings = context.settings
    state = context.state
    workspace = context.workspace
    population_year = _activitysim_population_year(state)

    formatted_print("ACTIVITY DEMAND MODEL")
    logger.info("[activity_demand] year=%s iteration=%s", inputs.year, inputs.iteration)
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
    del postprocess_outputs
    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.activity_demand,
    )
    return None
