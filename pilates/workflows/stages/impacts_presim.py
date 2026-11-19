from __future__ import annotations

import logging

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import impacts_presim

logger = logging.getLogger(__name__)


def run_impacts_presim_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: object,
    year: int,
    context: WorkflowRuntimeContext,
) -> None:
    """Run IMPACTS presim through Main's native step execution path."""
    del coupler
    settings = context.settings
    if not getattr(getattr(getattr(settings, "run", None), "models", None), "impacts", None):
        return
    state = context.state
    logger.info("[impacts_presim] year=%s run_id=%s", year, cr.current_run_id())
    _, outputs = execute_step(
        scenario=scenario,
        definition=impacts_presim,
        state=state,
        settings=settings,
        workspace=context.workspace,
        stage="impacts_presim",
        year=year,
        iteration=getattr(state, "iteration", None),
        phase="run",
    )
    del outputs
