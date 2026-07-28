from __future__ import annotations

import logging
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import flush_archive_queue
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import postprocessing

logger = logging.getLogger(__name__)


def run_postprocessing_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run the postprocessing stage.

    This stage executes the global postprocessing step after the year completes.
    It consolidates outputs (e.g., copying artifacts to external locations) and
    does not depend on coupler inputs because it reads directly from the
    workspace output tree.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state for year/stage coordination.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler forwarded to shared orchestration utilities.
    year : int
        Forecast year being postprocessed.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace
    logger.info("[postprocessing] year=%s run_id=%s", year, cr.current_run_id())

    _, outputs = execute_step(
        scenario=scenario,
        definition=postprocessing,
        state=state,
        settings=settings,
        workspace=workspace,
        stage="postprocessing",
        year=year,
        iteration=getattr(state, "iteration", None),
        phase="postprocess",
    )
    del outputs
    flush_archive_queue(timeout=300, fail_on_timeout=False)
