from __future__ import annotations

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder, make_postprocessing_step


def run_postprocessing_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    year: int,
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
    outputs_holder = StepOutputsHolder()
    postprocess_steps = [
        StepRef(
            name="postprocessing",
            step_func=make_postprocessing_step(),
            cache_mode="overwrite",
            load_inputs=False,
            model="postprocessing",
            year=year,
            iteration=getattr(state, "iteration", None),
            phase="postprocess",
        )
    ]
    run_workflow(
        stage_name="postprocessing",
        steps=postprocess_steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=str(year),
        iteration=getattr(state, "iteration", 0),
    )
