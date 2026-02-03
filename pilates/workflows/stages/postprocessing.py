from __future__ import annotations

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.utils.consist_config import build_step_consist_kwargs
from pilates.workflows.steps import make_postprocessing_step


def run_postprocessing_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
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
    year : int
        Forecast year being postprocessed.
    """
    postprocessing_step = make_postprocessing_step()
    postprocessing_config = build_step_config(
        fn=postprocessing_step,
        name=f"postprocessing_{year}",
        model="postprocessing",
        state=state,
        cache_mode="overwrite",
        load_inputs=False,
        runtime_kwargs=common_runtime_kwargs(
            settings=settings,
            state=state,
            workspace=workspace,
        ),
        consist_kwargs=build_step_consist_kwargs("postprocessing", settings),
    )
    scenario.run(**postprocessing_config.to_kwargs())
