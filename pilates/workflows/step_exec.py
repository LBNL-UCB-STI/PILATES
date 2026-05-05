from pilates.config.models import PilatesConfig
from pilates.workspace import Workspace
from workflow_state import WorkflowState


def warm_start_activities(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    """
    Run ActivitySim warm-start to update UrbanSim inputs with long-term choices.

    Parameters
    ----------
    settings : PilatesConfig
        Simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace used to resolve paths.
    """
    raise RuntimeError(
        "ActivitySim warm-start is deprecated and no longer supported. "
        "Disable `activitysim.warm_start_activities` to continue."
    )
