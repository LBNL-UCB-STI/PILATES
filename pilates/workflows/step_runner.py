from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pilates.config import PilatesConfig
    from pilates.workspace import Workspace
    from pilates.workflows.atlas_state import AtlasSubState
    from workflow_state import WorkflowState


def common_runtime_kwargs(
    *,
    settings: "PilatesConfig",
    state: "WorkflowState | AtlasSubState",
    workspace: "Workspace",
    **extras: Any,
) -> Dict[str, Any]:
    """
    Build runtime kwargs with the shared settings/state/workspace entries.

    Parameters
    ----------
    settings : object
        Simulation settings.
    state : object
        Workflow state.
    workspace : object
        Workspace instance.
    **extras : dict
        Additional runtime kwargs to include.

    Returns
    -------
    dict
        Runtime kwargs mapping ready for `scenario.run`.
    """
    return {
        "settings": settings,
        "state": state,
        "workspace": workspace,
        **extras,
    }
