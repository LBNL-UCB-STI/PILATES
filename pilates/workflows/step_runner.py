from typing import Any, Dict


def common_runtime_kwargs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
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
