from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.workflows.binding import (
    build_binding_plan,
    urbansim_datastore_selection_rules,
)
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.workflows.input_resolution import (
    resolved_value_for_key,
    selected_candidate_key,
)

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from pilates.workflows.surface import EnabledWorkflowSurface
    from workflow_state import WorkflowState

def build_urbansim_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build UrbanSim input paths and per-key log descriptions for a run year.

    Parameters
    ----------
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    year : int
        Simulation year for labeling and input resolution.

    Returns
    -------
    tuple of dict
        (inputs, descriptions) where descriptions are per-key log strings.

    Notes
    -----
    Input keys
        - ``usim_datastore_base_h5``: UrbanSim datastore treated as static/
          exogenous baseline input for the run year (H5). This is a semantic
          role used at workflow boundaries; in some runs it may resolve to the
          same physical mutable-input H5 as the current datastore handle.
        - ``usim_datastore_h5``: UrbanSim current mutable datastore used by
          active workflow steps for this year (H5). On later-year runs this
          usually points at the latest handoff datastore, while on start-year
          or fallback paths it may coincide with ``usim_datastore_base_h5``.
        - ``usim_mutable_data_dir``: UrbanSim mutable data directory used as
          the container input/output mount.
    Related outputs
        - UrbanSim runner/postprocessor update ``usim_datastore_h5`` for
          downstream ActivitySim/ATLAS runs.
        - TODO: Add any additional UrbanSim outputs (e.g., diagnostics or
          intermediate tables) that should be surfaced in logs or the coupler.
    """
    inputs: Dict[str, Any] = {}
    descriptions: Dict[str, str] = {}

    resolution = build_binding_plan(
        step_name="urbansim_input_selection",
        settings=settings,
        state=state,
        workspace=workspace,
        year=year,
        surface=surface,
        artifact_rules=urbansim_datastore_selection_rules(),
        required_keys=[USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5],
    )

    # Keep both semantic handles explicit even when they resolve to the same
    # physical file. The workflow uses ``base`` and ``current`` as distinct
    # roles for restart-sensitive provenance and downstream handoffs.
    for semantic_key in (USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5):
        value = resolved_value_for_key(
            resolved=resolution,
            key=semantic_key,
            coupler=None,
        )
        if value is None:
            continue
        inputs[semantic_key] = str(value)
        selected_key = selected_candidate_key(resolution, semantic_key)
        suffix = " (fallback)" if selected_key != semantic_key else ""
        if semantic_key == USIM_DATASTORE_BASE_H5:
            descriptions[semantic_key] = (
                f"UrbanSim base datastore for year {year}{suffix}"
            )
        else:
            descriptions[semantic_key] = (
                f"UrbanSim current datastore for year {year}{suffix}"
            )

    return inputs, descriptions
