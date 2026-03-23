from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.workflows.binding import ArtifactBindingRule, build_binding_plan
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
    from workflow_state import WorkflowState

def build_urbansim_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
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
          exogenous baseline input for the run year (H5).
        - ``usim_datastore_h5``: UrbanSim current mutable datastore used by
          active workflow steps for this year (H5).
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
        artifact_rules=(
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_BASE_H5,
                required=True,
                allow_fallback=True,
                preferred_keys=(
                    USIM_DATASTORE_BASE_H5,
                    USIM_DATASTORE_CURRENT_H5,
                ),
                fallback_provider="urbansim_inputs_for_year",
            ),
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_CURRENT_H5,
                required=True,
                allow_fallback=True,
                preferred_keys=(
                    USIM_DATASTORE_CURRENT_H5,
                    USIM_DATASTORE_BASE_H5,
                ),
                fallback_provider="urbansim_inputs_for_year",
            ),
        ),
        required_keys=[USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5],
    )

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
