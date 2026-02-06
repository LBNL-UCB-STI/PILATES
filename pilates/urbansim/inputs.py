from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.urbansim import postprocessor as usim_post
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
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

    usim_data_dir = Path(workspace.get_usim_mutable_data_dir())
    usim_input_fname = usim_post.get_usim_datastore_fname(settings, io="input")
    usim_input_path = usim_data_dir / usim_input_fname

    base_path: Optional[Path] = usim_input_path if usim_input_path.exists() else None

    current_path: Optional[Path] = None
    if base_path is not None:
        current_path = base_path
    elif not state.is_start_year():
        usim_output_fname = usim_post.get_usim_datastore_fname(
            settings, io="output", year=year
        )
        usim_output_path = usim_data_dir / usim_output_fname
        if usim_output_path.exists():
            current_path = usim_output_path

    if base_path is not None:
        inputs[USIM_DATASTORE_BASE_H5] = str(base_path)
        descriptions[USIM_DATASTORE_BASE_H5] = (
            f"UrbanSim base datastore for year {year}"
        )

    if current_path is not None:
        inputs[USIM_DATASTORE_CURRENT_H5] = str(current_path)
        descriptions[USIM_DATASTORE_CURRENT_H5] = (
            f"UrbanSim current datastore for year {year}"
        )

    # If only one path exists, keep both semantics available.
    if USIM_DATASTORE_BASE_H5 not in inputs and USIM_DATASTORE_CURRENT_H5 in inputs:
        inputs[USIM_DATASTORE_BASE_H5] = inputs[USIM_DATASTORE_CURRENT_H5]
        descriptions[USIM_DATASTORE_BASE_H5] = (
            f"UrbanSim base datastore for year {year} (fallback)"
        )
    if USIM_DATASTORE_CURRENT_H5 not in inputs and USIM_DATASTORE_BASE_H5 in inputs:
        inputs[USIM_DATASTORE_CURRENT_H5] = inputs[USIM_DATASTORE_BASE_H5]
        descriptions[USIM_DATASTORE_CURRENT_H5] = (
            f"UrbanSim current datastore for year {year} (fallback)"
        )

    return inputs, descriptions
