from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.workflows.artifact_constants import USIM_DATASTORE_H5

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


def build_atlas_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    coupler: Any,
    usim_datastore_h5_path: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build ATLAS input paths and per-key log descriptions for a sub-year.

    Parameters
    ----------
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    year : int
        ATLAS sub-year for labeling.
    coupler : object
        Consist coupler or compatible interface.
    usim_datastore_h5_path : str, optional
        Fallback UrbanSim datastore path when coupler has no value.

    Returns
    -------
    tuple of dict
        (inputs, descriptions) where descriptions are per-key log strings.

    Notes
    -----
    Input keys
        - ``usim_datastore_h5``: UrbanSim datastore used as the land use input
          for ATLAS scenario generation.
        - ``atlas_mutable_input_dir``: ATLAS mutable input directory (configs
          and data mounted into the container).
    Related outputs
        - ATLAS produces ``atlas_output_dir`` and may update
          ``usim_datastore_h5`` for subsequent model stages.
        - TODO: Document ATLAS outputs that should flow into downstream models
          or be logged as explicit expected outputs.
    """
    inputs: Dict[str, Any] = {}
    descriptions: Dict[str, str] = {}

    atlas_usim_input = None
    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        atlas_usim_input = get_value(USIM_DATASTORE_H5)
    if atlas_usim_input is None:
        atlas_usim_input = usim_datastore_h5_path

    atlas_mutable_input_dir = Path(workspace.get_atlas_mutable_input_dir())

    inputs[USIM_DATASTORE_H5] = atlas_usim_input
    inputs["atlas_mutable_input_dir"] = str(atlas_mutable_input_dir)

    descriptions[USIM_DATASTORE_H5] = f"UrbanSim datastore for ATLAS year {year}"
    descriptions["atlas_mutable_input_dir"] = "ATLAS mutable input directory"

    return inputs, descriptions
