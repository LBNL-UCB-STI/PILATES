from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.urbansim import postprocessor as usim_post

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
    """
    inputs: Dict[str, Any] = {}
    descriptions: Dict[str, str] = {}

    usim_data_dir = Path(workspace.get_usim_mutable_data_dir())
    usim_input_fname = usim_post.get_usim_datastore_fname(settings, io="input")
    usim_input_path = usim_data_dir / usim_input_fname

    if usim_input_path.exists():
        inputs["usim_datastore_h5"] = str(usim_input_path)
    elif not state.is_start_year():
        usim_output_fname = usim_post.get_usim_datastore_fname(
            settings, io="output", year=year
        )
        usim_output_path = usim_data_dir / usim_output_fname
        if usim_output_path.exists():
            inputs["usim_datastore_h5"] = str(usim_output_path)

    if usim_data_dir.exists():
        inputs["usim_mutable_data_dir"] = str(usim_data_dir)

    if "usim_datastore_h5" in inputs:
        descriptions["usim_datastore_h5"] = f"UrbanSim input datastore for year {year}"
    if "usim_mutable_data_dir" in inputs:
        descriptions["usim_mutable_data_dir"] = "UrbanSim mutable data directory"

    return inputs, descriptions
