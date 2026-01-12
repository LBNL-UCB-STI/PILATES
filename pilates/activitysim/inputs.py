from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.utils.coupler_helpers import artifact_to_path

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


def build_activitysim_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    iteration: int,
    coupler: Any,
    usim_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Optional[str]]]:
    """
    Build ActivitySim input paths from available sources.

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
    iteration : int
        Current supply-demand iteration.
    coupler : object
        Consist coupler or compatible interface.
    usim_inputs : dict, optional
        UrbanSim inputs mapping, used to source the datastore path.

    Returns
    -------
    tuple of dict
        (inputs, descriptions) where descriptions are per-key log strings.
        Descriptions may be None to skip logging for a given input.
    """
    inputs: Dict[str, Any] = {}
    descriptions: Dict[str, Optional[str]] = {}

    asim_data_dir = Path(workspace.get_asim_mutable_data_dir())
    if asim_data_dir.exists():
        inputs["asim_mutable_data_dir"] = str(asim_data_dir)
        descriptions["asim_mutable_data_dir"] = (
            f"ActivitySim mutable data dir for year {year}, iter {iteration}"
        )

    if usim_inputs and "usim_datastore_h5" in usim_inputs:
        inputs["usim_datastore_h5"] = usim_inputs["usim_datastore_h5"]
        descriptions["usim_datastore_h5"] = (
            f"UrbanSim datastore for ActivitySim year {year}, iter {iteration}"
        )

    zarr_skims_input = None
    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        zarr_skims_input = get_value("zarr_skims")
    if zarr_skims_input:
        inputs["zarr_skims"] = zarr_skims_input
        zarr_skims_path = artifact_to_path(zarr_skims_input, workspace)
        if zarr_skims_path:
            descriptions["zarr_skims"] = (
                f"ActivitySim compiled zarr skims for year {year}, iter {iteration}"
            )
        else:
            descriptions["zarr_skims"] = None

    return inputs, descriptions
