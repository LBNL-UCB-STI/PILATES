from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


def build_beam_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    iteration: int,
    coupler: Any,
    activity_demand_outputs: Optional[Any] = None,
    previous_beam_outputs: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """
    Build BEAM input mappings and optional mutable data directory log details.

    Parameters
    ----------
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    year : int
        Simulation year for labeling.
    iteration : int
        Current supply-demand iteration.
    coupler : object
        Consist coupler or compatible interface.
    activity_demand_outputs : object, optional
        ActivitySim outputs holder with a ``to_mapping()`` method.
    previous_beam_outputs : object, optional
        Previous BEAM outputs holder with a ``to_mapping()`` method.

    Returns
    -------
    tuple
        (inputs, beam_mutable_dir, beam_mutable_description). The mutable
        directory values are None when no directory exists.
    """
    inputs: Dict[str, Any] = {}

    if activity_demand_outputs:
        inputs.update(activity_demand_outputs.to_mapping())
    if previous_beam_outputs:
        inputs.update(previous_beam_outputs.to_mapping())

    zarr_skims_input = None
    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        zarr_skims_input = get_value("zarr_skims")
    if zarr_skims_input:
        inputs["zarr_skims"] = zarr_skims_input

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    if beam_mutable_dir.exists():
        return (
            inputs,
            str(beam_mutable_dir),
            f"BEAM mutable data dir for year {year}, iter {iteration}",
        )
    return inputs, None, None
