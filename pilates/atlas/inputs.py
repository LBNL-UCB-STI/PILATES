from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.generic.records import sanitize_artifact_key
from pilates.workflows.artifact_constants import USIM_DATASTORE_H5
from pilates.atlas.static_inputs import (
    ATLAS_STATIC_INPUTS_COMMON,
    ATLAS_STATIC_INPUTS_BY_SCENARIO,
)

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


def build_atlas_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    coupler: CouplerProtocol,
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
    coupler : CouplerProtocol
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

    inputs[USIM_DATASTORE_H5] = atlas_usim_input

    descriptions[USIM_DATASTORE_H5] = f"UrbanSim datastore for ATLAS year {year}"

    return inputs, descriptions


def atlas_static_input_keys(settings: PilatesConfig) -> Tuple[str, ...]:
    scenario = getattr(settings, "atlas", None)
    scenario_name = None
    if scenario is not None:
        scenario_name = getattr(settings.atlas, "scenario", None)
    scenario_key = str(scenario_name).lower() if scenario_name else None

    vehicle_type_mapping_by_scenario = {
        "baseline": "vehicle_type_mapping_baseline.csv",
        "ess_cons": "vehicle_type_mapping_ESS_const_220_price.csv",
        "zev_mandate": "vehicle_type_mapping_evMandForced2.csv",
    }

    relpaths = [
        relpath
        for relpath in ATLAS_STATIC_INPUTS_COMMON
        if not relpath.startswith("vehicle_type_mapping_")
    ]
    selected_mapping = vehicle_type_mapping_by_scenario.get(scenario_key)
    if selected_mapping is not None:
        relpaths.append(selected_mapping)

    if scenario_key:
        relpaths.extend(ATLAS_STATIC_INPUTS_BY_SCENARIO.get(scenario_key, []))

    keys = []
    for relpath in relpaths:
        rel_no_ext = relpath.rsplit(".", 1)[0]
        candidate = rel_no_ext.replace("/", "_")
        key = sanitize_artifact_key(candidate) or candidate
        keys.append(key)
    return tuple(keys)
