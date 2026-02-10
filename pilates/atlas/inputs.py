import re
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.generic.records import sanitize_artifact_key
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
)
from pilates.workflows.input_resolution import (
    first_resolved_key,
    resolve_preferred_step_input,
    resolve_step_inputs,
    resolved_value_for_key,
)
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
        - ``usim_datastore_h5``: UrbanSim current datastore used as the land
          use input for ATLAS scenario generation.
        - ``usim_datastore_base_h5``: UrbanSim base datastore for the run year.
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

    atlas_current_resolution = resolve_preferred_step_input(
        preferred_keys=[USIM_DATASTORE_CURRENT_H5, USIM_H5_UPDATED],
        coupler=coupler,
    )
    selected_current_key = first_resolved_key(
        atlas_current_resolution,
        [USIM_DATASTORE_CURRENT_H5, USIM_H5_UPDATED],
    )
    if selected_current_key is None and usim_datastore_h5_path is not None:
        atlas_current_resolution = resolve_step_inputs(
            keys=[USIM_DATASTORE_CURRENT_H5],
            fallback_inputs={USIM_DATASTORE_CURRENT_H5: usim_datastore_h5_path},
        )
        selected_current_key = USIM_DATASTORE_CURRENT_H5

    atlas_usim_input = (
        resolved_value_for_key(
            resolved=atlas_current_resolution,
            key=selected_current_key,
            coupler=coupler,
        )
        if selected_current_key is not None
        else None
    )

    atlas_base_resolution = resolve_step_inputs(
        keys=[USIM_DATASTORE_BASE_H5],
        coupler=coupler,
        fallback_inputs={USIM_DATASTORE_BASE_H5: atlas_usim_input}
        if atlas_usim_input is not None
        else None,
    )
    atlas_base_input = resolved_value_for_key(
        resolved=atlas_base_resolution,
        key=USIM_DATASTORE_BASE_H5,
        coupler=coupler,
    )
    if atlas_base_input is None:
        atlas_base_input = atlas_usim_input

    inputs[USIM_DATASTORE_CURRENT_H5] = atlas_usim_input
    inputs[USIM_DATASTORE_BASE_H5] = atlas_base_input

    descriptions[USIM_DATASTORE_CURRENT_H5] = (
        f"UrbanSim current datastore for ATLAS year {year}"
    )
    descriptions[USIM_DATASTORE_BASE_H5] = (
        f"UrbanSim base datastore for ATLAS year {year}"
    )

    return inputs, descriptions


_YEAR_SUFFIX = re.compile(r"_(\d{4})$")


def atlas_run_years(settings: PilatesConfig) -> Set[int]:
    """
    Determine ATLAS run years from global run configuration.

    ATLAS sub-runs advance in biannual cadence between configured run bounds.
    This is intentionally independent of ``run.vehicle_ownership_freq``, which
    controls higher-level stage cadence, not ATLAS internal sub-year steps.
    """
    run_cfg = getattr(settings, "run", None)
    if run_cfg is None:
        return set()
    start_year = getattr(run_cfg, "start_year", None)
    end_year = getattr(run_cfg, "end_year", None)
    if start_year is None or end_year is None:
        return set()
    return set(range(int(start_year), int(end_year) + 1, 2))


def atlas_static_input_relpaths(settings: PilatesConfig) -> Tuple[str, ...]:
    """
    Deterministic static ATLAS input relpaths based on settings.

    Rules align with `AtlasPreprocessor.copy_data_to_mutable_location()`:
    - scenario-specific adopt folder selection by `settings.atlas.scenario`
    - one selected `vehicle_type_mapping_*` file for known scenarios
    - unknown/no scenario falls back to all mapping files
    - year-stamped non-ADOPT files may be limited to configured ATLAS run years
    """
    scenario_name = getattr(getattr(settings, "atlas", None), "scenario", None)
    scenario_key = str(scenario_name).lower() if scenario_name else None

    vehicle_type_mapping_by_scenario = {
        "baseline": "vehicle_type_mapping_baseline.csv",
        "ess_cons": "vehicle_type_mapping_ESS_const_220_price.csv",
        "zev_mandate": "vehicle_type_mapping_evMandForced2.csv",
    }
    all_vehicle_mappings = [
        relpath
        for relpath in ATLAS_STATIC_INPUTS_COMMON
        if relpath.startswith("vehicle_type_mapping_")
    ]

    relpaths = [
        relpath
        for relpath in ATLAS_STATIC_INPUTS_COMMON
        if not relpath.startswith("vehicle_type_mapping_")
    ]
    selected_mapping = (
        vehicle_type_mapping_by_scenario.get(scenario_key) if scenario_key else None
    )
    if selected_mapping is not None:
        relpaths.append(selected_mapping)
    else:
        relpaths.extend(all_vehicle_mappings)

    if scenario_key:
        relpaths.extend(ATLAS_STATIC_INPUTS_BY_SCENARIO.get(scenario_key, []))
    else:
        for values in ATLAS_STATIC_INPUTS_BY_SCENARIO.values():
            relpaths.extend(values)

    run_years = atlas_run_years(settings)
    if run_years:
        filtered = []
        for relpath in relpaths:
            # ADOPT files are year-stamped snapshots that ATLAS may read from
            # neighboring years (for example, outyear 2021 needs *_2019 files).
            # Do not prune them by run-frequency year selection.
            if relpath.replace("\\", "/").startswith("adopt/"):
                filtered.append(relpath)
                continue
            rel_no_ext = relpath.rsplit(".", 1)[0]
            rel_key = rel_no_ext.replace("/", "_")
            match = _YEAR_SUFFIX.search(rel_key)
            if match and int(match.group(1)) not in run_years:
                continue
            filtered.append(relpath)
        relpaths = filtered

    # Preserve order while removing duplicates.
    seen = set()
    deduped = []
    for relpath in relpaths:
        if relpath in seen:
            continue
        seen.add(relpath)
        deduped.append(relpath)
    return tuple(deduped)


def atlas_static_input_keys(settings: PilatesConfig) -> Tuple[str, ...]:
    relpaths = atlas_static_input_relpaths(settings)

    keys = []
    for relpath in relpaths:
        rel_no_ext = relpath.rsplit(".", 1)[0]
        candidate = rel_no_ext.replace("\\", "/")
        key = sanitize_artifact_key(candidate) or candidate
        keys.append(key)
    return tuple(keys)
