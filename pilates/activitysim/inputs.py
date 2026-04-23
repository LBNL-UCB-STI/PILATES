import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    resolve_artifact_from_value,
    log_coupler_value,
)
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.binding import (
    activitysim_population_source_selection_rules,
    build_binding_plan,
)
from pilates.workflows.input_resolution import (
    resolved_value_for_key,
    selected_candidate_key,
)

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState
    from pilates.workflows.surface import EnabledWorkflowSurface


def build_activitysim_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    year: int,
    iteration: int,
    coupler: CouplerProtocol,
    usim_inputs: Optional[Dict[str, Any]] = None,
    *,
    include_omx_skims: bool = False,
    surface: "EnabledWorkflowSurface",
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
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    usim_inputs : dict, optional
        UrbanSim inputs mapping, used to source the datastore path.

    Returns
    -------
    tuple of dict
        (inputs, descriptions) where descriptions are per-key log strings.
        Descriptions may be None to skip logging for a given input.

    Notes
    -----
    Input keys
        - ``asim_mutable_data_dir``: ActivitySim input data/config directory
          containing household/person tables, land use, and settings files.
        - ``usim_population_source_h5``: UrbanSim datastore path that
          ActivitySim uses to read land use and demographic inputs (H5).
        - ``zarr_skims``: Travel time skims in Zarr format produced by
          ActivitySim compilation or updated by BEAM.
    Related outputs
        - ActivitySim compile produces ``zarr_skims`` for downstream runs.
        - ActivitySim main produces ``asim_output_dir`` and refreshes
          ``usim_datastore_h5`` via postprocessing for UrbanSim/ATLAS.
        - TODO: Document other ActivitySim outputs (e.g., in-memory demand
          stores) and confirm coupler keys expected by downstream steps.
    """
    inputs: Dict[str, Any] = {}
    descriptions: Dict[str, Optional[str]] = {}

    asim_data_dir = Path(workspace.get_asim_mutable_data_dir())
    if asim_data_dir.exists():
        explicit_inputs = {
            ASIM_HOUSEHOLDS_IN: "households.csv",
            ASIM_PERSONS_IN: "persons.csv",
            ASIM_LAND_USE_IN: "land_use.csv",
        }
        for key, filename in explicit_inputs.items():
            file_path = asim_data_dir / filename
            if file_path.exists():
                inputs[key] = str(file_path)
                descriptions[key] = f"ActivitySim input {filename} for year {year}"

        if include_omx_skims:
            omx_path = asim_data_dir / "skims.omx"
            if omx_path.exists():
                inputs[ASIM_OMX_SKIMS] = str(omx_path)
                descriptions[ASIM_OMX_SKIMS] = (
                    f"ActivitySim compile input skims (OMX) for year {year}"
                )

    land_use_enabled = surface.profile.land_use_enabled

    explicit_usim_inputs: Dict[str, Any] = {}
    if usim_inputs and not land_use_enabled:
        population_source = (
            usim_inputs.get(USIM_DATASTORE_BASE_H5)
            or usim_inputs.get(USIM_DATASTORE_CURRENT_H5)
        )
        if population_source is not None:
            explicit_usim_inputs[USIM_POPULATION_SOURCE_H5] = population_source

    usim_resolution = build_binding_plan(
        step_name="activitysim_input_selection",
        coupler=coupler,
        explicit_inputs=explicit_usim_inputs or usim_inputs,
        artifact_rules=activitysim_population_source_selection_rules(),
        required_keys=[USIM_POPULATION_SOURCE_H5],
        fallback_inputs=usim_inputs,
        surface=surface,
    )
    selected_usim_key = selected_candidate_key(
        usim_resolution,
        USIM_POPULATION_SOURCE_H5,
    )
    if selected_usim_key is not None:
        usim_value = resolved_value_for_key(
            resolved=usim_resolution,
            key=USIM_POPULATION_SOURCE_H5,
            coupler=coupler,
        )
        if usim_value is not None:
            inputs[USIM_POPULATION_SOURCE_H5] = usim_value
            descriptions[USIM_POPULATION_SOURCE_H5] = {
                USIM_POPULATION_SOURCE_H5: (
                    f"UrbanSim population-source datastore for ActivitySim year {year}, iter {iteration}"
                ),
                USIM_FORECAST_OUTPUT: (
                    f"UrbanSim forecast datastore used as the ActivitySim population source for year {year}, iter {iteration}"
                ),
                USIM_DATASTORE_CURRENT_H5: (
                    f"UrbanSim current datastore used as the ActivitySim population source for year {year}, iter {iteration}"
                ),
                USIM_DATASTORE_BASE_H5: (
                    f"UrbanSim base datastore used as the ActivitySim population source for year {year}, iter {iteration}"
                ),
            }.get(
                selected_usim_key,
                f"UrbanSim datastore used as the ActivitySim population source for year {year}, iter {iteration}",
            )

    zarr_skims_input = None
    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        zarr_skims_input = get_value(ZARR_SKIMS)
        log_coupler_value(
            key=ZARR_SKIMS,
            value=zarr_skims_input,
            workspace=workspace,
            context="activitysim_inputs.get",
        )
        zarr_skims_input = resolve_artifact_from_value(
            zarr_skims_input, key=ZARR_SKIMS, workspace=workspace
        )
        set_from_artifact = getattr(coupler, "set_from_artifact", None)
        if callable(set_from_artifact):
            set_from_artifact(ZARR_SKIMS, zarr_skims_input)
            log_coupler_value(
                key=ZARR_SKIMS,
                value=zarr_skims_input,
                workspace=workspace,
                context="activitysim_inputs.set",
            )
    if zarr_skims_input:
        inputs[ZARR_SKIMS] = zarr_skims_input
        zarr_skims_path = artifact_to_path(zarr_skims_input, workspace)
        if zarr_skims_path:
            descriptions[ZARR_SKIMS] = (
                f"ActivitySim compiled zarr skims for year {year}, iter {iteration}"
            )
        else:
            descriptions[ZARR_SKIMS] = None

    return inputs, descriptions
