from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from consist import (
    define_step,
    require_runtime_kwargs,
)

from pilates.config.models import PilatesConfig, SettingsLike
from pilates.workspace import Workspace
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import current_year, iteration_index
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.contracts import (
    ConfigContract,
    InputContract,
    StepCallInputs,
    StepDefinition,
)
from pilates.workflows.sim_state import ArtifactRef, SimState
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.steps.shared import (
    _build_step_call_inputs,
    _metadata_cache_options,
    _native_contract_output_paths,
    _native_execution_options,
)

logger = logging.getLogger(__name__)

POSTPROCESSING_SCHEMA_OUTPUTS = ["inexus_output", "mep_outputs"]

_POSTPROCESSING_INPUT_CONTRACT = InputContract(
    status="complete",
    config_contract=ConfigContract.payload(),
)


def _postprocessing_output_paths(
    *,
    settings: SettingsLike,
    state: Any,
    workspace: Any,
    step_inputs: StepCallInputs | None = None,
) -> dict[str, Path]:
    del step_inputs
    out_paths: dict[str, Path] = {}
    post_cfg = get_setting(settings, "postprocessing")
    if post_cfg is not None:
        year = resolve_forecast_year(state) or 0
        year_str = str(year)

        mep_folder = Path(post_cfg.mep_output_folder)
        if not mep_folder.is_absolute():
            mep_folder = Path(workspace.full_path) / mep_folder
        out_paths["mep_outputs"] = mep_folder / year_str

        out_folder = Path(post_cfg.output_folder)
        if not out_folder.is_absolute():
            out_folder = Path(workspace.full_path) / out_folder
        iter_idx = iteration_index(state)
        out_paths["inexus_output"] = (
            out_folder / f"inexus_output_{year_str}_iter{iter_idx}.csv.gz"
        )
    return out_paths


@define_step(
    model="postprocessing",
    name_template="postprocessing__y{year}__i{iteration}__phase_{phase}",
    input_binding="paths",
    tags=["postprocessing"],
    inputs={},
    optional_input_keys=(
        "events_parquet",
        "network",
        "linkstats",
        "households",
        "persons",
        "plans",
        "tours",
        "trips",
        "usim_datastore",
    ),
    schema_outputs=POSTPROCESSING_SCHEMA_OUTPUTS,
    outputs=_native_contract_output_paths(_postprocessing_output_paths),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _postprocessing_callable(
    *, settings: PilatesConfig, state: Any, workspace: Workspace, **kwargs: Any
) -> None:
    del kwargs
    if get_setting(settings, "postprocessing") is not None:
        from pilates.postprocessing.postprocessor import (
            copy_outputs_to_mep,
            process_event_file,
        )

        year = resolve_forecast_year(state) or current_year(state) or 0
        iter_idx = iteration_index(state)
        declared = _postprocessing_output_paths(
            settings=settings, state=state, workspace=workspace
        )
        inexus_target = declared.get("inexus_output")
        if inexus_target is not None:
            inexus_target.parent.mkdir(parents=True, exist_ok=True)
        mep_target = declared.get("mep_outputs")
        if mep_target is not None:
            mep_target.mkdir(parents=True, exist_ok=True)

        process_event_file(
            settings,
            year,
            iter_idx,
            output_path=inexus_target,
        )
        copy_outputs_to_mep(settings, year, iter_idx, workspace)


def _resolve_postprocessing_inputs(
    *,
    sim_state: SimState | None = None,
    settings: SettingsLike | None = None,
    state: Any = None,
    workspace: Any = None,
    **_: Any,
) -> StepCallInputs:
    del settings, state
    active_sim_state = sim_state if sim_state is not None else SimState()
    candidates: dict[str, ArtifactRef | None] = {
        "events_parquet": active_sim_state.network.events_parquet,
        "network": active_sim_state.network.network_csv,
        "linkstats": active_sim_state.network.linkstats,
        "households": (
            active_sim_state.demand.households
            if active_sim_state.demand.households is not None
            else active_sim_state.demand.households_in
        ),
        "persons": (
            active_sim_state.demand.persons
            if active_sim_state.demand.persons is not None
            else active_sim_state.demand.persons_in
        ),
        "plans": active_sim_state.demand.plans,
        "tours": active_sim_state.demand.tours,
        "trips": active_sim_state.demand.trips,
        "usim_datastore": (
            active_sim_state.land_use.current_h5
            if active_sim_state.land_use.current_h5 is not None
            else active_sim_state.land_use.forecast_output
        ),
    }

    destinations: dict[str, Path] = {}
    if workspace is not None:
        for role, val in candidates.items():
            if val is not None:
                src_p = artifact_to_path(val, workspace=workspace)
                if src_p:
                    destinations[role] = Path(src_p)

    return _build_step_call_inputs(
        candidates=candidates,
        required_roles=(),
        optional_roles=tuple(candidates.keys()),
        logical_destinations=destinations,
    )


def _project_postprocessing_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: SettingsLike | None = None,
    state: Any = None,
    workspace: Any = None,
    step_inputs: StepCallInputs | None = None,
) -> dict[str, Any]:
    del settings, state, workspace, step_inputs
    return dict(outputs)


postprocessing = StepDefinition(
    name="postprocessing",
    function=_postprocessing_callable,
    resolve_inputs=_resolve_postprocessing_inputs,
    project_outputs=_project_postprocessing_outputs,
    input_contract=_POSTPROCESSING_INPUT_CONTRACT,
    output_paths=_postprocessing_output_paths,
    execution_options=_native_execution_options,
    cache_options=_metadata_cache_options(),
    optional_output_keys=tuple(POSTPROCESSING_SCHEMA_OUTPUTS),
    archive_outputs=True,
)
