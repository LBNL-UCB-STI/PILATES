from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from consist import (
    BindingResult,
    CacheOptions,
    define_step,
    require_runtime_kwargs,
)

from pilates.config.models import PilatesConfig
from pilates.workspace import Workspace
from pilates.utils.state_access import current_year, iteration_index
from pilates.workflows.artifact_keys import (
    BEAM_NETWORK_FINAL,
    LINKSTATS,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
)
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.coupler_namespace import coupler_storage_value
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.contracts import (
    ConfigContract,
    InputContract,
    StepDefinition,
)
from pilates.workflows.steps.shared import (
    _native_contract_output_paths,
    _native_execution_options,
)

logger = logging.getLogger(__name__)

POSTPROCESSING_SCHEMA_OUTPUTS = ["inexus_output", "mep_outputs"]

_POSTPROCESSING_INPUT_CONTRACT = InputContract(
    status="complete",
    config_contract=ConfigContract.payload(),
)


from typing import Protocol, runtime_checkable


@runtime_checkable
class _HasForecastYear(Protocol):
    forecast_year: int | None


@runtime_checkable
class _HasPostprocessing(Protocol):
    postprocessing: Any | None


def _get_forecast_year(state: Any) -> int | None:
    if isinstance(state, _HasForecastYear) and state.forecast_year is not None:
        return int(state.forecast_year)
    yr = current_year(state)
    return int(yr) if yr is not None else None


def _postprocessing_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del resolved_inputs
    out_paths: dict[str, Path] = {}
    if isinstance(settings, _HasPostprocessing) and settings.postprocessing is not None:
        post_cfg = settings.postprocessing
        year = _get_forecast_year(state) or 0
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
    if isinstance(settings, _HasPostprocessing) and settings.postprocessing is not None:
        from pilates.postprocessing.postprocessor import (
            copy_outputs_to_mep,
            process_event_file,
        )

        year = current_year(state) or 0
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
    coupler: Any = None,
    settings: Any = None,
    state: Any = None,
    workspace: Any = None,
    **_: Any,
) -> ResolvedStepInputs:
    del settings
    year = current_year(state) or 0
    iteration = iteration_index(state)

    candidate_roles = {
        "events_parquet": (f"events_parquet_{year}_{iteration}", "events_parquet"),
        "network": (BEAM_NETWORK_FINAL,),
        "linkstats": (LINKSTATS,),
        "households": ("households_asim_out", "households_asim_in"),
        "persons": ("persons_asim_out", "persons_asim_in"),
        "plans": ("beam_plans_asim_out",),
        "tours": ("tours_asim_out",),
        "trips": ("trips_asim_out",),
        "usim_datastore": (USIM_DATASTORE_CURRENT_H5, USIM_FORECAST_OUTPUT),
    }

    inputs: dict[str, Any] = {}
    source_by_role: dict[str, str] = {}
    selected_key_by_role: dict[str, str] = {}
    destinations: dict[str, Path] = {}

    for logical_role, candidates in candidate_roles.items():
        found_val = None
        found_key = None
        for cand in candidates:
            val = coupler_storage_value(coupler, cand)
            if val is not None:
                found_val = val
                found_key = cand
                break
        if found_val is not None:
            inputs[logical_role] = found_val
            source_by_role[logical_role] = "coupler"
            selected_key_by_role[logical_role] = found_key or logical_role
            if workspace is not None:
                src_p = artifact_to_path(found_val, workspace=workspace)
                if src_p:
                    destinations[logical_role] = Path(src_p)
        else:
            source_by_role[logical_role] = "missing"

    return ResolvedStepInputs(
        step_name="postprocessing",
        binding=BindingResult(inputs=inputs),
        required_roles=(),
        optional_roles=tuple(candidate_roles.keys()),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=destinations,
    )


def _project_postprocessing_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any = None,
    state: Any = None,
    workspace: Any = None,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Any]:
    del settings, state, workspace, resolved_inputs
    return dict(outputs)


def _postprocessing_cache_options(**_: Any) -> CacheOptions:
    return CacheOptions(
        validate_cached_outputs="lazy",
        cache_hydration="metadata",
    )


postprocessing = StepDefinition(
    name="postprocessing",
    function=_postprocessing_callable,
    resolve_inputs=_resolve_postprocessing_inputs,
    project_outputs=_project_postprocessing_outputs,
    input_contract=_POSTPROCESSING_INPUT_CONTRACT,
    output_paths=_postprocessing_output_paths,
    execution_options=_native_execution_options,
    cache_options=_postprocessing_cache_options,
    optional_output_keys=tuple(POSTPROCESSING_SCHEMA_OUTPUTS),
    archive_outputs=True,
)
