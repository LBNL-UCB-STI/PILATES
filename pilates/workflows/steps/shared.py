from __future__ import annotations

# Coupler IO map (manual reference, update when wiring changes).
#
# Step                           Coupler inputs (input_keys)                                 Coupler outputs (keys written)
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# initialization                 (none)                                                      UrbanSim init outputs:
#                                                                                              - usim_datastore_h5
#                                                                                              - omx_skims
#                                                                                              - hh_size
#                                                                                              - income_rates
#                                                                                              - relmap
#                                                                                              - schools
#                                                                                              - school_districts
#
#                                                                                              ActivitySim init outputs:
#                                                                                              - canonical_zones
#                                                                                              - clipped_geoms (if exists)
#                                                                                              - (configs tracked via ActivitySim config adapter)
#
#                                                                                              ATLAS init outputs:
#                                                                                              - one key per non-readme file copied from
#                                                                                                atlas.host_input_folder (or pilates/atlas/atlas_input)
#                                                                                                after scenario filtering. Key is sanitized relative path.
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# urbansim_run                    usim_datastore_h5 + final_skims_omx (when required)          Raw outputs:
#                                                                                              - usim_forecast_output
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (derived from usim_forecast_output)
#
# urbansim_postprocess            usim_datastore_h5                                            Processed outputs:
#                                                                                              - usim_input_archive_<year>
#                                                                                              - usim_input_merged_<year>
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (mapped from usim_input_merged_<year>)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# atlas_preprocess                (none)                                                      (no coupler outputs logged in this step)
#
# atlas_run                       usim_datastore_h5                                            Raw outputs:
#                               + all static atlas input keys (if present)                    - one key per ATLAS CSV filename stem
#                               (input_keys)                                                    from expected_output_paths
#
#   Atlas static input keys (explicit; wildcards denote scenario/year variants):
#   Common (always eligible):
#   - accessbility2017
#   - accessbility_2015
#   - cpi
#   - modeaccessibility
#   - psid_names
#   - sfb_baseline
#   - taz_to_tract_sfbay
#   - vehicle_type_mapping_ESS_const_220_price (only if scenario=ess_cons)
#   - vehicle_type_mapping_baseline (only if scenario=baseline)
#   - vehicle_type_mapping_evMandForced2 (only if scenario=zev_mandate)
#
#   Scenario-specific (adopt/<scenario>/...):
#   - adopt_<scenario>_new_vehicle_annual_medians
#   - adopt_<scenario>_new_vehicle_representative_vehicle
#   - adopt_<scenario>_new_vehicles
#   - adopt_<scenario>_new_vehicles_biannual_values_<year>
#   - adopt_<scenario>_used_vehicles
#   - adopt_<scenario>_used_vehicles_<year>
#
# atlas_postprocess               atlas_run raw outputs (all keys above)                       Processed outputs:
#                               + usim_datastore_h5 (forecast datastore read directly)        - atlas_vehicles2_output
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if updated H5 exists)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# activitysim_preprocess          input_keys: usim_h5_updated (if present) OR usim_datastore_h5 Outputs:
#                               inputs (fallback): usim_datastore_h5 (path)                    - asim_land_use_in
#                                                + asim_mutable_configs_dir                     - asim_households_in
#                                                                                              - asim_persons_in
#                                                                                              - asim_omx_skims (if present)
#
# activitysim_run                 activitysim_preprocess outputs + OMX or zarr skims           Raw outputs (parquet allowlist; keys as listed):
#                                                                                              - households
#                                                                                              - persons
#                                                                                              - land_use
#                                                                                              - tours
#                                                                                              - trips
#                                                                                              - joint_tour_participants
#                                                                                              - person_windows
#                                                                                              - disaggregate_accessibility
#                                                                                              - proto_households
#                                                                                              - proto_persons
#                                                                                              - proto_persons_merged
#                                                                                              - proto_tours
#                                                                                              - proto_disaggregate_accessibility
#                                                                                              - school_destination_size
#                                                                                              - school_modeled_size
#                                                                                              - school_shadow_prices
#                                                                                              - workplace_destination_size
#                                                                                              - workplace_location_accessibility
#                                                                                              - workplace_modeled_size
#                                                                                              - workplace_shadow_prices
#
# activitysim_postprocess         activitysim_run raw outputs (all keys above)                 Processed outputs:
#                                                                                              - same allowlist as activitysim_run
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if updated H5 exists)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# beam_preprocess                 (none)                                                      Prepared inputs (from BEAM preprocessor):
#                                                                                              - BEAM_PLANS_IN
#                                                                                              - BEAM_HOUSEHOLDS_IN
#                                                                                              - BEAM_PERSONS_IN
#                                                                                              - LINKSTATS_WARMSTART
#                                                                                              - vehicles_beam_in (derived from ATLAS vehicles2)
#                                                                                              - plus any {file_stem}_beam_in created by
#                                                                                                preprocessor for other copied files
#
# beam_run                        beam_preprocess outputs (all keys above)                    Raw outputs (keys are base names below, with
#                                                                                              suffix _<year>_<iteration> and optional _sub<it>):
#                                                                                              Iteration-scoped outputs (files_to_get):
#                                                                                              - raw_od_skims
#                                                                                              - raw_od_skims_zarr
#                                                                                              - raw_origin_skims
#                                                                                              - linkstats
#                                                                                              - linkstats_unmodified
#                                                                                              - linkstats_parquet
#                                                                                              - linkstats_unmodified_parquet
#                                                                                              - beam_plans_out
#                                                                                              - beam_plans_xml
#                                                                                              - beam_experienced_plans_xml
#                                                                                              - beam_experienced_plans_scores
#                                                                                              - events
#                                                                                              - events_parquet
#                                                                                              - legs
#                                                                                              - route_history
#                                                                                              - final_vehicles
#                                                                                              - skims_taz
#                                                                                              - skims_taz_agg
#                                                                                              - skims_od
#                                                                                              - skims_od_agg
#                                                                                              - skims_od_vehicle_type
#                                                                                              - skims_od_vehicle_type_agg
#                                                                                              - skims_emissions
#                                                                                              - skims_emissions_agg
#                                                                                              - skims_ridehail_agg
#                                                                                              - skims_parking
#                                                                                              - skims_parking_agg
#                                                                                              - skims_transit_crowding
#                                                                                              - skims_transit_crowding_agg
#                                                                                              - skims_freight
#                                                                                              - skims_freight_agg
#                                                                                              - skims_travel_time_obs_sim
#                                                                                              - skims_travel_time_obs_sim_agg
#
#                                                                                              Top-level outputs (top_level_files):
#                                                                                              - beam_plans_final
#                                                                                              - beam_vehicles_final
#                                                                                              - beam_households_final
#                                                                                              - beam_persons_final
#                                                                                              - beam_population_final
#                                                                                              - beam_network_final
#                                                                                              - beam_output_plans_xml
#                                                                                              - beam_output_experienced_plans_xml
#                                                                                              - beam_output_vehicles_xml
#                                                                                              - beam_output_households_xml
#                                                                                              - beam_output_facilities_xml
#                                                                                              - beam_output_network_xml
#                                                                                              - beam_output_counts_xml
#
# beam_postprocess                selected beam_run outputs used by postprocessor:             Outputs:
#                               - events_parquet_<year>_<iter>[ _sub* ]                       - final_skims_omx OR zarr_skims
#                               - raw_od_skims_<year>_<iter>[ _sub* ]                         - linkstats (promoted latest)
#                               - raw_od_skims_zarr_<year>_<iter>[ _sub* ]                    - beam_plans_out (promoted latest)
#                               + zarr_skims (if present)
#
import inspect as pyinspect
import logging
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
)

from pilates.utils.beam_warmstart import (
    find_last_run_output_plans as find_last_run_output_plans,
)
from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,  # noqa: F401
    ActivitySimPreprocessOutputs,  # noqa: F401
    ActivitySimRunOutputs,  # noqa: F401
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,  # noqa: F401
    AtlasPreprocessOutputs,  # noqa: F401
    AtlasRunOutputs,  # noqa: F401
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,  # noqa: F401
    BeamPostprocessOutputs,  # noqa: F401
    BeamPreprocessOutputs,  # noqa: F401
    BeamRunOutputs,  # noqa: F401
)
from pilates.utils.consist_types import CouplerProtocol  # noqa: F401
from pilates.utils.coupler_helpers import (
    artifact_to_path,  # noqa: F401
    log_and_set_input as log_and_set_input,
    log_and_set_output as log_and_set_output,
    log_input_only as log_input_only,
    log_output_only as log_output_only,
    resolve_artifact_from_value as resolve_artifact_from_value,
)
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN as ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN as ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS as ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN as ASIM_PERSONS_IN,
    BEAM_EXPERIENCED_PLANS_XML as BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML as BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML as BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT as BEAM_PLANS_OUT,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5 as USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED as USIM_H5_UPDATED,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    USIM_FORECAST_OUTPUT,
    ZARR_SKIMS,
)
from pilates.workflows.step_exec import warm_start_activities as warm_start_activities
from pilates.workflows.catalog import (
    WORKFLOW_STEP_SPECS,
    workflow_step_key_match,
    workflow_step_spec_for_step_name,
)
from workflow_state import WorkflowState  # noqa: F401

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _log_step_records(
    *,
    record_items: Any,
    log_fn: Callable[..., Any],
    profile_schema_keys: Optional[set[str]] = None,
    profile_schema_suffixes: tuple[str, ...] = (),
    profile_schema_value: Any = True,
    extra_meta_fn: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
) -> None:
    """
    Log `(key, path, description)` record triples with optional schema profiling.

    Parameters
    ----------
    record_items : iterable
        Iterable yielding `(short_name, path, description)`.
    log_fn : callable
        Logging function accepting `key`, `path`, `description`, and `**meta`.
    profile_schema_keys : set[str], optional
        Keys for which `profile_file_schema` should be included.
    profile_schema_suffixes : tuple[str, ...], optional
        Path suffixes that should trigger `profile_file_schema`.
    profile_schema_value : Any, optional
        Value assigned to `profile_file_schema` when triggered.
    extra_meta_fn : callable, optional
        Callback returning additional metadata per record.
    """
    profile_schema_keys = profile_schema_keys or set()
    for short_name, path, description in record_items:
        path_str = str(path)
        meta: Dict[str, Any] = {}
        if short_name in profile_schema_keys or (
            profile_schema_suffixes and path_str.endswith(profile_schema_suffixes)
        ):
            meta["profile_file_schema"] = profile_schema_value
        if extra_meta_fn is not None:
            extra_meta = extra_meta_fn(short_name, path_str, description)
            if extra_meta:
                meta.update(extra_meta)
        log_fn(
            key=short_name,
            path=path_str,
            description=description,
            **meta,
        )


def _parse_prefixed_iteration_key(
    short_name: str, prefix: str
) -> Optional[Dict[str, Any]]:
    marker = f"{prefix}_"
    if not short_name.startswith(marker):
        return None
    tail = short_name[len(marker) :]
    parts = tail.split("_")
    if len(parts) < 2:
        return None
    try:
        year = int(parts[0])
        iteration = int(parts[1])
    except ValueError:
        return None
    payload: Dict[str, Any] = {
        "year": year,
        "iteration": iteration,
    }
    if len(parts) > 2 and parts[2].startswith("sub"):
        try:
            payload["beam_sub_iteration"] = int(parts[2][3:])
        except ValueError:
            pass
    return payload


def _beam_artifact_facets(short_name: str) -> Optional[Dict[str, Any]]:
    for prefix, family in (
        ("events_parquet", "events_parquet"),
        ("raw_od_skims", "raw_od_skims"),
        ("raw_od_skims_zarr", "raw_od_skims_zarr"),
        ("linkstats_parquet", "linkstats_parquet"),
    ):
        parsed = _parse_prefixed_iteration_key(short_name, prefix)
        if parsed is not None:
            return {"artifact_family": family, **parsed}

    if short_name.startswith("linkstats_unmodified_parquet__"):
        tokens = short_name.split("__")
        payload: Dict[str, Any] = {
            "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet"
        }
        for token in tokens[1:]:
            if token.startswith("y"):
                try:
                    payload["year"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("i"):
                try:
                    payload["iteration"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("phys_sim_iter"):
                try:
                    payload["phys_sim_iteration"] = int(token[len("phys_sim_iter") :])
                except ValueError:
                    return None
            elif token.startswith("beam_sub_iter"):
                try:
                    payload["beam_sub_iteration"] = int(token[len("beam_sub_iter") :])
                except ValueError:
                    return None
        if {"year", "iteration", "phys_sim_iteration"} <= set(payload.keys()):
            return payload
        return None

    return None


def _beam_log_facet_meta(short_name: str) -> Dict[str, Any]:
    facet = _beam_artifact_facets(short_name)
    if not facet:
        return {}
    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _beam_postprocess_split_facet_meta(short_name: str) -> Dict[str, Any]:
    if short_name.startswith("events_parquet_") and "_type_" in short_name:
        head, event_type = short_name.split("_type_", 1)
        parsed = _parse_prefixed_iteration_key(head, "events_parquet")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "events_parquet_split",
                    "event_type": event_type,
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    if short_name.startswith("path_traversal_links_"):
        parsed = _parse_prefixed_iteration_key(short_name, "path_traversal_links")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "path_traversal_links",
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    return {}


def _activitysim_output_facet_meta(
    short_name: str,
    *,
    year: int,
    iteration: int,
) -> Dict[str, Any]:
    family = None
    snapshot_meta: Dict[str, Any] = {}
    if short_name.endswith("_asim_out"):
        family = short_name[: -len("_asim_out")]
    elif short_name.startswith("asim_input_") and short_name.endswith("_archived"):
        family = "asim_input_archived"
        input_name = short_name.removeprefix("asim_input_").removesuffix("_archived")
        source_role_map = {
            "households_csv": ASIM_HOUSEHOLDS_IN,
            "persons_csv": ASIM_PERSONS_IN,
            "land_use_csv": ASIM_LAND_USE_IN,
            "skims_omx": ASIM_OMX_SKIMS,
            "skims_zarr": ZARR_SKIMS,
        }
        snapshot_meta = {
            "source_role": source_role_map.get(input_name, input_name),
            "snapshot_role": f"asim_input_{input_name}",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
        }
    elif short_name == ZARR_SKIMS:
        family = "zarr_skims"
    if family is None:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _urbansim_output_facet_meta(
    short_name: str,
    *,
    forecast_year: int,
) -> Dict[str, Any]:
    snapshot_meta: Dict[str, Any] = {}
    if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
        family = "usim_input_archive"
        snapshot_meta = {
            "source_role": USIM_DATASTORE_H5,
            "snapshot_role": "usim_input_archive",
            "snapshot_reason": "pre_merge_input",
            "storage_event": "snapshot_move",
        }
    elif short_name.startswith(USIM_INPUT_MERGED_PREFIX):
        family = "usim_input_merged"
        snapshot_meta = {
            "source_role": "usim_input_archive",
            "snapshot_role": "usim_input_merged",
            "snapshot_reason": "post_merge_handoff",
            "storage_event": "merged_h5_output",
        }
    elif short_name == USIM_FORECAST_OUTPUT:
        family = "usim_forecast_output"
    elif short_name == USIM_DATASTORE_H5:
        family = "usim_datastore_h5"
    elif short_name == USIM_DATASTORE_BASE_H5:
        family = "usim_datastore_base_h5"
    else:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": forecast_year,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _atlas_artifact_facet_meta(
    short_name: str,
    *,
    run_scenario: Optional[str],
    forecast_year: int,
    artifact_family: str = "atlas_input",
) -> Dict[str, Any]:
    key = short_name.replace("\\", "/")
    key_compact = key.replace("/", "_")

    input_group = "global"
    parsed_scenario = None
    input_year = None

    if key.startswith("adopt/") or key_compact.startswith("adopt_"):
        input_group = "adopt"
        if key.startswith("adopt/"):
            parts = key.split("/")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
        else:
            parts = key_compact.split("_")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
    elif key_compact.startswith("vehicle_type_mapping_"):
        input_group = "vehicle_type_mapping"
        if "baseline" in key_compact:
            parsed_scenario = "baseline"
        elif "evMandForced2" in key_compact:
            parsed_scenario = "zev_mandate"
        elif "ESS_const_220_price" in key_compact:
            parsed_scenario = "ess_cons"
    elif key_compact.startswith("atlas_vehicles2"):
        input_group = "vehicles2"
    elif key_compact.startswith("usim_"):
        input_group = "usim"

    tail = key_compact.rsplit("_", 1)
    if len(tail) == 2 and len(tail[1]) == 4 and tail[1].isdigit():
        input_year = int(tail[1])

    facet: Dict[str, Any] = {
        "artifact_family": artifact_family,
        "input_group": input_group,
        "forecast_year": forecast_year,
    }
    scenario_value = parsed_scenario or run_scenario
    if scenario_value:
        facet["scenario"] = str(scenario_value)
    if input_year is not None:
        facet["input_year"] = input_year

    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _declared_step_model(step_func: Callable[..., Any]) -> Optional[str]:
    meta = getattr(step_func, "__consist_step__", None)
    model = getattr(meta, "model", None)
    if isinstance(model, str) and model:
        return model
    return None


def _declared_step_name(step_func: Callable[..., Any]) -> Optional[str]:
    """Return the registered native identity for a decorated step callable."""
    for spec in WORKFLOW_STEP_SPECS:
        if spec.definition.function is step_func:
            return spec.step_name
    return None


def _step_meta_value(step_meta: Any, name: str) -> Any:
    if step_meta is None:
        return None
    direct = getattr(step_meta, name, None)
    if direct is not None:
        return direct
    extra = getattr(step_meta, "extra", None) or {}
    if isinstance(extra, Mapping):
        return extra.get(name)
    return None


def _provider_source_label(provider: Any) -> str:
    if provider is None:
        return "<none>"
    module = getattr(provider, "__module__", None)
    qualname = getattr(provider, "__qualname__", None) or getattr(
        provider, "__name__", None
    )
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return qualname
    return repr(provider)


def _provider_fix_location(provider: Any) -> str:
    source_file = pyinspect.getsourcefile(provider)
    if source_file:
        return source_file
    return _provider_source_label(provider)


def _invoke_contract_provider(
    provider: Callable[..., Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
) -> Any:
    signature = pyinspect.signature(provider)
    params = list(signature.parameters.values())
    if not params:
        return provider()

    context = SimpleNamespace(
        settings=settings,
        state=state,
        workspace=workspace,
        runtime_settings=settings,
        runtime_state=state,
        runtime_workspace=workspace,
    )
    keyword_values = {
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    accepts_var_kwargs = any(
        param.kind == pyinspect.Parameter.VAR_KEYWORD for param in params
    )
    kwargs = {
        name: value
        for name, value in keyword_values.items()
        if accepts_var_kwargs or name in signature.parameters
    }
    keyword_error: Optional[TypeError] = None
    if kwargs:
        try:
            return provider(**kwargs)
        except TypeError as exc:
            keyword_error = exc

    positional_params = [
        param
        for param in params
        if param.kind
        in (
            pyinspect.Parameter.POSITIONAL_ONLY,
            pyinspect.Parameter.POSITIONAL_OR_KEYWORD,
            pyinspect.Parameter.VAR_POSITIONAL,
        )
    ]
    required_positional_params = [
        param
        for param in positional_params
        if param.kind != pyinspect.Parameter.VAR_POSITIONAL
        and param.default is pyinspect.Parameter.empty
    ]
    accepts_single_context = (
        not accepts_var_kwargs
        and len(required_positional_params) == 1
        and required_positional_params[0].name not in {"settings", "state", "workspace"}
    )
    if accepts_single_context:
        return provider(context)

    if keyword_error is not None:
        raise keyword_error
    raise TypeError(
        "provider must accept settings/state/workspace keyword args or a single context object"
    )


def _resolve_contract_provider_mapping(
    provider: Any,
    *,
    step_name: str,
    provider_name: str,
    settings: Any,
    state: Any,
    workspace: Any,
) -> Optional[Mapping[str, Any]]:
    if provider is None:
        return None
    resolved = provider
    if callable(provider):
        try:
            resolved = _invoke_contract_provider(
                provider,
                settings=settings,
                state=state,
                workspace=workspace,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Step '{step_name}': failed to evaluate {provider_name} provider "
                f"{_provider_source_label(provider)}. Fix: inspect {_provider_fix_location(provider)}. "
                f"Original error: {exc}"
            ) from exc
    if resolved is None:
        return None
    if not isinstance(resolved, Mapping):
        raise RuntimeError(
            f"Step '{step_name}': {provider_name} provider {_provider_source_label(provider)} "
            f"must return a mapping, got {type(resolved).__name__}. "
            f"Fix: inspect {_provider_fix_location(provider)}."
        )
    return resolved


def _validate_contract_provider_keys(
    *,
    errors: list[str],
    step_name: str,
    direction: str,
    provider_name: str,
    provider: Any,
    required_keys: Sequence[str],
    optional_keys: Sequence[str],
    settings: Any,
    state: Any,
    workspace: Any,
) -> None:
    mapping = _resolve_contract_provider_mapping(
        provider,
        step_name=step_name,
        provider_name=provider_name,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if mapping is None:
        return

    canonical_provider_keys: Set[str] = set()
    for raw_key in mapping.keys():
        if not isinstance(raw_key, str):
            errors.append(
                f"Step '{step_name}': {provider_name} provider returned non-string {direction} key "
                f"{raw_key!r}. Fix: inspect {_provider_fix_location(provider)}."
            )
            continue
        match = workflow_step_key_match(step_name, raw_key, direction=direction)
        canonical_provider_keys.add(match.canonical_key)

    missing_required = [
        key for key in required_keys if key not in canonical_provider_keys
    ]
    if missing_required:
        errors.append(
            f"Step '{step_name}': {provider_name} provider {_provider_source_label(provider)} "
            f"is missing required catalog {direction} keys {missing_required}. "
            f"Provider returned {sorted(canonical_provider_keys)}. "
            f"Fix: inspect {_provider_fix_location(provider)}."
        )


def validate_workflow_step_contracts(
    *,
    declared_steps: Optional[Iterable[Callable[..., Any]]] = None,
    allow_untracked_declared: Optional[Set[str]] = None,
    step_refs: Optional[Iterable[Any]] = None,
    settings: Any = None,
    state: Any = None,
    workspace: Any = None,
    require_all_tracked_declared: bool = True,
) -> None:
    """Validate native decorated steps against the catalog contract."""
    del step_refs, state, workspace
    if declared_steps is None:
        return

    errors: list[str] = []
    declared_by_name: Dict[str, Callable[..., Any]] = {}
    for step in declared_steps:
        model = _declared_step_model(step)
        if model is None:
            errors.append(
                "Declared step callable is missing __consist_step__.model metadata"
            )
            continue
        name = _declared_step_name(step) or model
        if name in declared_by_name:
            errors.append(f"Duplicate declared step identity: {name}")
            continue
        declared_by_name[name] = step

    expected_names = {spec.step_name for spec in WORKFLOW_STEP_SPECS}
    declared_names = set(declared_by_name)
    if require_all_tracked_declared:
        missing = expected_names - declared_names
        if missing:
            errors.append(
                "Native definitions missing from declared steps: "
                + ", ".join(sorted(missing))
            )
    allowed = {"postprocessing"} | set(allow_untracked_declared or ())
    unexpected = declared_names - expected_names - allowed
    if unexpected:
        errors.append("Undeclared native step models: " + ", ".join(sorted(unexpected)))

    for name, step in declared_by_name.items():
        spec = workflow_step_spec_for_step_name(name)
        if spec is None:
            continue
        step_meta = getattr(step, "__consist_step__", None)
        schema_outputs = (
            getattr(step_meta, "schema_outputs", ()) if step_meta is not None else ()
        )
        if schema_outputs is not None and not set(spec.output_keys).issubset(
            set(schema_outputs)
        ):
            errors.append(
                f"Step '{name}' metadata schema outputs omit catalog outputs "
                + ", ".join(sorted(set(spec.output_keys) - set(schema_outputs)))
            )

    if errors:
        raise RuntimeError(
            "Workflow step contract validation failed:\n- " + "\n- ".join(errors)
        )
