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
# urbansim_preprocess             (none)                                                      Prepared inputs (from UrbansimPreprocessor._preprocess):
#                                                                                              - geoid_to_zone
#                                                                                              - usim_skims_input_updated (if BEAM skims copied)
#                                                                                              - plus pass-through of initialization inputs:
#                                                                                                usim_datastore_h5, omx_skims, hh_size, income_rates,
#                                                                                                relmap, schools, school_districts, usim_data_reference
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if the input datastore exists)
#
# urbansim_run                    prepared_inputs keys + usim_datastore_h5 (if present)       Raw outputs:
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
# activitysim_compile             (none)                                                      Outputs:
#                                                                                              - zarr_skims
#
# activitysim_run                 activitysim_preprocess outputs + zarr_skims                 Raw outputs (parquet allowlist; keys as listed):
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
import logging
import os
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

import h5py
from consist import define_step

from pilates.generic.records import RecordStore, FileRecord
from pilates.utils import consist_runtime as cr
from pilates.utils.beam_warmstart import (
    find_last_run_output_plans as find_last_run_output_plans,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_input as log_and_set_input,
    log_and_set_output as log_and_set_output,
    log_input_only as log_input_only,
    log_output_only as log_output_only,
    record_store_to_outputs,
    resolve_artifact_from_value as resolve_artifact_from_value,
    update_coupler_from_beam_outputs as update_coupler_from_beam_outputs,
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
    BEAM_R5_OSM_FILE,
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
from pilates.workflows.outputs_base import (
    declared_outputs_for_step_outputs_class,
    iter_step_output_items,
)
from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.workflows.catalog import (
    step_dependencies_from_catalog,
    step_outputs_classes_from_catalog,
)
from pilates.workflows.step_consist_meta import consist_step_meta
from workflow_state import WorkflowState

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def _warn_missing_coupler_inputs(
    coupler: Optional[CouplerProtocol],
    input_store: Optional[RecordStore],
    context: str,
) -> None:
    if coupler is None or input_store is None:
        return
    keys_attr = getattr(coupler, "keys", None)
    if not callable(keys_attr):
        return
    try:
        coupler_keys = set(keys_attr())
    except Exception:
        return
    missing = []
    for record in input_store.all_records():
        key = getattr(record, "short_name", None) or getattr(record, "unique_id", None)
        if key and key not in coupler_keys:
            missing.append(key)
    if missing:
        logger.warning(
            "[%s] Input RecordStore keys missing from coupler: %s",
            context,
            sorted(set(missing)),
        )


def _artifact_content_hash(value: Any) -> Optional[str]:
    """Extract a content hash from an artifact-like mapping value."""
    if value is None:
        return None
    for attr_name in ("content_hash", "hash"):
        content_hash = getattr(value, attr_name, None)
        if content_hash:
            return str(content_hash)
    return None


def _append_artifact_mapping_records(
    *,
    record_store: RecordStore,
    artifact_mapping: Optional[Mapping[str, Any]],
    workspace: Optional["Workspace"],
    description_prefix: str,
    key_aliases: Optional[Mapping[str, str]] = None,
) -> None:
    """Materialize artifact-like mapping values into a ``RecordStore``."""
    if not artifact_mapping:
        return
    aliases = key_aliases or {}
    for key, value in artifact_mapping.items():
        path = artifact_to_path(value, workspace)
        if path is None and isinstance(value, (str, os.PathLike)):
            path = os.fspath(value)
        if not path:
            continue
        record_store.add_record(
            FileRecord(
                file_path=str(path),
                short_name=aliases.get(key, key),
                description=f"{description_prefix}: {key}",
                content_hash=_artifact_content_hash(value),
            )
        )


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
        if (
            short_name in profile_schema_keys
            or (
                profile_schema_suffixes
                and path_str.endswith(profile_schema_suffixes)
            )
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


def _log_named_h5_tables(
    *,
    path: str,
    direction: str,
    table_keys: Dict[str, str],
    description_by_table: Optional[Dict[str, str]] = None,
    extra_meta_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
) -> None:
    """
    Log selected HDF5 datasets as individual ``h5_table`` artifacts.

    Parameters
    ----------
    path : str
        HDF5 container path.
    direction : str
        Consist artifact direction, usually ``"input"`` or ``"output"``.
    table_keys : dict
        Mapping of HDF5 table paths to artifact keys.
    description_by_table : dict, optional
        Optional mapping of HDF5 table paths to descriptions.
    extra_meta_fn : callable, optional
        Callback returning extra metadata for each `(artifact_key, table_path)`.
    """
    description_by_table = description_by_table or {}
    try:
        with h5py.File(path, "r") as h5_file:
            for table_path, artifact_key in table_keys.items():
                normalized_path = (
                    table_path if str(table_path).startswith("/") else f"/{table_path}"
                )
                if normalized_path not in h5_file:
                    logger.debug(
                        "Skipping HDF5 table log for missing dataset %s in %s",
                        normalized_path,
                        path,
                    )
                    continue
                meta: Dict[str, Any] = {
                    "profile_file_schema": True,
                    "h5_parent_key": artifact_key.rsplit("_table_", 1)[0]
                    if "_table_" in artifact_key
                    else artifact_key,
                    "h5_table_name": normalized_path.split("/")[-1],
                }
                if extra_meta_fn is not None:
                    extra_meta = extra_meta_fn(artifact_key, normalized_path)
                    if extra_meta:
                        meta.update(extra_meta)
                cr.log_h5_table(
                    path,
                    key=artifact_key,
                    table_path=normalized_path,
                    direction=direction,
                    description=description_by_table.get(
                        table_path, f"HDF5 table {normalized_path}"
                    ),
                    **meta,
                )
    except OSError:
        logger.debug("Skipping named HDF5 table logging for unreadable file %s", path)


def _parse_prefixed_iteration_key(short_name: str, prefix: str) -> Optional[Dict[str, Any]]:
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
                    payload["phys_sim_iteration"] = int(
                        token[len("phys_sim_iter") :]
                    )
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
    if short_name.endswith("_asim_out"):
        family = short_name[: -len("_asim_out")]
    elif short_name.startswith("asim_input_") and short_name.endswith("_archived"):
        family = "asim_input_archived"
    elif short_name == ZARR_SKIMS:
        family = "zarr_skims"
    if family is None:
        return {}
    return {
        "facet": {
            "artifact_family": family,
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
    if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
        family = "usim_input_archive"
    elif short_name.startswith(USIM_INPUT_MERGED_PREFIX):
        family = "usim_input_merged"
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


def _log_beam_r5_osm_input(
    *,
    tracker: Any,
    settings: "PilatesConfig",
    workspace: "Workspace",
) -> None:
    """Log the BEAM R5 OSM/mapdb file referenced by ingested BEAM config."""
    from pathlib import Path

    config_root = Path(workspace.get_beam_mutable_data_dir()) / settings.run.region

    try:
        from sqlmodel import Session, select
        from consist.models.beam import BeamConfigCache, BeamConfigIngestRunLink
    except Exception:
        logger.debug("SQLModel/Consist beam models unavailable; skipping OSM logging.")
        return

    current_run = cr.current_run()
    run_id = getattr(current_run, "id", None) if current_run else None
    if not run_id or tracker.db is None:
        return

    try:
        with Session(tracker.db.engine) as session:
            base_stmt = (
                select(
                    BeamConfigCache.value_str,
                    BeamConfigCache.content_hash,
                )
                .join(
                    BeamConfigIngestRunLink,
                    BeamConfigCache.content_hash
                    == BeamConfigIngestRunLink.content_hash,
                )
                .where(BeamConfigIngestRunLink.run_id == run_id)
            )
            config_name = settings.beam.config
            if config_name:
                base_stmt = base_stmt.where(
                    BeamConfigIngestRunLink.config_name == config_name
                )
            osm_rows = session.exec(
                base_stmt.where(
                    BeamConfigCache.key == "beam.routing.r5.osmFile"
                )
            ).all()
            if len(osm_rows) > 1:
                logger.warning(
                    "Multiple BEAM osmFile rows found for run_id=%s config=%s",
                    run_id,
                    config_name,
                )
            osm_value = osm_rows[0][0] if osm_rows else None
            osm_hash = osm_rows[0][1] if osm_rows else None
            logger.debug(
                "BEAM config osmFile resolved value: %s (run_id=%s, db=%s, config=%s, hash=%s)",
                osm_value,
                run_id,
                tracker.db.engine.url,
                config_name,
                osm_hash,
            )
            if osm_value == "/":
                all_osm_rows = session.exec(
                    select(
                        BeamConfigIngestRunLink.config_name,
                        BeamConfigCache.value_str,
                        BeamConfigCache.content_hash,
                    )
                    .join(
                        BeamConfigCache,
                        BeamConfigCache.content_hash
                        == BeamConfigIngestRunLink.content_hash,
                    )
                    .where(BeamConfigIngestRunLink.run_id == run_id)
                    .where(BeamConfigCache.key == "beam.routing.r5.osmFile")
                ).all()
                logger.warning(
                    "BEAM osmFile resolved to '/' for run_id=%s; rows=%s",
                    run_id,
                    all_osm_rows,
                )
            resolved_osm_path = None
            if osm_value and "${" not in osm_value:
                resolved_osm_path = osm_value
                if not os.path.isabs(resolved_osm_path):
                    resolved_osm_path = str(
                        (config_root / resolved_osm_path).resolve()
                    )
                if not os.path.exists(resolved_osm_path):
                    resolved_osm_path = None

            if resolved_osm_path is None:
                mapdb_row = session.exec(
                    base_stmt.where(
                        BeamConfigCache.key
                        == "beam.routing.r5.osmMapdbFile"
                    )
                ).first()
                mapdb_value = mapdb_row[0] if mapdb_row else None
                logger.debug(
                    "BEAM config osmMapdbFile resolved value: %s",
                    mapdb_value,
                )
                if mapdb_value and "${" not in mapdb_value:
                    resolved_osm_path = mapdb_value
                    if not os.path.isabs(resolved_osm_path):
                        resolved_osm_path = str(
                            (config_root / resolved_osm_path).resolve()
                        )
                    if not os.path.exists(resolved_osm_path):
                        resolved_osm_path = None

            if resolved_osm_path:
                cr.log_input(
                    resolved_osm_path,
                    key=BEAM_R5_OSM_FILE,
                    description=(
                        "BEAM R5 OSM input referenced by config"
                    ),
                )
    except Exception:
        logger.debug(
            "Failed to resolve/log BEAM R5 OSM file from config.",
            exc_info=True,
        )

StepOutputsT = TypeVar("StepOutputsT")
InputLogger = Callable[
    ["PilatesConfig", WorkflowState, "Workspace", "StepOutputsHolder"],
    Mapping[str, Any],
]
OutputLogger = Callable[
    [StepOutputsT, "PilatesConfig", WorkflowState, "Workspace", "StepOutputsHolder"],
    None,
]


class _PreprocessorExecutor(Protocol):
    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
    ) -> RecordStore: ...


class _RunnerExecutor(Protocol):
    def run(self, input_store: RecordStore, workspace: "Workspace") -> RecordStore: ...


class _PostprocessorExecutor(Protocol):
    def postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: "Workspace",
        model_run_hash: Optional[str] = None,
    ) -> RecordStore: ...


@dataclass
class StepOutputsHolder:
    """
    Accumulates typed step outputs across the workflow.

    This holder acts as the in-memory handoff between granular steps so that
    each model phase can consume the outputs produced by its predecessors
    without re-querying the coupler or filesystem.

    Attributes
    ----------
    activitysim_preprocess : ActivitySimPreprocessOutputs, optional
        Preprocess outputs.
    activitysim_run : ActivitySimRunOutputs, optional
        Run outputs.
    activitysim_postprocess : ActivitySimPostprocessOutputs, optional
        Postprocess outputs.
    beam_preprocess : BeamPreprocessOutputs, optional
        Preprocess outputs.
    beam_run : BeamRunOutputs, optional
        Run outputs.
    beam_postprocess : BeamPostprocessOutputs, optional
        Postprocess outputs.
    beam_full_skim : BeamFullSkimOutputs, optional
        Full-skim outputs.
    urbansim_preprocess : UrbanSimPreprocessOutputs, optional
        Preprocess outputs.
    urbansim_run : UrbanSimRunOutputs, optional
        Run outputs.
    urbansim_postprocess : UrbanSimPostprocessOutputs, optional
        Postprocess outputs.
    atlas_preprocess : AtlasPreprocessOutputs, optional
        Preprocess outputs.
    atlas_run : AtlasRunOutputs, optional
        Run outputs.
    atlas_postprocess : AtlasPostprocessOutputs, optional
        Postprocess outputs.
    """

    activitysim_preprocess: Optional[ActivitySimPreprocessOutputs] = None
    activitysim_run: Optional[ActivitySimRunOutputs] = None
    activitysim_postprocess: Optional[ActivitySimPostprocessOutputs] = None
    beam_preprocess: Optional[BeamPreprocessOutputs] = None
    beam_run: Optional[BeamRunOutputs] = None
    beam_postprocess: Optional[BeamPostprocessOutputs] = None
    beam_full_skim: Optional[BeamFullSkimOutputs] = None
    urbansim_preprocess: Optional[UrbanSimPreprocessOutputs] = None
    urbansim_run: Optional[UrbanSimRunOutputs] = None
    urbansim_postprocess: Optional[UrbanSimPostprocessOutputs] = None
    atlas_preprocess: Optional[AtlasPreprocessOutputs] = None
    atlas_run: Optional[AtlasRunOutputs] = None
    atlas_postprocess: Optional[AtlasPostprocessOutputs] = None

    def set_attribute(self, step_name: str, outputs: Any) -> None:
        """
        Set a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.
        outputs : Any
            Outputs object to store.
        """
        attr = step_name.replace("-", "_")
        setattr(self, attr, outputs)

    def get_attribute(self, step_name: str) -> Any:
        """
        Retrieve a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.

        Returns
        -------
        Any
            Outputs object or None if missing.
        """
        attr = step_name.replace("-", "_")
        return getattr(self, attr, None)


def _upstream_outputs_view(
    outputs_holder: StepOutputsHolder,
    *,
    current_step_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a non-null snapshot view of upstream step outputs.

    Parameters
    ----------
    outputs_holder : StepOutputsHolder
        Holder containing current in-memory step outputs.

    Returns
    -------
    dict
        Mapping of holder field name to output object for populated entries.
    """
    current_attr = (
        current_step_name.replace("-", "_") if isinstance(current_step_name, str) else None
    )
    upstream: Dict[str, Any] = {}
    for holder_field in fields(StepOutputsHolder):
        if current_attr and holder_field.name == current_attr:
            continue
        value = getattr(outputs_holder, holder_field.name, None)
        if value is not None:
            upstream[holder_field.name] = value
    return upstream


STEP_OUTPUTS_CLASSES = step_outputs_classes_from_catalog()


STEP_DEPENDENCIES = step_dependencies_from_catalog()

DEFAULT_UNTRACKED_STEP_NAMES = frozenset({"activitysim_compile", "postprocessing"})


def validate_step_ready(step_name: str, outputs_holder: StepOutputsHolder) -> None:
    """
    Validate that dependencies for a step are satisfied.

    This enforces the expected execution order (e.g., preprocess before run)
    so that downstream steps can rely on required outputs being present.

    Parameters
    ----------
    step_name : str
        Step name to validate.
    outputs_holder : StepOutputsHolder
        Holder containing upstream outputs.
    """
    spec = STEP_DEPENDENCIES.get(step_name)
    if not spec:
        logger.warning("No dependency spec for %s; skipping validation", step_name)
        return
    for holder_input_key in spec["holder_inputs"]:
        holder_attr = holder_input_key.replace("-", "_")
        if getattr(outputs_holder, holder_attr, None) is None:
            raise RuntimeError(
                f"{step_name} requires {holder_input_key} to complete first, "
                "but it has not been executed or failed."
            )


def _declared_step_model(step_func: Callable[..., Any]) -> Optional[str]:
    meta = getattr(step_func, "__consist_step__", None)
    model = getattr(meta, "model", None)
    if isinstance(model, str) and model:
        return model
    return None


def validate_workflow_step_contracts(
    *,
    declared_steps: Optional[Iterable[Callable[..., Any]]] = None,
    allow_untracked_declared: Optional[Set[str]] = None,
    step_refs: Optional[Iterable[Any]] = None,
) -> None:
    """
    Validate internal workflow step contracts.

    This is intended to run at startup to catch integration drift before
    expensive model execution starts.

    Checks performed:
    - ``StepOutputsHolder`` fields align with ``STEP_OUTPUTS_CLASSES`` keys.
    - ``STEP_OUTPUTS_CLASSES`` keys align with ``STEP_DEPENDENCIES`` keys.
    - Dependency specs reference known step names.
    - Optionally, declared step models are consistent with tracked step names.
    - For tracked declared steps, canonical outputs match step metadata outputs.
    - Deprecated ``StepRef.required_outputs`` overrides require rationale.
    """
    errors: list[str] = []

    def _normalize_output_keys(values: Any) -> list[str]:
        if not isinstance(values, Sequence) or isinstance(values, str):
            return []
        return [output for output in values if isinstance(output, str)]

    holder_fields = {f.name for f in fields(StepOutputsHolder)}
    output_class_keys = set(STEP_OUTPUTS_CLASSES.keys())
    dependency_keys = set(STEP_DEPENDENCIES.keys())
    tracked_step_names = holder_fields | output_class_keys | dependency_keys

    missing_output_class = holder_fields - output_class_keys
    if missing_output_class:
        errors.append(
            "Missing output classes for StepOutputsHolder fields: "
            + ", ".join(sorted(missing_output_class))
        )
    extra_output_class = output_class_keys - holder_fields
    if extra_output_class:
        errors.append(
            "STEP_OUTPUTS_CLASSES has keys not present on StepOutputsHolder: "
            + ", ".join(sorted(extra_output_class))
        )

    missing_dependency_spec = output_class_keys - dependency_keys
    if missing_dependency_spec:
        errors.append(
            "Missing STEP_DEPENDENCIES entries for tracked steps: "
            + ", ".join(sorted(missing_dependency_spec))
        )
    extra_dependency_spec = dependency_keys - output_class_keys
    if extra_dependency_spec:
        errors.append(
            "STEP_DEPENDENCIES has extra tracked keys not in STEP_OUTPUTS_CLASSES: "
            + ", ".join(sorted(extra_dependency_spec))
        )

    for step_name, spec in STEP_DEPENDENCIES.items():
        if not isinstance(spec, Mapping):
            errors.append(
                f"Dependency spec for {step_name!r} must be a mapping, got {type(spec).__name__}"
            )
            continue
        depends_on = spec.get("depends_on", [])
        holder_inputs = spec.get("holder_inputs", [])
        if not isinstance(depends_on, Sequence) or isinstance(depends_on, str):
            errors.append(
                f"STEP_DEPENDENCIES[{step_name!r}]['depends_on'] must be a sequence of step names"
            )
            depends_on = []
        if not isinstance(holder_inputs, Sequence) or isinstance(holder_inputs, str):
            errors.append(
                f"STEP_DEPENDENCIES[{step_name!r}]['holder_inputs'] must be a sequence of step names"
            )
            holder_inputs = []

        unknown_depends_on = set(depends_on) - tracked_step_names
        if unknown_depends_on:
            errors.append(
                f"STEP_DEPENDENCIES[{step_name!r}] depends_on unknown steps: "
                + ", ".join(sorted(unknown_depends_on))
            )
        unknown_holder_inputs = set(holder_inputs) - holder_fields
        if unknown_holder_inputs:
            errors.append(
                f"STEP_DEPENDENCIES[{step_name!r}] holder_inputs unknown holder fields: "
                + ", ".join(sorted(unknown_holder_inputs))
            )

    if declared_steps is not None:
        declared_counts: Dict[str, int] = {}
        declared_by_model: Dict[str, Callable[..., Any]] = {}
        undecorated_count = 0
        for step_func in declared_steps:
            step_model = _declared_step_model(step_func)
            if step_model is None:
                undecorated_count += 1
                continue
            declared_counts[step_model] = declared_counts.get(step_model, 0) + 1
            declared_by_model[step_model] = step_func

        if undecorated_count:
            errors.append(
                f"{undecorated_count} declared step callable(s) are missing __consist_step__.model metadata"
            )

        duplicate_declared = sorted(
            name for name, count in declared_counts.items() if count > 1
        )
        if duplicate_declared:
            errors.append(
                "Duplicate declared step model names: "
                + ", ".join(duplicate_declared)
            )

        declared_names = set(declared_counts.keys())
        missing_declared = tracked_step_names - declared_names
        if missing_declared:
            errors.append(
                "Tracked step names missing from declared steps: "
                + ", ".join(sorted(missing_declared))
            )

        allowed_untracked = set(DEFAULT_UNTRACKED_STEP_NAMES)
        if allow_untracked_declared:
            allowed_untracked |= set(allow_untracked_declared)

        unexpected_untracked = declared_names - tracked_step_names - allowed_untracked
        if unexpected_untracked:
            errors.append(
                "Declared step names are not tracked in holder/output/dependency maps "
                "(add tracking or allowlist explicitly): "
                + ", ".join(sorted(unexpected_untracked))
            )

        tracked_declared_names = sorted(declared_names & output_class_keys)
        for step_name in tracked_declared_names:
            outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
            step_func = declared_by_model.get(step_name)
            if outputs_class is None or step_func is None:
                continue
            canonical = list(declared_outputs_for_step_outputs_class(outputs_class))
            step_meta = getattr(step_func, "__consist_step__", None)
            metadata_outputs = _normalize_output_keys(getattr(step_meta, "outputs", None))
            if canonical != metadata_outputs:
                errors.append(
                    f"Step '{step_name}': canonical outputs {canonical} conflict with metadata outputs "
                    f"{metadata_outputs}. Fix: remove metadata override or update declared_outputs in "
                    f"{outputs_class.__name__}."
                )

    if step_refs is not None:
        for step_ref in step_refs:
            required_outputs = getattr(step_ref, "required_outputs", None)
            if required_outputs is None:
                continue
            rationale = getattr(step_ref, "required_outputs_rationale", None)
            step_name = getattr(step_ref, "name", "<unknown>")
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append(
                    f"Step '{step_name}': deprecated StepRef.required_outputs override requires "
                    "StepRef.required_outputs_rationale."
                )

    if errors:
        raise RuntimeError(
            "Workflow step contract validation failed:\n- " + "\n- ".join(errors)
        )


def require_common_runtime(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Require runtime kwargs shared across step functions.

    Parameters
    ----------
    *names : str
        Additional required runtime kwarg names.

    Returns
    -------
    callable
        Decorator that enforces the runtime kwargs.
    """
    return cr.require_runtime_kwargs("settings", "state", "workspace", *names)


def _schema_outputs_from_class(outputs_class: Type[StepOutputsT]) -> Optional[list[str]]:
    record_keys = getattr(outputs_class, "record_keys", None) or {}
    values = [value for value in record_keys.values() if isinstance(value, str)]
    unique = sorted(set(values))
    return unique or None


def _declared_outputs_from_class(
    outputs_class: Type[StepOutputsT],
) -> Optional[list[str]]:
    """
    Return strict output contract keys for a step outputs class when available.

    Precedence:
    1. Explicit ``declared_outputs`` class attribute.
    2. Fallback to required ``record_keys`` fields only.
    """
    declared = list(declared_outputs_for_step_outputs_class(outputs_class))
    if not declared:
        return None
    return declared


def _decorate_step_with_consist(
    *,
    step_func: Callable[..., Any],
    step_model: str,
    description: str,
    schema_outputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> Callable[..., Any]:
    """
    Attach native Consist step metadata to a workflow step function.
    """
    if hasattr(step_func, "__consist_step__"):
        return step_func

    kwargs: Dict[str, Any] = {
        "model": step_model,
        "description": description,
        "name_template": "{func_name}__y{year}__i{iteration}__phase_{phase}",
        "tags": tags or [step_model],
        **consist_step_meta(step_model),
    }
    if schema_outputs:
        kwargs["schema_outputs"] = schema_outputs
    if outputs:
        kwargs["outputs"] = outputs
    return define_step(**kwargs)(step_func)


def _execute_preprocess(
    preprocessor: _PreprocessorExecutor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a preprocessor using only the workspace.

    This phase typically prepares model-specific inputs (copying, formatting,
    or deriving tables) in the mutable workspace for the runner.

    Parameters
    ----------
    preprocessor : object
        Preprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder for upstream outputs (unused).

    Returns
    -------
    RecordStore
        Preprocessor outputs.
    """
    return preprocessor.preprocess(workspace)


def _execute_preprocess_typed(
    preprocessor: _PreprocessorExecutor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    outputs_class: Type[StepOutputsT],
    **kwargs: Any,
) -> StepOutputsT:
    record_store = _execute_preprocess(
        preprocessor,
        workspace,
        outputs_holder,
        **kwargs,
    )
    return _typed_outputs_from_record_store(
        record_store=record_store,
        outputs_class=outputs_class,
        workspace=workspace,
    )


def _build_required_input_store(
    *,
    outputs_holder: StepOutputsHolder,
    upstream_attr: str,
    missing_message: str,
    context: str,
    coupler: Optional[CouplerProtocol] = None,
    workspace: Optional["Workspace"] = None,
    extra_inputs: Optional[Mapping[str, Any]] = None,
    extra_input_description_prefix: str = "Executor extra input",
    warn_missing_coupler_inputs: bool = True,
) -> RecordStore:
    """
    Build a RecordStore from required upstream step outputs.

    Parameters
    ----------
    outputs_holder : StepOutputsHolder
        Holder containing typed step outputs.
    upstream_attr : str
        Attribute name on ``outputs_holder`` to resolve.
    missing_message : str
        RuntimeError message when upstream outputs are missing.
    context : str
        Context label for warning logs.
    coupler : CouplerProtocol, optional
        Coupler used for missing-input key warning checks.
    workspace : Workspace, optional
        Workspace used to resolve artifact-like extra inputs into paths.
    extra_inputs : mapping, optional
        Additional artifact-like inputs merged into the upstream input store.
    extra_input_description_prefix : str, default "Executor extra input"
        Description prefix for extra inputs materialized at the executor boundary.
    warn_missing_coupler_inputs : bool, default True
        Whether to emit warnings for RecordStore keys not present in coupler.

    Returns
    -------
    RecordStore
        Input store for runner/postprocessor execution.
    """
    def _content_hash_for(short_name: str) -> Optional[str]:
        for attr_name in (
            "input_hashes",
            "raw_output_hashes",
            "processed_output_hashes",
        ):
            hashes = getattr(upstream, attr_name, None)
            if isinstance(hashes, Mapping):
                content_hash = hashes.get(short_name)
                if content_hash:
                    return str(content_hash)
        return None

    upstream = getattr(outputs_holder, upstream_attr, None)
    if upstream is None:
        raise RuntimeError(missing_message)
    input_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(path),
                short_name=short_name,
                description=description,
                content_hash=_content_hash_for(short_name),
            )
            for short_name, path, description in iter_step_output_items(upstream)
        ]
    )
    if extra_inputs is not None:
        _append_artifact_mapping_records(
            record_store=input_store,
            artifact_mapping=extra_inputs,
            workspace=workspace,
            description_prefix=extra_input_description_prefix,
        )
    if warn_missing_coupler_inputs:
        _warn_missing_coupler_inputs(coupler, input_store, context)
    return input_store


def _typed_outputs_from_record_store(
    *,
    record_store: RecordStore,
    outputs_class: Type[StepOutputsT],
    workspace: "Workspace",
) -> StepOutputsT:
    return record_store_to_outputs(
        record_store=record_store,
        output_class=outputs_class,
        workspace=workspace,
    )

