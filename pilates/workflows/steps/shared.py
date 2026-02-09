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
#                                                                                              - usim_h5_updated
#                                                                                              - atlas_vehicles2_output
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
#                                                                                              - ATLAS_VEHICLES2_INPUT (if present)
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
    Sequence,
    Set,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

from consist import define_step

from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.beam_warmstart import find_last_run_output_plans
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
    log_output_only,
    record_store_to_outputs,
    resolve_artifact_from_value,
    update_coupler_from_beam_outputs,
)
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_PLANS_OUT,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_R5_OSM_FILE,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    USIM_FORECAST_OUTPUT,
    ZARR_SKIMS,
    ASIM_HOUSEHOLDS_IN,
    ASIM_PERSONS_IN,
    ASIM_LAND_USE_IN,
)
from pilates.workflows.step_exec import (
    Postprocessor,
    Preprocessor,
    Runner,
    run_postprocessor,
    run_preprocessor,
    run_runner,
    warm_start_activities,
)
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.beam.outputs import (
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


STEP_OUTPUTS_CLASSES = {
    "activitysim_preprocess": ActivitySimPreprocessOutputs,
    "activitysim_run": ActivitySimRunOutputs,
    "activitysim_postprocess": ActivitySimPostprocessOutputs,
    "beam_preprocess": BeamPreprocessOutputs,
    "beam_run": BeamRunOutputs,
    "beam_postprocess": BeamPostprocessOutputs,
    "urbansim_preprocess": UrbanSimPreprocessOutputs,
    "urbansim_run": UrbanSimRunOutputs,
    "urbansim_postprocess": UrbanSimPostprocessOutputs,
    "atlas_preprocess": AtlasPreprocessOutputs,
    "atlas_run": AtlasRunOutputs,
    "atlas_postprocess": AtlasPostprocessOutputs,
}


STEP_DEPENDENCIES = {
    "urbansim_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "urbansim_run": {
        "depends_on": ["urbansim_preprocess"],
        "holder_inputs": ["urbansim_preprocess"],
    },
    "urbansim_postprocess": {
        "depends_on": ["urbansim_run"],
        "holder_inputs": ["urbansim_run"],
    },
    "atlas_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "atlas_run": {
        "depends_on": ["atlas_preprocess"],
        "holder_inputs": ["atlas_preprocess"],
    },
    "atlas_postprocess": {
        "depends_on": ["atlas_run"],
        "holder_inputs": ["atlas_run"],
    },
    "activitysim_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "activitysim_run": {
        "depends_on": ["activitysim_preprocess"],
        "holder_inputs": ["activitysim_preprocess"],
    },
    "activitysim_postprocess": {
        "depends_on": ["activitysim_run"],
        "holder_inputs": ["activitysim_run"],
    },
    "beam_preprocess": {
        "depends_on": ["activitysim_postprocess"],
        "holder_inputs": ["activitysim_postprocess"],
    },
    "beam_run": {
        "depends_on": ["beam_preprocess"],
        "holder_inputs": ["beam_preprocess"],
    },
    "beam_postprocess": {
        "depends_on": ["beam_run"],
        "holder_inputs": ["beam_run"],
    },
}

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
    """
    errors: list[str] = []

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
        undecorated_count = 0
        for step_func in declared_steps:
            step_model = _declared_step_model(step_func)
            if step_model is None:
                undecorated_count += 1
                continue
            declared_counts[step_model] = declared_counts.get(step_model, 0) + 1

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


def _make_generic_step_function(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., RecordStore],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[InputLogger] = None,
    output_logger: Optional[OutputLogger] = None,
) -> Callable[..., None]:
    """
    Build a step function with common RecordStore-to-StepOutputs plumbing.

    The returned function executes a model component (preprocess/run/postprocess),
    converts its RecordStore outputs into a typed outputs dataclass, validates
    the outputs, logs any configured inputs/outputs for provenance, and stores
    the results in the shared outputs holder.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder used to store outputs for downstream steps.
    model_name : str
        Model identifier for logging.
    phase : str
        Step phase name (preprocess/run/postprocess).
    outputs_class : type
        StepOutputs dataclass type.
    component_getter : callable
        Callable that returns the component instance.
    component_executor : callable
        Callable that executes the component.
    outputs_holder_setter : callable
        Callback that stores outputs on the holder.
    input_logger : callable, optional
        Optional hook for logging step inputs.
    output_logger : callable, optional
        Optional hook for logging step outputs.

    Returns
    -------
    callable
        Step function compatible with Consist scenario execution.
    """

    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        logger.debug("Starting %s %s step", model_name, phase)
        factory = ModelFactory()
        component = component_getter(factory, state)

        extra_kwargs: Dict[str, Any] = {}
        if input_logger is not None:
            extra_kwargs = (
                input_logger(settings, state, workspace, outputs_holder) or {}
            )
            logger.debug(
                "%s %s input logger keys: %s",
                model_name,
                phase,
                list(extra_kwargs.keys()),
            )

        record_store = component_executor(
            component,
            workspace,
            outputs_holder,
            coupler=coupler,
            context=f"{model_name}_{phase}",
            **extra_kwargs,
            **kwargs,
        )
        if record_store is not None:
            try:
                record_keys = list(record_store.to_mapping().keys())
            except AttributeError:
                record_keys = []
            logger.debug(
                "%s %s record store keys: %s",
                model_name,
                phase,
                record_keys,
            )

        step_outputs = record_store_to_outputs(
            record_store=record_store,
            output_class=outputs_class,
            workspace=workspace,
        )
        step_outputs.validate()
        outputs_holder_setter(outputs_holder, step_outputs)

        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)

        logger.info("%s %s completed successfully", model_name, phase)

    step_model = f"{model_name}_{phase}"
    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=step_model,
        description=f"{model_name} {phase} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        outputs=_declared_outputs_from_class(outputs_class),
        tags=[model_name, phase],
    )


def _execute_preprocess(
    preprocessor: Preprocessor,
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
    return run_preprocessor(preprocessor, workspace)


def _execute_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "runner",
    extra_inputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a runner using upstream preprocess outputs.

    This phase performs the core model simulation (e.g., activity demand,
    land use, or traffic assignment) using the prepared inputs.

    Parameters
    ----------
    runner : object
        Runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.
    extra_inputs : RecordStore, optional
        Additional inputs to merge into the runner input store.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.activitysim_preprocess
    if upstream is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    input_store = upstream.to_record_store()
    if extra_inputs is not None:
        input_store += extra_inputs
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a postprocessor using upstream run outputs.

    This phase adapts raw model outputs for downstream consumption (e.g.,
    updating HDF5 inputs, producing summary outputs, or deriving skims).

    Parameters
    ----------
    postprocessor : object
        Postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.activitysim_run
    if upstream is None:
        raise RuntimeError("ActivitySim run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_beam_preprocess(
    preprocessor: Preprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "beam_preprocess",
    activity_demand_outputs: Optional[RecordStore] = None,
    previous_beam_outputs: Optional[RecordStore] = None,
    beam_preprocess_inputs: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM preprocessor with upstream RecordStore inputs.

    BEAM preprocess builds the runnable scenario inputs by combining
    ActivitySim demand outputs with warm-start data and optional ATLAS
    vehicle ownership inputs.

    Parameters
    ----------
    preprocessor : object
        BEAM preprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder for upstream outputs (unused).
    activity_demand_outputs : RecordStore, optional
        ActivitySim postprocess outputs.
    previous_beam_outputs : RecordStore, optional
        Previous BEAM outputs for warm starts.

    Returns
    -------
    RecordStore
        Preprocessor outputs.
    """
    combined = RecordStore()
    if activity_demand_outputs is not None:
        combined += activity_demand_outputs
    if previous_beam_outputs is not None:
        combined += previous_beam_outputs
    if beam_preprocess_inputs:
        # Bridge orchestration-level fallback inputs into the legacy BEAM
        # preprocessor contract, which expects ActivitySim-style short names.
        key_aliases = {
            BEAM_PLANS_IN: "beam_plans",
            BEAM_HOUSEHOLDS_IN: "households",
            BEAM_PERSONS_IN: "persons",
            LINKSTATS_WARMSTART: "linkstats",
        }
        for key, value in beam_preprocess_inputs.items():
            path = artifact_to_path(value, workspace)
            if path is None and isinstance(value, (str, os.PathLike)):
                path = os.fspath(value)
            if not path:
                continue
            combined.add_record(
                FileRecord(
                    file_path=str(path),
                    short_name=key_aliases.get(key, key),
                    description=f"BEAM preprocess provided input: {key}",
                )
            )
    _warn_missing_coupler_inputs(coupler, combined, context)
    return preprocessor.preprocess(workspace, combined)


def _execute_beam_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "beam_run",
    extra_inputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM runner using preprocess outputs.

    BEAM run performs the traffic assignment simulation, producing linkstats,
    skims, plans, and event outputs.

    Parameters
    ----------
    runner : object
        BEAM runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.
    extra_inputs : RecordStore, optional
        Additional inputs (e.g., zarr skims).

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    input_store = upstream.to_record_store()
    if extra_inputs is not None:
        input_store += extra_inputs
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_beam_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM postprocessor using run outputs.

    BEAM postprocess merges updated skims and writes final skim artifacts for
    downstream models.

    Parameters
    ----------
    postprocessor : object
        BEAM postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.beam_run
    if upstream is None:
        raise RuntimeError("BEAM run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_urbansim_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "urbansim_run",
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the UrbanSim runner using preprocess outputs.

    UrbanSim run performs the land-use forecast between the base year and
    forecast year, writing the UrbanSim datastore for downstream steps.

    Parameters
    ----------
    runner : object
        UrbanSim runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.urbansim_preprocess
    if upstream is None:
        raise RuntimeError("UrbanSim preprocess must complete first")
    input_store = upstream.to_record_store()
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_urbansim_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "urbansim_postprocess",
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the UrbanSim postprocessor using run outputs.

    UrbanSim postprocess prepares an updated input datastore for subsequent
    model stages (ActivitySim/ATLAS).

    Parameters
    ----------
    postprocessor : object
        UrbanSim postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.urbansim_run
    if upstream is None:
        raise RuntimeError("UrbanSim run must complete first")
    raw_outputs = upstream.to_record_store()
    _warn_missing_coupler_inputs(coupler, raw_outputs, context)
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_atlas_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "atlas_run",
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the ATLAS runner using preprocess outputs.

    ATLAS run simulates vehicle ownership evolution for the sub-year and
    produces household vehicle ownership outputs.

    Parameters
    ----------
    runner : object
        ATLAS runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.atlas_preprocess
    if upstream is None:
        raise RuntimeError("ATLAS preprocess must complete first")
    input_store = upstream.to_record_store()
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_atlas_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "atlas_postprocess",
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the ATLAS postprocessor using run outputs.

    ATLAS postprocess updates the UrbanSim HDF5 datastore and derives
    vehicles2 outputs for downstream BEAM runs.

    Parameters
    ----------
    postprocessor : object
        ATLAS postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.atlas_run
    if upstream is None:
        raise RuntimeError("ATLAS run must complete first")
    raw_outputs = upstream.to_record_store()
    _warn_missing_coupler_inputs(coupler, raw_outputs, context)
    return run_postprocessor(postprocessor, raw_outputs, workspace)
