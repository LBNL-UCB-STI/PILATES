import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import openmatrix as omx
import pandas as pd

from pilates.config import PilatesConfig
from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamRunOutputs,
    requires_omx_skim_output,
)
import pilates.utils.zone_utils as zone_utils

try:
    import xarray as xr
except Exception:
    print("FAILED TO LOAD XARRAY or ZARR")

from pilates.generic.postprocessor import GenericPostprocessor
from pilates.utils.coupler_helpers import artifact_to_existing_path, artifact_to_path
from pilates.utils.beam import get_beam_omx_skims_name
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workspace import Workspace
from pilates.utils.settings_helper import get as get_setting

if TYPE_CHECKING:
    from pilates.workflows.state import WorkflowState


from pilates.beam.tnc_zarr_pipeline import (
    TNC_CONSOLIDATION_MAP as TNC_CONSOLIDATION_MAP,
    _accumulate_all_completed_failed_trips as _accumulate_all_completed_failed_trips,
    _accumulate_all_completed_failed_trips_zarr as _accumulate_all_completed_failed_trips_zarr,
    _consolidate_tnc_data_zarr as _consolidate_tnc_data_zarr,
    _handle_transit_mode_availability as _handle_transit_mode_availability,
    _merge_beam_skims_to_zarr as _pipeline_merge_beam_skims_to_zarr,
    _merge_one_zarr_measure as _merge_one_zarr_measure,
    _merge_one_zarr_measure_from_zarr as _merge_one_zarr_measure_from_zarr,
    _merge_zarr_trip_counts as _merge_zarr_trip_counts,
    _merge_zarr_trip_counts_slice as _merge_zarr_trip_counts_slice,
    _normalize_beam_partial_time_period_coords as _normalize_beam_partial_time_period_coords,
    _postprocess_tnc_zarr as _postprocess_tnc_zarr,
    _process_all_tnc_logic as _pipeline_process_all_tnc_logic,
    _process_all_tnc_logic_zarr as _pipeline_process_all_tnc_logic_zarr,
    _transfer_tnc_provider_data_zarr as _transfer_tnc_provider_data_zarr,
    _transform_measure as _transform_measure,
    copy_skims_for_unobserved_modes as copy_skims_for_unobserved_modes,
)

logger = logging.getLogger(__name__)


def _activitysim_skims_target_enabled(settings: Any) -> bool:
    if isinstance(settings, PilatesConfig):
        return settings.run.models.activity_demand == "activitysim"
    activity_demand = get_setting(settings, "run.models.activity_demand", None)
    if activity_demand is not None:
        return activity_demand == "activitysim"
    return get_setting(settings, "activitysim", None) is not None


def _format_missing_beam_skim_error(
    *,
    expected_names: List[str],
    raw_output_files: List[str],
    forecast_year: Any,
    iteration: Any,
) -> str:
    skim_like_outputs = [
        name
        for name in raw_output_files
        if "skim" in name.lower() or "od" in name.lower()
    ]
    found_suffix = (
        f" Skim-like BEAM outputs found: {skim_like_outputs}."
        if skim_like_outputs
        else " No skim-like BEAM outputs were found."
    )
    return (
        "BEAM postprocess cannot update ActivitySim skims because BEAM did not "
        "emit a raw OD skim artifact for "
        f"year={forecast_year}, iteration={iteration}. Expected one of the "
        f"raw output short_names {expected_names}. Check the BEAM skim output "
        "configuration before restarting. ActivitySim cannot evolve on stale "
        f"skims.{found_suffix}"
    )


def find_latest_beam_iteration(beam_output_dir):
    """
    Return (path_to_latest_iteration_dir, latest_iteration_number).

    BEAM iteration directories are typically named `it.<n>` under an `ITERS` folder.
    Excludes already-archived regional iteration directories (year-\\d+-iteration-\\d+)
    unless beam_output_dir itself points into one.
    """
    if not beam_output_dir or not os.path.exists(beam_output_dir):
        return None, None

    is_already_scoped = bool(re.search(r"year-\d+-iteration-\d+", str(beam_output_dir)))

    iter_dirs = []
    for root, dirs, files in os.walk(beam_output_dir):
        if os.path.basename(root).startswith("."):
            dirs[:] = []
            continue
        if not is_already_scoped:
            # Do not traverse into already finalized past-year/iteration directories
            dirs[:] = [d for d in dirs if not re.match(r"^year-\d+-iteration-\d+$", d)]
        if "ITERS" in dirs:
            iter_dirs.append(os.path.join(root, "ITERS"))

    logger.info("Looking in directories {0}".format(iter_dirs))
    if not iter_dirs:
        return None, None

    last_iters_dir = max(iter_dirs, key=os.path.getmtime)
    it_prefix = "it."

    iteration_nums = []
    for dir_name in os.listdir(last_iters_dir):
        if dir_name.startswith(".") or not dir_name.startswith(it_prefix):
            continue
        try:
            iteration_nums.append(int(dir_name[len(it_prefix) :]))
        except ValueError:
            logger.warning(f"Skipping non-integer iteration directory: {dir_name}")

    logger.info("Looking in directories {0}".format(iteration_nums))
    if not iteration_nums:
        return None, None

    max_it_num = max(iteration_nums)
    target_iter_dir = os.path.join(last_iters_dir, f"{it_prefix}{max_it_num}")
    if not os.path.isdir(target_iter_dir):
        return None, None
    return target_iter_dir, max_it_num


def _sanitize_event_type(event_type: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(event_type)).strip("_")
    return safe or "unknown"


def split_events_parquet_by_type(
    events_path: str,
    event_types: Optional[List[str]] = None,
    *,
    output_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    create_path_traversal_links: bool = False,
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Split a BEAM events parquet file into per-type parquet files.

    Parameters
    ----------
    events_path : str
        Path to the combined events parquet.
    event_types : list[str], optional
        Event types to split. When None, uses all types present.
    output_dir : str, optional
        Output directory for split files (defaults to events_path parent).
    output_prefix : str, optional
        Prefix for output filenames (defaults to events filename stem).
    """
    # NOTE: DuckDB could do this faster with predicate pushdown and Parquet COPY.
    # Sketch:
    #   COPY (
    #     SELECT *, row_number() OVER () - 1 AS PathTraversalEventId
    #     FROM read_parquet('events.parquet')
    #     WHERE type = 'PathTraversal'
    #   ) TO '...PathTraversal.parquet' (FORMAT 'parquet');
    #   COPY (
    #     SELECT
    #       PathTraversalEventId,
    #       UNNEST(STRING_SPLIT(links, ','))::BIGINT AS linkId,
    #       UNNEST(STRING_SPLIT(linkTravelTime, ','))::DOUBLE AS travelTimeSeconds,
    #       ROW_NUMBER() OVER (PARTITION BY PathTraversalEventId) - 1 AS link_index
    #     FROM read_parquet('events.parquet')
    #     WHERE type = 'PathTraversal'
    #   ) TO '...PathTraversal.links.parquet' (FORMAT 'parquet');
    if not os.path.exists(events_path):
        logger.warning("Events parquet not found for split: %s", events_path)
        return {}, None
    if event_types is None:
        try:
            type_col = pd.read_parquet(events_path, columns=["type"])
        except Exception:
            logger.warning(
                "Failed to read 'type' column from events parquet: %s",
                events_path,
                exc_info=True,
            )
            return {}, None
        if "type" not in type_col.columns:
            logger.warning("Events parquet missing 'type' column: %s", events_path)
            return {}, None
        event_types = sorted(type_col["type"].dropna().unique().tolist())
    output_dir = output_dir or os.path.dirname(events_path)
    prefix = output_prefix or os.path.splitext(os.path.basename(events_path))[0]
    written_paths: Dict[str, str] = {}
    path_traversal_links_path: Optional[str] = None
    for event_type in event_types:
        try:
            subset = pd.read_parquet(
                events_path,
                filters=[("type", "==", event_type)],
                engine="pyarrow",
            )
        except Exception:
            logger.warning(
                "Failed to filter events parquet for type %s: %s",
                event_type,
                events_path,
                exc_info=True,
            )
            continue
        if subset.empty:
            continue
        subset = subset.reset_index(drop=True)
        subset[f"{_sanitize_event_type(event_type)}EventId"] = subset.index
        if create_path_traversal_links and event_type == "PathTraversal":
            links = subset.get("links")
            travel_times = subset.get("linkTravelTime")
            if links is not None and travel_times is not None:
                links_split = links.astype(str).str.split(",")
                travel_times_split = travel_times.astype(str).str.split(",")
                link_ids = links_split.explode()
                tt_vals = travel_times_split.explode()
                if len(link_ids) != len(tt_vals):
                    logger.warning(
                        "PathTraversal links and travel time lengths mismatch: %d vs %d",
                        len(link_ids),
                        len(tt_vals),
                    )
                link_table = pd.DataFrame(
                    {
                        "PathTraversalEventId": link_ids.index,
                        "link_index": link_ids.groupby(level=0).cumcount(),
                        "linkId": link_ids,
                        "travelTimeSeconds": tt_vals,
                    }
                )
                link_table["linkId"] = pd.to_numeric(
                    link_table["linkId"], errors="coerce"
                ).astype("Int64")
                link_table["travelTimeSeconds"] = pd.to_numeric(
                    link_table["travelTimeSeconds"], errors="coerce"
                )
                link_table = link_table.replace({"": pd.NA}).dropna(
                    subset=["linkId", "travelTimeSeconds"]
                )
                path_traversal_links_path = os.path.join(
                    output_dir,
                    f"{prefix}.PathTraversal.links.parquet",
                )
                link_table.to_parquet(path_traversal_links_path, index=False)
                subset = subset.drop(columns=["links", "linkTravelTime"])
        subset = subset.dropna(axis=1, how="all")
        safe_type = _sanitize_event_type(event_type)
        out_path = os.path.join(output_dir, f"{prefix}.{safe_type}.parquet")
        subset.to_parquet(out_path, index=False)
        written_paths[event_type] = out_path
    return written_paths, path_traversal_links_path


def _select_last_events_parquet_path(
    raw_outputs: BeamRunOutputs,
    year: int,
    iteration: int,
) -> Optional[Path]:
    target_name = f"events_parquet_{year}_{iteration}"
    best_sub: Optional[Path] = None
    best_sub_idx = -1
    for short_name, path in raw_outputs.raw_outputs.items():
        if short_name == target_name:
            return path
        if short_name.startswith(f"{target_name}_sub"):
            suffix = short_name.split("_sub", 1)[-1]
            try:
                sub_idx = int(suffix)
            except ValueError:
                continue
            if sub_idx > best_sub_idx:
                best_sub_idx = sub_idx
                best_sub = path
    return best_sub


def find_beam_iterations(beam_output_dir):
    iter_dirs = [
        os.path.join(root, dir)
        for root, dirs, files in os.walk(beam_output_dir)
        if not os.path.basename(root).startswith(".")
        for dir in dirs
        if dir == "ITERS"
    ]
    logger.info("Found ITERS directories: {0}".format(iter_dirs))

    if not iter_dirs:
        return {}, None

    # Select the most recently modified ITERS directory
    last_iters_dir = max(iter_dirs, key=os.path.getmtime)
    logger.info("Using ITERS directory: {0}".format(last_iters_dir))

    iteration_dict = {}
    it_prefix = "it."

    # Iterate over subdirectories and collect iteration numbers and paths
    for dir_name in os.listdir(last_iters_dir):
        if dir_name.startswith(it_prefix):
            try:
                it_num = int(dir_name[len(it_prefix) :])
                iteration_dict[it_num] = os.path.join(last_iters_dir, dir_name)
            except ValueError:
                logger.warning(f"Skipping non-integer iteration directory: {dir_name}")

    if not iteration_dict:
        return {}, None

    max_iteration = max(iteration_dict.keys())
    return iteration_dict, max_iteration


def find_iteration_file(
    iteration_path: str, iteration: int, file: str, file_type: Optional[str] = None
) -> Optional[str]:
    suffix = file_type if file_type is not None else ""
    path = os.path.join(iteration_path, f"{iteration}.{file}{suffix}")
    if os.path.exists(path):
        logger.debug(f"Found file at {path}")
        return path
    else:
        logger.debug(f"Could not find file at {path}")
        return None


def write_zarr_skim_as_omx_new(
    all_skims_path,
    settings: PilatesConfig,
    new_skim_name,
    exclude_tables=None,
    workspace: Optional[Workspace] = None,
):
    """
    Write the skims from the Zarr format to an OMX format.

    Parameters
    ----------
    all_skims_path : str
        Path to the main skims Zarr store.
    settings : PilatesConfig
        Settings dictionary, needed for 'region' and 'beam_local_mutable_data_folder'.
    new_skim_name : str
        Name of the new OMX skim file to be created (e.g., 'skims.omx').
    exclude_tables : list, optional
        A list of variable names (strings) from the Zarr dataset
        to exclude from being written to the OMX file. Defaults to None.

    Returns
    -------
    str or None
        The path to the newly created OMX file if successful, otherwise None.
    """
    logger.info(f"Starting conversion of Zarr skims to OMX at {all_skims_path}")

    region = settings.run.region
    beam_input_dir = (
        workspace.get_beam_mutable_data_dir()
        if workspace is not None
        else settings.beam.local_mutable_data_folder
    )

    if not region or not beam_input_dir:
        logger.error(
            "Settings 'region' or 'beam_local_input_folder' are not defined. Cannot determine output path."
        )
        return None

    target_skims_path = os.path.join(beam_input_dir, region, new_skim_name)

    if exclude_tables is None:
        exclude_tables = []

    skims_ds = None
    new_omx_file = None

    try:
        # Open the Zarr dataset
        skims_ds = xr.open_zarr(all_skims_path)
        # Ensure arrays are in memory so in-place edits are captured before re-writing.
        skims_ds = skims_ds.load()
        logger.info(f"Opened Zarr skims file: {all_skims_path}")

        zone_ids = None
        if "original_zone_ids" in skims_ds.attrs:
            zone_ids = skims_ds.attrs["original_zone_ids"]
            logger.info(
                "Using original zone IDs from Zarr attributes: %d zones",
                len(zone_ids),
            )
        elif "taz_ids" in skims_ds.attrs:
            zone_ids = skims_ds.attrs["taz_ids"]
            logger.info(f"Using TAZ IDs from attributes: {len(zone_ids)} zones")
        elif "otaz" in skims_ds.coords:
            if skims_ds["otaz"].attrs.get("preprocessed") != "zero-based-contiguous":
                zone_ids = skims_ds.coords["otaz"].values
                logger.info(f"Using zone IDs from coordinates: {len(zone_ids)} zones")
            elif workspace is not None:
                canonical_zones_df = zone_utils.load_canonical_zones(
                    settings, workspace
                )
                zone_ids = canonical_zones_df.index.tolist()
                logger.warning(
                    "Reconstructed zone IDs from canonical zones after zero-based preprocessing: %d zones",
                    len(zone_ids),
                )
            else:
                zone_ids = skims_ds.coords["otaz"].values
                logger.warning(
                    "Zarr uses zero-based coordinates and no canonical workspace was provided; "
                    "falling back to coordinate values for OMX mapping."
                )
        else:
            logger.warning("No zone coordinate found in Zarr file.")

        # PILATES' original ActivitySim OMX builder uses a sequential 1..N lookup.
        # Preserve canonical/original IDs in Zarr metadata, but emit the original
        # OMX lookup convention here so the exported file is interchangeable with
        # the pre-BEAM ActivitySim input OMXs.
        omx_zone_ids = list(range(1, int(skims_ds.sizes["otaz"]) + 1))

        # Get time periods
        time_periods = []
        if "time_period" in skims_ds.coords:
            time_periods = [str(s) for s in skims_ds.time_period.values]
            logger.info(f"Found {len(time_periods)} time periods: {time_periods}")

        # Prepare output file
        logger.info(f"Target output OMX path: {target_skims_path}")
        os.makedirs(os.path.dirname(target_skims_path), exist_ok=True)
        if os.path.exists(target_skims_path):
            logger.info(f"Deleting existing file: {target_skims_path}")
            os.remove(target_skims_path)

        # Create the new OMX file
        new_omx_file = omx.open_file(target_skims_path, "w")
        logger.info(f"Created new OMX file: {target_skims_path}")

        if omx_zone_ids:
            new_omx_file.create_mapping("zone_id", omx_zone_ids)
            logger.info(
                "Created 'zone_id' mapping with sequential 1-based IDs for %d zones",
                len(omx_zone_ids),
            )

        scaled_measures = {
            "TOTIVT",
            "IVT",
            "WACC",
            "IWAIT",
            "XWAIT",
            "WAUX",
            "WEGR",
            "DTIM",
            "FERRYIVT",
            "KEYIVT",
            "FAR",
        }

        logger.info("Writing matrices to OMX file...")
        written_count = 0

        for key in skims_ds.data_vars:
            if key in exclude_tables:
                logger.info(f"Skipping excluded variable: {key}")
                continue

            try:
                data_array = skims_ds[key]
                data = data_array.values
                measure_name = key.split("_")[-1] if "_" in key else key
                needs_descaling = (
                    measure_name in scaled_measures
                    and not key.startswith(("TNC_", "RH_"))
                )

                if data_array.ndim == 2:
                    data_to_write = np.nan_to_num(data).astype(np.float32)
                    if needs_descaling:
                        data_to_write = data_to_write / 100.0
                    new_omx_file[key] = data_to_write
                    written_count += 1
                    logger.debug(f"Wrote 2D matrix '{key}'")

                elif data_array.ndim == 3:
                    for t_idx, tp in enumerate(time_periods):
                        new_key = f"{key}__{tp}"
                        slice_data = data[:, :, t_idx]
                        data_to_write = np.nan_to_num(slice_data).astype(np.float32)
                        if needs_descaling:
                            data_to_write = data_to_write / 100.0
                        new_omx_file[new_key] = data_to_write

                        # Add attributes
                        parts = key.rsplit("_", 1)
                        if len(parts) == 2:
                            new_omx_file[new_key].attrs["mode"] = parts[0]
                            new_omx_file[new_key].attrs["measure"] = parts[1]
                        else:
                            new_omx_file[new_key].attrs["measure"] = key
                        new_omx_file[new_key].attrs["timePeriod"] = tp

                        written_count += 1
                        logger.debug(f"Wrote 3D slice '{new_key}'")

            except Exception as e:
                logger.error(f"Error writing variable '{key}' to OMX: {e}")

        logger.info(
            f"Finished writing {written_count} matrices to OMX file {target_skims_path}"
        )
        return target_skims_path

    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during Zarr to OMX conversion: {e}"
        )
        return None

    finally:
        # Ensure files are closed
        if new_omx_file:
            try:
                new_omx_file.close()
                logger.info("Closed OMX file")
            except Exception as e:
                logger.error(f"Error closing OMX file: {e}")

        if skims_ds:
            try:
                skims_ds.close()
                logger.info("Closed Zarr dataset")
            except Exception as e:
                logger.error(f"Error closing Zarr dataset: {e}")


def verify_skim_zone_order(
    settings, skim_file_path: str, workspace: "Workspace"
) -> List[str]:
    """
    Verifies that the zone order in a skim file (Zarr or OMX) matches the
    canonical order from the canonical_zones file.

    Args:
        settings: The PilatesConfig object.
        skim_file_path: Path to the skim file to verify.
        workspace: The current run's workspace object.

    Raises:
        ValueError: If the zone orders do not match.
    """
    logger.info(f"--- Verifying zone order for skim file: {skim_file_path} ---")

    # 1. Get Canonical Order
    canonical_zones_df = zone_utils.load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.tolist()
    logger.info(
        f"Successfully loaded canonical order for {len(canonical_order)} zones."
    )

    # 2. Get Skim File Order
    skim_order = None
    store = None
    if skim_file_path.endswith(".zarr") or skim_file_path.endswith(".zarr.zip"):
        try:
            store = xr.open_zarr(skim_file_path)
            if "original_zone_ids" in store.attrs:
                skim_order = store.attrs["original_zone_ids"]
                logger.info(
                    f"Extracted {len(skim_order)} zone IDs from Zarr root attributes."
                )
            elif "original_zone_ids" in store["otaz"].attrs:
                skim_order = store["otaz"].attrs["original_zone_ids"]
                logger.info(
                    f"Extracted {len(skim_order)} zone IDs from Zarr 'otaz' attributes (legacy)."
                )
            else:
                raise ValueError(
                    "Zarr file does not contain 'original_zone_ids' metadata in root or 'otaz' attributes."
                )
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Could not find 'original_zone_ids' on first attempt ({e}). Trying to consolidate metadata."
            )
            if store:
                store.close()
            try:
                import zarr

                zarr.consolidate_metadata(skim_file_path)
                store = xr.open_zarr(skim_file_path, consolidated=True)
                if "original_zone_ids" in store.attrs:
                    skim_order = store.attrs["original_zone_ids"]
                    logger.info(
                        f"Extracted {len(skim_order)} zone IDs after consolidating metadata."
                    )
                else:
                    raise ValueError(
                        "Zarr file does not contain 'original_zone_ids' metadata even after consolidation."
                    )
            except Exception as consolidation_error:
                raise ValueError(
                    f"Zarr file does not contain 'original_zone_ids' metadata. Consolidation attempt failed: {consolidation_error}"
                )

    elif skim_file_path.endswith(".omx"):
        with omx.open_file(skim_file_path, "r") as f:
            if "TAZ" in f.list_mappings():
                mapping_items = sorted(
                    f.mapping("TAZ").items(), key=lambda item: item[1]
                )
                skim_order = [
                    item[0].decode("utf-8")
                    if isinstance(item[0], (bytes, bytearray))
                    else str(item[0])
                    for item in mapping_items
                ]
                logger.info(f"Extracted {len(skim_order)} zone IDs from OMX mapping.")
            else:
                raise ValueError("OMX file does not contain 'TAZ' mapping.")
    else:
        logger.warning(
            f"Unsupported file type for zone order verification: {skim_file_path}"
        )
        return canonical_order

    # 3. Compare Orders
    try:
        if skim_order is None:
            raise ValueError("Could not extract zone order from skim file.")

        if len(skim_order) != len(canonical_order):
            msg = (
                f"Zone count mismatch! Canonical order has {len(canonical_order)} zones, "
                f"but skim file has {len(skim_order)} zones."
            )
            logger.error(msg)
            raise ValueError(msg)

        if skim_order != canonical_order:
            for i, (canonical_id, skim_id) in enumerate(
                zip(canonical_order, skim_order)
            ):
                if canonical_id != skim_id:
                    msg = (
                        f"FATAL: Skim zone order (from original_zone_ids attribute) does not match canonical order! "
                        f"This indicates silent data corruption is occurring in the BEAM run. "
                        f"First mismatch at index {i}: Canonical='{canonical_id}', Skim='{skim_id}'"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

        if store:
            expected_otaz_coords = np.arange(len(canonical_order))
            actual_otaz_coords = store.coords["otaz"].values

            if not np.array_equal(actual_otaz_coords, expected_otaz_coords):
                msg = (
                    f"FATAL: Zarr 'otaz' coordinates are not 0-based as expected! "
                    f"Expected 0-based coordinates up to {len(canonical_order) - 1}, "
                    f"but found coordinates starting with {actual_otaz_coords[0]} and ending with {actual_otaz_coords[-1]}. "
                    f"This indicates BEAM produced 1-based coordinates, which will cause ActivitySim to fail."
                )
                logger.error(msg)
                raise ValueError(msg)
    finally:
        if store:
            store.close()

    logger.info("--- Zone order verification successful. ---")

    return canonical_order


def _merge_beam_skims_to_zarr(
    all_skims_path,
    iteration_skims_path,
    beam_output_dir,
    settings,
    workspace,
    override=None,
    output_skims_path: Optional[str] = None,
) -> Optional[str]:
    """
    Merges current BEAM OMX skims into a Zarr skims store in a single pass.
    Reads from ``all_skims_path`` strictly read-only and writes the merged
    result to ``output_skims_path`` (or ``all_skims_path`` when omitted).
    """
    return _pipeline_merge_beam_skims_to_zarr(
        all_skims_path,
        iteration_skims_path,
        beam_output_dir,
        settings,
        workspace,
        override=override,
        output_skims_path=output_skims_path,
        verify_skim_zone_order_fn=verify_skim_zone_order,
        get_setting_fn=get_setting,
    )


def _process_all_tnc_logic_zarr(
    partial_skims_ds,
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict_all,
    previous_weight=0.9,
):
    """
    Process TNC/RH logic for Zarr format partial skims.
    Either consolidates TNC providers into RH modes or transfers them directly.
    """
    return _pipeline_process_all_tnc_logic_zarr(
        partial_skims_ds,
        skims_ds,
        timePeriods,
        settings,
        completed_failed_dict_all,
        previous_weight=previous_weight,
        get_setting_fn=get_setting,
    )


def _process_all_tnc_logic(
    partialSkims,
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict_all,
):
    """
    Handles all TNC/RH related skim processing:
    - If consolidate_tnc_fleets is True: Consolidates provider data -> RH_* in Zarr, then post-processes RH_*.
    - If consolidate_tnc_fleets is False: Transfers provider data -> TNC_* in Zarr, then post-processes TNC_*.
    """
    return _pipeline_process_all_tnc_logic(
        partialSkims,
        skims_ds,
        timePeriods,
        settings,
        completed_failed_dict_all,
        get_setting_fn=get_setting,
    )


def beam_postprocess_zarr_skims_output_path(
    workspace: Workspace,
    state: Optional["WorkflowState"] = None,
) -> str:
    """Return the declared output Zarr skims path for BEAM postprocessing."""
    year = resolve_forecast_year(state) if state is not None else None
    iteration = state.iteration if state is not None else None
    has_year = isinstance(year, int) and not isinstance(year, bool)
    has_iter = isinstance(iteration, int) and not isinstance(iteration, bool)
    if has_year and has_iter:
        return os.path.join(
            workspace.get_beam_output_dir(),
            ".pilates-consist-outputs",
            "beam_postprocess",
            f"year-{year}-iteration-{iteration}",
            "skims.zarr",
        )
    return os.path.join(
        workspace.get_beam_output_dir(),
        ".pilates-consist-outputs",
        "beam_postprocess",
        "skims.zarr",
    )


class BeamPostprocessor(GenericPostprocessor):
    """
    Postprocessor for BEAM model.
    """

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this postprocessor produces.

        Notes
        -----
        Output keys
            - ``final_skims_omx``: Final OMX skims written to the BEAM
              mutable data directory when OMX output is enabled.
            - ``zarr_skims``: Updated Zarr skims after BEAM postprocessing
              whenever ActivitySim is enabled.
        Related docs
            - See `pilates/beam/inputs.py` for the corresponding input
              descriptions used by BEAM and downstream models.
        """
        outputs: Dict[str, Any] = {}
        if _activitysim_skims_target_enabled(settings):
            outputs["zarr_skims"] = beam_postprocess_zarr_skims_output_path(
                workspace, state
            )

        if requires_omx_skim_output(settings):
            region = settings.run.region
            omx_name = get_beam_omx_skims_name(settings)
            outputs["final_skims_omx"] = os.path.join(
                workspace.get_beam_mutable_data_dir(), region, omx_name
            )
        return outputs

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.skim_format = get_setting(
            self.state.full_settings, "shared.skims.fname", "skimsActivitySimOD_current"
        )

    def _postprocess(
        self,
        raw_outputs: BeamRunOutputs,
        workspace: Workspace,
        zarr_skims: Optional[Any] = None,
    ) -> BeamPostprocessOutputs:
        """
        Postprocesses the raw outputs from a BEAM run by merging skims into a new Zarr store.
        """
        logger.info("Running BEAM postprocessor...")
        settings = self.state.full_settings

        raw_output_files = list(raw_outputs.raw_outputs.keys())
        split_events: Dict[str, Path] = {}
        split_event_links: Dict[str, Path] = {}

        events_path_obj = _select_last_events_parquet_path(
            raw_outputs, self.state.forecast_year, self.state.iteration
        )
        if events_path_obj is not None:
            events_path = str(events_path_obj)
            try:
                written_paths, links_path = split_events_parquet_by_type(
                    events_path,
                    create_path_traversal_links=True,
                )
                if written_paths:
                    logger.info(
                        "Wrote %d split events parquet files from %s",
                        len(written_paths),
                        events_path,
                    )
                    for event_type, out_path in written_paths.items():
                        safe_type = _sanitize_event_type(event_type)
                        short_name = (
                            f"events_parquet_{self.state.forecast_year}_"
                            f"{self.state.iteration}_type_{safe_type}"
                        )
                        resolved = artifact_to_path(out_path, workspace)
                        if resolved is not None:
                            split_events[short_name] = Path(resolved)
                if links_path:
                    short_name = (
                        f"path_traversal_links_{self.state.forecast_year}_"
                        f"{self.state.iteration}"
                    )
                    resolved = artifact_to_path(links_path, workspace)
                    if resolved is not None:
                        split_event_links[short_name] = Path(resolved)
            except Exception:
                logger.warning(
                    "Failed to split events parquet for %s", events_path, exc_info=True
                )

        input_skims_path = artifact_to_existing_path(
            zarr_skims,
            workspace=workspace,
        )
        if input_skims_path is None and _activitysim_skims_target_enabled(settings):
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                input_skims_path = candidate

        output_skims_path = beam_postprocess_zarr_skims_output_path(
            workspace, self.state
        )
        active_skims_path: Optional[str] = None

        zarr_skim_name = (
            f"{self.skim_format}_{self.state.forecast_year}_{self.state.iteration}"
        )
        omx_skim_name = (
            f"raw_od_skims_{self.state.forecast_year}_{self.state.iteration}"
        )

        # List of possible skim names to check
        possible_skim_names = [
            zarr_skim_name,
            f"raw_od_skims_zarr_{self.state.forecast_year}_{self.state.iteration}",
            omx_skim_name,
        ]

        skim_name_found = None
        for name in possible_skim_names:
            if name in raw_output_files:
                if name != zarr_skim_name:
                    logger.debug(
                        f"Could not find skim file with name {zarr_skim_name}, but found {name}. Proceeding with this file."
                    )
                skim_name_found = name
                break

        if not skim_name_found:
            for name in raw_output_files:
                if "skims" in name and name.endswith(".zarr"):
                    logger.warning(
                        f"Could not find skim file with expected names, but found {name}. Proceeding with this file."
                    )
                    skim_name_found = name
                    break

        merged_any = False
        if not skim_name_found:
            logger.warning(
                "Raw BEAM OD skims file not found in raw_outputs. Skim merging will be skipped."
            )
            if _activitysim_skims_target_enabled(settings):
                raise RuntimeError(
                    _format_missing_beam_skim_error(
                        expected_names=possible_skim_names,
                        raw_output_files=raw_output_files,
                        forecast_year=self.state.forecast_year,
                        iteration=self.state.iteration,
                    )
                )

        elif input_skims_path is None or not os.path.exists(input_skims_path):
            logger.warning(
                f"Input Zarr skims file not found at {input_skims_path}. Cannot proceed with merging."
            )
        else:
            skim_name = skim_name_found

            all_skims_found = [
                str(path)
                for key, path in raw_outputs.raw_outputs.items()
                if key == skim_name
            ]

            logger.info("Here are all the skim files found")
            logger.info(all_skims_found)

            beam_output_dir = workspace.get_beam_output_dir()

            merged_sources = []
            for it, it_skim in enumerate(all_skims_found):
                raw_od_skims_path = os.path.join(
                    workspace.full_path,
                    it_skim,
                )
                merged_sources.append(raw_od_skims_path)

                logger.info(f"Processing {raw_od_skims_path} for iteration {it}")

                source_skims_path = (
                    input_skims_path
                    if it == 0
                    else (active_skims_path or output_skims_path)
                )
                updated_skims_path = _merge_beam_skims_to_zarr(
                    all_skims_path=source_skims_path,
                    iteration_skims_path=raw_od_skims_path,
                    beam_output_dir=beam_output_dir,
                    settings=settings,
                    workspace=workspace,
                    override=raw_od_skims_path,
                    output_skims_path=output_skims_path,
                )

                if updated_skims_path:
                    merged_any = True
                    active_skims_path = updated_skims_path

        zarr_path: Optional[Path] = None
        final_omx_output_path: Optional[Path] = None

        if merged_any and active_skims_path is not None:
            resolved_zarr = artifact_to_path(active_skims_path, workspace)
            if resolved_zarr is not None:
                zarr_path = Path(resolved_zarr)

        if requires_omx_skim_output(settings):
            logger.info(
                "Writing skims to OMX file for UrbanSim, ATLAS, or other downstream models..."
            )
            if active_skims_path is None or not os.path.exists(active_skims_path):
                logger.error(
                    "Main Zarr skims file not available. Cannot write skims to OMX."
                )
            else:
                try:
                    # Exclude intermediate trip/failure tables from the final OMX
                    vars_to_exclude = []
                    with xr.open_zarr(active_skims_path) as skims_ds:
                        for key in skims_ds.data_vars:
                            if (
                                key.endswith("_TRIPS")
                                or key.endswith("_FAILURES")
                                or key.endswith("_REJECTIONPROB")
                            ):
                                vars_to_exclude.append(key)

                    final_omx_path = write_zarr_skim_as_omx_new(
                        active_skims_path,
                        settings,
                        get_beam_omx_skims_name(settings),
                        exclude_tables=vars_to_exclude,
                        workspace=workspace,
                    )

                    if final_omx_path:
                        resolved_omx = artifact_to_path(final_omx_path, workspace)
                        if resolved_omx is not None:
                            final_omx_output_path = Path(resolved_omx)
                except Exception as e:
                    logger.error(f"Failed to write skims to OMX: {e}")

        return BeamPostprocessOutputs(
            zarr_skims=zarr_path,
            final_skims_omx=final_omx_output_path,
            split_events=split_events,
            split_event_links=split_event_links,
        )

    def postprocess(
        self,
        raw_outputs: BeamRunOutputs,
        workspace: Workspace,
        zarr_skims: Optional[Any] = None,
    ) -> BeamPostprocessOutputs:
        """
        Postprocess typed BEAM run outputs and return typed outputs.
        """
        if not isinstance(raw_outputs, BeamRunOutputs):
            raise TypeError("BeamPostprocessor.postprocess expects BeamRunOutputs")
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(
            raw_outputs,
            workspace,
            zarr_skims=zarr_skims,
        )
