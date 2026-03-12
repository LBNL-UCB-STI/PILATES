import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import openmatrix as omx
import pandas as pd

from pilates.config import PilatesConfig
from pilates.beam.outputs import BeamPostprocessOutputs, BeamRunOutputs
import pilates.utils.zone_utils as zone_utils
from pilates.utils.zone_utils import ensure_0_based_and_flag_zarr_skims

try:
    import xarray as xr
except Exception:
    print("FAILED TO LOAD XARRAY or ZARR")

from pilates.activitysim.preprocessor import zone_order
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workspace import Workspace
from pilates.utils.settings_helper import get as get_setting

if TYPE_CHECKING:
    from workflow_state import WorkflowState


logger = logging.getLogger(__name__)

TNC_CONSOLIDATION_MAP = {
    "TNC_SINGLE": "RH_SOLO",
    "TNC_POOLED": "RH_SHARED",
}


def _activitysim_skims_target_enabled(settings: PilatesConfig) -> bool:
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None)
    return getattr(model_cfg, "activity_demand", None) == "activitysim"


def find_latest_beam_iteration(beam_output_dir):
    """
    Return (path_to_latest_iteration_dir, latest_iteration_number).

    BEAM iteration directories are typically named `it.<n>` under an `ITERS` folder.
    """
    iter_dirs = [
        os.path.join(root, dir_name)
        for root, dirs, files in os.walk(beam_output_dir)
        if not os.path.basename(root).startswith(".")
        for dir_name in dirs
        if dir_name == "ITERS"
    ]
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
    return os.path.join(last_iters_dir, f"{it_prefix}{max_it_num}"), max_it_num


def find_produced_od_skims(beam_output_dir, suffix="csv.gz"):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    od_skims_path = os.path.join(
        iteration_dir, "{0}.skimsActivitySimOD_current.{1}".format(it_num, suffix)
    )
    logger.info("expecting skims at {0}".format(od_skims_path))
    return od_skims_path


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


def _select_last_events_parquet_record(
    raw_outputs: RecordStore,
    year: int,
    iteration: int,
) -> Optional[FileRecord]:
    target_name = f"events_parquet_{year}_{iteration}"
    best_sub = None
    best_sub_idx = -1
    for record in raw_outputs.all_records():
        short_name = getattr(record, "short_name", "")
        if short_name == target_name:
            return record
        if short_name.startswith(f"{target_name}_sub"):
            suffix = short_name.split("_sub", 1)[-1]
            try:
                sub_idx = int(suffix)
            except ValueError:
                continue
            if sub_idx > best_sub_idx:
                best_sub_idx = sub_idx
                best_sub = record
    return best_sub


def find_produced_linkstats(beam_output_dir, suffix="csv.gz"):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    od_skims_path = os.path.join(
        iteration_dir, "{0}.linkstats.{1}".format(it_num, suffix)
    )
    logger.info("expecting linkstats at {0}".format(od_skims_path))
    return od_skims_path


def find_produced_plans(beam_output_dir, suffix="csv.gz"):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    od_skims_path = os.path.join(iteration_dir, "{0}.plans.{1}".format(it_num, suffix))
    logger.info("expecting output plans at {0}".format(od_skims_path))
    return od_skims_path


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
    iteration_path: str, iteration: int, file: str, file_type: str = ""
) -> Optional[str]:
    path = os.path.join(iteration_path, f"{iteration}.{file}{file_type}")
    if os.path.exists(path):
        logger.debug(f"Found file at {path}")
        return path
    else:
        logger.warning(f"Could not find file at {path}")
        return None


def find_produced_origin_skims(beam_output_dir):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    ridehail_skims_path = os.path.join(iteration_dir, f"{it_num}.skimsRidehail.csv.gz")
    logger.info("expecting skims at {0}".format(ridehail_skims_path))
    return ridehail_skims_path


def copy_skims_for_unobserved_modes(mapping, skims_ds):
    """
    Copy skim data from one mode to others based on a mapping.
    Operates on the Zarr dataset directly.

    Parameters
    ----------
    mapping : dict
        Mapping from source mode to list of target modes (e.g., {"SOV": ["SOVTOLL", ...]}).
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    """
    logger.info("Copying skims for unobserved modes...")
    for fromMode, toModes in mapping.items():
        # Quick fix for config issue where a string is provided instead of a list
        if isinstance(toModes, str):
            toModes = [toModes]

        all_vars = list(skims_ds.data_vars)

        for var_name in all_vars:
            if var_name.startswith(fromMode + "_") and not any(
                m in var_name for m in ["_TRIPS", "_FAILURES"]
            ):
                if "TOLL" in var_name.split("_")[0]:
                    continue

                # Correctly extract measure name for complex mode names
                measure_name = var_name[len(fromMode) + 1 :]

                for toMode in toModes:
                    target_var_name = f"{toMode}_{measure_name}"

                    if target_var_name in all_vars:
                        if skims_ds[var_name].shape == skims_ds[target_var_name].shape:
                            skims_ds[target_var_name].data[:] = skims_ds[var_name].data[
                                :
                            ]
                            logger.info(
                                f"Copied data from '{var_name}' to '{target_var_name}'"
                            )
                        else:
                            logger.warning(
                                f"Shape mismatch when copying from '{var_name}' to '{target_var_name}'. Skipping."
                            )
                    else:
                        logger.debug(
                            f"Target variable '{target_var_name}' not found in dataset. Skipping copy."
                        )

    logger.info("Completed copying skims for unobserved modes.")


def _postprocess_tnc_zarr(
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict,
    use_rh_modes=False,
):
    """
    Applies TNC/RH-specific post-processing rules (REJECTIONPROB, IWAIT filling,
    DDIST/TOTIVT interpolation, FAR assignment) to Zarr skims.
    Operates on either TNC_* or RH_* variables based on `use_rh_modes`.

    Parameters:
    -----------
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    timePeriods : list of str
        List of time period names.
    settings : dict
        Settings dictionary, needed for SOV keys and potential FAR value.
    completed_failed_dict : dict
        Dictionary containing completed and failed trip counts aggregated across
        all time periods for each mode path. Keyed by mode path, values are
        [completed_trips_3d_array, failed_trips_3d_array]. This dict should
        contain counts for the modes being processed (either TNC providers or RH).
    use_rh_modes : bool
        If True, process RH_* modes. If False, process TNC_* provider modes.
    """
    logger.info("Applying TNC/RH-specific post-processing...")
    if use_rh_modes:
        # Process consolidated RH modes
        modes_to_process = [
            target_mode
            for source_mode, target_mode in TNC_CONSOLIDATION_MAP.items()
            if f"{target_mode}_TRIPS"
            in skims_ds.data_vars  # Only process if the consolidated variable exists
        ]
        logger.info(f"Post-processing consolidated RH modes: {modes_to_process}")
    else:
        # Process original TNC provider modes
        # Find provider-specific TNC modes like TNC_SINGLE_UBER, TNC_POOLED_LYFT
        all_vars = list(skims_ds.data_vars)
        tnc_provider_modes = set()
        for var_name in all_vars:
            if var_name.startswith("TNC_") and "_TRIPS" in var_name:
                parts = var_name.split("_")
                if len(parts) >= 3 and parts[2] not in [
                    "TRIPS",
                    "FAILURES",
                    "REJECTIONPROB",
                    "IWAIT",
                    "DTIM",
                    "DDIST",
                    "TOTIVT",
                    "FAR",
                ]:
                    tnc_provider_modes.add("_".join(parts[:3]))  # e.g., TNC_SINGLE_UBER
                elif len(parts) >= 2 and parts[1] in ["SINGLE", "POOLED"]:
                    key_parts = var_name.split("__")  # split off period
                    if len(key_parts) > 0:
                        measure_part = key_parts[0]  # e.g. TNC_SINGLE_UBER_TRIPS
                        # Split from the right, looking for measure names
                        measure_found = False
                        for common_measure in [
                            "TRIPS",
                            "FAILURES",
                            "REJECTIONPROB",
                            "IWAIT",
                            "DTIM",
                            "DDIST",
                            "TOTIVT",
                            "FAR",
                        ]:
                            if measure_part.endswith("_" + common_measure):
                                mode_name = measure_part[
                                    : -len("_" + common_measure)
                                ]  # e.g., TNC_SINGLE_UBER
                                tnc_provider_modes.add(mode_name)
                                measure_found = True
                                break
                        if not measure_found:
                            logger.debug(
                                f"Could not parse TNC provider mode from variable: {var_name}"
                            )

        modes_to_process = list(tnc_provider_modes)
        logger.info(f"Post-processing original TNC provider modes: {modes_to_process}")

    if not modes_to_process:
        logger.info("No TNC/RH modes found in skims dataset to post-process.")
        return

    # Get SOV skims needed for interpolation ratios
    sov_dist_da = skims_ds.get("SOV_DIST")
    sov_time_da = skims_ds.get("SOV_TIME")
    if (sov_dist_da is None or sov_time_da is None) and any(
        mode.startswith("RH_") for mode in modes_to_process
    ):
        logger.warning(
            "SOV_DIST or SOV_TIME missing, DDIST/TOTIVT interpolation for RH modes will be skipped."
        )

    for mode in modes_to_process:
        logger.debug(f"Post-processing mode: {mode}")

        # Get completed and failed trip data for this mode (3D arrays)
        # These should already be in skims_ds if merge/consolidation happened
        completed_key = f"{mode}_TRIPS"
        failed_key = f"{mode}_FAILURES"

        completed_da = skims_ds.get(completed_key)
        failed_da = skims_ds.get(failed_key)

        if completed_da is None or failed_da is None:
            logger.warning(
                f"Missing {completed_key} or {failed_key} for TNC/RH post-processing. Skipping {mode}."
            )
            continue

        completed_3d = np.nan_to_num(completed_da.data)
        failed_3d = np.nan_to_num(failed_da.data)

        # Get other relevant TNC/RH data arrays
        rejection_prob_da = skims_ds.get(f"{mode}_REJECTIONPROB")
        iwait_da = skims_ds.get(f"{mode}_IWAIT")
        ddist_da = skims_ds.get(f"{mode}_DDIST")
        totivt_da = skims_ds.get(f"{mode}_TOTIVT")
        far_da = skims_ds.get(f"{mode}_FAR")

        # --- REJECTIONPROB ---
        if rejection_prob_da is not None:
            logger.debug(f"Calculating REJECTIONPROB for {mode}")
            total_trips_3d = completed_3d + failed_3d
            # Calculate OD-level probability where total trips > 0
            valid_ods_mask_3d = total_trips_3d > 0

            rejection_prob_3d = np.zeros_like(completed_3d, dtype=np.float32)
            np.divide(
                failed_3d,
                total_trips_3d,
                out=rejection_prob_3d,
                where=valid_ods_mask_3d,
            )

            # Apply origin-level probability where origin total trips > 0 (This matches original OMX logic)
            completed_sum_by_origin_3d = completed_3d.sum(
                axis=1
            )  # Sum over destination dimension
            failed_sum_by_origin_3d = failed_3d.sum(
                axis=1
            )  # Sum over destination dimension
            total_trips_origin_3d = completed_sum_by_origin_3d + failed_sum_by_origin_3d
            valid_origins_mask_3d = (
                total_trips_origin_3d > 0
            )  # Mask is (zones, periods)

            # Calculate origin-level probs
            origin_probs_3d = np.zeros_like(
                completed_sum_by_origin_3d, dtype=np.float32
            )
            np.divide(
                failed_sum_by_origin_3d,
                total_trips_origin_3d,
                out=origin_probs_3d,
                where=valid_origins_mask_3d,
            )  # origin_probs_3d shape (zones, periods)

            # Apply origin-level probability to rows where origin total > 0
            for tp_idx in range(len(timePeriods)):
                valid_origins_tp = valid_origins_mask_3d[:, tp_idx]  # (zones,)
                origin_probs_tp = origin_probs_3d[:, tp_idx]  # (zones,)
                rejection_prob_da.data[valid_origins_tp, :, tp_idx] = origin_probs_tp[
                    valid_origins_tp, None
                ]  # Apply row-wise

            logger.debug(f"Updated REJECTIONPROB for {mode}")

        # --- IWAIT ---
        if iwait_da is not None:
            logger.debug(f"Processing IWAIT for {mode}")

            # Calculate weighted mean wait time by origin *using the TRIPS data from skims_ds*
            # Sum waitTime * completed per origin, divide by sum completed per origin
            # Use the data already in the Zarr slice (`iwait_da.data`) which was merged/consolidated
            # Apply scaling (100) if needed - _merge_zarr_skim does this. If consolidate logic does it,
            # the data should already be scaled. Assume data is already scaled (e.g. in 0.01 minutes).
            # The postprocessing logic *should* happen on scaled data.

            current_iwait_data_3d = np.nan_to_num(
                iwait_da.data
            )  # Data should be in 0.01 minutes from merge/consolidate
            completed_3d_scaled = (
                completed_3d  # Completed is count, no scaling needed for weight
            )

            sum_weighted_wait_3d = np.nansum(
                current_iwait_data_3d * completed_3d_scaled, axis=1
            )  # Shape (zones, periods)
            sum_completed_origin_3d = np.nansum(
                completed_3d_scaled, axis=1
            )  # Shape (zones, periods)

            # Handle division by zero - origins with no completed trips will have NaN mean initially
            weighted_mean_by_origin_3d = np.full_like(
                sum_completed_origin_3d, np.nan, dtype=np.float32
            )
            valid_origins_for_mean_3d = sum_completed_origin_3d != 0
            np.divide(
                sum_weighted_wait_3d,
                sum_completed_origin_3d,
                out=weighted_mean_by_origin_3d,
                where=valid_origins_for_mean_3d,
            )  # Shape (zones, periods)

            # Identify cells to fill: where completed is 0 AND the origin had some completed trips
            # Apply this filling logic period by period for clarity
            for tp_idx in range(len(timePeriods)):
                completed_tp = completed_3d[:, :, tp_idx]  # (zones, zones)
                valid_origins_tp = valid_origins_for_mean_3d[:, tp_idx]  # (zones,)
                weighted_mean_tp = weighted_mean_by_origin_3d[:, tp_idx]  # (zones,)

                # Mask for cells where completed == 0 AND the origin has at least one completed trip *in this period*
                mask_to_fill_with_average = (completed_tp == 0) & (
                    valid_origins_tp[:, None]
                )  # (zones, zones)

                # Apply the weighted mean by origin
                if mask_to_fill_with_average.any():
                    # Get the origin index for each cell to fill
                    origin_indices_to_fill = np.where(mask_to_fill_with_average)[
                        0
                    ]  # Row indices (origins)
                    iwait_da.data[mask_to_fill_with_average, tp_idx] = weighted_mean_tp[
                        origin_indices_to_fill
                    ]

                    logger.debug(
                        f"Filled {mask_to_fill_with_average.sum()} {mode} IWAIT values in {timePeriods[tp_idx]} with origin weighted average."
                    )

            # Handle bad values (NaNs) that might still exist if an origin had no completed trips at all
            # These remain 0 after _merge_zarr_skim or _consolidate_tnc_data
            # Check if any NaNs were introduced or remain (they shouldn't be if init was zeros/nan)
            # iwait_da.data[:, :, tp_idx] = iwait_slice # Update Zarr slice - not needed if modifying .data in place

            logger.debug(f"Updated IWAIT for {mode}")

        # --- DDIST and TOTIVT (Interpolation using SOV) ---
        # Requires SOV_DIST and SOV_TIME skims to be present in skims_ds

        # DDIST
        if ddist_da is not None and sov_dist_da is not None:
            logger.debug(f"Processing DDIST for {mode}")
            current_ddist_data_3d = np.nan_to_num(
                ddist_da.data
            )  # Data should be in miles/feet? Assume consistent units
            sov_dist_data_3d = np.nan_to_num(sov_dist_da.data)

            # Cells where we can calculate the ratio: SOV_DIST > 0, MODE_TRIPS > 0, MODE_DDIST > 0
            # Use 3D arrays for this calculation across all periods
            mask_for_ratio_3d = (
                (sov_dist_data_3d > 0)
                & (completed_3d > 0)
                & (current_ddist_data_3d > 0)
            )

            ratio = np.nan
            # Calculate overall weighted average ratio if there are any valid points across all periods
            if mask_for_ratio_3d.any():
                # Calculate weighted average ratio: (MODE_DDIST * Completed).sum() / (SOV_DIST * Completed).sum()
                weighted_mode_ddist_3d = (
                    current_ddist_data_3d[mask_for_ratio_3d]
                    * completed_3d[mask_for_ratio_3d]
                )
                weighted_sov_dist_3d = (
                    sov_dist_data_3d[mask_for_ratio_3d]
                    * completed_3d[mask_for_ratio_3d]
                )
                sum_weighted_sov = np.sum(weighted_sov_dist_3d)
                if sum_weighted_sov > 0:
                    ratio = np.sum(weighted_mode_ddist_3d) / sum_weighted_sov

                ratios_individual = (
                    current_ddist_data_3d[mask_for_ratio_3d]
                    / sov_dist_data_3d[mask_for_ratio_3d]
                )
                logger.info(
                    f"Observed {mode} DDIST/SOV DIST ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {mode}. "
                    f"Interpolating {np.sum(~mask_for_ratio_3d):.0f} missing values."
                )
            else:
                logger.info(
                    f"No data points available to calculate {mode} DDIST/SOV DIST ratio for {mode}."
                )

            # Apply minimum ratio check (from feature branch logic)
            min_ratio = 0.8
            if not np.isnan(ratio) and ratio < min_ratio:
                logger.warning(
                    f"Calculated {mode} DDIST/SOV DIST ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}."
                )
                ratio = min_ratio

            # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
            mask_to_interpolate_3d = ~mask_for_ratio_3d
            if not np.isnan(ratio) and mask_to_interpolate_3d.any():
                # Interpolate using SOV_DIST where needed
                # Ensure we don't use SOV_DIST values that are zero for interpolation
                mask_interpolation_valid_sov_3d = mask_to_interpolate_3d & (
                    sov_dist_data_3d > 0
                )
                # Apply interpolated DDIST value (distance measure, no 100x scaling)
                ddist_da.data[mask_interpolation_valid_sov_3d] = (
                    sov_dist_data_3d[mask_interpolation_valid_sov_3d] * ratio
                )  # Assumes SOV_DIST is in consistent units (miles/feet)
                logger.debug(
                    f"Interpolated {mask_interpolation_valid_sov_3d.sum()} {mode} DDIST values using SOV_DIST ratio."
                )

            # Handle remaining NaNs if any (e.g., where SOV_DIST was also 0 or ratio wasn't calculable)
            if np.isnan(ddist_da.data).any():
                nan_count = np.isnan(ddist_da.data).sum()
                ddist_da.data[np.isnan(ddist_da.data)] = (
                    0.0  # Default remaining NaNs to 0
                )
                logger.debug(f"Set {nan_count} remaining NaN {mode} DDIST values to 0.")

            logger.debug(f"Updated DDIST for {mode}")
        elif ddist_da is not None:
            logger.warning(
                f"{mode} DDIST cannot be interpolated: SOV_DIST is missing from skims."
            )

        # TOTIVT
        if totivt_da is not None and sov_time_da is not None:
            logger.debug(f"Processing TOTIVT for {mode}")
            current_totivt_data_3d = np.nan_to_num(totivt_da.data)
            sov_time_data_3d = np.nan_to_num(sov_time_da.data)

            # Cells where we can calculate the ratio: SOV_TIME > 0, MODE_TRIPS > 0, MODE_TOTIVT > 0
            # Use 3D arrays for this calculation across all periods
            mask_for_ratio_3d = (
                (sov_time_data_3d > 0)
                & (completed_3d > 0)
                & (current_totivt_data_3d > 0)
            )

            ratio = np.nan
            # Calculate overall weighted average ratio if there are any valid points across all periods
            if mask_for_ratio_3d.any():
                # Calculate weighted average ratio: (MODE_TOTIVT * Completed).sum() / (SOV_TIME * Completed).sum()
                weighted_mode_totivt_3d = (
                    current_totivt_data_3d[mask_for_ratio_3d]
                    * completed_3d[mask_for_ratio_3d]
                )
                weighted_sov_time_3d = (
                    sov_time_data_3d[mask_for_ratio_3d]
                    * completed_3d[mask_for_ratio_3d]
                )
                sum_weighted_sov = np.sum(weighted_sov_time_3d)
                if sum_weighted_sov > 0:
                    ratio = np.sum(weighted_mode_totivt_3d) / sum_weighted_sov

                ratios_individual = (
                    current_totivt_data_3d[mask_for_ratio_3d]
                    / sov_time_data_3d[mask_for_ratio_3d]
                )
                logger.info(
                    f"Observed {mode} TOTIVT/SOV TIME ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {mode}. "
                    f"Interpolating {np.sum(~mask_for_ratio_3d):.0f} missing values."
                )
            else:
                logger.info(
                    f"No data points available to calculate {mode} TOTIVT/SOV TIME ratio for {mode}."
                )

            # Apply minimum ratio check (from feature branch logic)
            min_ratio = 0.8  # Same minimum as DDIST in feature branch
            if not np.isnan(ratio) and ratio < min_ratio:
                logger.warning(
                    f"Calculated {mode} TOTIVT/SOV TIME ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}."
                )
                ratio = min_ratio

            # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
            mask_to_interpolate_3d = ~mask_for_ratio_3d
            if not np.isnan(ratio) and mask_to_interpolate_3d.any():
                # Interpolate using SOV_TIME where needed
                # Ensure we don't use SOV_TIME values that are zero for interpolation
                mask_interpolation_valid_sov_3d = mask_to_interpolate_3d & (
                    sov_time_data_3d > 0
                )
                totivt_da.data[mask_interpolation_valid_sov_3d] = (
                    sov_time_data_3d[mask_interpolation_valid_sov_3d] * ratio
                )
                logger.debug(
                    f"Interpolated {mask_interpolation_valid_sov_3d.sum()} {mode} TOTIVT values using SOV_TIME ratio."
                )

            # Handle remaining NaNs if any (e.g., where SOV_TIME was also 0 or ratio wasn't calculable)
            if np.isnan(totivt_da.data).any():
                nan_count = np.isnan(totivt_da.data).sum()
                totivt_da.data[np.isnan(totivt_da.data)] = (
                    0.0  # Default remaining NaNs to 0
                )
                logger.debug(
                    f"Set {nan_count} remaining NaN {mode} TOTIVT values to 0."
                )

            logger.debug(f"Updated TOTIVT for {mode}")
        elif totivt_da is not None:
            logger.warning(
                f"{mode} TOTIVT cannot be interpolated: SOV_TIME is missing from skims."
            )

        # --- FAR ---
        if far_da is not None:
            logger.debug(f"Processing FAR for {mode}")
            # Assuming FAR should be set to a default value where it wasn't observed or is zero
            # Original logic seems to set FAR to 0 except where explicitly calculated, or set to a fixed value.
            # The OMX code did not explicitly modify FAR, but the postprocessor had commented out code.
            # Let's default to 0 where TRIPS are zero, or a fixed value if settings provide one.
            default_far = 0.0  # Replaces legacy tnc_default_far setting

            # Set FAR to default_far where TRIPS are zero or NaN
            mask_zero_trips = (completed_3d + failed_3d) == 0
            mask_nan_far = np.isnan(far_da.data)
            mask_to_set = mask_zero_trips | mask_nan_far

            if mask_to_set.any():
                far_da.data[mask_to_set] = default_far
                logger.debug(
                    f"Set {mask_to_set.sum()} {mode} FAR values to default {default_far}"
                )

            # Ensure FAR is not negative
            negative_far = far_da.data < 0
            if negative_far.any():
                logger.warning(
                    f"Found {negative_far.sum()} negative FAR values for {mode}. Setting to 0."
                )
                far_da.data[negative_far] = 0.0

            logger.debug(f"Updated FAR for {mode}")

    logger.info("Completed TNC/RH-specific post-processing.")


def clear_skim_cache(asim_local_output_dir):
    skims_path = os.path.join(asim_local_output_dir, "cache")
    if os.path.exists(skims_path):
        logger.info(
            "Deleting skims cache at {0}. Eventually we should modify it in place".format(
                skims_path
            )
        )
        shutil.rmtree(skims_path)
    else:
        logger.warning("Did not find skim cache to delete")


def _accumulate_all_completed_failed_trips_zarr(partial_skims_ds, timePeriods):
    """
    Accumulates completed and failed trip counts from all modes/providers
    in the partial Zarr dataset.

    Parameters:
    -----------
    partial_skims_ds : xarray.Dataset
        The Zarr dataset containing the partial skims from BEAM.
    timePeriods : list of str
        List of time period names.

    Returns:
    --------
    dict
        Dictionary where keys are mode paths and values are tuples:
        (completed_trips_3d_array, failed_trips_3d_array).
    """

    if "time_period" in partial_skims_ds.coords:
        zarr_time_periods = [str(tp) for tp in partial_skims_ds.time_period.values]
        if zarr_time_periods != timePeriods:
            logger.warning(
                f"Time periods mismatch: Zarr has {zarr_time_periods}, "
                f"expected {timePeriods}. Using Zarr periods."
            )

    logger.info(
        "Accumulating completed and failed trip counts from partial Zarr skims..."
    )
    completed_failed_dict = {}

    # Get all data variables
    for var_name in partial_skims_ds.data_vars:
        # Check if it's a TRIPS or FAILURES variable
        is_trips = var_name.endswith("_TRIPS")
        is_failures = var_name.endswith("_FAILURES")

        if is_trips or is_failures:
            # Extract the mode path
            if is_trips:
                mode = var_name[: -len("_TRIPS")]
            else:  # is_failures
                mode = var_name[: -len("_FAILURES")]

            if mode not in completed_failed_dict:
                # Get the shape from the data variable
                shape = partial_skims_ds[var_name].shape
                completed_failed_dict[mode] = [
                    np.zeros(shape, dtype=np.float32),  # Completed trips 3D
                    np.zeros(shape, dtype=np.float32),  # Failed trips 3D
                ]

            try:
                data_3d = np.nan_to_num(partial_skims_ds[var_name].values)

                if is_trips:
                    completed_failed_dict[mode][0] = data_3d
                else:  # is_failures
                    completed_failed_dict[mode][1] = data_3d

            except Exception as e:
                logger.error(f"Error reading partial Zarr skim {var_name}: {e}")

    logger.info(
        f"Accumulated trip counts for {len(completed_failed_dict)} modes/providers."
    )
    return completed_failed_dict


def _accumulate_all_completed_failed_trips(partialSkims, timePeriods):
    """
    Accumulates completed and failed trip counts from all modes/providers
    in the partial OMX skim file.

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM.
    timePeriods : list of str
        List of time period names.

    Returns:
    --------
    dict
        Dictionary where keys are mode paths (e.g., "SOV", "TNC_SINGLE_UBER", "WLK_TRN_WLK")
        and values are tuples: (completed_trips_3d_array, failed_trips_3d_array).
    """
    logger.info("Accumulating completed and failed trip counts from partial skims...")
    completed_failed_dict = {}

    try:
        # Get shape from a known 3D matrix, fallback to 2D shape if needed
        sample_matrix = next(
            (
                m
                for m in partialSkims.list_matrices()
                if m.endswith("__" + timePeriods[0])
            ),
            None,
        )
        if sample_matrix:
            out_array_shape_2d = partialSkims[sample_matrix].shape
            out_array_shape_3d = list(out_array_shape_2d) + [len(timePeriods)]
        else:
            # Fallback if no matrices found with period suffix
            logger.warning(
                "No partial skim matrices found with period suffix. Cannot determine shape for trip counts."
            )
            return {}

    except Exception as e:
        logger.error(
            f"Error determining partial skim shape: {e}. Cannot accumulate trip counts."
        )
        return {}

    for omx_key in partialSkims.list_matrices():
        # Split key into measure_part and period
        parts = omx_key.split("__")
        if len(parts) != 2:
            continue  # Skip keys not in PATH_MEASURE__PERIOD format

        measure_part = parts[0]  # e.g. SOV_TRIPS or TNC_SINGLE_UBER_TRIPS
        period = parts[1]  # e.g. AM

        if period not in timePeriods:
            continue  # Skip keys for irrelevant periods

        # Check if it's a TRIPS or FAILURES matrix
        is_trips = measure_part.endswith("_TRIPS")
        is_failures = measure_part.endswith("_FAILURES")

        if is_trips or is_failures:
            # Extract the mode path from the measure_part
            # e.g., SOV from SOV_TRIPS, TNC_SINGLE_UBER from TNC_SINGLE_UBER_TRIPS
            if is_trips:
                mode = measure_part[: -len("_TRIPS")]
            else:  # is_failures
                mode = measure_part[: -len("_FAILURES")]

            if mode not in completed_failed_dict:
                completed_failed_dict[mode] = [
                    np.zeros(
                        out_array_shape_3d, dtype=np.float32
                    ),  # Completed trips 3D
                    np.zeros(out_array_shape_3d, dtype=np.float32),  # Failed trips 3D
                ]

            try:
                tp_idx = timePeriods.index(period)
                data_2d = np.nan_to_num(
                    partialSkims[omx_key][:]
                )  # Get 2D numpy array for this period

                # Ensure shape compatibility
                if data_2d.shape != out_array_shape_2d:
                    logger.warning(
                        f"Shape mismatch for partial skim {omx_key}. Expected {out_array_shape_2d}, got {data_2d.shape}. Skipping accumulation."
                    )
                    continue

                if is_trips:
                    completed_failed_dict[mode][0][:, :, tp_idx] = data_2d
                else:  # is_failures
                    completed_failed_dict[mode][1][:, :, tp_idx] = data_2d

            except ValueError:
                logger.warning(
                    f"Period '{period}' from key '{omx_key}' not found in settings periods. Skipping."
                )
            except Exception as e:
                logger.error(
                    f"Error reading partial skim {omx_key} for trip count accumulation: {e}"
                )

    logger.info(
        f"Accumulated trip counts for {len(completed_failed_dict)} modes/providers."
    )
    # logger.debug(f"Modes with trip counts: {list(completed_failed_dict.keys())}") # Can be very verbose

    return completed_failed_dict


def _transform_measure(
    input_vals, completed, failed, measure, path, transit_scale_factor=100.0
):
    """
    Transforms skim values based on measure type, path, and trip completion statistics.
    This is applied to a single 2D slice (one time period).

    Parameters:
    -----------
    input_vals : ndarray (2D)
        The new observed values for a single time period from the partial skim.
    completed : ndarray (2D)
        Count of completed trips for each OD pair for the current time period (2D slice).
    failed : ndarray (2D)
        Count of failed trips for each OD pair for the current time period (2D slice).
    measure : str
        The measure name (e.g., "IWAIT", "TOTIVT").
    path : str
        The mode path (e.g., "SOV", "WLK_TRN_WLK", "TNC_SINGLE_UBER", "RH_SOLO").

    Returns:
    --------
    tuple (mask, vals, to_cancel)
        mask: Boolean array (2D) indicating which cells to update with `vals`.
        vals: Values (1D, matching mask.sum()) to assign to the masked cells.
        to_cancel: Boolean array (2D) indicating cells to zero out due to high failure rate (for IVT/TOTIVT).
                   Returns None for other measures.
    """
    # Basic mask - where there are completed trips and valid input values
    valid = ~np.isnan(input_vals)
    completed_mask = completed > 0
    basic_mask = valid & completed_mask
    # Handle negative values - seems reasonable to treat as 0
    negative = input_vals < 0
    if np.any(negative):
        logger.debug(
            f"Found {np.sum(negative)} negative values in input_vals for measure {measure} and path {path} in one period. Setting them to 0."
        )
        input_vals[negative] = 0.0

    measures_is_scaled_for_transit = {
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
    # Apply 100x scaling for specific measures only if the mode is NOT TNC/RH
    scaling = (
        transit_scale_factor
        if measure in measures_is_scaled_for_transit
        and not (path.startswith("TNC_") or path.startswith("RH_"))
        else 1.0
    )

    # Handle measures that need scaling but no other special logic
    if measure in measures_is_scaled_for_transit and measure not in ["TOTIVT", "IVT"]:
        return basic_mask, input_vals[basic_mask] * scaling, None

    # Handle travel time measures with penalty logic (only TOTIVT/IVT) for NON-TNC/RH
    elif measure in ["TOTIVT", "IVT"]:
        if path.startswith("TNC_") or path.startswith("RH_"):
            # TNC/RH TOTIVT/IVT is handled by the interpolation logic in post-processing.
            # Just copy values, no scaling here.
            return basic_mask, input_vals[basic_mask], None
        else:
            # Apply penalty logic for non-TNC/RH modes (like transit)
            to_cancel = (failed > 5) & (failed > (1 * completed))
            # Update where not cancelled
            update_mask = basic_mask & ~to_cancel
            # Apply scaling for transit
            scaled_vals = input_vals[update_mask] * scaling
            return update_mask, scaled_vals, to_cancel

    # Handle DIST (simple assignment where completed > 0)
    elif measure == "DIST":
        # TNC DDIST interpolation is handled in post-processing
        if path.startswith("TNC_") or path.startswith("RH_"):
            # For TNC/RH, DIST is handled by interpolation, so just copy raw values for now.
            return basic_mask, input_vals[basic_mask], None
        else:
            # For others, just apply direct value where completed > 0
            return basic_mask, input_vals[basic_mask], None

    # Handle non-scaled measures (simple assignment where completed > 0)
    # Includes COST, TOLLs, etc.
    # TNC FAR and REJECTIONPROB are handled in post-processing, so they are excluded here.
    elif measure not in ["REJECTIONPROB", "FAR"]:
        return basic_mask, input_vals[basic_mask], None

    # Default case for measures handled entirely in post-processing (e.g., REJECTIONPROB, FAR)
    logger.debug(
        f"Measure {measure} for path {path} will be handled in post-processing. No direct merge."
    )
    return np.zeros_like(input_vals, dtype=bool), np.array([]), None


def _merge_one_zarr_measure_from_zarr(
    partial_skims_ds,
    skims_ds,
    path,
    measure,
    timePeriods,
    completed_3d,
    failed_3d,
):
    """
    Merges a single measure's data from Zarr partial skims into the master Zarr dataset.
    """
    source_var_name = f"{path}_{measure}"
    target_var_name = source_var_name

    # Check if source variable exists in partial skims
    if source_var_name not in partial_skims_ds.data_vars:
        logger.debug(f"Variable {source_var_name} not found in partial Zarr skims.")
        return None, None

    # Ensure the target variable exists in skims_ds
    if target_var_name not in skims_ds.data_vars:
        logger.warning(f"Target Zarr variable '{target_var_name}' not found. Ignoring")
        return None, None

    target_da = skims_ds[target_var_name]
    source_da = partial_skims_ds[source_var_name]

    # Get the data
    input_vals_3d = source_da.values

    # Apply transformations for each time period
    for tpIdx in range(input_vals_3d.shape[2]):
        input_vals_tp = input_vals_3d[:, :, tpIdx]
        completed_tp = completed_3d[:, :, tpIdx]
        failed_tp = failed_3d[:, :, tpIdx]

        # Apply transformations using the helper function
        mask_update, vals_update, to_cancel_tp = _transform_measure(
            input_vals_tp,
            completed_tp,
            failed_tp,
            measure,
            path,
        )

        # Apply updates
        current_slice = target_da.data[:, :, tpIdx]
        if mask_update.any():
            current_slice[mask_update] = vals_update

        # Handle cancellations
        if to_cancel_tp is not None and to_cancel_tp.any():
            cancellation_count = to_cancel_tp.sum()
            current_slice[to_cancel_tp] = 0.0
            logger.debug(f"  Canceled {cancellation_count} ODs for {source_var_name}")

        target_da.data[:, :, tpIdx] = current_slice

    return path, measure


def _merge_one_zarr_measure(
    partialSkims,
    skims_ds,
    path,
    measure,
    timePeriods,
    completed_3d,
    failed_3d,
):
    """
    Merges a single measure's data for a given path and all time periods
    from OMX partial skims into a Zarr DataArray, applying measure-specific transformations.

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM.
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    path : str
        The mode path (e.g., "SOV", "WLK_TRN_WLK").
    measure : str
        The measure name (e.g., "TOTIVT", "DIST").
    timePeriods : list of str
        List of time period names.
    completed_3d : ndarray (3D)
        Completed trip counts for this path across all periods [zones, zones, periods].
    failed_3d : ndarray (3D)
        Failed trip counts for this path across all periods [zones, zones, periods].

    Returns:
    --------
    tuple (path, measure_name)
        The path and measure name of the skim that was processed.
    """
    target_var_name = f"{path}_{measure}"  # e.g., SOV_TOTIVT

    # Ensure the target variable exists in skims_ds
    # Use SOV_TRIPS shape/coords as a default if target var doesn't exist
    if target_var_name not in skims_ds.data_vars:
        logger.warning(
            f"Target Zarr variable '{target_var_name}' not found. Creating with SOV_TRIPS shape."
        )
        try:
            zarr_shape = skims_ds["SOV_TRIPS"].shape
            zarr_coords = skims_ds["SOV_TRIPS"].coords
            zarr_dims = skims_ds["SOV_TRIPS"].dims
            skims_ds[target_var_name] = xr.DataArray(
                np.zeros(zarr_shape, dtype=np.float32),
                coords=zarr_coords,
                dims=zarr_dims,
                name=target_var_name,
            )
        except KeyError:
            logger.error(
                f"Cannot create variable {target_var_name}: SOV_TRIPS not found to get shape/coords."
            )
            return None, None  # Indicate skipping
    target_da = skims_ds[target_var_name]

    # Iterate through each time period slice
    for tpIdx, tp in enumerate(timePeriods):
        # Construct the key for the partial skims in OMX format
        partial_key = f"{path}_{measure}__{tp}"

        if partial_key in partialSkims:
            try:
                input_vals_tp = partialSkims[partial_key][
                    :
                ]  # Get 2D numpy array for this period

                # Ensure target DA slice exists for this period
                if target_da.ndim < 3 or target_da.shape[-1] <= tpIdx:
                    logger.warning(
                        f"Target Zarr variable {target_var_name} does not have enough time period dimensions for period {tp} (index {tpIdx}). Skipping merge for this period."
                    )
                    continue  # Skip this period

                # Ensure shape compatibility
                if input_vals_tp.shape != target_da.shape[:2]:
                    logger.warning(
                        f"Shape mismatch for partial skim {partial_key}. Expected {target_da.shape[:2]}, got {input_vals_tp.shape}. Skipping merge for this period."
                    )
                    continue

                # Get the 2D completed/failed slices for the current time period
                completed_tp = completed_3d[:, :, tpIdx]
                failed_tp = failed_3d[:, :, tpIdx]

                # Apply transformations using the helper function
                # _transform_measure expects 2D completed/failed arrays
                mask_update, vals_update, to_cancel_tp = _transform_measure(
                    input_vals_tp,
                    completed_tp,
                    failed_tp,
                    measure,
                    path,  # Pass path here
                )

                # Apply updates to the Zarr DataArray slice using the mask
                current_slice = target_da.data[:, :, tpIdx]
                if mask_update.any():
                    current_slice[mask_update] = vals_update

                # Handle cancellations for IVT/TOTIVT
                if to_cancel_tp is not None and to_cancel_tp.any():
                    cancellation_count = to_cancel_tp.sum()
                    current_slice[to_cancel_tp] = 0.0  # Set canceled values to 0
                    logger.debug(
                        f"  Canceled {cancellation_count} ODs for {partial_key}"
                    )

                # Update the slice in the DataArray's data
                target_da.data[:, :, tpIdx] = current_slice

            except Exception as e:
                logger.error(f"Error processing skim slice {partial_key}: {e}")
                # Continue to the next time period

        else:
            logger.debug(
                f"Partial skims missing key {partial_key}. Skipping merge for this period."
            )

    # Note: SOV/HOV copying logic removed from here - handled by copy_skims_for_unobserved_modes

    return path, measure


def _merge_zarr_trip_counts(
    allSkims,
    path,
    completed,
    failed,
    previous_weight=0.9,
):
    key_completed = f"{path}_TRIPS"
    key_failed = f"{path}_FAILURES"
    any_observed = (completed + failed) > 0
    # Ensure keys exist before attempting to access .data
    if key_completed in allSkims and key_failed in allSkims:
        prev_completed = np.nan_to_num(allSkims[key_completed].data)
        prev_failed = np.nan_to_num(allSkims[key_failed].data)
        logger.info(
            f"For {path} previously had {prev_completed[any_observed].sum():.0f} completed trips and {prev_failed[any_observed].sum():.0f} failed trips"
        )
        prev_completed = np.ceil(previous_weight * prev_completed + completed).astype(
            int
        )
        prev_failed = np.ceil(previous_weight * prev_failed + failed).astype(int)
        allSkims[key_completed].data[
            :
        ] = prev_completed  # Use [:] to modify data in place
        allSkims[key_failed].data[:] = prev_failed  # Use [:] to modify data in place
        logger.info(
            f"Now we have {prev_completed[any_observed].sum():.0f} completed trips and {prev_failed[any_observed].sum():.0f} failed trips"
        )
    else:
        logger.warning(
            f"Skipping trip counts merge for {path} as {key_completed} or {key_failed} does not exist in target skims file"
        )


def _merge_zarr_trip_counts_slice(
    allSkims,
    path,
    completed_tp,
    failed_tp,
    tp_idx,
    previous_weight=0.9,
):
    key_completed = f"{path}_TRIPS"
    key_failed = f"{path}_FAILURES"
    if key_completed in allSkims and key_failed in allSkims:
        prev_completed_tp = np.nan_to_num(allSkims[key_completed].data[:, :, tp_idx])
        prev_failed_tp = np.nan_to_num(allSkims[key_failed].data[:, :, tp_idx])
        prev_completed_tp = np.ceil(
            previous_weight * prev_completed_tp + completed_tp
        ).astype(int)
        prev_failed_tp = np.ceil(
            previous_weight * prev_failed_tp + failed_tp
        ).astype(int)
        allSkims[key_completed].data[:, :, tp_idx] = prev_completed_tp
        allSkims[key_failed].data[:, :, tp_idx] = prev_failed_tp
    else:
        logger.warning(
            f"Skipping trip counts merge for {path} at time index {tp_idx} as {key_completed} or {key_failed} does not exist in target skims file"
        )


def _handle_transit_mode_availability(skims_ds, timePeriods):
    """
    Handles transit mode availability by checking specific transit modes.

    For OD pairs with 50+ successful general transit trips, any specific transit mode
    with 0 successful trips will be marked as unavailable (TOTIVT = 0).
    Only marks OD pairs where TOTIVT is currently > 0 or np.nan.
    Operates directly on the Zarr dataset.

    Parameters:
    -----------
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    timePeriods : list of str
        List of time period names.
    """
    # General transit mode
    general_transit_path = "WLK_TRN_WLK"

    # List of specific transit modes to check
    specific_transit_modes = [
        "WLK_LOC_WLK",  # Local bus
        "WLK_HVY_WLK",  # Heavy rail
        "WLK_COM_WLK",  # Commuter rail
        "WLK_EXP_WLK",  # Express bus
        # Add any other specific transit modes as needed
    ]

    # Create a summary table header using logger
    logger.info(f"{'=' * 80}")
    logger.info("Transit Mode Availability Analysis")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Period':<6} | {'Mode':<12} | {'ODs w/50+ Transit':<18} | {'ODs w/0 Mode Trips':<18} | {'Changed':<10} | {'% Changed':<10}"
    )
    logger.info(
        f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 10}"
    )

    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}

    # Process each time period
    for tp in timePeriods:
        tp_idx = tp_to_idx[tp]

        # Check if general transit trip counts exist
        general_trips_key = f"{general_transit_path}_TRIPS"  # Zarr key format
        general_trips_da = skims_ds.get(general_trips_key)

        if general_trips_da is None:
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {'NO DATA':<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}"
            )
            continue

        # Find OD pairs with at least 50 successful transit trips for this period slice
        general_transit_trips = general_trips_da.data[:, :, tp_idx]  # Get 2D slice
        mask_significant_transit = general_transit_trips >= 50

        significant_count = np.sum(mask_significant_transit)

        if not np.any(mask_significant_transit):
            # No significant transit trips for this period, nothing to check
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {0:<18} | {0:<18} | {0:<10} | {0:<10.1f}"
            )
            continue

        # Check each specific transit mode
        for mode in specific_transit_modes:
            # Get trip counts for this mode for this period slice
            mode_trips_key = f"{mode}_TRIPS"  # Zarr key format
            mode_trips_da = skims_ds.get(mode_trips_key)

            if mode_trips_da is None:
                logger.info(
                    f"{tp:<6} | {mode:<12} | {significant_count:<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}"
                )
                continue

            mode_trips = mode_trips_da.data[:, :, tp_idx]  # Get 2D slice

            # Find OD pairs where general transit has 50+ trips but this mode has 0
            mask_mode_unused = mask_significant_transit & (mode_trips == 0)

            unused_count = np.sum(mask_mode_unused)

            # Mark this mode as unavailable for these OD pairs by setting TOTIVT = 0
            totivt_key = f"{mode}_TOTIVT"  # Zarr key format
            totivt_da = skims_ds.get(totivt_key)
            changed_count = 0

            if totivt_da is not None and unused_count > 0:
                # Get current TOTIVT values slice
                current_totivt_slice = totivt_da.data[:, :, tp_idx]

                # Only mark as unavailable if current TOTIVT is > 0 or np.nan within the mask_mode_unused area
                mask_to_change = mask_mode_unused & (
                    (current_totivt_slice > 0) | np.isnan(current_totivt_slice)
                )
                changed_count = np.sum(mask_to_change)

                # Update values directly using .data slice
                if changed_count > 0:
                    current_totivt_slice[mask_to_change] = 0.0
                    totivt_da.data[:, :, tp_idx] = (
                        current_totivt_slice  # Ensure changes are reflected
                    )

            percent_changed = (
                (changed_count / significant_count) * 100
                if significant_count > 0
                else 0
            )

            logger.info(
                f"{tp:<6} | {mode:<12} | {significant_count:<18} | {unused_count:<18} | {changed_count:<10} | {percent_changed:<10.1f}"
            )

    logger.info(f"{'=' * 80}")


def _consolidate_tnc_data_zarr(
    partialSkims,
    skims_ds,
    timePeriods,
    completed_failed_dict_providers,
    settings,
):
    """
    Consolidates skims from multiple TNC providers (e.g., TNC_SINGLE_UBER, TNC_SINGLE_LYFT)
    into single consolidated skims (e.g., RH_SOLO, RH_SHARED) within the Zarr dataset.
    Reads from partialSkims (OMX) and writes to skims_ds (Zarr).

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM (provider-specific).
    skims_ds : xarray.Dataset
        The target Zarr dataset (main skims file). Consolidated data is written here.
    timePeriods : list of str
        List of time period names.
    completed_failed_dict_providers : dict
        Dictionary containing completed and failed trip counts (3D arrays)
        for each TNC *provider* mode (e.g., TNC_SINGLE_UBER, TNC_POOLED_LYFT).
        Used for weighted averages.
    settings : dict
        Settings dictionary.

    Returns:
    --------
    set
        A set containing consolidated RH mode names.
    """
    logger.info("Starting TNC fleet consolidation from OMX to Zarr...")

    all_partial_vars = partialSkims.list_matrices()
    # Get Zarr shape/coords from an existing 3D variable
    try:
        zarr_shape = skims_ds["SOV_TRIPS"].shape
        zarr_coords = skims_ds["SOV_TRIPS"].coords
        zarr_dims = skims_ds["SOV_TRIPS"].dims
        if zarr_shape[-1] != len(timePeriods):
            logger.warning(
                f"Zarr time period dimension ({zarr_shape[-1]}) does not match settings periods ({len(timePeriods)}). Using Zarr shape."
            )
            # Update tp_to_idx to match Zarr if necessary, or raise error. Let's assume settings is correct.
            # For safety, ensure the Zarr has enough space for all periods in settings.
            if zarr_shape[-1] < len(timePeriods):
                logger.error(
                    f"Zarr shape {zarr_shape} has fewer time periods than in settings ({len(timePeriods)}). Cannot consolidate."
                )
                return {}  # Indicate failure or no consolidation
            # If Zarr has more periods, we only populate those listed in settings.
    except KeyError:
        logger.error(
            "Zarr dataset does not contain 'SOV_TRIPS'. Cannot determine shape and coordinates for new variables."
        )
        return {}  # Indicate failure or no consolidation
    except Exception as e:
        logger.error(f"Error getting Zarr shape/coords: {e}. Cannot consolidate.")
        return {}

    # Group TNC provider variables from partialSkims by (base_mode, measure)
    grouped_sources = (
        {}
    )  # Key: (base_mode, measure), Value: list of (provider, period, omx_key)

    for omx_key in all_partial_vars:
        # Split key into measure_part and period (e.g., TNC_SINGLE_UBER_IWAIT__AM -> TNC_SINGLE_UBER_IWAIT, AM)
        parts = omx_key.split("__")
        if len(parts) != 2:
            continue  # Skip keys not in PATH_MEASURE__PERIOD format

        measure_part = parts[0]  # e.g. TNC_SINGLE_UBER_IWAIT
        period = parts[1]  # e.g. AM

        if not measure_part.startswith("TNC_") or period not in timePeriods:
            continue  # Only process TNC keys for relevant periods

        # Parse measure_part: TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
        measure_part_parts = measure_part.split("_")
        if len(measure_part_parts) < 3:
            logger.debug(
                f"Skipping TNC key with unexpected format (too few parts): {omx_key}"
            )
            continue

        tnc_prefix = measure_part_parts[0]  # Should be "TNC"
        base_mode = measure_part_parts[1]  # e.g. "SINGLE", "POOLED"
        # Check if base_mode maps to a consolidated RH mode
        if f"{tnc_prefix}_{base_mode}" not in TNC_CONSOLIDATION_MAP:
            logger.debug(f"Skipping TNC key with unknown base mode pattern: {omx_key}")
            continue

        # Identify provider and measure. The first part after TNC_BASEMODE that is NOT a known measure prefix
        # is likely part of the provider name, UNLESS the measure name itself has underscores.
        # Let's try to find the measure name by splitting from the right, checking against common measures.
        measure = None
        provider_parts = []
        remaining_parts = list(measure_part_parts[2:])  # Parts after TNC_BASEMODE

        # Reverse the remaining parts and check if the rightmost form a known measure
        reversed_remaining = remaining_parts[::-1]
        for i in range(len(reversed_remaining)):
            potential_measure_parts = reversed_remaining[: i + 1][
                ::-1
            ]  # e.g., ['IWAIT'] or ['KEY', 'IVT']
            potential_measure = "_".join(potential_measure_parts)

            # Check against common measures or specific ones with underscores
            if potential_measure in [
                "TRIPS",
                "FAILURES",
                "REJECTIONPROB",
                "IWAIT",
                "FAR",
                "DDIST",
                "TOTIVT",
            ]:
                measure = potential_measure
                provider_parts = remaining_parts[
                    : len(remaining_parts) - (i + 1)
                ]  # Remaining parts are the provider
                break  # Found the measure

        if measure is None:
            logger.warning(
                f"Could not identify measure name in TNC key: {omx_key}. Skipping."
            )
            continue

        provider = (
            "_".join(provider_parts) if provider_parts else "DEFAULT"
        )  # Use "DEFAULT" if no provider part
        base_mode_key = f"{tnc_prefix}_{base_mode}"  # e.g. "TNC_SINGLE"

        if (base_mode_key, measure) not in grouped_sources:
            grouped_sources[(base_mode_key, measure)] = []

        grouped_sources[(base_mode_key, measure)].append((provider, period, omx_key))

    logger.debug(f"Grouped TNC sources from OMX: {grouped_sources}")

    consolidated_modes = set()

    # First, consolidate TRIPS and FAILURES to get total counts per RH_ mode
    for base_mode_key, measure in [(k[0], k[1]) for k in grouped_sources.keys()]:
        if measure not in ["TRIPS", "FAILURES"]:
            continue  # Only process trips/failures first

        target_mode = TNC_CONSOLIDATION_MAP[base_mode_key]  # e.g. "RH_SOLO"
        target_var_name = f"{target_mode}_{measure}"

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
            logger.debug(
                f"Creating target variable '{target_var_name}' in Zarr dataset."
            )
            skims_ds[target_var_name] = xr.DataArray(
                np.zeros(zarr_shape, dtype=np.float32),
                coords=zarr_coords,
                dims=zarr_dims,
                name=target_var_name,
            )
        target_da = skims_ds[target_var_name]

        for tp_idx, period in enumerate(timePeriods):
            consolidated_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
            for provider, period_key, omx_key in grouped_sources.get(
                (base_mode_key, measure), []
            ):
                if period_key != period:
                    continue
                if omx_key in partialSkims:
                    try:
                        input_vals_tp = np.nan_to_num(
                            partialSkims[omx_key][:]
                        )  # 2D numpy array for this period
                        if input_vals_tp.shape == zarr_shape[:2]:
                            consolidated_tp += input_vals_tp
                        else:
                            logger.warning(
                                f"Shape mismatch for partial skim {omx_key}. Expected {zarr_shape[:2]}, got {input_vals_tp.shape}. Skipping sum."
                            )
                    except Exception as e:
                        logger.error(
                            f"Error reading partial skim {omx_key} for consolidation: {e}"
                        )
                else:
                    logger.debug(f"Partial skim {omx_key} not found. Skipping.")
            target_da.data[:, :, tp_idx] = consolidated_tp
        consolidated_modes.add(target_mode)

    # Now, consolidate other measures using weighted average
    for (base_mode_key, measure), sources in grouped_sources.items():
        if measure in ["TRIPS", "FAILURES", "REJECTIONPROB"]:
            continue  # Already handled TRIPS/FAILURES, REJECTIONPROB calculated later

        target_mode = TNC_CONSOLIDATION_MAP[base_mode_key]  # e.g. "RH_SOLO"
        target_var_name = f"{target_mode}_{measure}"

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
            logger.debug(
                f"Creating target variable '{target_var_name}' in Zarr dataset."
            )
            skims_ds[target_var_name] = xr.DataArray(
                np.zeros(zarr_shape, dtype=np.float32),  # Initialize with zeros
                coords=zarr_coords,
                dims=zarr_dims,
                name=target_var_name,
            )
        target_da = skims_ds[target_var_name]

        for tp_idx, period in enumerate(timePeriods):
            sum_weighted_values_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
            sum_weights_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
            for provider, period_key, omx_key in sources:
                if period_key != period:
                    continue
                if omx_key in partialSkims:
                    try:
                        input_vals_tp = np.nan_to_num(partialSkims[omx_key][:])
                        provider_mode_key = f"TNC_{base_mode}_{provider}"
                        if provider_mode_key in completed_failed_dict_providers:
                            provider_completed_3d, _ = completed_failed_dict_providers[
                                provider_mode_key
                            ]
                            completed_tp = np.nan_to_num(
                                provider_completed_3d[:, :, tp_idx]
                            )
                            if (
                                input_vals_tp.shape == zarr_shape[:2]
                                and completed_tp.shape == zarr_shape[:2]
                            ):
                                input_vals_tp[input_vals_tp < 0] = 0.0
                                sum_weighted_values_tp += input_vals_tp * completed_tp
                                sum_weights_tp += completed_tp
                            else:
                                logger.warning(
                                    f"Shape mismatch for partial skim {omx_key} or its trips data. Skipping weighted average contribution."
                                )
                        else:
                            logger.debug(
                                f"Completed trip data for provider mode {provider_mode_key} not found in provider dict. Skipping weighted average contribution for {omx_key}."
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing partial skim {omx_key} for weighted consolidation: {e}"
                        )
                else:
                    logger.debug(f"Partial skim {omx_key} not found. Skipping.")

            consolidated_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
            valid_weights_mask = sum_weights_tp != 0
            np.divide(
                sum_weighted_values_tp,
                sum_weights_tp,
                out=consolidated_tp,
                where=valid_weights_mask,
            )
            target_da.data[:, :, tp_idx] = consolidated_tp

    logger.info("Completed TNC fleet consolidation from OMX to Zarr.")

    # Return consolidated counts for use in post-processing
    return consolidated_modes


# New function to transfer TNC provider data from OMX to Zarr without consolidation
def _transfer_tnc_provider_data_zarr(
    partialSkims,
    skims_ds,
    timePeriods,
    completed_failed_dict_providers,
    settings,
):
    """
    Transfers provider-specific TNC skims from partialSkims (OMX) to skims_ds (Zarr).
    Does NOT consolidate. This is for the case where consolidation is disabled.

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM (provider-specific).
    skims_ds : xarray.Dataset
        The target Zarr dataset (main skims file). Provider data is written here.
    timePeriods : list of str
        List of time period names.
    completed_failed_dict_providers : dict
        Dictionary containing completed and failed trip counts (3D arrays)
        for each TNC *provider* mode. Used by _merge_one_zarr_measure.
    settings : dict
        Settings dictionary.

    Returns:
    --------
    list
        List of Zarr variable names that were created/updated (TNC provider skims).
    """
    logger.info("Starting TNC provider data transfer from OMX to Zarr...")

    all_partial_vars = partialSkims.list_matrices()
    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}
    transferred_vars = set()

    # Group TNC provider variables from partialSkims by (mode, measure) (where mode is provider specific)
    # Key: (mode, measure), Value: list of (period, omx_key)
    grouped_sources = {}

    # Define TNC measures with underscores (if any)
    TNC_MEASURES_WITH_UNDERSCORES = set()  # Add any such measures if needed

    for omx_key in all_partial_vars:
        # Split key into measure_part and period
        parts = omx_key.split("__")
        if len(parts) != 2:
            continue  # Skip keys not in PATH_MEASURE__PERIOD format

        measure_part = parts[0]  # e.g. TNC_SINGLE_UBER_IWAIT
        period = parts[1]  # e.g. AM

        if not measure_part.startswith("TNC_") or period not in timePeriods:
            continue  # Only process TNC keys for relevant periods

        # Parse measure_part to get the full provider mode name and the measure
        measure = None
        mode = None

        measure_part_parts = measure_part.split("_")
        if len(measure_part_parts) < 3:
            logger.debug(
                f"Skipping TNC key with unexpected format (too few parts): {omx_key}"
            )
            continue

        # Assume format TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
        # Let's try to find the measure name by splitting from the right
        remaining_parts = list(measure_part_parts[2:])
        reversed_remaining = remaining_parts[::-1]
        for i in range(len(reversed_remaining)):
            potential_measure_parts = reversed_remaining[: i + 1][::-1]
            potential_measure = "_".join(potential_measure_parts)

            if (
                potential_measure
                in [
                    "TRIPS",
                    "FAILURES",
                    "REJECTIONPROB",
                    "IWAIT",
                    "XWAIT",
                    "WACC",
                    "WAUX",
                    "WEGR",
                    "DTIM",
                    "DDIST",
                    "TOTIVT",
                    "FAR",
                    "COST",
                ]
                or potential_measure in TNC_MEASURES_WITH_UNDERSCORES
            ):
                measure = potential_measure
                mode_parts = measure_part_parts[
                    : len(measure_part_parts) - (i + 1)
                ]  # Parts before the measure
                mode = "_".join(mode_parts)  # e.g., TNC_SINGLE_UBER
                break

        if mode is None or measure is None:
            logger.warning(
                f"Could not parse TNC provider mode or measure from key: {omx_key}. Skipping."
            )
            continue

        if (mode, measure) not in grouped_sources:
            grouped_sources[(mode, measure)] = []

        grouped_sources[(mode, measure)].append((period, omx_key))

    # Get Zarr shape/coords from an existing 3D variable
    try:
        zarr_shape = skims_ds["SOV_TRIPS"].shape
        zarr_coords = skims_ds["SOV_TRIPS"].coords
        zarr_dims = skims_ds["SOV_TRIPS"].dims
        if zarr_shape[-1] < len(timePeriods):
            logger.error(
                f"Zarr shape {zarr_shape} has fewer time periods than in settings ({len(timePeriods)}). Cannot transfer TNC data."
            )
            return []  # Indicate failure
    except KeyError:
        logger.error(
            "Zarr dataset does not contain 'SOV_TRIPS'. Cannot determine shape and coordinates for new TNC provider variables."
        )
        return []  # Indicate failure
    except Exception as e:
        logger.error(f"Error getting Zarr shape/coords: {e}. Cannot transfer TNC data.")
        return []

    # Transfer data for each mode/measure pair
    for (mode, measure), sources in grouped_sources.items():
        target_var_name = f"{mode}_{measure}"  # e.g. TNC_SINGLE_UBER_IWAIT

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
            logger.debug(
                f"Creating target variable '{target_var_name}' in Zarr dataset."
            )
            # Check if the mode exists in the completed_failed_dict_providers
            # This is needed to get the shape. Fallback to SOV_TRIPS shape.
            try:
                # Use the shape from the provider counts if available
                provider_completed_shape = completed_failed_dict_providers[mode][
                    0
                ].shape
                if provider_completed_shape[:2] != zarr_shape[
                    :2
                ] or provider_completed_shape[-1] < len(timePeriods):
                    logger.warning(
                        f"Provider completed counts shape {provider_completed_shape} for mode {mode} is incompatible with Zarr shape {zarr_shape} or settings periods {len(timePeriods)}. Falling back to SOV_TRIPS shape."
                    )
                    raise KeyError  # Fallback
                current_zarr_shape = provider_completed_shape
            except KeyError:
                # Fallback to default Zarr shape if provider shape isn't reliable
                current_zarr_shape = zarr_shape
                logger.debug(
                    f"Using default Zarr shape {current_zarr_shape} for mode {mode}"
                )

            skims_ds[target_var_name] = xr.DataArray(
                np.zeros(current_zarr_shape, dtype=np.float32),  # Initialize with zeros
                coords=zarr_coords,  # Use SOV_TRIPS coords as a default
                dims=zarr_dims,  # Use SOV_TRIPS dims as a default
                name=target_var_name,
            )
        target_da = skims_ds[target_var_name]
        transferred_vars.add(target_var_name)

        # Get completed/failed counts for this mode from the provider dict
        completed_3d, failed_3d = completed_failed_dict_providers.get(
            mode, (None, None)
        )
        if completed_3d is None or failed_3d is None:
            logger.warning(
                f"Completed/failed trip data for mode {mode} not found in provider dict. Skipping merge for {target_var_name}."
            )
            continue  # Cannot merge without counts

        # Merge data from all relevant periods
        for period, omx_key in sources:
            if omx_key in partialSkims:
                try:
                    tp_idx = tp_to_idx[period]
                    # Ensure the target DA has enough dimensions for this period
                    if target_da.ndim < 3 or target_da.shape[-1] <= tp_idx:
                        logger.warning(
                            f"Target Zarr variable {target_var_name} does not have enough time period dimensions for period {period} (index {tp_idx}). Skipping merge for this period."
                        )
                        continue  # Skip this period if target DA is too small

                    input_vals_tp = partialSkims[omx_key][
                        :
                    ]  # Get 2D numpy array for this period

                    # Ensure shape compatibility before merging
                    if input_vals_tp.shape != target_da.shape[:2]:
                        logger.warning(
                            f"Shape mismatch for partial skim {omx_key}. Expected {target_da.shape[:2]}, got {input_vals_tp.shape}. Skipping merge for this period."
                        )
                        continue

                    # Apply transformations using the helper function _transform_measure
                    # _transform_measure expects 2D completed/failed arrays
                    completed_tp = completed_3d[:, :, tp_idx]
                    failed_tp = failed_3d[:, :, tp_idx]

                    mask_update, vals_update, to_cancel_tp = _transform_measure(
                        input_vals_tp,
                        completed_tp,
                        failed_tp,
                        measure,
                        mode,  # Pass mode name here
                    )

                    # Apply updates to the Zarr DataArray slice using the mask
                    current_slice = target_da.data[:, :, tp_idx]
                    if mask_update.any():
                        current_slice[mask_update] = vals_update

                    # Handle cancellations for IVT/TOTIVT
                    if to_cancel_tp is not None and to_cancel_tp.any():
                        cancellation_count = to_cancel_tp.sum()
                        current_slice[to_cancel_tp] = 0.0  # Set canceled values to 0
                        logger.debug(f"Canceled {cancellation_count} ODs for {omx_key}")

                    # Update the slice in the DataArray's data
                    target_da.data[:, :, tp_idx] = current_slice

                except Exception as e:
                    logger.error(
                        f"Error reading partial skim {omx_key} for transfer: {e}"
                    )
            else:
                logger.debug(f"Partial skim {omx_key} not found. Skipping.")

    logger.info("Completed TNC provider data transfer from OMX to Zarr.")
    return list(transferred_vars)


def write_zarr_skim_as_omx_new(
    all_skims_path, settings: PilatesConfig, new_skim_name, exclude_tables=None
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
    beam_input_dir = settings.beam.local_mutable_data_folder

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

        # Get zone IDs - simplified logic
        if "taz_ids" in skims_ds.attrs:
            # BEAM stores actual TAZ IDs in attributes
            zone_ids = skims_ds.attrs["taz_ids"]
            logger.info(f"Using TAZ IDs from attributes: {len(zone_ids)} zones")
        else:
            # Fall back to coordinate values
            zone_ids = skims_ds.coords["otaz"].values
            logger.info(f"Using zone IDs from coordinates: {len(zone_ids)} zones")

        # Ensure zone_ids are integers if they represent numbers
        try:
            zone_ids = [int(z) for z in zone_ids]
        except ValueError:
            # Keep as strings if they can't be converted to int
            zone_ids = [str(z) for z in zone_ids]

        # Get time periods
        time_periods = []
        if "time_period" in skims_ds.coords:
            time_periods = [str(s) for s in skims_ds.time_period.values]
            logger.info(f"Found {len(time_periods)} time periods: {time_periods}")

        # Prepare output file
        logger.info(f"Target output OMX path: {target_skims_path}")
        if os.path.exists(target_skims_path):
            logger.info(f"Deleting existing file: {target_skims_path}")
            os.remove(target_skims_path)

        # Create the new OMX file
        new_omx_file = omx.open_file(target_skims_path, "w")
        logger.info(f"Created new OMX file: {target_skims_path}")

        # Add zone mapping
        new_omx_file.create_mapping("zone_id", zone_ids)
        logger.info(f"Created 'zone_id' mapping with {len(zone_ids)} zones")

        # Write matrices - NO SCALING CHANGES
        logger.info("Writing matrices to OMX file...")
        written_count = 0

        for key in skims_ds.data_vars:
            if key in exclude_tables:
                logger.info(f"Skipping excluded variable: {key}")
                continue

            try:
                data_array = skims_ds[key]
                data = data_array.values

                if data_array.ndim == 2:
                    # Write 2D matrix directly
                    data_to_write = np.nan_to_num(data).astype(np.float32)
                    new_omx_file[key] = data_to_write
                    written_count += 1
                    logger.debug(f"Wrote 2D matrix '{key}'")

                elif data_array.ndim == 3:
                    # Write slices for each time period
                    for t_idx, tp in enumerate(time_periods):
                        new_key = f"{key}__{tp}"
                        slice_data = data[:, :, t_idx]
                        data_to_write = np.nan_to_num(slice_data).astype(np.float32)
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


def write_zarr_skim_as_omx(
    all_skims_path, settings: PilatesConfig, new_skim_name, exclude_tables=None
):
    """
    Write the skims from the Zarr format to an OMX format.

    Parameters
    ----------
    all_skims_path : str
        Path to the main skims Zarr store.
    settings : PilatesConfig
        Settings dictionary, needed for 'region' and 'beam_local_input_folder'.
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

    region = get_setting(settings, "run.region")
    beam_input_dir = get_setting(settings, "beam.local_input_folder")

    if not region or not beam_input_dir:
        logger.error(
            "Settings 'region' or 'beam_local_input_folder' are not defined. Cannot determine output path."
        )
        return None

    target_skims_path = os.path.join(beam_input_dir, region, new_skim_name)

    if exclude_tables is None:
        exclude_tables = []

    skims_ds = None  # Initialize Zarr dataset variable
    new_omx_file = None  # Initialize OMX file variable

    try:
        # Open the Zarr dataset
        skims_ds = xr.open_zarr(all_skims_path)
        logger.info(f"Opened Zarr skims file: {all_skims_path}")

        # --- Get Zone IDs from stored attributes ---
        zone_ids = None
        if "original_zone_ids" in skims_ds.attrs:
            zone_ids = np.array(skims_ds.attrs["original_zone_ids"])
            logger.info(
                f"Using original zone IDs from Zarr attributes: {len(zone_ids)} zones"
            )
        elif "otaz" in skims_ds.coords:
            # Fallback: check if zones are still original
            if skims_ds["otaz"].attrs.get("preprocessed") != "zero-based-contiguous":
                zone_ids = skims_ds.coords["otaz"].values
                logger.info(f"Using zone IDs from coordinates: {len(zone_ids)} zones")
            else:
                logger.error(
                    "Zarr uses zero-based zones but no original zone mapping found!"
                )
                # Try to reconstruct from settings
                zone_ids = zone_order(
                    settings, get_setting(settings, "run.start_year", 2015)
                )
                logger.warning(
                    f"Reconstructed zone IDs from settings: {len(zone_ids)} zones"
                )
        else:
            logger.warning("No zone coordinate found in Zarr file.")

        time_periods = []
        if "time_period" in skims_ds.coords:
            # Ensure time periods are strings for OMX keys
            time_periods = [str(s) for s in skims_ds.time_period.values]
            logger.info(
                f"Found {len(time_periods)} time periods in Zarr: {time_periods}"
            )
        else:
            logger.warning(
                "Zarr skims file does not have 'time_period' coordinate. 3D variables cannot be written with period suffixes."
            )

        # --- Prepare output file ---
        logger.info(f"Target output OMX path: {target_skims_path}")
        if os.path.exists(target_skims_path):
            logger.info(f"Deleting existing file: {target_skims_path}")
            try:
                os.remove(target_skims_path)
            except OSError as e:
                logger.error(
                    f"Error deleting existing file {target_skims_path}: {e}. Aborting."
                )
                return None

        # Create the new OMX file using 'w' mode as we just deleted it
        new_omx_file = omx.open_file(target_skims_path, "w")
        logger.info(f"Created new OMX file: {target_skims_path}")

        # --- Add Zone Mapping FIRST ---
        if zone_ids is not None and len(zone_ids) > 0:
            try:
                # Ensure zone_ids are integers
                zone_ids = np.array(zone_ids, dtype=int)
                new_omx_file.create_mapping("zone_id", zone_ids, overwrite=True)
                logger.info(
                    f"Created 'zone_id' mapping in OMX file with {len(zone_ids)} zones."
                )
            except Exception as e:
                logger.error(f"Error creating zone mapping in OMX file: {e}.")

        # --- Handle Data Scaling ---
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

        # --- Write Matrices from Zarr to OMX ---
        logger.info("Writing matrices to OMX file...")
        written_count = 0
        for key in skims_ds.data_vars:
            if key in exclude_tables:
                logger.info(f"Skipping excluded variable: {key}")
                continue

            try:
                data_array = skims_ds[key]
                data = data_array.values

                # Check if this measure needs descaling
                measure_name = key.split("_")[-1] if "_" in key else key
                needs_descaling = (
                    measure_name in scaled_measures
                    and not key.startswith(("TNC_", "RH_"))
                )

                if data_array.ndim == 2:
                    # Write 2D matrix
                    data_to_write = np.nan_to_num(data).astype(np.float32)
                    if needs_descaling:
                        data_to_write = data_to_write / 100.0
                        logger.debug(f"Descaled 2D matrix '{key}' by 100x")
                    new_omx_file[key] = data_to_write
                    written_count += 1

                elif data_array.ndim == 3:
                    # Write slices for each time period
                    for t_idx, tp in enumerate(time_periods):
                        new_key = f"{key}__{tp}"
                        slice_data = data[:, :, t_idx]
                        data_to_write = np.nan_to_num(slice_data).astype(np.float32)
                        if needs_descaling:
                            data_to_write = data_to_write / 100.0
                            logger.debug(f"Descaled 3D slice '{new_key}' by 100x")
                        new_omx_file[new_key] = data_to_write

                        # Add attributes
                        strsplit = key.rsplit("_", 1)
                        new_omx_file[new_key].attrs["measure"] = strsplit[-1]
                        new_omx_file[new_key].attrs["timePeriod"] = tp
                        if len(strsplit) == 2:
                            new_omx_file[new_key].attrs["mode"] = strsplit[0]
                        written_count += 1

            except Exception as e:
                logger.error(f"Error writing variable '{key}' to OMX: {e}")

        logger.info(
            f"Finished writing {written_count} matrices to OMX file {target_skims_path}."
        )
        return target_skims_path

    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during Zarr to OMX conversion: {e}"
        )
        return None  # Indicate failure

    finally:
        # Ensure files are closed
        if new_omx_file:
            try:
                new_omx_file.close()
                logger.info("Closed OMX file.")
            except Exception as e:
                logger.error(f"Error closing OMX file {target_skims_path}: {e}")

        if skims_ds:
            try:
                skims_ds.close()
                logger.info("Closed Zarr dataset.")
            except Exception as e:
                logger.error(f"Error closing Zarr dataset {all_skims_path}: {e}")


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
                skim_order = [item[0] for item in mapping_items]
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
) -> Optional[str]:
    """
    Merges current BEAM OMX skims into the main Zarr skims file.
    Handles TNC consolidation if enabled.
    Records provenance for all skims lineage.
    Creates zarr version snapshots for skim evolution tracking.

    Parameters
    ----------
    state : WorkflowState, optional
        Current workflow state (for year/iteration tracking)
    run_id : str, optional
        PILATES run ID (for zarr snapshot tracking)
    """
    logger.info(f"Starting merge of current BEAM skims into Zarr at {all_skims_path}")

    # Load canonical order for coordinate validation and correction
    canonical_zones_df = zone_utils.load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.tolist()
    logger.info(f"DEBUG: Loaded canonical order for {len(canonical_order)} zones.")

    # Determine input format and find skims
    partialSkims = None
    partial_skims_ds = None

    if override is None:
        # First try to find Zarr skims

        if os.path.exists(iteration_skims_path):
            current_skims_path = iteration_skims_path
        else:
            current_skims_path = None
    else:
        current_skims_path = override
        # Detect format from extension

    if current_skims_path is None or not os.path.exists(current_skims_path):
        logger.warning("No current skims found. Skipping merge.")
        return None

    canonical_order = verify_skim_zone_order(settings, current_skims_path, workspace)

    if current_skims_path.endswith(".zarr"):
        input_format = "zarr"
        partial_skims_ds = xr.open_zarr(current_skims_path)
        logger.info(
            f"DEBUG: Partial (BEAM) skims otaz coords: {partial_skims_ds.coords['otaz'].values[:5]}...{partial_skims_ds.coords['otaz'].values[-5:]}"
        )
        if "time_period" in partial_skims_ds.coords:
            logger.debug(
                "Partial BEAM skims time_period coords for %s: %s",
                current_skims_path,
                list(partial_skims_ds.coords["time_period"].values),
            )
    else:
        input_format = "omx"
        partialSkims = omx.open_file(current_skims_path, mode="r")

    # Ensure the main skims.zarr is 0-based and flagged before opening for merge
    try:
        ensure_0_based_and_flag_zarr_skims(all_skims_path, settings, workspace)
    except Exception as e:
        logger.error(f"Failed to ensure 0-based and flag main Zarr skims: {e}")
        return all_skims_path

    # Open the Zarr dataset in append mode to modify it in place
    try:
        skims_ds = xr.open_zarr(all_skims_path)
        logger.info(f"Opened Zarr skims file: {all_skims_path}")
        logger.info(
            f"DEBUG: Initial skims_ds otaz coords: {skims_ds.coords['otaz'].values[:5]}...{skims_ds.coords['otaz'].values[-5:]}"
        )

        # Store original zone IDs if needed
        if "original_zone_ids" not in skims_ds.attrs:
            if partial_skims_ds and "original_zone_ids" in partial_skims_ds.attrs:
                # Copy from partial skims if available
                skims_ds.attrs["original_zone_ids"] = partial_skims_ds.attrs[
                    "original_zone_ids"
                ]
            else:
                # Get from settings
                canonical_zones_df = zone_utils.load_canonical_zones(
                    settings, workspace
                )
                original_zone_ids = canonical_zones_df.index.tolist()
                skims_ds.attrs["original_zone_ids"] = original_zone_ids
    except Exception as e:
        logger.error(f"Failed to open target Zarr skims file {all_skims_path}: {e}")
        if partial_skims_ds:
            partial_skims_ds.close()
        return None

    timePeriods = get_setting(settings, "shared.skims.periods")
    consolidate_tnc_fleets = get_setting(settings, "consolidate_tnc_fleets", True)

    # Step 1: Accumulate completed and failed trips based on format
    completed_failed_dict_all = {}
    if input_format == "zarr" and partial_skims_ds:
        completed_failed_dict_all = _accumulate_all_completed_failed_trips_zarr(
            partial_skims_ds, timePeriods
        )
    elif input_format == "omx" and partialSkims:
        completed_failed_dict_all = _accumulate_all_completed_failed_trips(
            partialSkims, timePeriods
        )

    previous_weight = get_setting(settings, "beam.skim_previous_weight", 0.9)

    # Merge trip counts
    for path, (completed, failed) in completed_failed_dict_all.items():
        _merge_zarr_trip_counts(
            skims_ds, path, completed, failed, previous_weight=previous_weight
        )

    if partial_skims_ds or partialSkims:
        try:
            if input_format == "zarr":
                # You'll need to create a Zarr version of _process_all_tnc_logic
                tnc_modes_processed = _process_all_tnc_logic_zarr(
                    partial_skims_ds,
                    skims_ds,
                    timePeriods,
                    settings,
                    completed_failed_dict_all,
                    previous_weight=previous_weight,
                )
            else:
                tnc_modes_processed = _process_all_tnc_logic(
                    partialSkims,
                    skims_ds,
                    timePeriods,
                    settings,
                    completed_failed_dict_all,
                )
            logger.info(f"Processed TNC modes: {tnc_modes_processed}")
            logger.info(
                f"DEBUG: skims_ds otaz coords after TNC logic: {skims_ds.coords['otaz'].values[:5]}...{skims_ds.coords['otaz'].values[-5:]}"
            )
        except Exception as e:
            logger.error(f"Failed to process TNC logic: {e}", exc_info=True)
    else:
        logger.warning(
            "No new OMX skims found. Reconstructing TNC/RH trip counts from existing Zarr data for post-processing."
        )
        tnc_modes_for_postprocess = set()
        all_zarr_vars = list(skims_ds.data_vars)
        if consolidate_tnc_fleets:
            for target_mode in TNC_CONSOLIDATION_MAP.values():
                if f"{target_mode}_TRIPS" in all_zarr_vars:
                    tnc_modes_for_postprocess.add(target_mode)
        else:
            # Find existing TNC provider modes in Zarr
            for var_name in all_zarr_vars:
                if var_name.startswith("TNC_") and "_TRIPS" in var_name:
                    mode = var_name.rsplit("_TRIPS", 1)[0]  # Extract mode name
                    tnc_modes_for_postprocess.add(mode)

        # Reconstruct completed_failed_dict for these modes from Zarr
        completed_failed_dict_from_zarr = {}
        for mode in tnc_modes_for_postprocess:
            trips_key = f"{mode}_TRIPS"
            failures_key = f"{mode}_FAILURES"
            if trips_key in skims_ds.data_vars and failures_key in skims_ds.data_vars:
                completed_failed_dict_from_zarr[mode] = (
                    np.nan_to_num(skims_ds[trips_key].data),
                    np.nan_to_num(skims_ds[failures_key].data),
                )
        if tnc_modes_for_postprocess:
            # Run post-processing on existing TNC/RH data in Zarr
            _postprocess_tnc_zarr(
                skims_ds,
                timePeriods,
                settings,
                completed_failed_dict_from_zarr,
                use_rh_modes=consolidate_tnc_fleets,
            )
        else:
            logger.info("No existing TNC/RH modes found in Zarr for post-processing.")

    if input_format == "zarr" and partial_skims_ds:
        # Process Zarr format
        grouped_partial_sources = {}

        for var_name in partial_skims_ds.data_vars:
            # Parse path and measure from variable name
            # Similar logic to OMX parsing but simpler since format is PATH_MEASURE
            parts = var_name.rsplit("_", 1)
            if len(parts) == 2:
                path, measure = parts

                # Skip TNC/RH paths
                if path.startswith("TNC_") or path.startswith("RH_"):
                    continue

                # Skip if no trip counts
                if path not in completed_failed_dict_all:
                    continue

                grouped_partial_sources[(path, measure)] = [var_name]

        # Process each group
        for (path, measure), var_names in grouped_partial_sources.items():
            completed_3d, failed_3d = completed_failed_dict_all[path]
            _merge_one_zarr_measure_from_zarr(
                partial_skims_ds,
                skims_ds,
                path,
                measure,
                timePeriods,
                completed_3d,
                failed_3d,
            )

    elif (
        input_format == "omx" and partialSkims
    ):  # Only merge if we have new partial skims
        # Group partial skims by (path, measure)
        grouped_partial_sources = {}  # Key: (path, measure), Value: list of omx_key

        for omx_key in partialSkims.list_matrices():
            # Split key into measure_part and period
            parts = omx_key.split("__")
            if len(parts) != 2:
                continue  # Skip keys not in PATH_MEASURE__PERIOD format

            measure_part = parts[0]
            period = parts[1]

            if period not in timePeriods:
                continue

            # Determine path and measure
            # Try parsing from right for standard measures
            measure = None
            path = None
            measure_part_parts = measure_part.split("_")
            reversed_parts = measure_part_parts[::-1]

            for i in range(len(reversed_parts)):
                potential_measure_parts = reversed_parts[: i + 1][::-1]
                potential_measure = "_".join(potential_measure_parts)
                potential_path_parts = measure_part_parts[
                    : len(measure_part_parts) - (i + 1)
                ]
                potential_path = "_".join(potential_path_parts)

                # Check if the potential measure is a common one
                if potential_measure in [
                    "TRIPS",
                    "FAILURES",
                    "REJECTIONPROB",
                    "IWAIT",
                    "XWAIT",
                    "WACC",
                    "WAUX",
                    "WEGR",
                    "DTIM",
                    "DDIST",
                    "TOTIVT",
                    "FAR",
                    "COST",
                    "DIST",
                    "TIME",
                    "BTOLL",
                    "VTOLL",
                    "KEYIVT",
                    "FERRYIVT",
                ]:
                    measure = potential_measure
                    path = potential_path
                    break

            if path is None or measure is None:
                logger.debug(
                    f"Could not parse path/measure from key {omx_key}. Skipping."
                )
                continue

            # Skip TNC/RH paths if they were handled by consolidation/transfer
            if path.startswith("TNC_") or path.startswith("RH_"):
                # If consolidation was on, TNC_* and RH_* were handled by _process_all_tnc_logic
                # If consolidation was off, TNC_* were handled by _process_all_tnc_logic
                # In either case, skip them here.
                if consolidate_tnc_fleets:
                    # If consolidating, skip original TNC provider paths and the new RH paths
                    # Need to check if 'path' is an original provider path or a consolidated RH path
                    is_provider_path = path.startswith("TNC_")  # e.g. TNC_SINGLE_UBER
                    is_consolidated_path = (
                        path in TNC_CONSOLIDATION_MAP.values()
                    )  # e.g. RH_SOLO
                    if is_provider_path or is_consolidated_path:
                        logger.debug(
                            f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC/RH and handled elsewhere (consolidation enabled)."
                        )
                        continue
                else:
                    # If not consolidating, skip original TNC provider paths (handled by transfer)
                    if path.startswith("TNC_"):
                        logger.debug(
                            f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC and handled elsewhere (consolidation disabled)."
                        )
                        continue

            # Add to the grouping for non-TNC/non-RH paths
            if (path, measure) not in grouped_partial_sources:
                grouped_partial_sources[(path, measure)] = []
            grouped_partial_sources[(path, measure)].append(omx_key)

        logger.info(
            f"Merging {len(grouped_partial_sources)} non-TNC/non-RH (path, measure) groups from partial skims."
        )

        # Process each non-TNC/non-RH (path, measure) group
        processed_non_tnc_vars = set()
        for (path, measure), omx_keys in grouped_partial_sources.items():
            # Ensure we have completed/failed counts for this path
            if path not in completed_failed_dict_all:
                logger.warning(
                    f"Completed/failed trip data for path {path} not found. Skipping merge for measure {measure}."
                )
                continue

            completed_3d, failed_3d = completed_failed_dict_all[path]

            # Call the generic merge function for this measure/path
            processed_path, processed_measure = _merge_one_zarr_measure(
                partialSkims,
                skims_ds,
                path,
                measure,
                timePeriods,
                completed_3d,
                failed_3d,
            )
            if processed_path is not None:
                processed_non_tnc_vars.add(f"{processed_path}_{processed_measure}")

        logger.info(
            f"Completed merging {len(processed_non_tnc_vars)} non-TNC/non-RH variables."
        )

    # Step 4: Handle transit mode availability (uses TRIPS/FAILURES data already in skims_ds)
    # This should run regardless of whether new OMX skims were merged, to ensure consistency.
    logger.info("Applying transit mode availability rules.")
    _handle_transit_mode_availability(skims_ds, timePeriods)

    # Step 5: Copy skims for unobserved modes (e.g., SOV -> HOV/SOVTOLL)
    # This should run after the base skims (like SOV) are finalized.
    # This should run regardless of whether new OMX skims were merged.
    logger.info("Copying skims for unobserved modes based on mapping.")
    mapping = get_setting(
        settings,
        "beam.ridehail_path_map",
        {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]},
    )  # Add setting
    copy_skims_for_unobserved_modes(mapping, skims_ds)

    # Step 6: Save the updated Zarr dataset
    logger.info(f"Started writing updated zarr skims to {all_skims_path}")
    try:
        # Defensively ensure coordinates are correct before saving, to counteract any
        # potential corruption or misalignment that may have occurred during the merge.
        expected_coords = np.arange(len(canonical_order))
        otaz_corrected = False
        dtaz_corrected = False
        if not np.array_equal(skims_ds.coords["otaz"].values, expected_coords):
            logger.warning(
                "Correcting 'otaz' coordinates to be 0-based before final save."
            )
            skims_ds.coords["otaz"].data[:] = expected_coords
            otaz_corrected = True
        if not np.array_equal(skims_ds.coords["dtaz"].values, expected_coords):
            logger.warning(
                "Correcting 'dtaz' coordinates to be 0-based before final save."
            )
            skims_ds.coords["dtaz"].data[:] = expected_coords
            dtaz_corrected = True

        skims_ds.to_zarr(
            all_skims_path, mode="a", consolidated=False, zarr_format=2
        )

        import zarr

        zgroup = zarr.open_group(all_skims_path, mode="a")
        if otaz_corrected and "otaz" in zgroup:
            zgroup["otaz"][:] = expected_coords
        if dtaz_corrected and "dtaz" in zgroup:
            zgroup["dtaz"][:] = expected_coords
        zgroup.attrs["ZARR_WRITE_TIME"] = time.time()
        if "original_zone_ids" in skims_ds.attrs:
            zgroup.attrs["original_zone_ids"] = skims_ds.attrs["original_zone_ids"]

        zarr.consolidate_metadata(all_skims_path)
        logger.info("Completed writing zarr skims successfully.")
        merge_successful = (
            True  # Indicate merge/post-processing was attempted successfully
        )
    except Exception as e:
        logger.error(f"FAILED to write updated zarr skims to {all_skims_path}: {e}")
        raise

        # Independent verification of the written Zarr file
    try:
        with xr.open_zarr(all_skims_path) as verified_ds:
            logger.info(
                f"DEBUG: VERIFICATION - Zarr file on disk ({all_skims_path}) otaz coords: {verified_ds.coords['otaz'].values[:5]}...{verified_ds.coords['otaz'].values[-5:]}"
            )
            if "time_period" in verified_ds.coords:
                logger.debug(
                    "Post-write time_period coords for %s: %s",
                    all_skims_path,
                    list(verified_ds.coords["time_period"].values),
                )
    except Exception as e:
        logger.error(
            f"DEBUG: VERIFICATION FAILED - Could not open or verify {all_skims_path}: {e}"
        )

    # Close the datasets
    if partialSkims:
        partialSkims.close()
    if partial_skims_ds:
        partial_skims_ds.close()
    if skims_ds:
        skims_ds.close()

    # Return the updated Zarr skims path whenever we successfully wrote them.
    # This ensures downstream logging captures the rewritten master skims even
    # if no new OMX skims were found.
    if merge_successful:
        return all_skims_path
    logger.warning("No new OMX skims found, returning None.")
    return None


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
    consolidate_tnc_fleets = get_setting(settings, "consolidate_tnc_fleets", True)
    tnc_modes_processed = set()

    if consolidate_tnc_fleets:
        # Consolidate TNC providers into RH modes
        logger.info("Consolidating TNC fleets into RH modes from Zarr partial skims...")

        # Group providers by their target RH mode
        rh_consolidation = {}  # target_mode -> list of provider modes
        for provider_mode, target_mode in TNC_CONSOLIDATION_MAP.items():
            if provider_mode in completed_failed_dict_all:
                if target_mode not in rh_consolidation:
                    rh_consolidation[target_mode] = []
                rh_consolidation[target_mode].append(provider_mode)
                tnc_modes_processed.add(provider_mode)

        # Process each RH target mode
        for target_mode, provider_modes in rh_consolidation.items():
            logger.info(f"Consolidating {provider_modes} into {target_mode}")

            # Accumulate trip counts across providers per time period
            for tp_idx in range(len(timePeriods)):
                consolidated_completed_tp = np.zeros(
                    completed_failed_dict_all[provider_modes[0]][0].shape[:2],
                    dtype=np.float32,
                )
                consolidated_failed_tp = np.zeros_like(consolidated_completed_tp)
                for provider_mode in provider_modes:
                    completed, failed = completed_failed_dict_all[provider_mode]
                    consolidated_completed_tp += completed[:, :, tp_idx]
                    consolidated_failed_tp += failed[:, :, tp_idx]
                _merge_zarr_trip_counts_slice(
                    skims_ds,
                    target_mode,
                    consolidated_completed_tp,
                    consolidated_failed_tp,
                    tp_idx,
                    previous_weight=previous_weight,
                )

            # Process each measure for this RH mode
            # Get all measures from the first provider as a template
            first_provider = provider_modes[0]
            provider_measures = set()

            for var_name in partial_skims_ds.data_vars:
                if var_name.startswith(f"{first_provider}_") and not (
                    var_name.endswith("_TRIPS") or var_name.endswith("_FAILURES")
                ):
                    # Extract measure name
                    measure = var_name[len(first_provider) + 1 :]
                    provider_measures.add(measure)

            # Consolidate each measure
            for measure in provider_measures:
                target_var_name = f"{target_mode}_{measure}"

                # Create target variable if it doesn't exist
                if target_var_name not in skims_ds.data_vars:
                    # Use first provider's variable as template
                    template_var = f"{first_provider}_{measure}"
                    if template_var in partial_skims_ds.data_vars:
                        skims_ds[target_var_name] = xr.DataArray(
                            np.zeros(
                                partial_skims_ds[template_var].shape, dtype=np.float32
                            ),
                            coords=partial_skims_ds[template_var].coords,
                            dims=partial_skims_ds[template_var].dims,
                            name=target_var_name,
                        )

                # Consolidate data from all providers
                if target_var_name in skims_ds.data_vars:
                    for tp_idx in range(len(timePeriods)):
                        consolidated_data_tp = np.zeros(
                            skims_ds[target_var_name].shape[:2], dtype=np.float32
                        )
                        total_weights_tp = np.zeros_like(consolidated_data_tp)

                        for provider_mode in provider_modes:
                            provider_var = f"{provider_mode}_{measure}"
                            if provider_var in partial_skims_ds.data_vars:
                                provider_data_tp = partial_skims_ds[
                                    provider_var
                                ].data[:, :, tp_idx]
                                provider_completed_tp = completed_failed_dict_all[
                                    provider_mode
                                ][0][:, :, tp_idx]

                                mask = provider_completed_tp > 0
                                consolidated_data_tp[mask] += (
                                    provider_data_tp[mask] * provider_completed_tp[mask]
                                )
                                total_weights_tp[mask] += provider_completed_tp[mask]

                        mask = total_weights_tp > 0
                        consolidated_data_tp[mask] = (
                            consolidated_data_tp[mask] / total_weights_tp[mask]
                        )

                        completed_tp = skims_ds[f"{target_mode}_TRIPS"].data[
                            :, :, tp_idx
                        ]
                        failed_tp = skims_ds[f"{target_mode}_FAILURES"].data[
                            :, :, tp_idx
                        ]

                        mask_update, vals_update, to_cancel = _transform_measure(
                            consolidated_data_tp,
                            completed_tp,
                            failed_tp,
                            measure,
                            target_mode,
                        )

                        if mask_update.any():
                            skims_ds[target_var_name].data[:, :, tp_idx][
                                mask_update
                            ] = vals_update

                        if to_cancel is not None and to_cancel.any():
                            skims_ds[target_var_name].data[:, :, tp_idx][to_cancel] = 0.0

            tnc_modes_processed.add(target_mode)

    else:
        # Transfer TNC provider modes directly without consolidation
        logger.info(
            "Transferring TNC provider modes directly from Zarr partial skims..."
        )

        # Find all TNC provider modes in partial skims
        tnc_provider_modes = set()
        for var_name in partial_skims_ds.data_vars:
            if var_name.startswith("TNC_") and "_TRIPS" in var_name:
                mode = var_name[: -len("_TRIPS")]
                tnc_provider_modes.add(mode)

        # Process each TNC provider mode
        for provider_mode in tnc_provider_modes:
            if provider_mode not in completed_failed_dict_all:
                continue

            completed, failed = completed_failed_dict_all[provider_mode]

            # Get all measures for this provider
            provider_measures = set()
            for var_name in partial_skims_ds.data_vars:
                if var_name.startswith(f"{provider_mode}_") and not (
                    var_name.endswith("_TRIPS") or var_name.endswith("_FAILURES")
                ):
                    measure = var_name[len(provider_mode) + 1 :]
                    provider_measures.add(measure)

            # Transfer each measure
            for measure in provider_measures:
                _merge_one_zarr_measure_from_zarr(
                    partial_skims_ds,
                    skims_ds,
                    provider_mode,
                    measure,
                    timePeriods,
                    completed,
                    failed,
                )

            tnc_modes_processed.add(provider_mode)

    # Run post-processing on the TNC/RH modes
    if tnc_modes_processed:
        completed_failed_for_postprocess = {
            mode: completed_failed_dict_all.get(mode, (None, None))
            for mode in tnc_modes_processed
            if mode in completed_failed_dict_all or mode in skims_ds.data_vars
        }

        # For consolidated RH modes, use the consolidated counts
        if consolidate_tnc_fleets:
            for target_mode in rh_consolidation:
                if f"{target_mode}_TRIPS" in skims_ds.data_vars:
                    completed_failed_for_postprocess[target_mode] = (
                        np.nan_to_num(skims_ds[f"{target_mode}_TRIPS"].values),
                        np.nan_to_num(skims_ds[f"{target_mode}_FAILURES"].values),
                    )

        _postprocess_tnc_zarr(
            skims_ds,
            timePeriods,
            settings,
            completed_failed_for_postprocess,
            use_rh_modes=consolidate_tnc_fleets,
        )

    return tnc_modes_processed


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

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM.
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    timePeriods : list of str
        List of time period names.
    settings : dict
        Settings dictionary.
    completed_failed_dict_all : dict
        Dictionary containing completed and failed trip counts (3D arrays)
        for ALL modes/providers in the partial skims.

    Returns:
    --------
    set
        A set of Zarr variable names that were created/updated by this function (TNC provider or RH consolidated).
    """
    consolidate_tnc_fleets = get_setting(settings, "consolidate_tnc_fleets", True)
    processed_vars = set()
    tnc_modes_to_postprocess = set()
    completed_failed_dict_for_postprocess = (
        {}
    )  # Counts for the modes that will be post-processed

    if consolidate_tnc_fleets:
        logger.info("TNC fleet consolidation enabled.")
        # Step 2a: Consolidate TNC provider data into RH_* variables in skims_ds
        # This function returns the consolidated RH modes
        consolidated_modes = _consolidate_tnc_data_zarr(
            partialSkims,
            skims_ds,
            timePeriods,
            completed_failed_dict_all,
            settings,
        )
        # Add consolidated RH modes to the list for post-processing
        tnc_modes_to_postprocess.update(consolidated_modes)

        # The consolidated variables (RH_*) are now in skims_ds.
        # We need to get their names to return them as handled.
        for target_mode in consolidated_modes:
            for (
                measure
            ) in (
                skims_ds.data_vars
            ):  # Iterate through ALL skims_ds vars to find the ones belonging to this mode
                if measure.startswith(f"{target_mode}_"):
                    processed_vars.add(measure)

    else:
        logger.info("TNC fleet consolidation disabled. Transferring provider data.")
        # Step 2a: Transfer TNC provider data into TNC_* variables in skims_ds
        # This function returns the names of variables transferred.
        transferred_vars = _transfer_tnc_provider_data_zarr(
            partialSkims,
            skims_ds,
            timePeriods,
            completed_failed_dict_all,
            settings,
        )
        processed_vars.update(transferred_vars)

        # Identify the TNC provider modes that were transferred
        # Iterate through the transferred variable names to find the modes
        for var_name in transferred_vars:
            parts = var_name.split("_")
            if len(parts) >= 3 and parts[0] == "TNC":
                # Assuming format TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
                # Need to parse the mode name correctly again
                measure_part = var_name.rsplit("_", 1)[
                    0
                ]  # Remove measure at the end for 3D vars like RH_SOLO_TRIPS
                if measure_part.endswith("__"):  # Handle 2D vars like DIST__AM
                    measure_part = var_name.split("__")[0]
                elif "_" in var_name.split("__")[0]:  # Handle 3D vars
                    measure_part = var_name.split("__")[0]

                # Find the mode name from the beginning of the measure_part
                mode_name = None
                measure_name_check = measure_part
                while "_" in measure_name_check:
                    potential_mode = measure_name_check
                    potential_measure_end = measure_name_check.rsplit("_", 1)[-1]
                    if potential_measure_end.upper() in [
                        "TRIPS",
                        "FAILURES",
                        "REJECTIONPROB",
                        "IWAIT",
                        "DDIST",
                        "TOTIVT",
                        "FAR",
                    ]:
                        # Found a potential measure end, the rest is the mode
                        mode_name = potential_mode.rsplit("_", 1)[0]
                        break
                    measure_name_check = potential_mode.rsplit("_", 1)[
                        0
                    ]  # Chop off the last part and try again

                if mode_name and mode_name.startswith("TNC_"):
                    tnc_modes_to_postprocess.add(mode_name)
                    # Use the provider counts for post-processing
                    if mode_name in completed_failed_dict_all:
                        completed_failed_dict_for_postprocess[mode_name] = (
                            completed_failed_dict_all[mode_name]
                        )
                    else:
                        logger.warning(
                            f"Completed/failed counts not found for provider mode {mode_name} in completed_failed_dict_all. Cannot post-process."
                        )

        logger.info(
            f"Identified {len(tnc_modes_to_postprocess)} TNC provider modes for post-processing."
        )

    # Step 2b: Run TNC/RH specific post-processing on the data now in skims_ds
    # This function needs the completed/failed counts for the modes it's processing.
    if tnc_modes_to_postprocess:
        _postprocess_tnc_zarr(
            skims_ds,
            timePeriods,
            settings,
            completed_failed_dict_for_postprocess,
            use_rh_modes=consolidate_tnc_fleets,
        )
    else:
        logger.info(
            "No TNC/RH modes to post-process after consolidation/transfer step."
        )

    return processed_vars


def validate_zone_alignment(zarr_path, land_use_df):
    """Validate that zarr zones align with land_use zones."""
    ds = xr.open_zarr(zarr_path)

    # Get zones from zarr
    zarr_zones = ds.coords["otaz"].values

    # Get zones from land_use
    if f"_original_{land_use_df.index.name}" in land_use_df:
        land_use_zones = land_use_df[f"_original_{land_use_df.index.name}"].values
    else:
        land_use_zones = land_use_df.index.values

    # Check if they match
    if not np.array_equal(zarr_zones, land_use_zones):
        logger.warning(
            f"Zone mismatch! Zarr has {len(zarr_zones)} zones, land_use has {len(land_use_zones)}"
        )
        logger.warning(f"First 10 zarr zones: {zarr_zones[:10]}")
        logger.warning(f"First 10 land_use zones: {land_use_zones[:10]}")

        # Check if same zones, different order
        if set(zarr_zones) == set(land_use_zones):
            logger.info("Same zones but different order - reindexing will be required")
        else:
            missing_in_zarr = set(land_use_zones) - set(zarr_zones)
            missing_in_land_use = set(zarr_zones) - set(land_use_zones)
            if missing_in_zarr:
                logger.error(f"Zones in land_use but not zarr: {missing_in_zarr}")
            if missing_in_land_use:
                logger.error(f"Zones in zarr but not land_use: {missing_in_land_use}")
        return False
    return True


def trim_inaccessible_ods_zarr(all_skims_path, settings):
    """
    Zero out inaccessible ODs in Zarr-format skims, similar to trim_inaccessible_ods for OMX.
    Uses direct .data access for in-place modification.
    """
    logger.info("Starting trim of inaccessible ODs in Zarr skims.")
    try:
        skims = xr.open_zarr(all_skims_path)  # Open in append/write mode
    except Exception as e:
        logger.error(
            f"Failed to open skims file {all_skims_path} for trimming inaccessible ODs: {e}"
        )
        return

    try:
        order = zone_order(
            settings, get_setting(settings, "run.start_year", 2015)
        )  # Use .get for start_year safety
        periods = get_setting(settings, "shared.skims.periods")
        transit_paths_settings = get_setting(
            settings, "shared.skims.transit_paths", {}
        )  # Use .get for safety
    except Exception as e:
        logger.error(f"Error reading settings for trimming: {e}. Skipping trim.")
        skims.close()
        return

    if not transit_paths_settings:
        logger.warning(
            "No 'transit_paths' defined in settings for trimming inaccessible ODs. Skipping trim."
        )
        skims.close()
        return

    num_zones = len(order)

    # Calculate total trips per OD pair across all non-RH transit/walk/bike modes for each period
    # Initialize totalTrips as a 3D array [zones, zones, periods]
    totalTrips = np.zeros((num_zones, num_zones, len(periods)), dtype=np.float32)

    all_vars = list(skims.data_vars)

    for var_name in all_vars:
        parts = var_name.split("__")
        if (
            len(parts) == 2
        ):  # Variable name is in PATH_MEASURE__PERIOD format (old OMX style, sometimes kept in Zarr)
            measure_part, tp = parts
            if tp in periods:
                # Check if it's a TRIPS matrix for a non-RH mode
                if "TRIPS" in measure_part and not measure_part.startswith("RH_"):
                    try:
                        tp_idx = periods.index(tp)
                        if skims[var_name].ndim == 2 and skims[var_name].shape == (
                            num_zones,
                            num_zones,
                        ):
                            totalTrips[:, :, tp_idx] += np.nan_to_num(
                                skims[var_name].data
                            )
                        else:
                            logger.debug(
                                f"Skipping '{var_name}' for total trips calculation due to unexpected shape {skims[var_name].shape}."
                            )
                    except ValueError:
                        logger.warning(
                            f"Period '{tp}' from key '{var_name}' not found in settings periods. Skipping."
                        )
                    except Exception as e:
                        logger.error(
                            f"Error reading '{var_name}' for total trips calculation: {e}"
                        )

        elif (
            skims[var_name].ndim == 3
        ):  # Variable name is in PATH_MEASURE format (new Zarr style)
            # Check if it's a TRIPS matrix for a non-RH mode
            if var_name.endswith("_TRIPS") and not var_name.startswith("RH_"):
                try:
                    # Assuming 3D shape is [zones, zones, periods]
                    if skims[var_name].shape[:2] == (num_zones, num_zones) and skims[
                        var_name
                    ].shape[-1] >= len(periods):
                        # Sum across all periods we care about (as defined by settings)
                        for tp_idx in range(len(periods)):
                            totalTrips[:, :, tp_idx] += np.nan_to_num(
                                skims[var_name].data[:, :, tp_idx]
                            )
                    else:
                        logger.debug(
                            f"Skipping '{var_name}' for total trips calculation due to unexpected shape {skims[var_name].shape}."
                        )
                except Exception as e:
                    logger.error(
                        f"Error reading '{var_name}' for total trips calculation: {e}"
                    )

    # Calculate total trips BY ZONE (sum across rows and columns for each zone) for each period
    completedAllTripsByOandD_3d = totalTrips.sum(axis=1) + totalTrips.sum(
        axis=0
    )  # Shape (zones, periods)

    # Now, check transit modes for trimming
    for path, metrics in transit_paths_settings.items():
        trip_name_zarr = f"{path}_TRIPS"  # Zarr key format
        fail_name_zarr = f"{path}_FAILURES"  # Zarr key format

        trip_da = skims.get(trip_name_zarr)
        fail_da = skims.get(fail_name_zarr)

        if trip_da is None or fail_da is None:
            logger.debug(
                f"Skipping trim for {path}: missing {trip_name_zarr} or {fail_name_zarr} in skims_ds."
            )
            continue

        # Get 3D completed/failed transit trips for this mode
        try:
            completedTransitTrips_3d = np.nan_to_num(
                trip_da.data[:, :, : len(periods)]
            )  # Slice to match settings periods
            failedTransitTrips_3d = np.nan_to_num(
                fail_da.data[:, :, : len(periods)]
            )  # Slice to match settings periods
            if completedTransitTrips_3d.shape[:2] != (
                num_zones,
                num_zones,
            ) or completedTransitTrips_3d.shape[-1] != len(periods):
                logger.warning(
                    f"Shape mismatch for {trip_name_zarr} or {fail_name_zarr} ({completedTransitTrips_3d.shape}) vs expected ({num_zones}, {num_zones}, {len(periods)}). Skipping trim for {path}."
                )
                continue
        except Exception as e:
            logger.error(
                f"Error reading trip data for {path} from Zarr: {e}. Skipping trim for this path."
            )
            continue

        # Calculate completed and failed transit trips BY ZONE for each period
        completedTransitTripsByOandD_3d = completedTransitTrips_3d.sum(
            axis=1
        ) + completedTransitTrips_3d.sum(
            axis=0
        )  # Shape (zones, periods)
        failedTransitTripsByOandD_3d = failedTransitTrips_3d.sum(
            axis=1
        ) + failedTransitTrips_3d.sum(
            axis=0
        )  # Shape (zones, periods)

        # Determine which zones to delete for each period
        # Condition: (Total trips > 1000) & (Failed transit trips > 200) & (Completed transit trips == 0)
        toDelete_3d = (
            (completedAllTripsByOandD_3d > 1000)
            & (failedTransitTripsByOandD_3d > 200)
            & (completedTransitTripsByOandD_3d == 0)
        )  # Shape (zones, periods)

        # Apply trimming for each period where zones need deletion
        for tpIdx, period in enumerate(periods):
            zones_to_delete_tp = toDelete_3d[
                :, tpIdx
            ]  # Boolean array (zones,) for this period
            num_zones_to_delete_tp = np.sum(zones_to_delete_tp)

            if num_zones_to_delete_tp > 0:
                logger.info(
                    f"Trimming {path} service for {num_zones_to_delete_tp} zones in {period} because no completed transit trips were observed for these zones (conditions met)."
                )

                # Apply deletion mask to metrics for this path and period
                for metric in metrics:
                    name_zarr = f"{path}_{metric}"  # Zarr key format
                    metric_da = skims.get(name_zarr)

                    if metric_da is not None:
                        try:
                            # Ensure the DataArray has enough dimensions and size for the period
                            if (
                                metric_da.ndim == 3
                                and metric_da.shape[-1] > tpIdx
                                and metric_da.shape[:2] == (num_zones, num_zones)
                            ):
                                # Get the 2D slice for this period
                                arr_slice = metric_da.data[:, :, tpIdx]
                                # Apply the deletion mask. This sets values to 0 where the origin OR destination zone is in `zones_to_delete_tp`.
                                arr_slice[
                                    zones_to_delete_tp[:, None]
                                    | zones_to_delete_tp[None, :]
                                ] = 0.0
                                # Update the data in the Zarr array
                                metric_da.data[:, :, tpIdx] = arr_slice
                                logger.debug(
                                    f"Trimmed {name_zarr} for {period} for {num_zones_to_delete_tp} zones."
                                )
                            elif (
                                metric_da.ndim == 2
                                and len(periods) == 1
                                and periods[0] == period
                                and metric_da.shape == (num_zones, num_zones)
                            ):
                                # Handle 2D case (single period)
                                arr_slice = metric_da.data[:, :]
                                arr_slice[
                                    zones_to_delete_tp[:, None]
                                    | zones_to_delete_tp[None, :]
                                ] = 0.0
                                metric_da.data[:, :] = arr_slice
                                logger.debug(
                                    f"Trimmed {name_zarr} (2D) for {period} for {num_zones_to_delete_tp} zones."
                                )
                            else:
                                logger.warning(
                                    f"Skipping trim for metric {name_zarr} in {period} due to unexpected shape {metric_da.shape} or period mismatch."
                                )
                        except Exception as e:
                            logger.error(
                                f"Error applying trim mask for metric {name_zarr} in {period}: {e}. Skipping."
                            )
                    else:
                        logger.debug(
                            f"Skipping trim for metric {name_zarr} in {period}: variable not found in skims_ds."
                        )

    # Save the updated Zarr dataset after trimming
    logger.info(
        f"Started writing zarr skims after trimming inaccessible ODs to {all_skims_path}"
    )
    try:
        skims.to_zarr(
            all_skims_path, mode="w", consolidated=True, zarr_format=2
        )  # Use mode='w' to overwrite
        logger.info("Completed writing zarr skims after trimming successfully.")
    except Exception as e:
        logger.error(
            f"FAILED to write zarr skims after trimming inaccessible ODs to {all_skims_path}: {e}"
        )

    skims.close()


class BeamPostprocessor(GenericPostprocessor):
    """
    Postprocessor for BEAM model.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this postprocessor expects from the workflow.
        """
        beam_output_dir = workspace.get_beam_output_dir()
        zarr_path = None
        if getattr(settings, "activitysim", None) is not None:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        return {
            "beam_output_dir": (
                beam_output_dir if os.path.exists(beam_output_dir) else None
            ),
            "zarr_skims": zarr_path,
        }

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
              when OMX output is disabled.
        Related docs
            - See `pilates/beam/inputs.py` for the corresponding input
              descriptions used by BEAM and downstream models.
        """
        write_omx = get_setting(settings, "write_skims_to_omx", False) or (
            get_setting(settings, "run.models.land_use") == "urbansim"
        )
        if write_omx:
            region = settings.run.region
            omx_name = settings.shared.skims.fname
            final_omx_path = os.path.join(
                workspace.get_beam_mutable_data_dir(), region, omx_name
            )
            return {"final_skims_omx": final_omx_path}

        if _activitysim_skims_target_enabled(settings):
            zarr_path = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            return {"zarr_skims": zarr_path}
        return {}

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_data = ["zarr_skims", "raw_od_skims"]
        self.skim_format = get_setting(
            self.state.full_settings, "shared.skims.fname", "skimsActivitySimOD_current"
        )

    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocesses the raw outputs from a BEAM run by merging skims into the main Zarr store.
        """
        logger.info("Running BEAM postprocessor...")
        settings = self.state.full_settings

        raw_output_files = [record.short_name for record in raw_outputs.all_records()]
        processed_records = []

        events_record = _select_last_events_parquet_record(
            raw_outputs, self.state.forecast_year, self.state.iteration
        )
        if events_record is not None and getattr(events_record, "file_path", None):
            events_path = events_record.file_path
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
                        processed_records.append(
                            FileRecord(
                                file_path=out_path,
                                year=self.state.forecast_year,
                                short_name=short_name,
                                description=(
                                    "BEAM events parquet (type: %s)" % event_type
                                ),
                            )
                        )
                if links_path:
                    short_name = (
                        f"path_traversal_links_{self.state.forecast_year}_"
                        f"{self.state.iteration}"
                    )
                    processed_records.append(
                        FileRecord(
                            file_path=links_path,
                            year=self.state.forecast_year,
                            short_name=short_name,
                            description="BEAM PathTraversal link travel times",
                        )
                    )
            except Exception:
                logger.warning(
                    "Failed to split events parquet for %s", events_path, exc_info=True
                )

        all_skims_path = None
        if _activitysim_skims_target_enabled(settings):
            all_skims_path = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )

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
                    logger.warning(
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

        if not skim_name_found:
            logger.warning(
                "Raw BEAM OD skims file not found in raw_outputs. Skim merging will be skipped, but post-processing on existing Zarr will proceed."
            )

        elif all_skims_path is None or not os.path.exists(all_skims_path):
            logger.warning(
                f"Target Zarr skims file not found at {all_skims_path}. Cannot proceed with merging."
            )
        else:
            skim_name = skim_name_found

            all_skims_found = [
                record.file_path
                for record in raw_outputs.all_records()
                if record.short_name == skim_name
            ]

            logger.info("Here are all the skim files found")
            logger.info(all_skims_found)

            beam_output_dir = workspace.get_beam_output_dir()

            merged_sources = []
            merged_any = False
            for it, it_skim in enumerate(all_skims_found):

                raw_od_skims_path = os.path.join(
                    workspace.full_path,
                    it_skim,
                )
                merged_sources.append(raw_od_skims_path)

                logger.info(f"Processing {raw_od_skims_path} for iteration {it}")

                # Call the main merging and post-processing logic for Zarr skims
                updated_skims_path = _merge_beam_skims_to_zarr(
                    all_skims_path=all_skims_path,
                    iteration_skims_path=raw_od_skims_path,
                    beam_output_dir=beam_output_dir,
                    settings=settings,
                    workspace=workspace,
                    override=raw_od_skims_path,
                )

                # The main output is the modified Zarr store.
                if updated_skims_path:
                    merged_any = True

            if merged_any:
                processed_records.append(
                    FileRecord(
                        file_path=all_skims_path,
                        year=self.state.current_year,
                        description="Zarr skims store updated with BEAM outputs.",
                        short_name="zarr_skims",
                    )
                )

        if (
            get_setting(settings, "write_skims_to_omx", False)
            or get_setting(settings, "run.models.land_use") == "urbansim"
        ):
            logger.info(
                "Writing skims to OMX file for UrbanSim or other downstream models..."
            )
            if all_skims_path is None or not os.path.exists(all_skims_path):
                logger.error(
                    "Main Zarr skims file not available. Cannot write skims to OMX."
                )
            else:
                try:
                    # Exclude intermediate trip/failure tables from the final OMX
                    vars_to_exclude = []
                    with xr.open_zarr(all_skims_path) as skims_ds:
                        for key in skims_ds.data_vars:
                            if (
                                key.endswith("_TRIPS")
                                or key.endswith("_FAILURES")
                                or key.endswith("_REJECTIONPROB")
                            ):
                                vars_to_exclude.append(key)

                    final_omx_path = write_zarr_skim_as_omx_new(
                        all_skims_path,
                        settings,
                        get_setting(settings, "shared.skims.fname"),
                        exclude_tables=vars_to_exclude,
                    )

                    if final_omx_path:
                        processed_records.append(
                            FileRecord(
                                file_path=final_omx_path,
                                short_name="final_skims_omx",
                                description=(
                                    "Final skims converted to OMX format for "
                                    "ActivitySim/UrbanSim."
                                ),
                            )
                        )
                except Exception as e:
                    logger.error(f"Failed to write skims to OMX: {e}")

        output_store = RecordStore(recordList=processed_records)
        return output_store

    def postprocess(
        self,
        raw_outputs: BeamRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> BeamPostprocessOutputs:
        """
        Postprocess typed BEAM run outputs and return typed outputs.
        """
        if not isinstance(raw_outputs, BeamRunOutputs):
            raise TypeError("BeamPostprocessor.postprocess expects BeamRunOutputs")
        self.state.set_sub_stage_progress("postprocessor")
        input_store = RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
                for short_name, path, description in raw_outputs._iter_record_items()
            ]
        )
        output_store = self._postprocess(input_store, workspace, model_run_hash)
        mapping = output_store.to_mapping()

        zarr_path = artifact_to_path(mapping.get("zarr_skims"), workspace)
        final_omx_path = artifact_to_path(mapping.get("final_skims_omx"), workspace)

        split_events: Dict[str, Path] = {}
        split_event_links: Dict[str, Path] = {}
        for key, value in mapping.items():
            if key.startswith("events_parquet_") and "_type_" in key:
                path = artifact_to_path(value, workspace)
                if path is not None:
                    split_events[key] = Path(path)
            elif key.startswith("path_traversal_links_"):
                path = artifact_to_path(value, workspace)
                if path is not None:
                    split_event_links[key] = Path(path)

        return BeamPostprocessOutputs(
            zarr_skims=Path(zarr_path) if zarr_path is not None else None,
            final_skims_omx=Path(final_omx_path) if final_omx_path is not None else None,
            split_events=split_events,
            split_event_links=split_event_links,
        )
