import concurrent.futures
import logging
import os
import shutil

import numpy as np
import openmatrix as omx
import pandas as pd

try:
    import xarray as xr
except:
    print("FAILED TO LOAD XARRAY")

# import pickle
# import cloudpickle
# import dill
#
# np.seterr(divide='ignore')
#
# dill.settings['recurse'] = True
#
# pickle.ForkingPickler = cloudpickle.Pickler
#
# import multiprocessing as mp

# mp.set_start_method('spawn', True)
# from multiprocessing import Pool, cpu_count

from pilates.activitysim.preprocessor import zone_order

logger = logging.getLogger(__name__)

TNC_CONSOLIDATION_MAP = {
    "TNC_SINGLE": "RH_SOLO",
    "TNC_POOLED": "RH_SHARED",
}


def find_latest_beam_iteration(beam_output_dir):
    iter_dirs = [os.path.join(root, dir) for root, dirs, files in os.walk(beam_output_dir) if
                 not root.startswith('.') for dir in dirs if dir == "ITERS"]
    logger.info("Looking in directories {0}".format(iter_dirs))
    if not iter_dirs:
        return None, None
    last_iters_dir = max(iter_dirs, key=os.path.getmtime)
    all_iteration_dir = [it for it in os.listdir(last_iters_dir) if not it.startswith('.')]
    logger.info("Looking in directories {0}".format(all_iteration_dir))
    if not all_iteration_dir:
        return None, None
    it_prefix = "it."
    max_it_num = max(dir_name[len(it_prefix):] for dir_name in all_iteration_dir)
    return os.path.join(last_iters_dir, it_prefix + str(max_it_num)), max_it_num


def find_not_taken_dir_name(dir_name):
    for x in range(1, 99999):
        testing_name = f"{dir_name}_{x}"
        if not os.path.exists(testing_name):
            return testing_name
    raise RuntimeError(f"Cannot find an appropriate not taken directory for {dir_name}")


def rename_beam_output_directory(beam_output_dir, settings, year, replanning_iteration_number=0):
    iteration_output_directory, _ = find_latest_beam_iteration(beam_output_dir)
    beam_run_output_dir = os.path.join(*iteration_output_directory.split(os.sep)[:-2])
    new_iteration_output_directory = os.path.join(beam_output_dir, settings['region'],
                                                  "year-{0}-iteration-{1}".format(year, replanning_iteration_number))
    if os.path.exists(new_iteration_output_directory):
        os.rename(new_iteration_output_directory, find_not_taken_dir_name(new_iteration_output_directory))
    try:
        os.rename(beam_run_output_dir, new_iteration_output_directory)
    except FileNotFoundError:
        logger.warning("Files {0} not found. Adding a slash".format(beam_run_output_dir))
        os.rename("/" + str(beam_run_output_dir), new_iteration_output_directory)


def find_produced_od_skims(beam_output_dir, suffix="csv.gz"):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    od_skims_path = os.path.join(iteration_dir, "{0}.skimsActivitySimOD_current.{1}".format(it_num, suffix))
    logger.info("expecting skims at {0}".format(od_skims_path))
    return od_skims_path


def reorder(path):
    split = path.split('_')
    if len(split) == 6:
        split[3], split[5] = split[5], split[3]
        return '_'.join(split)
    elif len(split) == 4:
        split[1], split[3] = split[3], split[1]
        return '_'.join(split)
    elif len(split) == 5:
        split[2], split[4] = split[4], split[2]
        return '_'.join(split)
    else:
        print('STOP ', path)


def find_produced_origin_skims(beam_output_dir):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    ridehail_skims_path = os.path.join(iteration_dir, f"{it_num}.skimsRidehail.csv.gz")
    logger.info("expecting skims at {0}".format(ridehail_skims_path))
    return ridehail_skims_path


def _merge_skim(inputMats, outputMats, path, timePeriod, measures, transit_scale_factor=100.0):
    complete_key = f"{path}_TRIPS__{timePeriod}"
    failed_key = f"{path}_FAILURES__{timePeriod}"
    completed, failed = None, None

    if complete_key in inputMats.keys():
        completed = np.array(inputMats[complete_key]).copy()
        if f"{path}_TOTIVT__{timePeriod}" in inputMats.keys():
            shouldNotBeZero = (completed > 0) & (np.array(inputMats[f"{path}_TOTIVT__{timePeriod}"]) == 0)
            if shouldNotBeZero.any():
                logger.warning(
                    f"In BEAM outputs for {path} in {timePeriod} we have {shouldNotBeZero.sum()} completed trips with time = 0"
                )
                completed[shouldNotBeZero] = 0
        failed = np.array(inputMats[failed_key])

        logger.info(
            f"Adding {np.nan_to_num(completed).sum()} valid trips and {np.nan_to_num(failed).sum()} impossible trips to skim {complete_key}, where {np.nan_to_num(np.array(outputMats[complete_key])).sum()} had existed before"
        )
        logger.info(
            f"Of the {np.nan_to_num(completed).sum()} completed trips, {np.nan_to_num(completed[outputMats[complete_key][:] == 0]).sum()} were to a previously unobserved OD"
        )
        toCancel = []
        toPenalize = []

        for measure in measures:
            inputKey = f"{path}_{measure}__{timePeriod}"
            if path in ["WALK", "BIKE"]:
                if measure == "DIST":
                    outputKey = f"{path}DIST"
                else:
                    outputKey = f"{path}_{measure}"
            else:
                outputKey = inputKey

            if (outputKey in outputMats) and (inputKey in inputMats):
                if measure == "TRIPS":
                    outputMats[outputKey][completed > 0] += completed[completed > 0]
                elif measure == "FAILURES":
                    outputMats[outputKey][failed > 0] += failed[failed > 0]
                elif measure == "DIST":
                    outputMats[outputKey][completed > 0] = 0.5 * (
                            outputMats[outputKey][completed > 0] + inputMats[inputKey][completed > 0]
                    )
                elif measure in ["IWAIT", "XWAIT", "WACC", "WAUX", "WEGR", "DTIM", "DDIST", "FERRYIVT"]:
                    valid = ~np.isnan(inputMats[inputKey][:])
                    outputMats[outputKey][(completed > 0) & valid] = inputMats[inputKey][
                                                                         (completed > 0) & valid] * transit_scale_factor
                elif measure in ["TOTIVT", "IVT"]:
                    inputKeyKEYIVT = f"{path}_KEYIVT__{timePeriod}"
                    outputKeyKEYIVT = inputKeyKEYIVT
                    if (inputKeyKEYIVT in inputMats.keys()) & (outputKeyKEYIVT in outputMats.keys()):
                        additionalFilter = (outputMats[outputKeyKEYIVT][:] > 0)
                    else:
                        additionalFilter = False
                    outputTravelTime = np.array(outputMats[outputKey])
                    toCancel = (failed > 3) & (failed > (6 * completed))
                    previouslyNonZero = ((outputTravelTime > 0) | additionalFilter) & toCancel
                    toPenalize = (failed > completed) & ~toCancel & ((outputTravelTime > 0) | additionalFilter)
                    if toCancel.sum() > 0:
                        logger.info(
                            f"Marking {toCancel.sum()} {path} trips completely impossible in {timePeriod}. There were {completed[toCancel].sum()} completed trips but {failed[toCancel].sum()} failed trips in these ODs. Previously, {previouslyNonZero.sum()} were nonzero"
                        )
                        logger.info(
                            f"There are now {((completed > 0) & (outputTravelTime > 0)).sum()} observed ODs, {(outputTravelTime == 0).sum()} impossible ODs, and {((completed == 0) & (outputTravelTime > 0)).sum()} default ODs"
                        )
                    toAllow = ~toCancel & ~toPenalize & ~np.isnan(inputMats[inputKey][:])
                    outputMats[outputKey][toAllow] = inputMats[inputKey][toAllow] * 100
                    if (inputKeyKEYIVT in inputMats.keys()) & (outputKeyKEYIVT in outputMats.keys()):
                        outputMats[outputKeyKEYIVT][toAllow] = inputMats[inputKeyKEYIVT][toAllow] * 100
                elif not measure.endswith("TOLL"):
                    outputMats[outputKey][completed > 0] = inputMats[inputKey][completed > 0]
                    if path.startswith('SOV_'):
                        for sub in ['SOVTOLL_', 'HOV2_', 'HOV2TOLL_', 'HOV3_', 'HOV3TOLL_']:
                            newKey = f"{path}_{measure.replace('SOV_', sub)}__{timePeriod}"
                            outputMats[newKey][completed > 0] = inputMats[inputKey][completed > 0]
                            logger.info(
                                f"Adding {np.nan_to_num(completed).sum()} valid trips and {np.nan_to_num(failed).sum()} impossible trips to skim {newKey}")

                badVals = np.sum(np.isnan(outputMats[outputKey][:]))
                if badVals > 0:
                    logger.warning(f"Total number of {badVals} skim values are NaN for skim {outputKey}")
            elif outputKey in outputMats:
                logger.warning(f"Target skims are missing key {outputKey}")
            else:
                logger.warning(f"BEAM skims are missing key {outputKey}")

        if toCancel.sum() > 0:
            for measure in measures:
                if measure not in ["TRIPS", "FAILURES"]:
                    key = f"{path}_{measure}__{timePeriod}"
                    try:
                        outputMats[key][toCancel] = 0.0
                    except:
                        logger.warning(f"Tried to cancel {toCancel.sum()} trips for key {key} but couldn't find key")

        if ("TOTIVT" in measures) & ("IWAIT" in measures) & ("KEYIVT" in measures):
            if toPenalize.sum() > 0:
                inputKey = f"{path}_IWAIT__{timePeriod}"
                outputMats[inputKey][toPenalize] = inputMats[inputKey][toPenalize] * (failed[toPenalize] + 1) / (
                        completed[toPenalize] + 1
                )
    else:
        logger.info(f"No input skim for mode {path} and time period {timePeriod}, with key {complete_key}")

    return (path, timePeriod), (completed, failed)


def simplify(input, timePeriod, mode, utf=False, expand=False):
    # TODO: This is a hack
    hdf_utf = input[{"mode": mode.encode('utf-8'), "timePeriod": timePeriod.encode('utf-8')}]
    hdf = input[{"mode": mode, "timePeriod": timePeriod}]
    originalDictUtf = {sk.name: sk for sk in hdf}
    originalDict = {sk.name: sk for sk in hdf_utf}
    bruteForceDict = {name: input[name] for name in input.list_matrices() if
                      (name.startswith(mode) & name.endswith(timePeriod))}
    if originalDict is None:
        if originalDictUtf is None:
            originalDict = bruteForceDict
        else:
            originalDict = originalDictUtf.update(originalDictUtf)
    else:
        originalDict.update(bruteForceDict)
        if originalDictUtf is not None:
            originalDict.update(originalDictUtf)
    newDict = dict()
    if expand:
        for key, item in originalDict.items():
            if key.startswith('SOV_'):
                newKey = key.replace('SOV_', 'SOVTOLL_')
                newDict[newKey] = item
                newKey = key.replace('SOV_', 'HOV2_')
                newDict[newKey] = item
                newKey = key.replace('SOV_', 'HOV2TOLL_')
                newDict[newKey] = item
                newKey = key.replace('SOV_', 'HOV3_')
                newDict[newKey] = item
                newKey = key.replace('SOV_', 'HOV3TOLL_')
                newDict[newKey] = item
    originalDict.update(newDict)
    return originalDict


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
        # Find all skim variables starting with the source mode
        # Exclude TRIPS and FAILURES as these represent observed demand, not skim values
        # Exclude measures that might have mode-specific meanings or are handled separately
        # e.g., "TOLL" skims should not be copied onto non-toll modes, but the mapping is SOV -> SOVTOLL etc.
        # The original code copied *all* relevant skim keys. Let's refine this.
        # A better approach is to specify which measures should be copied.
        # However, sticking to the original behavior of copying most things except TRIPS/FAILURES
        # and assuming the measures list in settings covers what's needed.
        # Let's copy based on the existence of the target key format in the dataset.

        # Get all variable names in the dataset
        all_vars = list(skims_ds.data_vars)

        for var_name in all_vars:
            # Check if the variable name starts with the source mode path
            # and is not TRIPS, FAILURES, or TNC specific (handled elsewhere)
            if var_name.startswith(fromMode + "_") and not any(m in var_name for m in ["_TRIPS", "_FAILURES"]):
                 # Check if this measure should be copied (avoid copying e.g. SOV_TOLL to HOV)
                 if "TOLL" in var_name.split('_')[0]: # Check if the mode itself contains TOLL
                      continue # Don't use TOLL skims as sources to copy FROM

                 # Extract the measure name (assuming format SOV_MEASURE or SOVTOLL_MEASURE)
                 measure_parts = var_name.split('_', 1)
                 if len(measure_parts) < 2:
                     continue # Skip if not in expected format

                 measure_name = measure_parts[1] # e.g., "TOTIVT"

                 # For each target mode, construct the expected target variable name
                 for toMode in toModes:
                     target_var_name = f"{toMode}_{measure_name}"

                     # Check if the target variable exists in the dataset
                     if target_var_name in all_vars:
                         # Perform the copy using .data
                         # Check shape compatibility - they should be the same (zones, zones, periods)
                         if skims_ds[var_name].shape == skims_ds[target_var_name].shape:
                             skims_ds[target_var_name].data[:] = skims_ds[var_name].data[:]
                             logger.info(f"Copied data from '{var_name}' to '{target_var_name}'")
                         else:
                             logger.warning(f"Shape mismatch when copying from '{var_name}' to '{target_var_name}'. Skipping.")
                     else:
                         logger.debug(f"Target variable '{target_var_name}' not found in dataset. Skipping copy.")

    logger.info("Completed copying skims for unobserved modes.")

def _postprocess_tnc_zarr(skims_ds, timePeriods, settings, completed_failed_dict, use_rh_modes=False):
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
    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}

    if use_rh_modes:
        # Process consolidated RH modes
        modes_to_process = [
            target_mode for source_mode, target_mode in TNC_CONSOLIDATION_MAP.items()
            if f"{target_mode}_TRIPS" in skims_ds.data_vars # Only process if the consolidated variable exists
        ]
        logger.info(f"Post-processing consolidated RH modes: {modes_to_process}")
    else:
        # Process original TNC provider modes
        # Find provider-specific TNC modes like TNC_SINGLE_UBER, TNC_POOLED_LYFT
        all_vars = list(skims_ds.data_vars)
        tnc_provider_modes = set()
        for var_name in all_vars:
            if var_name.startswith("TNC_") and "_TRIPS" in var_name:
                parts = var_name.split('_')
                if len(parts) >= 3 and parts[2] not in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT",
                                                          "DTIM", "DDIST", "TOTIVT", "FAR"]:
                     tnc_provider_modes.add("_".join(parts[:3])) # e.g., TNC_SINGLE_UBER
                elif len(parts) >= 2 and parts[1] in ["SINGLE", "POOLED"]:
                     key_parts = var_name.split('__') # split off period
                     if len(key_parts) > 0:
                         measure_part = key_parts[0] # e.g. TNC_SINGLE_UBER_TRIPS
                         # Split from the right, looking for measure names
                         measure_found = False
                         for common_measure in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT",
                                           "DTIM", "DDIST", "TOTIVT", "FAR"]:
                              if measure_part.endswith("_" + common_measure):
                                   mode_name = measure_part[:-len("_" + common_measure)] # e.g., TNC_SINGLE_UBER
                                   tnc_provider_modes.add(mode_name)
                                   measure_found = True
                                   break
                         if not measure_found:
                             logger.debug(f"Could not parse TNC provider mode from variable: {var_name}")

        modes_to_process = list(tnc_provider_modes)
        logger.info(f"Post-processing original TNC provider modes: {modes_to_process}")

    if not modes_to_process:
        logger.info("No TNC/RH modes found in skims dataset to post-process.")
        return

    # Get SOV skims needed for interpolation ratios
    sov_dist_da = skims_ds.get("SOV_DIST")
    sov_time_da = skims_ds.get("SOV_TIME")
    if (sov_dist_da is None or sov_time_da is None) and any(mode.startswith("RH_") for mode in modes_to_process):
         logger.warning("SOV_DIST or SOV_TIME missing, DDIST/TOTIVT interpolation for RH modes will be skipped.")


    for mode in modes_to_process:
        logger.debug(f"Post-processing mode: {mode}")

        # Get completed and failed trip data for this mode (3D arrays)
        # These should already be in skims_ds if merge/consolidation happened
        completed_key = f"{mode}_TRIPS"
        failed_key = f"{mode}_FAILURES"

        completed_da = skims_ds.get(completed_key)
        failed_da = skims_ds.get(failed_key)

        if completed_da is None or failed_da is None:
            logger.warning(f"Missing {completed_key} or {failed_key} for TNC/RH post-processing. Skipping {mode}.")
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
             np.divide(failed_3d, total_trips_3d,
                       out=rejection_prob_3d,
                       where=valid_ods_mask_3d)

             # Apply origin-level probability where origin total trips > 0 (This matches original OMX logic)
             completed_sum_by_origin_3d = completed_3d.sum(axis=1) # Sum over destination dimension
             failed_sum_by_origin_3d = failed_3d.sum(axis=1) # Sum over destination dimension
             total_trips_origin_3d = completed_sum_by_origin_3d + failed_sum_by_origin_3d
             valid_origins_mask_3d = total_trips_origin_3d > 0 # Mask is (zones, periods)

             # Calculate origin-level probs
             origin_probs_3d = np.zeros_like(completed_sum_by_origin_3d, dtype=np.float32)
             np.divide(failed_sum_by_origin_3d, total_trips_origin_3d,
                       out=origin_probs_3d,
                       where=valid_origins_mask_3d) # origin_probs_3d shape (zones, periods)

             # Apply origin-level probability to rows where origin total > 0
             for tp_idx in range(len(timePeriods)):
                  valid_origins_tp = valid_origins_mask_3d[:, tp_idx] # (zones,)
                  origin_probs_tp = origin_probs_3d[:, tp_idx] # (zones,)
                  rejection_prob_da.data[valid_origins_tp, :, tp_idx] = origin_probs_tp[valid_origins_tp, None] # Apply row-wise

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

            current_iwait_data_3d = np.nan_to_num(iwait_da.data) # Data should be in 0.01 minutes from merge/consolidate
            completed_3d_scaled = completed_3d # Completed is count, no scaling needed for weight

            sum_weighted_wait_3d = np.nansum(current_iwait_data_3d * completed_3d_scaled, axis=1) # Shape (zones, periods)
            sum_completed_origin_3d = np.nansum(completed_3d_scaled, axis=1) # Shape (zones, periods)

            # Handle division by zero - origins with no completed trips will have NaN mean initially
            weighted_mean_by_origin_3d = np.full_like(sum_completed_origin_3d, np.nan, dtype=np.float32)
            valid_origins_for_mean_3d = sum_completed_origin_3d != 0
            np.divide(sum_weighted_wait_3d, sum_completed_origin_3d,
                       out=weighted_mean_by_origin_3d,
                       where=valid_origins_for_mean_3d) # Shape (zones, periods)

            # Identify cells to fill: where completed is 0 AND the origin had some completed trips
            # Apply this filling logic period by period for clarity
            for tp_idx in range(len(timePeriods)):
                 completed_tp = completed_3d[:, :, tp_idx] # (zones, zones)
                 valid_origins_tp = valid_origins_for_mean_3d[:, tp_idx] # (zones,)
                 weighted_mean_tp = weighted_mean_by_origin_3d[:, tp_idx] # (zones,)

                 # Mask for cells where completed == 0 AND the origin has at least one completed trip *in this period*
                 mask_to_fill_with_average = (completed_tp == 0) & (valid_origins_tp[:, None]) # (zones, zones)

                 # Apply the weighted mean by origin
                 if mask_to_fill_with_average.any():
                      # Get the origin index for each cell to fill
                      origin_indices_to_fill = np.where(mask_to_fill_with_average)[0] # Row indices (origins)
                      iwait_da.data[mask_to_fill_with_average, tp_idx] = weighted_mean_tp[origin_indices_to_fill]

                      logger.debug(f"Filled {mask_to_fill_with_average.sum()} {mode} IWAIT values in {timePeriods[tp_idx]} with origin weighted average.")


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
            current_ddist_data_3d = np.nan_to_num(ddist_da.data) # Data should be in miles/feet? Assume consistent units
            sov_dist_data_3d = np.nan_to_num(sov_dist_da.data)

            # Cells where we can calculate the ratio: SOV_DIST > 0, MODE_TRIPS > 0, MODE_DDIST > 0
            # Use 3D arrays for this calculation across all periods
            mask_for_ratio_3d = (sov_dist_data_3d > 0) & (completed_3d > 0) & (current_ddist_data_3d > 0)

            ratio = np.nan
            # Calculate overall weighted average ratio if there are any valid points across all periods
            if mask_for_ratio_3d.any():
                 # Calculate weighted average ratio: (MODE_DDIST * Completed).sum() / (SOV_DIST * Completed).sum()
                 weighted_mode_ddist_3d = current_ddist_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
                 weighted_sov_dist_3d = sov_dist_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
                 sum_weighted_sov = np.sum(weighted_sov_dist_3d)
                 if sum_weighted_sov > 0:
                     ratio = np.sum(weighted_mode_ddist_3d) / sum_weighted_sov

                 ratios_individual = current_ddist_data_3d[mask_for_ratio_3d] / sov_dist_data_3d[mask_for_ratio_3d]
                 logger.info(
                    f"Observed {mode} DDIST/SOV DIST ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {mode}. "
                    f"Interpolating {np.sum(~mask_for_ratio_3d):.0f} missing values."
                 )
            else:
                 logger.info(f"No data points available to calculate {mode} DDIST/SOV DIST ratio for {mode}.")

            # Apply minimum ratio check (from feature branch logic)
            min_ratio = 0.8
            if not np.isnan(ratio) and ratio < min_ratio:
                 logger.warning(f"Calculated {mode} DDIST/SOV DIST ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}.")
                 ratio = min_ratio

            # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
            mask_to_interpolate_3d = ~mask_for_ratio_3d
            if not np.isnan(ratio) and mask_to_interpolate_3d.any():
                 # Interpolate using SOV_DIST where needed
                 # Ensure we don't use SOV_DIST values that are zero for interpolation
                 mask_interpolation_valid_sov_3d = mask_to_interpolate_3d & (sov_dist_data_3d > 0)
                 # Apply interpolated DDIST value (distance measure, no 100x scaling)
                 ddist_da.data[mask_interpolation_valid_sov_3d] = sov_dist_data_3d[mask_interpolation_valid_sov_3d] * ratio  # Assumes SOV_DIST is in consistent units (miles/feet)
                 logger.debug(f"Interpolated {mask_interpolation_valid_sov_3d.sum()} {mode} DDIST values using SOV_DIST ratio.")

            # Handle remaining NaNs if any (e.g., where SOV_DIST was also 0 or ratio wasn't calculable)
            if np.isnan(ddist_da.data).any():
                nan_count = np.isnan(ddist_da.data).sum()
                ddist_da.data[np.isnan(ddist_da.data)] = 0.0 # Default remaining NaNs to 0
                logger.debug(f"Set {nan_count} remaining NaN {mode} DDIST values to 0.")

            logger.debug(f"Updated DDIST for {mode}")
        elif ddist_da is not None:
             logger.warning(f"{mode} DDIST cannot be interpolated: SOV_DIST is missing from skims.")


        # TOTIVT
        if totivt_da is not None and sov_time_da is not None:
            logger.debug(f"Processing TOTIVT for {mode}")
            current_totivt_data_3d = np.nan_to_num(totivt_da.data)
            sov_time_data_3d = np.nan_to_num(sov_time_da.data)

            # Cells where we can calculate the ratio: SOV_TIME > 0, MODE_TRIPS > 0, MODE_TOTIVT > 0
            # Use 3D arrays for this calculation across all periods
            mask_for_ratio_3d = (sov_time_data_3d > 0) & (completed_3d > 0) & (current_totivt_data_3d > 0)

            ratio = np.nan
            # Calculate overall weighted average ratio if there are any valid points across all periods
            if mask_for_ratio_3d.any():
                 # Calculate weighted average ratio: (MODE_TOTIVT * Completed).sum() / (SOV_TIME * Completed).sum()
                 weighted_mode_totivt_3d = current_totivt_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
                 weighted_sov_time_3d = sov_time_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
                 sum_weighted_sov = np.sum(weighted_sov_time_3d)
                 if sum_weighted_sov > 0:
                     ratio = np.sum(weighted_mode_totivt_3d) / sum_weighted_sov

                 ratios_individual = current_totivt_data_3d[mask_for_ratio_3d] / sov_time_data_3d[mask_for_ratio_3d]
                 logger.info(
                    f"Observed {mode} TOTIVT/SOV TIME ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {mode}. "
                    f"Interpolating {np.sum(~mask_for_ratio_3d):.0f} missing values."
                 )
            else:
                 logger.info(f"No data points available to calculate {mode} TOTIVT/SOV TIME ratio for {mode}.")


            # Apply minimum ratio check (from feature branch logic)
            min_ratio = 0.8 # Same minimum as DDIST in feature branch
            if not np.isnan(ratio) and ratio < min_ratio:
                 logger.warning(f"Calculated {mode} TOTIVT/SOV TIME ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}.")
                 ratio = min_ratio

            # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
            mask_to_interpolate_3d = ~mask_for_ratio_3d
            if not np.isnan(ratio) and mask_to_interpolate_3d.any():
                 # Interpolate using SOV_TIME where needed
                 # Ensure we don't use SOV_TIME values that are zero for interpolation
                 mask_interpolation_valid_sov_3d = mask_to_interpolate_3d & (sov_time_data_3d > 0)
                 totivt_da.data[mask_interpolation_valid_sov_3d] = sov_time_data_3d[mask_interpolation_valid_sov_3d] * ratio
                 logger.debug(f"Interpolated {mask_interpolation_valid_sov_3d.sum()} {mode} TOTIVT values using SOV_TIME ratio.")

            # Handle remaining NaNs if any (e.g., where SOV_TIME was also 0 or ratio wasn't calculable)
            if np.isnan(totivt_da.data).any():
                nan_count = np.isnan(totivt_da.data).sum()
                totivt_da.data[np.isnan(totivt_da.data)] = 0.0 # Default remaining NaNs to 0
                logger.debug(f"Set {nan_count} remaining NaN {mode} TOTIVT values to 0.")


            logger.debug(f"Updated TOTIVT for {mode}")
        elif totivt_da is not None:
             logger.warning(f"{mode} TOTIVT cannot be interpolated: SOV_TIME is missing from skims.")

        # --- FAR ---
        if far_da is not None:
            logger.debug(f"Processing FAR for {mode}")
            # Assuming FAR should be set to a default value where it wasn't observed or is zero
            # Original logic seems to set FAR to 0 except where explicitly calculated, or set to a fixed value.
            # The OMX code did not explicitly modify FAR, but the postprocessor had commented out code.
            # Let's default to 0 where TRIPS are zero, or a fixed value if settings provide one.
            default_far = settings.get('tnc_default_far', 0.0) # Add this setting

            # Set FAR to default_far where TRIPS are zero or NaN
            mask_zero_trips = (completed_3d + failed_3d) == 0
            mask_nan_far = np.isnan(far_da.data)
            mask_to_set = mask_zero_trips | mask_nan_far

            if mask_to_set.any():
                 far_da.data[mask_to_set] = default_far
                 logger.debug(f"Set {mask_to_set.sum()} {mode} FAR values to default {default_far}")

            # Ensure FAR is not negative
            negative_far = far_da.data < 0
            if negative_far.any():
                 logger.warning(f"Found {negative_far.sum()} negative FAR values for {mode}. Setting to 0.")
                 far_da.data[negative_far] = 0.0

            logger.debug(f"Updated FAR for {mode}")


    logger.info("Completed TNC/RH-specific post-processing.")


def clear_skim_cache(asim_local_output_dir):
    skims_path = os.path.join(asim_local_output_dir, "cache")
    if os.path.exists(skims_path):
        logger.info("Deleting skims cache at {0}. Eventually we should modify it in place".format(skims_path))
        shutil.rmtree(skims_path)
    else:
        logger.warning("Did not find skim cache to delete")

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
        sample_matrix = next((m for m in partialSkims.list_matrices() if m.endswith("__" + timePeriods[0])), None)
        if sample_matrix:
            out_array_shape_2d = partialSkims[sample_matrix].shape
            out_array_shape_3d = list(out_array_shape_2d) + [len(timePeriods)]
        else:
            # Fallback if no matrices found with period suffix
             logger.warning("No partial skim matrices found with period suffix. Cannot determine shape for trip counts.")
             return {}

    except Exception as e:
         logger.error(f"Error determining partial skim shape: {e}. Cannot accumulate trip counts.")
         return {}


    for omx_key in partialSkims.list_matrices():
        # Split key into measure_part and period
        parts = omx_key.split('__')
        if len(parts) != 2:
             continue # Skip keys not in PATH_MEASURE__PERIOD format

        measure_part = parts[0] # e.g. SOV_TRIPS or TNC_SINGLE_UBER_TRIPS
        period = parts[1] # e.g. AM

        if period not in timePeriods:
             continue # Skip keys for irrelevant periods

        # Check if it's a TRIPS or FAILURES matrix
        is_trips = measure_part.endswith("_TRIPS")
        is_failures = measure_part.endswith("_FAILURES")

        if is_trips or is_failures:
             # Extract the mode path from the measure_part
             # e.g., SOV from SOV_TRIPS, TNC_SINGLE_UBER from TNC_SINGLE_UBER_TRIPS
             if is_trips:
                  mode = measure_part[:-len("_TRIPS")]
             else: # is_failures
                  mode = measure_part[:-len("_FAILURES")]

             if mode not in completed_failed_dict:
                 completed_failed_dict[mode] = [
                     np.zeros(out_array_shape_3d, dtype=np.float32), # Completed trips 3D
                     np.zeros(out_array_shape_3d, dtype=np.float32)  # Failed trips 3D
                 ]

             try:
                  tp_idx = timePeriods.index(period)
                  data_2d = np.nan_to_num(partialSkims[omx_key][:]) # Get 2D numpy array for this period

                  # Ensure shape compatibility
                  if data_2d.shape != out_array_shape_2d:
                       logger.warning(f"Shape mismatch for partial skim {omx_key}. Expected {out_array_shape_2d}, got {data_2d.shape}. Skipping accumulation.")
                       continue

                  if is_trips:
                       completed_failed_dict[mode][0][:, :, tp_idx] = data_2d
                  else: # is_failures
                       completed_failed_dict[mode][1][:, :, tp_idx] = data_2d

             except ValueError:
                  logger.warning(f"Period '{period}' from key '{omx_key}' not found in settings periods. Skipping.")
             except Exception as e:
                  logger.error(f"Error reading partial skim {omx_key} for trip count accumulation: {e}")


    logger.info(f"Accumulated trip counts for {len(completed_failed_dict)} modes/providers.")
    # logger.debug(f"Modes with trip counts: {list(completed_failed_dict.keys())}") # Can be very verbose

    return completed_failed_dict

def _accumulate_completed_failed_trips(partialSkims, timePeriods):
    completed_failed_dict = {}
    out_array_shape = list(partialSkims.shape()) + [len(timePeriods)]

    for tpIdx, tp in enumerate(timePeriods):
        for key in partialSkims.list_matrices():
            if key.endswith(f"_TRIPS__{tp}"):
                mode = key.rsplit('_', 3)[0]  # Extract mode from key (e.g., WLK_TRN_WLK)
                if mode not in completed_failed_dict:
                    completed_failed_dict[mode] = [np.zeros(out_array_shape, dtype=np.float32),
                                                   np.zeros(out_array_shape, dtype=np.float32)]
                completed_failed_dict[mode][0][:, :, tpIdx] = np.nan_to_num(partialSkims[key][:])
            elif key.endswith(f"_FAILURES__{tp}"):
                mode = key.rsplit('_', 3)[0]  # Extract mode from key (e.g., WLK_TRN_WLK)
                if mode not in completed_failed_dict:
                    completed_failed_dict[mode] = [np.zeros(out_array_shape, dtype=np.float32),
                                                   np.zeros(out_array_shape, dtype=np.float32)]
                completed_failed_dict[mode][1][:, :, tpIdx] = np.nan_to_num(partialSkims[key][:])

    return completed_failed_dict


def _transform_measure(input_vals, completed, failed, measure, path, transit_scale_factor=100.0):
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
    completed_mask = (completed > 0)
    basic_mask = valid & completed_mask
    # Handle negative values - seems reasonable to treat as 0
    negative = (input_vals < 0)
    if np.any(negative):
        logger.debug(f"Found {np.sum(negative)} negative values in input_vals for measure {measure} and path {path} in one period. Setting them to 0.")
        input_vals[negative] = 0.0

    measures_in_01_minutes = {"TOTIVT", "IVT", "WACC", "IWAIT", "XWAIT", "WAUX", "WEGR", "DTIM", "FERRYIVT", "KEYIVT"}
    # Apply 100x scaling for specific measures only if the mode is NOT TNC/RH
    scaling = transit_scale_factor if measure in measures_in_01_minutes and not (path.startswith("TNC_") or path.startswith("RH_")) else 1.0


    # Handle measures that need scaling
    if measure in measures_in_01_minutes:
        # TNC IWAIT filling, DDIST, TOTIVT for TNC/RH are handled in post-processing, not here
        # This function is for the initial merge/transfer based on *observed* data.
        # The interpolation/filling logic is separate.
        # So, apply basic scaled assignment where completed > 0 and input is valid.
        return basic_mask, input_vals[basic_mask] * scaling, None

    # Handle travel time measures with penalty logic (only TOTIVT/IVT) for NON-TNC/RH
    elif measure in ["TOTIVT", "IVT"]:
        # The penalty logic applies only to non-TNC/RH modes (like transit)

        if path.startswith("TNC_") or path.startswith("RH_"):
             # TNC/RH TOTIVT/IVT is handled by the interpolation logic in post-processing

             is_tnc_rh = path.startswith("TNC_") or path.startswith("RH_")

             if is_tnc_rh:
                  return basic_mask, input_vals[basic_mask], None
             else:
                 # Apply penalty logic for non-TNC/RH modes (like transit)
                 to_cancel = (failed > 5) & (failed > (1 * completed))

                 # To penalize: failed > completed, NOT canceled, valid input, and some completed trips
                 to_penalize = (failed > completed) & ~to_cancel & valid & completed_mask

                 # To allow: NOT canceled, NOT penalized, valid input, and some completed trips
                 to_allow = valid & completed_mask & ~to_cancel & ~to_penalize

                 # Prepare result values for cells to update
                 result_vals = np.zeros_like(input_vals[basic_mask]) # Only allocate for cells covered by basic_mask

                 # Apply penalty where needed (within the basic_mask)
                 mask_penalize_subset = to_penalize[basic_mask] # Subset of basic_mask that is also to_penalize
                 if mask_penalize_subset.any():
                     penalty_factor = (failed[basic_mask][mask_penalize_subset] + 1) / (completed[basic_mask][mask_penalize_subset] + 1)
                     result_vals[mask_penalize_subset] = input_vals[basic_mask][mask_penalize_subset] * penalty_factor * scaling

                 # Use regular values where allowed (within the basic_mask)
                 mask_allow_subset = to_allow[basic_mask] # Subset of basic_mask that is also to_allow
                 if mask_allow_subset.any():
                      result_vals[mask_allow_subset] = input_vals[basic_mask][mask_allow_subset] * scaling

                 # The update mask is the basic_mask itself, but the values are calculated based on penalty/allowance
                 return basic_mask, result_vals, to_cancel


    # Handle DIST (simple assignment where completed > 0)
    elif measure == "DIST":
        # TNC DDIST interpolation is handled in post-processing
        if path.startswith("TNC_") or path.startswith("RH_"):
             # For TNC/RH, DIST is handled by interpolation
             return np.zeros_like(input_vals, dtype=bool), np.array([]), None
        else:
             # For others, just apply direct value where completed > 0
             return basic_mask, input_vals[basic_mask], None


    # Handle non-scaled measures (simple assignment where completed > 0)
    # Includes COST, TOLLs, FAR, REJECTIONPROB etc.
    # TNC FAR and REJECTIONPROB are handled in post-processing
    elif measure not in ["REJECTIONPROB", "FAR"]: # Exclude measures handled in post-processing
         if path.startswith("TNC_") or path.startswith("RH_"):
             # For TNC/RH, these are either interpolated (DDIST/TOTIVT handled above)
             # or just assigned directly if observed (COST, non-interpolated DDIST/TOTIVT).
             # Let's include COST here for TNC/RH - it's a direct transfer where observed.
             if measure == "COST":
                  # For TNC/RH COST, simple assignment where completed > 0
                  return basic_mask, input_vals[basic_mask], None
             else:
                  # Other non-excluded measures for TNC/RH (e.g. TOLLs) - simple assignment where completed > 0
                  return basic_mask, input_vals[basic_mask], None

         else:
             # For non-TNC/RH, simple assignment where completed > 0
             return basic_mask, input_vals[basic_mask], None


    # Default case (e.g., REJECTIONPROB, FAR - handled in post-processing)
    # Should not be reached for measures intended to be processed here.
    logger.warning(f"Measure {measure} for path {path} fell through _transform_measure logic.")
    return np.zeros_like(input_vals, dtype=bool), np.array([]), None

def _merge_one_zarr_measure(partialSkims, skims_ds, path, measure, timePeriods, completed_3d, failed_3d):
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
    target_var_name = f"{path}_{measure}" # e.g., SOV_TOTIVT

    # Ensure the target variable exists in skims_ds
    # Use SOV_TRIPS shape/coords as a default if target var doesn't exist
    if target_var_name not in skims_ds.data_vars:
         logger.warning(f"Target Zarr variable '{target_var_name}' not found. Creating with SOV_TRIPS shape.")
         try:
             zarr_shape = skims_ds['SOV_TRIPS'].shape
             zarr_coords = skims_ds['SOV_TRIPS'].coords
             zarr_dims = skims_ds['SOV_TRIPS'].dims
             skims_ds[target_var_name] = xr.DataArray(
                 np.zeros(zarr_shape, dtype=np.float32),
                 coords=zarr_coords,
                 dims=zarr_dims,
                 name=target_var_name
             )
         except KeyError:
              logger.error(f"Cannot create variable {target_var_name}: SOV_TRIPS not found to get shape/coords.")
              return None, None # Indicate skipping
    target_da = skims_ds[target_var_name]


    # Iterate through each time period slice
    for tpIdx, tp in enumerate(timePeriods):
        # Construct the key for the partial skims in OMX format
        partial_key = f"{path}_{measure}__{tp}"

        if partial_key in partialSkims:
            try:
                input_vals_tp = partialSkims[partial_key][:] # Get 2D numpy array for this period

                # Ensure target DA slice exists for this period
                if target_da.ndim < 3 or target_da.shape[-1] <= tpIdx:
                    logger.warning(f"Target Zarr variable {target_var_name} does not have enough time period dimensions for period {tp} (index {tpIdx}). Skipping merge for this period.")
                    continue # Skip this period

                # Ensure shape compatibility
                if input_vals_tp.shape != target_da.shape[:2]:
                     logger.warning(f"Shape mismatch for partial skim {partial_key}. Expected {target_da.shape[:2]}, got {input_vals_tp.shape}. Skipping merge for this period.")
                     continue

                # Get the 2D completed/failed slices for the current time period
                completed_tp = completed_3d[:, :, tpIdx]
                failed_tp = failed_3d[:, :, tpIdx]

                # Apply transformations using the helper function
                # _transform_measure expects 2D completed/failed arrays
                mask_update, vals_update, to_cancel_tp = _transform_measure(
                    input_vals_tp, completed_tp, failed_tp, measure, path # Pass path here
                )

                # Apply updates to the Zarr DataArray slice using the mask
                current_slice = target_da.data[:, :, tpIdx]
                if mask_update.any():
                    current_slice[mask_update] = vals_update
                    # Optional: Log weighted average change for this slice/period
                    # weights = completed_tp[mask_update]
                    # if weights.sum() > 0:
                    #      before_vals = current_slice[mask_update]
                    #      after_vals = vals_update
                    #      weighted_avg_before = np.average(before_vals, weights=weights)
                    #      weighted_avg_after = np.average(after_vals, weights=weights)
                    #      logger.debug(f"  {tp}: Wtd Avg Change for {target_var_name}: {weighted_avg_before:.2f} -> {weighted_avg_after:.2f}")


                # Handle cancellations for IVT/TOTIVT
                if to_cancel_tp is not None and to_cancel_tp.any():
                     cancellation_count = to_cancel_tp.sum()
                     current_slice[to_cancel_tp] = 0.0 # Set canceled values to 0
                     logger.debug(f"  Canceled {cancellation_count} ODs for {partial_key}")

                # Update the slice in the DataArray's data
                target_da.data[:, :, tpIdx] = current_slice

            except Exception as e:
                logger.error(f"Error processing skim slice {partial_key}: {e}")
                # Continue to the next time period

        else:
            logger.debug(f"Partial skims missing key {partial_key}. Skipping merge for this period.")

    # Note: SOV/HOV copying logic removed from here - handled by copy_skims_for_unobserved_modes

    return path, measure


def _merge_zarr_trip_counts(allSkims, path, completed, failed):
    key_completed = f"{path}_TRIPS"
    key_failed = f"{path}_FAILURES"
    # Ensure keys exist before attempting to access .data
    if key_completed in allSkims and key_failed in allSkims:
        prev_completed = np.nan_to_num(allSkims[key_completed].data)
        prev_failed = np.nan_to_num(allSkims[key_failed].data)
        logger.info(f"For {path} previously had {prev_completed.sum():.0f} completed trips and {prev_failed.sum():.0f} failed trips")
        prev_completed += completed
        prev_failed += failed
        allSkims[key_completed].data[:] = prev_completed # Use [:] to modify data in place
        allSkims[key_failed].data[:] = prev_failed # Use [:] to modify data in place
        logger.info(f"Now we have {prev_completed.sum():.0f} completed trips and {prev_failed.sum():.0f} failed trips")
    else:
        logger.warning(f"Skipping trip counts merge for {path} as {key_completed} or {key_failed} does not exist in target skims file")



def _merge_zarr_skim(partialSkims, skims_da, completed_failed_dict, timePeriods):
    """
    Merges a single measure's data for all time periods from OMX partial skims
    into a Zarr DataArray, applying measure-specific transformations.

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM.
    skims_da : xarray.DataArray
        The target Zarr DataArray for the current measure (e.g., skims['SOV_TOTIVT']).
        Expected dimensions: (origin, destination, time_period).
    completed_failed_dict : dict
        Dictionary containing completed and failed trip counts aggregated across
        all time periods for each mode path. Keyed by mode path, values are
        [completed_trips_3d_array, failed_trips_3d_array].
    timePeriods : list of str
        List of time period names.

    Returns:
    --------
    tuple (path, measure_name)
        The path and measure name of the skim that was processed.
    """
    # Extract path and measure name from the DataArray name
    # Expected name format: "PATH_MEASURE" (e.g., "SOV_TOTIVT")
    path_measure_name = skims_da.name
    parts = path_measure_name.rsplit('_', 1)
    if len(parts) != 2:
        logger.warning(f"Skipping skim '{path_measure_name}' due to unexpected name format.")
        return None, None # Indicate skipping

    path, measure_name = path_measure_name.rsplit('_', 1) # Re-parse to handle names like WLK_TRN_WLK_TOTIVT

    # Skip TNC-specific measures that are handled in post-processing
    if path.startswith("TNC") and measure_name in ["REJECTIONPROB", "IWAIT", "DDIST", "TOTIVT"]:
        # These will be calculated in the post-processing step
        return path, measure_name # Indicate processed, but no data merged here

    # Get the completed and failed trips for the current mode path
    # These are 3D arrays [zones, zones, time periods]
    completed_3d, failed_3d = completed_failed_dict.get(path, (None, None))

    if completed_3d is None or failed_3d is None:
        logger.debug(f"No completed/failed trip data found for mode path {path}. Skipping merge for {path_measure_name}.")
        return path, measure_name # Indicate processed, but no data merged

    weighted_avgs = {}
    cancellation_counts = {}

    # Iterate through each time period slice in the DataArray
    for tpIdx, tp in enumerate(timePeriods):
        # Construct the key for the partial skims in OMX format
        partial_key = f"{path}_{measure_name}__{tp}"

        # Get the 2D completed/failed slices for the current time period
        completed_tp = completed_3d[:, :, tpIdx]
        failed_tp = failed_3d[:, :, tpIdx]

        if partial_key in partialSkims:
            try:
                input_vals_tp = partialSkims[partial_key][:] # Get 2D numpy array for this period

                # Apply transformations using the helper function
                # _transform_measure expects 2D completed/failed arrays
                mask_update, vals_update, to_cancel_tp = _transform_measure(
                    input_vals_tp, completed_tp, failed_tp, measure_name, path
                )

                # Apply updates to the Zarr DataArray slice using the mask
                if mask_update.any():
                    # Use .data for direct, in-place modification
                    current_slice = skims_da.data[:, :, tpIdx]

                    # Calculate weighted averages for logging *before* update
                    # Use completed trips for the weight
                    weights = completed_tp[mask_update]
                    before_vals = current_slice[mask_update]
                    after_vals = vals_update

                    if np.sum(weights) > 0:
                         weighted_avg_before = np.average(before_vals, weights=weights)
                         weighted_avg_after = np.average(after_vals, weights=weights)
                    else:
                         weighted_avg_before = np.nan
                         weighted_avg_after = np.nan
                    weighted_avgs[tp] = (weighted_avg_before, weighted_avg_after)

                    # Apply the updated values
                    current_slice[mask_update] = after_vals
                    skims_da.data[:, :, tpIdx] = current_slice # Ensure changes are reflected (might be redundant with [:] but safer)


                # Handle cancellations for IVT/TOTIVT
                if to_cancel_tp is not None and to_cancel_tp.any():
                     cancellation_counts[tp] = to_cancel_tp.sum()
                     current_slice = skims_da.data[:, :, tpIdx]
                     current_slice[to_cancel_tp] = 0.0 # Set canceled values to 0
                     skims_da.data[:, :, tpIdx] = current_slice # Ensure changes reflected

            except Exception as e:
                logger.error(f"Error processing skim slice {partial_key}: {e}")
                # Continue to the next time period

        else:
            logger.debug(f"Partial skims missing key {partial_key}. Skipping merge for this period.")

    # Log weighted averages summary if any updates occurred
    if weighted_avgs:
        summary = "; ".join(
            f"{tp}: before={before:.2f}, after={after:.2f}"
            for tp, (before, after) in weighted_avgs.items()
            if not np.isnan(before) or not np.isnan(after)
        )
        if summary:
             logger.info(f"Weighted average update for {path_measure_name} (by completed trips): {summary}")
        else:
             logger.debug(f"No weighted average updates for {path_measure_name} (no completed trips in updated cells).")

    # Log cancellation summary if any occurred
    if cancellation_counts:
        summary = "; ".join(
            f"{tp}: {count} ODs" for tp, count in cancellation_counts.items() if count > 0
        )
        if summary:
            logger.info(f"Canceled ODs for {path_measure_name}: {summary}")

    # Note: SOV/HOV copying logic removed from here.
    # Note: TNC specific logic removed from here.

    return path, measure_name


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
    logger.info(f"Transit Mode Availability Analysis")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Period':<6} | {'Mode':<12} | {'ODs w/50+ Transit':<18} | {'ODs w/0 Mode Trips':<18} | {'Changed':<10} | {'% Changed':<10}")
    logger.info(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 10}")

    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}

    # Process each time period
    for tp in timePeriods:
        tp_idx = tp_to_idx[tp]

        # Check if general transit trip counts exist
        general_trips_key = f"{general_transit_path}_TRIPS" # Zarr key format
        general_trips_da = skims_ds.get(general_trips_key)

        if general_trips_da is None:
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {'NO DATA':<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}")
            continue

        # Find OD pairs with at least 50 successful transit trips for this period slice
        general_transit_trips = general_trips_da.data[:, :, tp_idx] # Get 2D slice
        mask_significant_transit = general_transit_trips >= 50

        significant_count = np.sum(mask_significant_transit)

        if not np.any(mask_significant_transit):
            # No significant transit trips for this period, nothing to check
            logger.info(f"{tp:<6} | {'ALL MODES':<12} | {0:<18} | {0:<18} | {0:<10} | {0:<10.1f}")
            continue

        # Check each specific transit mode
        for mode in specific_transit_modes:
            # Get trip counts for this mode for this period slice
            mode_trips_key = f"{mode}_TRIPS" # Zarr key format
            mode_trips_da = skims_ds.get(mode_trips_key)

            if mode_trips_da is None:
                logger.info(
                    f"{tp:<6} | {mode:<12} | {significant_count:<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}")
                continue

            mode_trips = mode_trips_da.data[:, :, tp_idx] # Get 2D slice

            # Find OD pairs where general transit has 50+ trips but this mode has 0
            mask_mode_unused = mask_significant_transit & (mode_trips == 0)

            unused_count = np.sum(mask_mode_unused)

            # Mark this mode as unavailable for these OD pairs by setting TOTIVT = 0
            totivt_key = f"{mode}_TOTIVT" # Zarr key format
            totivt_da = skims_ds.get(totivt_key)
            changed_count = 0

            if totivt_da is not None and unused_count > 0:
                # Get current TOTIVT values slice
                current_totivt_slice = totivt_da.data[:, :, tp_idx]

                # Only mark as unavailable if current TOTIVT is > 0 or np.nan within the mask_mode_unused area
                mask_to_change = mask_mode_unused & ((current_totivt_slice > 0) | np.isnan(current_totivt_slice))
                changed_count = np.sum(mask_to_change)

                # Update values directly using .data slice
                if changed_count > 0:
                    current_totivt_slice[mask_to_change] = 0.0
                    totivt_da.data[:, :, tp_idx] = current_totivt_slice # Ensure changes are reflected

            percent_changed = (changed_count / significant_count) * 100 if significant_count > 0 else 0

            logger.info(
                f"{tp:<6} | {mode:<12} | {significant_count:<18} | {unused_count:<18} | {changed_count:<10} | {percent_changed:<10.1f}")

    logger.info(f"{'=' * 80}")


def _consolidate_tnc_data_zarr(partialSkims, skims_ds, timePeriods, completed_failed_dict_providers, settings):
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
    dict
        A dictionary containing consolidated completed/failed trip counts (3D arrays)
        for the RH_SOLO and RH_SHARED modes.
    """
    logger.info("Starting TNC fleet consolidation from OMX to Zarr...")

    all_partial_vars = partialSkims.list_matrices()
    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}
    # Get Zarr shape/coords from an existing 3D variable
    try:
        zarr_shape = skims_ds['SOV_TRIPS'].shape
        zarr_coords = skims_ds['SOV_TRIPS'].coords
        zarr_dims = skims_ds['SOV_TRIPS'].dims
        if zarr_shape[-1] != len(timePeriods):
             logger.warning(f"Zarr time period dimension ({zarr_shape[-1]}) does not match settings periods ({len(timePeriods)}). Using Zarr shape.")
             # Update tp_to_idx to match Zarr if necessary, or raise error. Let's assume settings is correct.
             # For safety, ensure the Zarr has enough space for all periods in settings.
             if zarr_shape[-1] < len(timePeriods):
                  logger.error(f"Zarr shape {zarr_shape} has fewer time periods than in settings ({len(timePeriods)}). Cannot consolidate.")
                  return {} # Indicate failure or no consolidation
             # If Zarr has more periods, we only populate those listed in settings.
    except KeyError:
        logger.error("Zarr dataset does not contain 'SOV_TRIPS'. Cannot determine shape and coordinates for new variables.")
        return {} # Indicate failure or no consolidation
    except Exception as e:
         logger.error(f"Error getting Zarr shape/coords: {e}. Cannot consolidate.")
         return {}


    # Group TNC provider variables from partialSkims by (base_mode, measure)
    grouped_sources = {} # Key: (base_mode, measure), Value: list of (provider, period, omx_key)

    for omx_key in all_partial_vars:
         # Split key into measure_part and period (e.g., TNC_SINGLE_UBER_IWAIT__AM -> TNC_SINGLE_UBER_IWAIT, AM)
         parts = omx_key.split('__')
         if len(parts) != 2:
              continue # Skip keys not in PATH_MEASURE__PERIOD format

         measure_part = parts[0] # e.g. TNC_SINGLE_UBER_IWAIT
         period = parts[1] # e.g. AM

         if not measure_part.startswith("TNC_") or period not in timePeriods:
              continue # Only process TNC keys for relevant periods

         # Parse measure_part: TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
         measure_part_parts = measure_part.split('_')
         if len(measure_part_parts) < 3:
              logger.debug(f"Skipping TNC key with unexpected format (too few parts): {omx_key}")
              continue

         tnc_prefix = measure_part_parts[0] # Should be "TNC"
         base_mode = measure_part_parts[1] # e.g. "SINGLE", "POOLED"
         # Check if base_mode maps to a consolidated RH mode
         if f"{tnc_prefix}_{base_mode}" not in TNC_CONSOLIDATION_MAP:
              logger.debug(f"Skipping TNC key with unknown base mode pattern: {omx_key}")
              continue

         # Identify provider and measure. The first part after TNC_BASEMODE that is NOT a known measure prefix
         # is likely part of the provider name, UNLESS the measure name itself has underscores.
         # Let's try to find the measure name by splitting from the right, checking against common measures.
         measure = None
         provider_parts = []
         remaining_parts = list(measure_part_parts[2:]) # Parts after TNC_BASEMODE

         # Reverse the remaining parts and check if the rightmost form a known measure
         reversed_remaining = remaining_parts[::-1]
         for i in range(len(reversed_remaining)):
              potential_measure_parts = reversed_remaining[:i+1][::-1] # e.g., ['IWAIT'] or ['KEY', 'IVT']
              potential_measure = "_".join(potential_measure_parts)

              # Check against common measures or specific ones with underscores
              if potential_measure in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT", "FAR", "DDIST", "TOTIVT"]:
                   measure = potential_measure
                   provider_parts = remaining_parts[:len(remaining_parts) - (i+1)] # Remaining parts are the provider
                   break # Found the measure

         if measure is None:
              logger.warning(f"Could not identify measure name in TNC key: {omx_key}. Skipping.")
              continue

         provider = "_".join(provider_parts) if provider_parts else "DEFAULT" # Use "DEFAULT" if no provider part
         base_mode_key = f"{tnc_prefix}_{base_mode}" # e.g. "TNC_SINGLE"

         if (base_mode_key, measure) not in grouped_sources:
              grouped_sources[(base_mode_key, measure)] = []

         grouped_sources[(base_mode_key, measure)].append((provider, period, omx_key))

    logger.debug(f"Grouped TNC sources from OMX: {grouped_sources}")

    consolidated_completed_failed_dict = {} # Store consolidated counts for RH_ modes

    # First, consolidate TRIPS and FAILURES to get total counts per RH_ mode
    for base_mode_key, measure in [(k[0], k[1]) for k in grouped_sources.keys()]:
        if measure not in ["TRIPS", "FAILURES"]:
            continue # Only process trips/failures first

        target_mode = TNC_CONSOLIDATION_MAP[base_mode_key] # e.g. "RH_SOLO"
        target_var_name = f"{target_mode}_{measure}"

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
             logger.debug(f"Creating target variable '{target_var_name}' in Zarr dataset.")
             skims_ds[target_var_name] = xr.DataArray(
                 np.zeros(zarr_shape, dtype=np.float32),
                 coords=zarr_coords,
                 dims=zarr_dims,
                 name=target_var_name
             )
        target_da = skims_ds[target_var_name]

        # Initialize accumulation array for this measure
        consolidated_data_3d = np.zeros(zarr_shape, dtype=np.float32)

        # Accumulate data from all relevant provider sources
        for provider, period, omx_key in grouped_sources.get((base_mode_key, measure), []):
             if omx_key in partialSkims:
                 try:
                     tp_idx = tp_to_idx[period]
                     input_vals_tp = np.nan_to_num(partialSkims[omx_key][:]) # Get 2D numpy array for this period
                     # Ensure shape compatibility before summing
                     if input_vals_tp.shape == zarr_shape[:2]: # Compare (zones, zones)
                          consolidated_data_3d[:, :, tp_idx] += input_vals_tp
                     else:
                          logger.warning(f"Shape mismatch for partial skim {omx_key}. Expected {zarr_shape[:2]}, got {input_vals_tp.shape}. Skipping sum.")
                 except Exception as e:
                     logger.error(f"Error reading partial skim {omx_key} for consolidation: {e}")
             else:
                  logger.debug(f"Partial skim {omx_key} not found. Skipping.")

        # Assign the summed data to the target Zarr variable
        target_da.data[:] = consolidated_data_3d
        logger.debug(f"Summed {measure} into '{target_var_name}'. Total: {consolidated_data_3d.sum():.0f}")

        # Store the consolidated counts
        if target_mode not in consolidated_completed_failed_dict:
             # Initialize with zeros of the correct shape
             consolidated_completed_failed_dict[target_mode] = [
                 np.zeros(zarr_shape, dtype=np.float32),
                 np.zeros(zarr_shape, dtype=np.float32)
             ]

        if measure == "TRIPS":
             consolidated_completed_failed_dict[target_mode][0][:] = consolidated_data_3d
        elif measure == "FAILURES":
             consolidated_completed_failed_dict[target_mode][1][:] = consolidated_data_3d

    # Now, consolidate other measures using weighted average
    for (base_mode_key, measure), sources in grouped_sources.items():
        if measure in ["TRIPS", "FAILURES", "REJECTIONPROB"]:
            continue # Already handled TRIPS/FAILURES, REJECTIONPROB calculated later

        target_mode = TNC_CONSOLIDATION_MAP[base_mode_key] # e.g. "RH_SOLO"
        target_var_name = f"{target_mode}_{measure}"

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
             logger.debug(f"Creating target variable '{target_var_name}' in Zarr dataset.")
             skims_ds[target_var_name] = xr.DataArray(
                 np.zeros(zarr_shape, dtype=np.float32), # Initialize with zeros
                 coords=zarr_coords,
                 dims=zarr_dims,
                 name=target_var_name
             )
        target_da = skims_ds[target_var_name]

        # Initialize accumulation arrays (weighted sum and sum of weights) for this measure
        sum_weighted_values_3d = np.zeros(zarr_shape, dtype=np.float32)
        sum_weights_3d = np.zeros(zarr_shape, dtype=np.float32) # Sum of completed trips

        # Accumulate data from all relevant provider sources
        for provider, period, omx_key in sources:
             if omx_key in partialSkims:
                 try:
                      tp_idx = tp_to_idx[period]
                      input_vals_tp = np.nan_to_num(partialSkims[omx_key][:]) # Get 2D data

                      # Get corresponding TRIPS data for this provider and period from the provider dict
                      provider_mode_key = f"TNC_{base_mode}_{provider}" # e.g. TNC_SINGLE_UBER
                      if provider_mode_key in completed_failed_dict_providers:
                           provider_completed_3d, _ = completed_failed_dict_providers[provider_mode_key]
                           completed_tp = np.nan_to_num(provider_completed_3d[:, :, tp_idx]) # Get 2D completed trips for this provider/period

                           # Ensure shapes match
                           if input_vals_tp.shape == zarr_shape[:2] and completed_tp.shape == zarr_shape[:2]:
                                # Handle negative values in input_vals_tp before weighting
                                input_vals_tp[input_vals_tp < 0] = 0.0

                                # Weighted value = measure_value * completed_trips
                                weighted_values_tp = input_vals_tp * completed_tp
                                sum_weighted_values_3d[:, :, tp_idx] += weighted_values_tp
                                sum_weights_3d[:, :, tp_idx] += completed_tp # Sum of completed trips are the weights
                           else:
                                logger.warning(f"Shape mismatch for partial skim {omx_key} or its trips data. Skipping weighted average contribution.")
                      else:
                           logger.debug(f"Completed trip data for provider mode {provider_mode_key} not found in provider dict. Skipping weighted average contribution for {omx_key}.")
                 except Exception as e:
                      logger.error(f"Error processing partial skim {omx_key} for weighted consolidation: {e}")
             else:
                  logger.debug(f"Partial skim {omx_key} not found. Skipping.")

        # Calculate the weighted average for the entire 3D array
        consolidated_data_3d = np.zeros(zarr_shape, dtype=np.float32) # Initialize with zeros
        valid_weights_mask_3d = sum_weights_3d != 0
        # Use np.divide with 'where' and 'out' for safe division
        np.divide(sum_weighted_values_3d, sum_weights_3d,
                  out=consolidated_data_3d, # Write results into the consolidated data array
                  where=valid_weights_mask_3d)

        # Assign the calculated data to the target Zarr variable
        target_da.data[:] = consolidated_data_3d
        mean_weighted_avg = np.nanmean(consolidated_data_3d[valid_weights_mask_3d]) if valid_weights_mask_3d.any() else np.nan
        logger.debug(f"Calculated weighted average for {measure} into '{target_var_name}'. Mean (weighted): {mean_weighted_avg:.4f}")

    logger.info("Completed TNC fleet consolidation from OMX to Zarr.")

    # Return consolidated counts for use in post-processing
    return consolidated_completed_failed_dict


# New function to transfer TNC provider data from OMX to Zarr without consolidation
def _transfer_tnc_provider_data_zarr(partialSkims, skims_ds, timePeriods, completed_failed_dict_providers, settings):
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

    for omx_key in all_partial_vars:
         # Split key into measure_part and period
         parts = omx_key.split('__')
         if len(parts) != 2:
              continue # Skip keys not in PATH_MEASURE__PERIOD format

         measure_part = parts[0] # e.g. TNC_SINGLE_UBER_IWAIT
         period = parts[1] # e.g. AM

         if not measure_part.startswith("TNC_") or period not in timePeriods:
              continue # Only process TNC keys for relevant periods

         # Parse measure_part to get the full provider mode name and the measure
         measure = None
         mode = None

         measure_part_parts = measure_part.split('_')
         if len(measure_part_parts) < 3:
             logger.debug(f"Skipping TNC key with unexpected format (too few parts): {omx_key}")
             continue

         # Assume format TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
         # Let's try to find the measure name by splitting from the right
         remaining_parts = list(measure_part_parts[2:])
         reversed_remaining = remaining_parts[::-1]
         for i in range(len(reversed_remaining)):
             potential_measure_parts = reversed_remaining[:i+1][::-1]
             potential_measure = "_".join(potential_measure_parts)

             if potential_measure in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT", "XWAIT", "WACC",
                                      "WAUX", "WEGR", "DTIM", "DDIST", "TOTIVT", "FAR", "COST"] or \
                potential_measure in TNC_MEASURES_WITH_UNDERSCORES:
                  measure = potential_measure
                  mode_parts = measure_part_parts[:len(measure_part_parts) - (i+1)] # Parts before the measure
                  mode = "_".join(mode_parts) # e.g., TNC_SINGLE_UBER
                  break

         if mode is None or measure is None:
             logger.warning(f"Could not parse TNC provider mode or measure from key: {omx_key}. Skipping.")
             continue

         if (mode, measure) not in grouped_sources:
              grouped_sources[(mode, measure)] = []

         grouped_sources[(mode, measure)].append((period, omx_key))


    # Get Zarr shape/coords from an existing 3D variable
    try:
        zarr_shape = skims_ds['SOV_TRIPS'].shape
        zarr_coords = skims_ds['SOV_TRIPS'].coords
        zarr_dims = skims_ds['SOV_TRIPS'].dims
        if zarr_shape[-1] < len(timePeriods):
             logger.error(f"Zarr shape {zarr_shape} has fewer time periods than in settings ({len(timePeriods)}). Cannot transfer TNC data.")
             return [] # Indicate failure
    except KeyError:
        logger.error("Zarr dataset does not contain 'SOV_TRIPS'. Cannot determine shape and coordinates for new TNC provider variables.")
        return [] # Indicate failure
    except Exception as e:
         logger.error(f"Error getting Zarr shape/coords: {e}. Cannot transfer TNC data.")
         return []


    # Transfer data for each mode/measure pair
    for (mode, measure), sources in grouped_sources.items():
        target_var_name = f"{mode}_{measure}" # e.g. TNC_SINGLE_UBER_IWAIT

        # Ensure target variable exists in Zarr
        if target_var_name not in skims_ds.data_vars:
             logger.debug(f"Creating target variable '{target_var_name}' in Zarr dataset.")
             # Check if the mode exists in the completed_failed_dict_providers
             # This is needed to get the shape. Fallback to SOV_TRIPS shape.
             try:
                  # Use the shape from the provider counts if available
                  provider_completed_shape = completed_failed_dict_providers[mode][0].shape
                  if provider_completed_shape[:2] != zarr_shape[:2] or provider_completed_shape[-1] < len(timePeriods):
                      logger.warning(f"Provider completed counts shape {provider_completed_shape} for mode {mode} is incompatible with Zarr shape {zarr_shape} or settings periods {len(timePeriods)}. Falling back to SOV_TRIPS shape.")
                      raise KeyError # Fallback
                  current_zarr_shape = provider_completed_shape
             except KeyError:
                  # Fallback to default Zarr shape if provider shape isn't reliable
                  current_zarr_shape = zarr_shape
                  logger.debug(f"Using default Zarr shape {current_zarr_shape} for mode {mode}")

             skims_ds[target_var_name] = xr.DataArray(
                 np.zeros(current_zarr_shape, dtype=np.float32), # Initialize with zeros
                 coords=zarr_coords, # Use SOV_TRIPS coords as a default
                 dims=zarr_dims, # Use SOV_TRIPS dims as a default
                 name=target_var_name
             )
        target_da = skims_ds[target_var_name]
        transferred_vars.add(target_var_name)

        # Get completed/failed counts for this mode from the provider dict
        completed_3d, failed_3d = completed_failed_dict_providers.get(mode, (None, None))
        if completed_3d is None or failed_3d is None:
             logger.warning(f"Completed/failed trip data for mode {mode} not found in provider dict. Skipping merge for {target_var_name}.")
             continue # Cannot merge without counts

        # Merge data from all relevant periods
        for period, omx_key in sources:
             if omx_key in partialSkims:
                 try:
                     tp_idx = tp_to_idx[period]
                     # Ensure the target DA has enough dimensions for this period
                     if target_da.ndim < 3 or target_da.shape[-1] <= tp_idx:
                         logger.warning(f"Target Zarr variable {target_var_name} does not have enough time period dimensions for period {period} (index {tp_idx}). Skipping merge for this period.")
                         continue # Skip this period if target DA is too small

                     input_vals_tp = partialSkims[omx_key][:] # Get 2D numpy array for this period

                     # Ensure shape compatibility before merging
                     if input_vals_tp.shape != target_da.shape[:2]:
                         logger.warning(f"Shape mismatch for partial skim {omx_key}. Expected {target_da.shape[:2]}, got {input_vals_tp.shape}. Skipping merge for this period.")
                         continue

                     # Apply transformations using the helper function _transform_measure
                     # _transform_measure expects 2D completed/failed arrays
                     completed_tp = completed_3d[:, :, tp_idx]
                     failed_tp = failed_3d[:, :, tp_idx]

                     mask_update, vals_update, to_cancel_tp = _transform_measure(
                         input_vals_tp, completed_tp, failed_tp, measure, mode # Pass mode name here
                     )

                     # Apply updates to the Zarr DataArray slice using the mask
                     current_slice = target_da.data[:, :, tp_idx]
                     if mask_update.any():
                         current_slice[mask_update] = vals_update

                     # Handle cancellations for IVT/TOTIVT
                     if to_cancel_tp is not None and to_cancel_tp.any():
                          cancellation_count = to_cancel_tp.sum()
                          current_slice[to_cancel_tp] = 0.0 # Set canceled values to 0
                          logger.debug(f"Canceled {cancellation_count} ODs for {omx_key}")

                     # Update the slice in the DataArray's data
                     target_da.data[:, :, tp_idx] = current_slice


                 except Exception as e:
                     logger.error(f"Error reading partial skim {omx_key} for transfer: {e}")
             else:
                  logger.debug(f"Partial skim {omx_key} not found. Skipping.")

    logger.info("Completed TNC provider data transfer from OMX to Zarr.")
    return list(transferred_vars)


def write_zarr_skim_as_omx(all_skims_path, settings, new_skim_name, exclude_tables=None):
    """
    Write the skims from the Zarr format to an OMX format.

    Parameters
    ----------
    all_skims_path : str
        Path to the main skims file
    settings : dict
        Settings dictionary
    new_skim_name : str
        Name of the new skim to be created

    Returns
    -------
    None
    """
    region = settings['region']
    beam_input_dir = settings['beam_local_input_folder']
    skims_fname = settings['skims_fname']
    if exclude_tables is None:
        exclude_tables = []

    target_skims_path = os.path.join(beam_input_dir, region, new_skim_name)
    skims = xr.open_zarr(all_skims_path)
    logger.info(f"Deleting current skims file {target_skims_path} and replacing them with new omx skims")
    if os.path.exists(target_skims_path):
        os.remove(target_skims_path)
    new_omx_file = omx.open_file(target_skims_path, 'a')
    time_periods = [s for s in skims.time_period.values]
    for key in skims.keys():
        # Get the data for this key
        if key in exclude_tables:
            continue
        data = skims[key].values
        logger.info(f"Writing {key} with shape {data.shape} to {target_skims_path}")
        if len(data.shape) == 2:
            new_omx_file[key] = data
        elif len(data.shape) == 3:
            for t_idx, tp in enumerate(time_periods):
                new_key = f"{key}__{tp}"
                new_omx_file[new_key] = data[:, :, t_idx]
    logger.info(f"Done writing skims to {target_skims_path} with shape {new_omx_file.shape()}")
    new_omx_file.close()
    skims.close()


def merge_current_zarr_od_skims(all_skims_path, beam_output_dir, settings, override=None):
    """
    Merges current BEAM OMX skims into the main Zarr skims file.
    Handles TNC consolidation if enabled.
    """
    logger.info(f"Starting merge of current BEAM OMX skims into Zarr at {all_skims_path}")

    if override is None:
        current_omx_skims_path = find_produced_od_skims(beam_output_dir, "omx")
    else:
        current_omx_skims_path = override

    if current_omx_skims_path is None or not os.path.exists(current_omx_skims_path):
        logger.warning(f"No current OMX skims found at {current_omx_skims_path}. Skipping merge.")
        # Check if the Zarr file exists; if not, ActivitySim won't run, so this is a failure state.
        # Assume Zarr was created by asim_pre.create_zarr_skims
        if not os.path.exists(all_skims_path):
            logger.error(f"Target Zarr skims file not found at {all_skims_path} and no new OMX skims available. Cannot proceed.")
            return None
        try:
            skims_ds = xr.open_zarr(all_skims_path) # Open existing for possible post-processing if it's a replan iteration
            skims_ds.close() # Just check if it's readable
        except Exception as e:
            logger.error(f"Failed to open existing Zarr skims file at {all_skims_path}: {e}. Cannot proceed.")
            return None

        logger.info("Existing Zarr skims file found, but no new OMX skims to merge. Proceeding with post-processing on existing Zarr if needed.")
        # Need to decide if post-processing should run even without new skims. Yes, probably.
        # We still need to run TNC postprocessing, transit availability, etc.
        # The rest of this function handles that.

        # The completed_failed_dict and grouped_partial_sources will be empty/None,
        # so the merging loops will be skipped, but post-processing steps will run
        # on the existing data in skims_ds.
        partialSkims = None # Ensure partialSkims is None if file not found

    else:
        try:
            partialSkims = omx.open_file(current_omx_skims_path, mode='r')
            logger.info(f"Opened partial skims file: {current_omx_skims_path}")
        except Exception as e:
            logger.error(f"Failed to open partial skims file {current_omx_skims_path}: {e}. Skipping merge.")
            partialSkims = None # Ensure partialSkims is None if opening fails
            if not os.path.exists(all_skims_path):
                 logger.error(f"Target Zarr skims file not found at {all_skims_path} and failed to open new OMX skims. Cannot proceed.")
                 return None # Indicate failure

    # Open the Zarr dataset in append mode to modify it in place
    try:
        skims_ds = xr.open_zarr(all_skims_path)
        logger.info(f"Opened Zarr skims file: {all_skims_path}")
    except Exception as e:
        logger.error(f"Failed to open target Zarr skims file {all_skims_path}: {e}. Cannot proceed.")
        if partialSkims:
             partialSkims.close()
        return None # Indicate failure


    timePeriods = settings["periods"]
    consolidate_tnc_fleets = settings.get('consolidate_tnc_fleets', True)

    # Step 1: Accumulate completed and failed trips from partial skims (all modes/providers)
    completed_failed_dict_all = {}
    if partialSkims:
        completed_failed_dict_all = _accumulate_all_completed_failed_trips(partialSkims, timePeriods)


    # Step 2: Process TNC/RH skims (either consolidate or transfer) and run their specific post-processing
    # This step handles creating/updating TNC_* or RH_* variables in skims_ds and running their post-processing.
    # It needs the provider-level completed/failed counts if not consolidating.
    tnc_modes_processed = set() # Keep track of modes handled by the TNC/RH logic
    if partialSkims:
         # If consolidating, _process_all_tnc_logic will use provider counts internally
         # and run postprocessing on consolidated RH modes.
         # If not consolidating, it will transfer provider data and run postprocessing
         # on TNC_* provider modes.
         tnc_modes_processed = _process_all_tnc_logic(partialSkims, skims_ds, timePeriods, settings, completed_failed_dict_all)
    else:
         # If no new partial skims, still need to run TNC post-processing on existing Zarr data
         # This requires completed/failed counts to be available in the Zarr.
         # This is complex - ideally, trip counts are always merged first.
         # Let's assume trip counts for TNC/RH are already in skims_ds if partialSkims is None
         # (e.g., from a previous iteration). We need to reconstruct the completed_failed_dict
         # from the Zarr data itself in this case.
         logger.warning("No new OMX skims found. Reconstructing TNC/RH trip counts from existing Zarr data for post-processing.")
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
                       mode = var_name.rsplit('_TRIPS', 1)[0] # Extract mode name
                       tnc_modes_for_postprocess.add(mode)

         # Reconstruct completed_failed_dict for these modes from Zarr
         completed_failed_dict_from_zarr = {}
         for mode in tnc_modes_for_postprocess:
              trips_key = f"{mode}_TRIPS"
              failures_key = f"{mode}_FAILURES"
              if trips_key in skims_ds.data_vars and failures_key in skims_ds.data_vars:
                   completed_failed_dict_from_zarr[mode] = (
                       np.nan_to_num(skims_ds[trips_key].data),
                       np.nan_to_num(skims_ds[failures_key].data)
                   )
         if tnc_modes_for_postprocess:
             # Run post-processing on existing TNC/RH data in Zarr
             _postprocess_tnc_zarr(skims_ds, timePeriods, settings, completed_failed_dict_from_zarr, use_rh_modes=consolidate_tnc_fleets)
         else:
             logger.info("No existing TNC/RH modes found in Zarr for post-processing.")


    # Step 3: Process non-TNC/non-RH skims and run standard merging
    if partialSkims: # Only merge if we have new partial skims
        # Group partial skims by (path, measure)
        grouped_partial_sources = {} # Key: (path, measure), Value: list of omx_key

        for omx_key in partialSkims.list_matrices():
             # Split key into measure_part and period
             parts = omx_key.split('__')
             if len(parts) != 2:
                  continue # Skip keys not in PATH_MEASURE__PERIOD format

             measure_part = parts[0]
             period = parts[1]

             if period not in timePeriods:
                  continue

             # Determine path and measure
             # Try parsing from right for standard measures
             measure = None
             path = None
             measure_part_parts = measure_part.split('_')
             reversed_parts = measure_part_parts[::-1]

             for i in range(len(reversed_parts)):
                  potential_measure_parts = reversed_parts[:i+1][::-1]
                  potential_measure = "_".join(potential_measure_parts)
                  potential_path_parts = measure_part_parts[:len(measure_part_parts) - (i+1)]
                  potential_path = "_".join(potential_path_parts)

                  # Check if the potential measure is a common one
                  if potential_measure in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT", "XWAIT", "WACC",
                                           "WAUX", "WEGR", "DTIM", "DDIST", "TOTIVT", "FAR", "COST", "DIST", "TIME",
                                           "TOLL", "KTOLL", "HTOLL", "LTTOLL", "FTTOLL", "VTOLL", "KEYIVT", "FERRYIVT", # Added common measures
                                           ]:
                       measure = potential_measure
                       path = potential_path
                       break

             if path is None or measure is None:
                  logger.debug(f"Could not parse path/measure from key {omx_key}. Skipping.")
                  continue

             # Skip TNC/RH paths if they were handled by consolidation/transfer
             if path.startswith("TNC_") or path.startswith("RH_"):
                 # If consolidation was on, TNC_* and RH_* were handled by _process_all_tnc_logic
                 # If consolidation was off, TNC_* were handled by _process_all_tnc_logic
                 # In either case, skip them here.
                 # Unless it's a consolidated RH_ mode created *in this step* and needs standard merge?
                 # No, the RH_ modes created by consolidate should *not* go through the generic merge.
                 # The post-processing function handles RH_ specific calculations *after* consolidation.
                 if consolidate_tnc_fleets:
                      # If consolidating, skip original TNC provider paths and the new RH paths
                      # Need to check if 'path' is an original provider path or a consolidated RH path
                      is_provider_path = path.startswith("TNC_") # e.g. TNC_SINGLE_UBER
                      is_consolidated_path = path in TNC_CONSOLIDATION_MAP.values() # e.g. RH_SOLO
                      if is_provider_path or is_consolidated_path:
                           logger.debug(f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC/RH and handled elsewhere (consolidation enabled).")
                           continue
                 else:
                      # If not consolidating, skip original TNC provider paths (handled by transfer)
                      if path.startswith("TNC_"):
                           logger.debug(f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC and handled elsewhere (consolidation disabled).")
                           continue

             # Add to the grouping for non-TNC/non-RH paths
             if (path, measure) not in grouped_partial_sources:
                  grouped_partial_sources[(path, measure)] = []
             grouped_partial_sources[(path, measure)].append(omx_key)


        logger.info(f"Merging {len(grouped_partial_sources)} non-TNC/non-RH (path, measure) groups from partial skims.")

        # Process each non-TNC/non-RH (path, measure) group
        processed_non_tnc_vars = set()
        for (path, measure), omx_keys in grouped_partial_sources.items():
             # Ensure we have completed/failed counts for this path
             if path not in completed_failed_dict_all:
                  logger.warning(f"Completed/failed trip data for path {path} not found. Skipping merge for measure {measure}.")
                  continue

             completed_3d, failed_3d = completed_failed_dict_all[path]

             # Call the generic merge function for this measure/path
             processed_path, processed_measure = _merge_one_zarr_measure(
                 partialSkims, skims_ds, path, measure, timePeriods, completed_3d, failed_3d
             )
             if processed_path is not None:
                  processed_non_tnc_vars.add(f"{processed_path}_{processed_measure}")

        logger.info(f"Completed merging {len(processed_non_tnc_vars)} non-TNC/non-RH variables.")

    # Step 4: Handle transit mode availability (uses TRIPS/FAILURES data already in skims_ds)
    # This should run regardless of whether new OMX skims were merged, to ensure consistency.
    logger.info("Applying transit mode availability rules.")
    _handle_transit_mode_availability(skims_ds, timePeriods)


    # Step 5: Copy skims for unobserved modes (e.g., SOV -> HOV/SOVTOLL)
    # This should run after the base skims (like SOV) are finalized.
    # This should run regardless of whether new OMX skims were merged.
    logger.info("Copying skims for unobserved modes based on mapping.")
    mapping = settings.get('unobserved_skim_copy_map', {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]}) # Add setting
    copy_skims_for_unobserved_modes(mapping, skims_ds)

    # Step 6: Save the updated Zarr dataset
    logger.info(f"Started writing updated zarr skims to {all_skims_path}")
    # Use mode='w' to overwrite with the updated data, consolidated=True for performance
    # Ensure zarr_version=2 for compatibility
    try:
        skims_ds.to_zarr(all_skims_path, mode='w', consolidated=True, zarr_version=2)
        logger.info("Completed writing zarr skims successfully.")
        merge_successful = True # Indicate merge/post-processing was attempted successfully
    except Exception as e:
         logger.error(f"FAILED to write updated zarr skims to {all_skims_path}: {e}")
         merge_successful = False # Indicate failure

    # Close the datasets
    skims_ds.close()
    if partialSkims:
        partialSkims.close()

    # Return the path to the new OMX skims *if* a new one was found and processed.
    # This is used by the caller to check if skims were updated.
    if partialSkims is not None and merge_successful:
         return current_omx_skims_path
    else:  # partialSkims is None or writing failed
         logger.warning("No new OMX skims found, returning None.")
         return None # No new skims to indicate success/failure for



# New function to orchestrate TNC/RH processing
def _process_all_tnc_logic(partialSkims, skims_ds, timePeriods, settings, completed_failed_dict_all):
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
    consolidate_tnc_fleets = settings.get('consolidate_tnc_fleets', True)
    processed_vars = set()
    tnc_modes_to_postprocess = set()
    completed_failed_dict_for_postprocess = {} # Counts for the modes that will be post-processed

    if consolidate_tnc_fleets:
        logger.info("TNC fleet consolidation enabled.")
        # Step 2a: Consolidate TNC provider data into RH_* variables in skims_ds
        # This function also returns the consolidated RH counts
        consolidated_completed_failed_dict = _consolidate_tnc_data_zarr(
            partialSkims, skims_ds, timePeriods, completed_failed_dict_all, settings
        )
        # Add consolidated RH modes to the list for post-processing
        tnc_modes_to_postprocess.update(consolidated_completed_failed_dict.keys())
        # Use the consolidated counts for post-processing
        completed_failed_dict_for_postprocess = consolidated_completed_failed_dict

        # The consolidated variables (RH_*) are now in skims_ds.
        # We need to get their names to return them as handled.
        for target_mode in consolidated_completed_failed_dict.keys():
             for measure in skims_ds.data_vars: # Iterate through ALL skims_ds vars to find the ones belonging to this mode
                  if measure.startswith(f"{target_mode}_"):
                       processed_vars.add(measure)


    else:
        logger.info("TNC fleet consolidation disabled. Transferring provider data.")
        # Step 2a: Transfer TNC provider data into TNC_* variables in skims_ds
        # This function returns the names of variables transferred.
        transferred_vars = _transfer_tnc_provider_data_zarr(
            partialSkims, skims_ds, timePeriods, completed_failed_dict_all, settings
        )
        processed_vars.update(transferred_vars)

        # Identify the TNC provider modes that were transferred
        # Iterate through the transferred variable names to find the modes
        for var_name in transferred_vars:
             parts = var_name.split('_')
             if len(parts) >= 3 and parts[0] == "TNC":
                  # Assuming format TNC_BASEMODE_PROVIDER_MEASURE or TNC_BASEMODE_MEASURE
                  # Need to parse the mode name correctly again
                  measure_part = var_name.rsplit('_', 1)[0] # Remove measure at the end for 3D vars like RH_SOLO_TRIPS
                  if measure_part.endswith("__"): # Handle 2D vars like DIST__AM
                     measure_part = var_name.split('__')[0]
                  elif "_" in var_name.split("__")[0]: # Handle 3D vars
                     measure_part = var_name.split("__")[0]


                  # Find the mode name from the beginning of the measure_part
                  mode_name = None
                  measure_name_check = measure_part
                  while '_' in measure_name_check:
                       potential_mode = measure_name_check
                       potential_measure_end = measure_name_check.rsplit('_', 1)[-1]
                       if potential_measure_end.upper() in ["TRIPS", "FAILURES", "REJECTIONPROB", "IWAIT", "DDIST", "TOTIVT", "FAR"]:
                             # Found a potential measure end, the rest is the mode
                             mode_name = potential_mode.rsplit('_', 1)[0]
                             break
                       measure_name_check = potential_mode.rsplit('_', 1)[0] # Chop off the last part and try again

                  if mode_name and mode_name.startswith("TNC_"):
                       tnc_modes_to_postprocess.add(mode_name)
                       # Use the provider counts for post-processing
                       if mode_name in completed_failed_dict_all:
                            completed_failed_dict_for_postprocess[mode_name] = completed_failed_dict_all[mode_name]
                       else:
                            logger.warning(f"Completed/failed counts not found for provider mode {mode_name} in completed_failed_dict_all. Cannot post-process.")


        logger.info(f"Identified {len(tnc_modes_to_postprocess)} TNC provider modes for post-processing.")


    # Step 2b: Run TNC/RH specific post-processing on the data now in skims_ds
    # This function needs the completed/failed counts for the modes it's processing.
    if tnc_modes_to_postprocess:
         _postprocess_tnc_zarr(
             skims_ds, timePeriods, settings, completed_failed_dict_for_postprocess,
             use_rh_modes=consolidate_tnc_fleets
         )
    else:
         logger.info("No TNC/RH modes to post-process after consolidation/transfer step.")

    return processed_vars

def merge_current_omx_od_skims(all_skims_path, beam_output_dir, settings):
    """
    Merge current OMX skims from BEAM into the main skims file.

    Parameters
    ----------
    all_skims_path : str
        Path to the main skims file
    beam_output_dir : str
        Path to the BEAM output directory
    settings : dict
        Settings dictionary

    Returns
    -------
    str
        Path to the current skims file
    """
    skims = omx.open_file(all_skims_path, 'a')
    current_skims_path = find_produced_od_skims(beam_output_dir, "omx")
    partialSkims = omx.open_file(current_skims_path, mode='r')
    iterable = [(
        path, timePeriod, vals[1].to_list()) for (path, timePeriod), vals
        in
        pd.Series(partialSkims.listMatrices()).str.rsplit('_', n=3, expand=True).groupby([0, 3])]

    # GIVING UP ON PARALLELIZING THIS FOR NOW. see below for attempts that didn't work for some reason or another

    # results = Parallel(n_jobs=-1)(delayed(_merge_skim)(x) for x in iterable)
    # p = mp.Pool(4)
    # result = [p.apply_async(_merge_skim, args=((simplify(partialSkims, timePeriod, path, True),
    #                                             simplify(skims, timePeriod, path, False), path,
    #                                             timePeriod, vals))) for (path, timePeriod, vals) in iterable]
    result = [_merge_skim(simplify(partialSkims, timePeriod, path, True),
                          simplify(skims, timePeriod, path, False), path,
                          timePeriod, vals) for (path, timePeriod, vals) in iterable]

    discover_impossible_ods(result, skims, skims.list_matrices())
    mapping = {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]}
    copy_skims_for_unobserved_modes(mapping, skims, skims.list_matrices())

    order = zone_order(settings, settings['start_year'])
    zone_id = np.arange(1, len(order) + 1)

    # Generint offset
    skims.create_mapping('zone_id', zone_id, overwrite=True)

    skims.close()
    partialSkims.close()
    return current_skims_path

def trim_inaccessible_ods_zarr(all_skims_path, settings):
    """
    Zero out inaccessible ODs in Zarr-format skims, similar to trim_inaccessible_ods for OMX.
    Uses direct .data access for in-place modification.
    """
    logger.info("Starting trim of inaccessible ODs in Zarr skims.")
    try:
        skims = xr.open_zarr(all_skims_path) # Open in append/write mode
    except Exception as e:
        logger.error(f"Failed to open skims file {all_skims_path} for trimming inaccessible ODs: {e}")
        return

    try:
        order = zone_order(settings, settings.get('start_year', 2015)) # Use .get for start_year safety
        periods = settings["periods"]
        transit_paths_settings = settings.get('transit_paths', {}) # Use .get for safety
    except Exception as e:
        logger.error(f"Error reading settings for trimming: {e}. Skipping trim.")
        skims.close()
        return

    if not transit_paths_settings:
         logger.warning("No 'transit_paths' defined in settings for trimming inaccessible ODs. Skipping trim.")
         skims.close()
         return

    tp_to_idx = {p: i for i, p in enumerate(periods)}
    num_zones = len(order)

    # Calculate total trips per OD pair across all non-RH transit/walk/bike modes for each period
    # Initialize totalTrips as a 3D array [zones, zones, periods]
    totalTrips = np.zeros((num_zones, num_zones, len(periods)), dtype=np.float32)

    all_vars = list(skims.data_vars)

    for var_name in all_vars:
         parts = var_name.split('__')
         if len(parts) == 2: # Variable name is in PATH_MEASURE__PERIOD format (old OMX style, sometimes kept in Zarr)
              measure_part, tp = parts
              if tp in periods:
                   # Check if it's a TRIPS matrix for a non-RH mode
                   if 'TRIPS' in measure_part and not measure_part.startswith('RH_'):
                        try:
                             tp_idx = periods.index(tp)
                             if skims[var_name].ndim == 2 and skims[var_name].shape == (num_zones, num_zones):
                                  totalTrips[:, :, tp_idx] += np.nan_to_num(skims[var_name].data)
                             else:
                                  logger.debug(f"Skipping '{var_name}' for total trips calculation due to unexpected shape {skims[var_name].shape}.")
                        except ValueError:
                             logger.warning(f"Period '{tp}' from key '{var_name}' not found in settings periods. Skipping.")
                        except Exception as e:
                            logger.error(f"Error reading '{var_name}' for total trips calculation: {e}")

         elif skims[var_name].ndim == 3: # Variable name is in PATH_MEASURE format (new Zarr style)
             # Check if it's a TRIPS matrix for a non-RH mode
             if var_name.endswith("_TRIPS") and not var_name.startswith("RH_"):
                  try:
                       # Assuming 3D shape is [zones, zones, periods]
                       if skims[var_name].shape[:2] == (num_zones, num_zones) and skims[var_name].shape[-1] >= len(periods):
                           # Sum across all periods we care about (as defined by settings)
                           for tp_idx in range(len(periods)):
                                totalTrips[:, :, tp_idx] += np.nan_to_num(skims[var_name].data[:, :, tp_idx])
                       else:
                            logger.debug(f"Skipping '{var_name}' for total trips calculation due to unexpected shape {skims[var_name].shape}.")
                  except Exception as e:
                        logger.error(f"Error reading '{var_name}' for total trips calculation: {e}")

    # Calculate total trips BY ZONE (sum across rows and columns for each zone) for each period
    completedAllTripsByOandD_3d = totalTrips.sum(axis=1) + totalTrips.sum(axis=0) # Shape (zones, periods)


    # Now, check transit modes for trimming
    for path, metrics in transit_paths_settings.items():
        trip_name_zarr = f"{path}_TRIPS" # Zarr key format
        fail_name_zarr = f"{path}_FAILURES" # Zarr key format

        trip_da = skims.get(trip_name_zarr)
        fail_da = skims.get(fail_name_zarr)

        if trip_da is None or fail_da is None:
             logger.debug(f"Skipping trim for {path}: missing {trip_name_zarr} or {fail_name_zarr} in skims_ds.")
             continue

        # Get 3D completed/failed transit trips for this mode
        try:
            completedTransitTrips_3d = np.nan_to_num(trip_da.data[:, :, :len(periods)]) # Slice to match settings periods
            failedTransitTrips_3d = np.nan_to_num(fail_da.data[:, :, :len(periods)]) # Slice to match settings periods
            if completedTransitTrips_3d.shape[:2] != (num_zones, num_zones) or completedTransitTrips_3d.shape[-1] != len(periods):
                 logger.warning(f"Shape mismatch for {trip_name_zarr} or {fail_name_zarr} ({completedTransitTrips_3d.shape}) vs expected ({num_zones}, {num_zones}, {len(periods)}). Skipping trim for {path}.")
                 continue
        except Exception as e:
             logger.error(f"Error reading trip data for {path} from Zarr: {e}. Skipping trim for this path.")
             continue


        # Calculate completed and failed transit trips BY ZONE for each period
        completedTransitTripsByOandD_3d = completedTransitTrips_3d.sum(axis=1) + completedTransitTrips_3d.sum(axis=0) # Shape (zones, periods)
        failedTransitTripsByOandD_3d = failedTransitTrips_3d.sum(axis=1) + failedTransitTrips_3d.sum(axis=0) # Shape (zones, periods)


        # Determine which zones to delete for each period
        # Condition: (Total trips > 1000) & (Failed transit trips > 200) & (Completed transit trips == 0)
        toDelete_3d = (completedAllTripsByOandD_3d > 1000) & (failedTransitTripsByOandD_3d > 200) & (completedTransitTripsByOandD_3d == 0) # Shape (zones, periods)


        # Apply trimming for each period where zones need deletion
        for tpIdx, period in enumerate(periods):
            zones_to_delete_tp = toDelete_3d[:, tpIdx] # Boolean array (zones,) for this period
            num_zones_to_delete_tp = np.sum(zones_to_delete_tp)

            if num_zones_to_delete_tp > 0:
                logger.info(f"Trimming {path} service for {num_zones_to_delete_tp} zones in {period} because no completed transit trips were observed for these zones (conditions met).")

                # Apply deletion mask to metrics for this path and period
                for metric in metrics:
                    name_zarr = f"{path}_{metric}" # Zarr key format
                    metric_da = skims.get(name_zarr)

                    if metric_da is not None:
                        try:
                             # Ensure the DataArray has enough dimensions and size for the period
                             if metric_da.ndim == 3 and metric_da.shape[-1] > tpIdx and metric_da.shape[:2] == (num_zones, num_zones):
                                  # Get the 2D slice for this period
                                  arr_slice = metric_da.data[:, :, tpIdx]
                                  # Apply the deletion mask. This sets values to 0 where the origin OR destination zone is in `zones_to_delete_tp`.
                                  arr_slice[zones_to_delete_tp[:, None] | zones_to_delete_tp[None, :]] = 0.0
                                  # Update the data in the Zarr array
                                  metric_da.data[:, :, tpIdx] = arr_slice
                                  logger.debug(f"Trimmed {name_zarr} for {period} for {num_zones_to_delete_tp} zones.")
                             elif metric_da.ndim == 2 and len(periods) == 1 and periods[0] == period and metric_da.shape == (num_zones, num_zones):
                                  # Handle 2D case (single period)
                                  arr_slice = metric_da.data[:, :]
                                  arr_slice[zones_to_delete_tp[:, None] | zones_to_delete_tp[None, :]] = 0.0
                                  metric_da.data[:, :] = arr_slice
                                  logger.debug(f"Trimmed {name_zarr} (2D) for {period} for {num_zones_to_delete_tp} zones.")
                             else:
                                  logger.warning(f"Skipping trim for metric {name_zarr} in {period} due to unexpected shape {metric_da.shape} or period mismatch.")
                        except Exception as e:
                             logger.error(f"Error applying trim mask for metric {name_zarr} in {period}: {e}. Skipping.")
                    else:
                        logger.debug(f"Skipping trim for metric {name_zarr} in {period}: variable not found in skims_ds.")

    # Save the updated Zarr dataset after trimming
    logger.info(f"Started writing zarr skims after trimming inaccessible ODs to {all_skims_path}")
    try:
        skims.to_zarr(all_skims_path, mode='w', consolidated=True, zarr_version=2) # Use mode='w' to overwrite
        logger.info("Completed writing zarr skims after trimming successfully.")
    except Exception as e:
         logger.error(f"FAILED to write zarr skims after trimming inaccessible ODs to {all_skims_path}: {e}")

    skims.close()


def trim_inaccessible_ods(settings, working_dir):
    all_skims_path = os.path.join(working_dir, settings['asim_local_mutable_data_folder'], "skims.omx")
    order = zone_order(settings, settings['start_year'])
    skims = omx.open_file(str(all_skims_path), "a")
    all_mats = skims.list_matrices()
    totalTrips = dict()
    for period in settings["periods"]:
        totalTrips[period] = np.zeros((len(order), len(order)))
    for mat in all_mats:
        if ('TRIPS__' in mat) & ('RH_' not in mat):
            tp = mat[-2:]
            totalTrips[tp] += np.array(skims[mat])
    for period in settings["periods"]:
        completedAllTripsByOandD = totalTrips[period].sum(axis=0) + totalTrips[period].sum(axis=1)
        for path, metrics in settings['transit_paths'].items():
            trip_name = "{0}_TRIPS__{1}".format(path, period)
            fail_name = "{0}_FAILURES__{1}".format(path, period)
            if trip_name in all_mats:
                completedTransitTrips = np.array(skims[trip_name])
                failedTransitTrips = np.array(skims[fail_name])
                completedTransitTripsByOandD = completedTransitTrips.sum(axis=0) + completedTransitTrips.sum(axis=1)
                failedTransitTripsByOandD = failedTransitTrips.sum(axis=0) + failedTransitTrips.sum(axis=1)
                toDelete = np.squeeze((completedAllTripsByOandD > 1000) & (failedTransitTripsByOandD > 200) & (
                        completedTransitTripsByOandD == 0))
                logger.info("Deleting all {0} service for {1} zones in {2} "
                            "because no trips were observed".format(path, np.sum(toDelete), period))
                for metric in metrics:
                    name = "{0}_{1}__{2}".format(path, metric, period)
                    if name in all_mats:
                        skims[name][toDelete[:, None] | toDelete[None, :]] = 0.0
    skims.close()


def discover_impossible_ods(result, skims, mats):
    # return (path, timePeriod), (completed, failed)
    allMats = mats
    metricsPerPath = dict()
    for (path, tp), _ in result:
        if path not in metricsPerPath.keys():
            try:
                metricsPerPath[path] = set([mat.split('_')[3] for mat in allMats if mat.startswith(path)])
            except:
                continue
    timePeriods = np.unique([b for (a, b), _ in result])
    # WALK TRANSIT:
    for tp in timePeriods:
        completed = {(a, b): c for (a, b), (c, d) in result if
                     a.startswith('WLK') & a.endswith('WLK') & (b == tp) & ('TRN' not in a)}
        failed = {(a, b): d for (a, b), (c, d) in result if
                  a.startswith('WLK') & a.endswith('WLK') & (b == tp) & ('TRN' not in a)}
        totalCompleted = np.nansum(list(completed.values()), axis=0)
        totalFailed = np.nansum(list(failed.values()), axis=0)
        for (path, _), mat in completed.items():
            for metric in metricsPerPath[path]:
                name = '_'.join([path, metric, '', tp])
                if (name in allMats) & (metric not in ["TRIPS", "FAILURES"]):
                    toDelete = (mat == 0) & (totalCompleted > 50) & (totalFailed > 50) & (skims[name][:] > 0)
                    if np.any(toDelete):
                        print(
                            "Deleting {0} ODs for {1} in the {2} because after 50 transit trips "
                            "there no one has chosen it".format(
                                toDelete.sum(), name, tp))
                        skims[name][toDelete] = 0


def merge_current_od_skims(all_skims_path, previous_skims_path, beam_output_dir):
    current_skims_path = find_produced_od_skims(beam_output_dir)
    if (current_skims_path is None) | (previous_skims_path == current_skims_path):
        # this means beam has not produced the skims
        logger.error("No skims found in directory {0}, defaulting to {1}".format(beam_output_dir, current_skims_path))
        return previous_skims_path

    schema = {
        "origin": str,
        "destination": str,
        "DEBUG_TEXT": str,
    }
    index_columns = ['timePeriod', 'pathType', 'origin', 'destination']

    all_skims = pd.read_csv(all_skims_path, dtype=schema, index_col=index_columns, na_values=["∞"])
    cur_skims = pd.read_csv(current_skims_path, dtype=schema, index_col=index_columns, na_values=["∞"])
    for col in cur_skims.columns:  # Handle new skim columns
        if col not in all_skims.columns:
            all_skims[col] = 0.0
    all_skims = pd.concat([cur_skims, all_skims.loc[all_skims.index.difference(cur_skims.index, sort=False)]])
    all_skims = all_skims.reset_index()
    all_skims.to_csv(all_skims_path, index=False)
    return current_skims_path


def hourToTimeBin(hour: int):
    if hour < 3:
        return 'EV'
    elif hour < 6:
        return 'EA'
    elif hour < 10:
        return 'AM'
    elif hour < 15:
        return 'MD'
    elif hour < 19:
        return 'PM'
    else:
        return 'EV'


def aggregateInTimePeriod(df):
    if df['completedRequests'].sum() > 0:
        totalCompletedRequests = df['completedRequests'].sum()
        waitTime = (df['waitTime'] * df['completedRequests']).sum() / totalCompletedRequests / 60.
        costPerMile = (df['costPerMile'] * df['completedRequests']).sum() / totalCompletedRequests
        observations = df['observations'].sum()
        unmatchedRequestPortion = 1. - totalCompletedRequests / observations
        return pd.Series({"waitTimeInMinutes": waitTime, "costPerMile": costPerMile,
                          "unmatchedRequestPortion": unmatchedRequestPortion, "observations": observations,
                          "completedRequests": totalCompletedRequests})
    else:
        observations = df['observations'].sum()
        return pd.Series({"waitTimeInMinutes": 6.0, "costPerMile": 5.0,
                          "unmatchedRequestPortion": 1.0, "observations": observations,
                          "completedRequests": 0})


# noinspection PyUnresolvedReferences
def merge_current_omx_origin_skims(all_skims_path, previous_skims_path, beam_output_dir, measure_map):
    current_skims_path = find_produced_origin_skims(beam_output_dir)

    rawInputSchema = {
        "tazId": str,
        "hour": int,
        "reservationType": str,
        "waitTime": float,
        "costPerMile": float,
        "unmatchedRequestsPercent": float,
        "observations": int,
        "iterations": int
    }

    cur_skims = pd.read_csv(current_skims_path, dtype=rawInputSchema, na_values=["∞"])
    cur_skims['timePeriod'] = cur_skims['hour'].apply(hourToTimeBin)
    cur_skims.rename(columns={'tazId': 'origin'}, inplace=True)
    cur_skims['completedRequests'] = cur_skims['observations'] * (1. - cur_skims['unmatchedRequestsPercent'] / 100.0)
    cur_skims = cur_skims.groupby(['timePeriod', 'reservationType', 'origin']).apply(aggregateInTimePeriod)
    cur_skims['failures'] = cur_skims['observations'] - cur_skims['completedRequests']
    skims = omx.open_file(all_skims_path, 'a')
    idx = pd.Index(np.array(list(skims.mapping("zone_id").keys()), dtype=str).copy())
    for (timePeriod, reservationType), _df in cur_skims.groupby(level=[0, 1]):
        df = _df.loc[(timePeriod, reservationType)].reindex(idx, fill_value=0.0)

        trips = "RH_{0}_{1}__{2}".format(reservationType.upper(), 'TRIPS', timePeriod.upper())
        failures = "RH_{0}_{1}__{2}".format(reservationType.upper(), 'FAILURES', timePeriod.upper())
        rejectionprob = "RH_{0}_{1}__{2}".format(reservationType.upper(), 'REJECTIONPROB', timePeriod.upper())

        logger.info(
            "Adding {0} complete trips and {1} failed trips to skim {2}".format(int(df['completedRequests'].sum()),
                                                                                int(df['failures'].sum()),
                                                                                trips))

        skims[trips][:] = skims[trips][:] * 0.5 + df.loc[:, 'completedRequests'].values[:, None]
        skims[failures][:] = skims[trips][:] * 0.5 + df.loc[:, 'failures'].values[:, None]
        skims[rejectionprob][:] = np.nan_to_num(skims[failures][:] / (skims[trips][:] + skims[failures][:]))

        wait = "RH_{0}_{1}__{2}".format(reservationType.upper(), 'WAIT', timePeriod.upper())
        originalWaitTime = np.array(
            skims[wait])  # Hack due to pytables issue https://github.com/PyTables/PyTables/issues/310
        originalWaitTime[df['completedRequests'] > 0, :] = df.loc[
                                                               df['completedRequests'] > 0, measure_map['WAIT']].values[
                                                           :,
                                                           None]
        skims[wait][:] = originalWaitTime
    skims.close()


def merge_current_origin_skims(all_skims_path, previous_skims_path, beam_output_dir):
    current_skims_path = find_produced_origin_skims(beam_output_dir)
    if (current_skims_path is None) | (previous_skims_path == current_skims_path):
        # this means beam has not produced the skims
        logger.error("no skims produced from path {0}".format(current_skims_path))
        return previous_skims_path

    rawInputSchema = {
        "tazId": str,
        "hour": int,
        "reservationType": str,
        "waitTime": float,
        "costPerMile": float,
        "unmatchedRequestsPercent": float,
        "observations": int,
        "iterations": int
    }

    aggregatedInput = {
        "origin": str,
        "timePeriod": str,
        "reservationType": str,
        "waitTimeInMinutes": float,
        "costPerMile": float,
        "unmatchedRequestPortion": float,
        "observations": int
    }

    index_columns = ['timePeriod', 'reservationType', 'origin']

    all_skims = pd.read_csv(all_skims_path, dtype=aggregatedInput, na_values=["∞"])
    all_skims.set_index(index_columns, drop=True, inplace=True)
    cur_skims = pd.read_csv(current_skims_path, dtype=rawInputSchema, na_values=["∞"])
    cur_skims['timePeriod'] = cur_skims['hour'].apply(hourToTimeBin)
    cur_skims.rename(columns={'tazId': 'origin'}, inplace=True)
    cur_skims['completedRequests'] = cur_skims['observations'] * (1. - cur_skims['unmatchedRequestsPercent'] / 100.)
    cur_skims = cur_skims.groupby(['timePeriod', 'reservationType', 'origin']).apply(aggregateInTimePeriod)
    all_skims = pd.concat([cur_skims, all_skims.loc[all_skims.index.difference(cur_skims.index, sort=False)]])
    if all_skims.index.duplicated().sum() > 0:
        logger.warning("Duplicated values in index: \n {0}".format(all_skims.loc[all_skims.duplicated()]))
        all_skims.drop_duplicates(inplace=True)
    all_skims.to_csv(all_skims_path, index=True)
    cur_skims['totalWaitTimeInMinutes'] = cur_skims['waitTimeInMinutes'] * cur_skims['completedRequests']
    totals = cur_skims.groupby(['timePeriod', 'reservationType']).sum()
    totals['matchedPercent'] = totals['completedRequests'] / totals['observations']
    totals['meanWaitTimeInMinutes'] = totals['totalWaitTimeInMinutes'] / totals['completedRequests']
    logger.info("Ridehail matching summary: \n {0}".format(totals[['meanWaitTimeInMinutes', 'matchedPercent']]))
    logger.info("Total requests: \n {0}".format(totals['observations'].sum()))
    logger.info("Total completed requests: \n {0}".format(totals['completedRequests'].sum()))
