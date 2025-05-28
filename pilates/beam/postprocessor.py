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


def _merge_skim(inputMats, outputMats, path, timePeriod, measures):
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
                                                                         (completed > 0) & valid] * 100.0
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

def _postprocess_tnc_skims_zarr(skims_ds, timePeriods, settings):
    """
    Applies TNC-specific post-processing rules (REJECTIONPROB, IWAIT filling,
    DDIST/TOTIVT interpolation, FAR assignment) to Zarr skims.

    Parameters:
    -----------
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    timePeriods : list of str
        List of time period names.
    settings : dict
        Settings dictionary, needed for SOV keys.
    """
    logger.info("Applying TNC-specific post-processing...")
    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}
    tnc_modes = [k.rsplit('_', 1)[0] for k in skims_ds.data_vars if k.startswith("TNC_") and "_TRIPS" in k]
    # Ensure unique TNC modes
    tnc_modes = list(set(tnc_modes))

    if not tnc_modes:
        logger.info("No TNC modes found in skims dataset. Skipping TNC post-processing.")
        return

    # Get SOV skims needed for interpolation ratios
    sov_dist_da = skims_ds.get("SOV_DIST")
    sov_time_da = skims_ds.get("SOV_TIME")

    for tnc_mode in tnc_modes:
        logger.debug(f"Post-processing mode: {tnc_mode}")

        # Get completed and failed trip data for this mode (3D arrays)
        completed_key = f"{tnc_mode}_TRIPS"
        failed_key = f"{tnc_mode}_FAILURES"

        completed_da = skims_ds.get(completed_key)
        failed_da = skims_ds.get(failed_key)

        if completed_da is None or failed_da is None:
            logger.warning(f"Missing {completed_key} or {failed_key} for TNC post-processing. Skipping {tnc_mode}.")
            continue

        completed_3d = completed_da.data
        failed_3d = failed_da.data

        # Get other relevant TNC data arrays
        rejection_prob_da = skims_ds.get(f"{tnc_mode}_REJECTIONPROB")
        iwait_da = skims_ds.get(f"{tnc_mode}_IWAIT")
        ddist_da = skims_ds.get(f"{tnc_mode}_DDIST")
        totivt_da = skims_ds.get(f"{tnc_mode}_TOTIVT")

        for tp in timePeriods:
            tp_idx = tp_to_idx[tp]

            completed_tp = completed_3d[:, :, tp_idx]
            failed_tp = failed_3d[:, :, tp_idx]
            completed_tp_sum = completed_tp.sum()
            failed_tp_sum = failed_tp.sum()

            if completed_tp_sum == 0 and failed_tp_sum == 0:
                 logger.debug(f"No trips for {tnc_mode} in {tp}. Skipping post-processing for this period.")
                 continue # Skip post-processing for this period if no data

            logger.debug(f"Processing {tnc_mode} for period {tp}")

            # --- REJECTIONPROB ---
            if rejection_prob_da is not None:
                rejection_prob_slice = rejection_prob_da.data[:, :, tp_idx]
                # Original logic:
                # Calculate OD-level probability where total trips > 0
                total_trips_od = completed_tp + failed_tp
                valid_ods_mask = total_trips_od > 0
                rejection_prob_slice[valid_ods_mask] = failed_tp[valid_ods_mask] / total_trips_od[valid_ods_mask]

                # Calculate origin-level probability where origin total trips > 0
                completed_sum_by_origin = completed_tp.sum(axis=1)
                failed_sum_by_origin = failed_tp.sum(axis=1)
                total_trips_origin = completed_sum_by_origin + failed_sum_by_origin
                valid_origins_mask = total_trips_origin > 0

                # Apply origin-level probability to rows where origin total > 0
                # This overwrites the OD-level probability for these rows. This matches the original behavior.
                origin_probs = failed_sum_by_origin[valid_origins_mask] / total_trips_origin[valid_origins_mask]
                rejection_prob_slice[valid_origins_mask, :] = origin_probs[:, None]

                rejection_prob_da.data[:, :, tp_idx] = rejection_prob_slice # Update Zarr slice
                logger.debug(f"Updated REJECTIONPROB for {tnc_mode} in {tp}")


            # --- IWAIT ---
            if iwait_da is not None:
                iwait_slice = iwait_da.data[:, :, tp_idx]

                # Calculate weighted mean wait time by origin *only* for OD pairs with completed trips
                # This seems incorrect - the weighted mean should be over all data points *used* to calculate it in BEAM.
                # However, the original code calculates weighted mean based on `waitTimes * completed` where `waitTimes`
                # comes from the input skim. Let's interpret this as "mean wait time experienced by completed trips originating from a zone".
                # It then uses this mean to fill NaNs/zeros where `completed == 0`.

                # Use the data from the partial skim for this calculation if available
                partial_iwait_key = f"{tnc_mode}_IWAIT__{tp}"
                if partial_iwait_key in partialSkims:
                    input_iwait_tp = partialSkims[partial_iwait_key][:]

                    # Calculate weighted mean by origin using the partial skim data
                    # Sum waitTime * completed per origin, divide by sum completed per origin
                    sum_weighted_wait = np.nansum(input_iwait_tp * completed_tp, axis=1)
                    sum_completed_origin = np.nansum(completed_tp, axis=1)
                    # Handle division by zero - origins with no completed trips will have NaN mean
                    weighted_mean_by_origin = np.divide(sum_weighted_wait, sum_completed_origin,
                                                         out=np.full_like(sum_weighted_wait, np.nan),
                                                         where=sum_completed_origin != 0)

                    # Identify cells in the Zarr slice to fill: where completed is 0 AND the origin had some completed trips
                    origins_with_data = sum_completed_origin > 0
                    mask_fill = (completed_tp == 0)[:, None] & origins_with_data[None, :] # Check origins column-wise, apply row-wise
                    # Correct mask application: mask_fill should be 2D based on OD
                    mask_fill = (completed_tp == 0) & (sum_completed_origin[:, None] > 0)

                    # Identify cells to directly update (where completed > 0)
                    mask_direct_update = completed_tp > 0

                    # Apply updates
                    if mask_direct_update.any():

                        # Use the value already in the Zarr slice from _merge_zarr_skim (which applied scaling)
                        # Cells where completed>0 and input was valid already have the scaled input value.
                        # Cells where completed=0 and input was valid might have 0 or nan depending on initial state.
                        # Cells where completed>0 and input was nan are unchanged.

                        # Mask for cells where completed == 0 AND the origin has at least one completed trip
                        mask_to_fill_with_average = (completed_tp == 0) & (sum_completed_origin[:, None] > 0)

                        # Apply the weighted mean by origin
                        if mask_to_fill_with_average.any():
                             origin_indices_to_fill = np.where(mask_to_fill_with_average)[0] # Get row indices (origins)
                             iwait_slice[mask_to_fill_with_average] = weighted_mean_by_origin[origin_indices_to_fill]

                             logger.debug(f"Filled {mask_to_fill_with_average.sum()} TNC IWAIT values in {tp} with origin weighted average.")

                    # Handle bad values (NaNs) that might still exist if an origin had no completed trips
                    if np.isnan(iwait_slice).any():
                         nan_count = np.isnan(iwait_slice).sum()
                         # Decide how to handle remaining NaNs. Defaulting to 0 might be okay.
                         iwait_slice[np.isnan(iwait_slice)] = 0.0
                         logger.debug(f"Set {nan_count} remaining NaN TNC IWAIT values in {tp} to 0.")

                    iwait_da.data[:, :, tp_idx] = iwait_slice # Update Zarr slice
                    logger.debug(f"Updated IWAIT for {tnc_mode} in {tp}")

                else:
                    logger.debug(f"Missing partial skim {partial_iwait_key} for TNC IWAIT post-processing.")


            # --- DDIST and TOTIVT (Interpolation using SOV) ---
            # Requires SOV_DIST and SOV_TIME skims
            # Assuming DDIST uses SOV_DIST and TOTIVT uses SOV_TIME as basis

            # DDIST
            if ddist_da is not None and sov_dist_da is not None:
                ddist_slice = ddist_da.data[:, :, tp_idx]
                sov_dist_slice = sov_dist_da.data[:, :, tp_idx]

                # Cells where we can calculate the ratio: SOV_DIST > 0, TNC_TRIPS > 0, TNC_DDIST > 0
                mask_for_ratio = (sov_dist_slice > 0) & (completed_tp > 0) & (ddist_slice > 0)

                ratio = np.nan
                if mask_for_ratio.any():
                     # Calculate weighted average ratio: (TNC_DDIST * Completed).sum() / (SOV_DIST * Completed).sum()
                     weighted_tnc_ddist = ddist_slice[mask_for_ratio] * completed_tp[mask_for_ratio]
                     weighted_sov_dist = sov_dist_slice[mask_for_ratio] * completed_tp[mask_for_ratio]
                     sum_weighted_sov = weighted_sov_dist.sum()
                     if sum_weighted_sov > 0:
                         ratio = weighted_tnc_ddist.sum() / sum_weighted_sov

                     ratios_individual = ddist_slice[mask_for_ratio] / sov_dist_slice[mask_for_ratio]
                     logger.info(
                        f"Observed TNC DDIST/SOV DIST ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {tnc_mode} in {tp}. "
                        f"Interpolating {np.sum(~mask_for_ratio):.0f} missing values."
                     )
                else:
                     logger.info(f"No data points available to calculate TNC DDIST/SOV DIST ratio for {tnc_mode} in {tp}.")

                # Apply minimum ratio check (from feature branch logic)
                min_ratio = 0.8
                if ratio is not np.nan and ratio < min_ratio:
                     logger.warning(f"Calculated TNC DDIST/SOV DIST ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}.")
                     ratio = min_ratio

                # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
                mask_to_interpolate = ~mask_for_ratio
                if ratio is not np.nan and mask_to_interpolate.any():
                     # Interpolate using SOV_DIST where needed
                     # Ensure we don't use SOV_DIST values that are zero for interpolation
                     mask_interpolation_valid_sov = mask_to_interpolate & (sov_dist_slice > 0)
                     ddist_slice[mask_interpolation_valid_sov] = sov_dist_slice[mask_interpolation_valid_sov] * ratio
                     logger.debug(f"Interpolated {mask_interpolation_valid_sov.sum()} TNC DDIST values in {tp} using SOV_DIST ratio.")

                # Handle remaining NaNs if any (e.g., where SOV_DIST was also 0 or ratio wasn't calculable)
                if np.isnan(ddist_slice).any():
                    nan_count = np.isnan(ddist_slice).sum()
                    ddist_slice[np.isnan(ddist_slice)] = 0.0 # Default remaining NaNs to 0
                    logger.debug(f"Set {nan_count} remaining NaN TNC DDIST values in {tp} to 0.")


                ddist_da.data[:, :, tp_idx] = ddist_slice # Update Zarr slice
                logger.debug(f"Updated DDIST for {tnc_mode} in {tp}")


            # TOTIVT
            if totivt_da is not None and sov_time_da is not None:
                totivt_slice = totivt_da.data[:, :, tp_idx]
                sov_time_slice = sov_time_da.data[:, :, tp_idx]

                # Cells where we can calculate the ratio: SOV_TIME > 0, TNC_TRIPS > 0, TNC_TOTIVT > 0
                mask_for_ratio = (sov_time_slice > 0) & (completed_tp > 0) & (totivt_slice > 0)

                ratio = np.nan
                if mask_for_ratio.any():
                     # Calculate weighted average ratio: (TNC_TOTIVT * Completed).sum() / (SOV_TIME * Completed).sum()
                     weighted_tnc_totivt = totivt_slice[mask_for_ratio] * completed_tp[mask_for_ratio]
                     weighted_sov_time = sov_time_slice[mask_for_ratio] * completed_tp[mask_for_ratio]
                     sum_weighted_sov = weighted_sov_time.sum()
                     if sum_weighted_sov > 0:
                         ratio = weighted_tnc_totivt.sum() / sum_weighted_sov

                     ratios_individual = totivt_slice[mask_for_ratio] / sov_time_slice[mask_for_ratio]
                     logger.info(
                        f"Observed TNC TOTIVT/SOV TIME ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {tnc_mode} in {tp}. "
                        f"Interpolating {np.sum(~mask_for_ratio):.0f} missing values."
                     )
                else:
                     logger.info(f"No data points available to calculate TNC TOTIVT/SOV TIME ratio for {tnc_mode} in {tp}.")


                # Apply minimum ratio check (from feature branch logic)
                min_ratio = 0.8 # Same minimum as DDIST in feature branch
                if ratio is not np.nan and ratio < min_ratio:
                     logger.warning(f"Calculated TNC TOTIVT/SOV TIME ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}.")
                     ratio = min_ratio

                # Interpolate where ratio was calculated and cells were NOT used for ratio calculation
                mask_to_interpolate = ~mask_for_ratio
                if ratio is not np.nan and mask_to_interpolate.any():
                     # Interpolate using SOV_TIME where needed
                     # Ensure we don't use SOV_TIME values that are zero for interpolation
                     mask_interpolation_valid_sov = mask_to_interpolate & (sov_time_slice > 0)
                     totivt_slice[mask_interpolation_valid_sov] = sov_time_slice[mask_interpolation_valid_sov] * ratio
                     logger.debug(f"Interpolated {mask_interpolation_valid_sov.sum()} TNC TOTIVT values in {tp} using SOV_TIME ratio.")

                # Handle remaining NaNs if any (e.g., where SOV_TIME was also 0 or ratio wasn't calculable)
                if np.isnan(totivt_slice).any():
                    nan_count = np.isnan(totivt_slice).sum()
                    totivt_slice[np.isnan(totivt_slice)] = 0.0 # Default remaining NaNs to 0
                    logger.debug(f"Set {nan_count} remaining NaN TNC TOTIVT values in {tp} to 0.")

                totivt_da.data[:, :, tp_idx] = totivt_slice # Update Zarr slice
                logger.debug(f"Updated TOTIVT for {tnc_mode} in {tp}")

    logger.info("Completed TNC-specific post-processing.")


def clear_skim_cache(asim_local_output_dir):
    skims_path = os.path.join(asim_local_output_dir, "cache")
    if os.path.exists(skims_path):
        logger.info("Deleting skims cache at {0}. Eventually we should modify it in place".format(skims_path))
        shutil.rmtree(skims_path)
    else:
        logger.warning("Did not find skim cache to delete")


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


def _transform_measure(input_vals, completed, failed, measure, path):
    """
    Transforms skim values based on measure type, path, and trip completion statistics.

    Parameters:
    -----------
    input_vals : ndarray
        The new observed values from the partial skim for a single time period.
    completed : ndarray (2D)
        Count of completed trips for each OD pair for the current time period.
    failed : ndarray (2D)
        Count of failed trips for each OD pair for the current time period.
    measure : str
        The measure name (e.g., "IWAIT", "TOTIVT").
    path : str
        The mode path (e.g., "SOV", "WLK_TRN_WLK", "TNC_SINGLE").

    Returns:
    --------
    tuple (mask, vals, to_cancel)
        mask: Boolean array (2D) indicating which cells to update with `vals`.
        vals: Values (1D) to assign to the masked cells.
        to_cancel: Boolean array (2D) indicating cells to zero out due to high failure rate (for IVT/TOTIVT).
                   Returns None for other measures.
    """
    # Basic mask - where there are completed trips and valid input values
    valid = ~np.isnan(input_vals)
    completed_mask = (completed > 0)
    basic_mask = valid & completed_mask
    negative = (input_vals < 0)
    if np.any(negative):
        logger.warning(f"Found {np.sum(negative)} negative values in input_vals for measure {measure} and path {path}. Setting them to 0.")
        input_vals[negative] = 0.0  # Set negative values to 0

    # Determine scaling factor based on path (mode)
    scaling = 100.0 if not path.startswith("TNC") else 1.0

    # Handle measures that need scaling
    if measure in ["IWAIT", "XWAIT", "WACC", "WAUX", "WEGR", "DTIM", "FERRYIVT"]:
        # Note: DDIST and TOTIVT for TNC are handled in post-processing, not here
        if path.startswith("TNC") and measure in ["DDIST", "TOTIVT"]:
             # These are handled separately in post-processing
             return np.zeros_like(input_vals, dtype=bool), np.array([]), None
        if measure == "IWAIT" and path.startswith("TNC"):
             # TNC IWAIT filling is handled separately in post-processing
             return basic_mask, input_vals[basic_mask] * scaling, None # Still apply basic scaling if direct data used

        return basic_mask, input_vals[basic_mask] * scaling, None

    # Handle travel time measures with penalty logic
    elif measure in ["TOTIVT", "IVT"]:
        # TNC TOTIVT interpolation is handled separately in post-processing
        if path.startswith("TNC") and measure == "TOTIVT":
             return np.zeros_like(input_vals, dtype=bool), np.array([]), None

        # Feature branch cancellation condition: failed > 3 and failed > 1*completed
        # Original cancellation condition: failed > 3 and failed > 6*completed
        # Using the feature branch condition:
        to_cancel = (failed > 3) & (failed > (1 * completed))

        # To penalize: failed > completed, NOT canceled, valid input, and some completed trips
        to_penalize = (failed > completed) & ~to_cancel & valid & completed_mask

        # To allow: NOT canceled, NOT penalized, valid input, and some completed trips
        to_allow = valid & completed_mask & ~to_cancel & ~to_penalize

        # Prepare result array
        result_vals = np.zeros_like(input_vals)

        # Apply penalty where needed
        if to_penalize.any():
            penalty_factor = (failed[to_penalize] + 1) / (completed[to_penalize] + 1)
            result_vals[to_penalize] = input_vals[to_penalize] * penalty_factor

        # Use regular values where allowed (with scaling)
        if to_allow.any():
            result_vals[to_allow] = input_vals[to_allow] * scaling

        # Combined mask for cells to update
        # Note: Canceled cells are explicitly set to 0 *after* this function returns
        # by the caller (_merge_zarr_skim), so the update_mask only covers penalized and allowed cells.
        update_mask = to_penalize | to_allow

        return update_mask, result_vals[update_mask], to_cancel

    # Handle DIST (special averaging in old code, keep simple assignment for Zarr)
    elif measure == "DIST":
        return basic_mask, input_vals[basic_mask], None

    # Handle non-TOLL measures (simple assignment)
    # Note: FAR for TNC is handled in post-processing
    elif not measure.endswith("TOLL"):
        # Simple assignment where completed > 0
        return completed_mask, input_vals[completed_mask], None

    # Default case (e.g., TOLL measures are copied later if SOV, or handled by other logic)
    return np.zeros_like(input_vals, dtype=bool), np.array([]), None


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

    path, measure_name = parts

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



def merge_current_zarr_od_skims(all_skims_path, previous_skims_path, beam_output_dir, settings, override=None):
    """
    Merges current BEAM OMX skims into the main Zarr skims file.
    """
    # Set parallel to False explicitly
    parallel = False

    skims_ds = xr.open_zarr(all_skims_path)

    if override is None:
        current_skims_path = find_produced_od_skims(beam_output_dir, "omx")
    else:
        current_skims_path = override

    if current_skims_path is None or not os.path.exists(current_skims_path):
        logger.warning(f"No current OMX skims found at {current_skims_path}. Skipping merge.")
        skims_ds.close()
        return None # Indicate no merge happened

    try:
        partialSkims = omx.open_file(current_skims_path, mode='r')
    except Exception as e:
        logger.error(f"Failed to open partial skims file {current_skims_path}: {e}")
        skims_ds.close()
        return None # Indicate merge failed

    timePeriods = settings["periods"]

    # Step 1: Accumulate completed and failed trips from partial skims
    logger.info("Accumulating completed and failed trip counts from partial skims...")
    completed_failed_dict = _accumulate_completed_failed_trips(partialSkims, timePeriods)

    # Step 2: Merge completed and failed trips into the Zarr skims Dataset
    # Iterate through modes found in the partial skims
    logger.info("Merging trip counts into Zarr skims...")
    for path, (completed, failed) in completed_failed_dict.items():
         # completed and failed here are 3D arrays [zones, zones, periods]
        if f"{path}_TRIPS" in skims_ds and f"{path}_FAILURES" in skims_ds:
            _merge_zarr_trip_counts(skims_ds, path, completed, failed)
        else:
            logger.debug(f"Skipping trip counts merge for mode {path} as TRIPS or FAILURES variable does not exist in target skims.")


    # Step 3: Merge other measures using _merge_zarr_skim
    # _merge_zarr_skim processes one variable (measure) for all periods.
    logger.info("Merging other skim measures into Zarr skims...")
    processed_vars = set()
    # Build a set of variables in partial skims (excluding time period suffix)
    partial_skim_measures = set()
    for key in partialSkims.list_matrices():
         parts = key.rsplit('_', 3)
         if len(parts) == 4:
              path, measure, _, tp = parts
              if measure not in ["TRIPS", "FAILURES"]:
                   partial_skim_measures.add(f"{path}_{measure}")

    for var_name in skims_ds.data_vars:
        # Check if this variable corresponds to a measure present in the partial skims
        # and is not TRIPS/FAILURES (already handled).
        if var_name in partial_skim_measures:
            logger.debug(f"Processing variable: {var_name}")
            try:
                # Pass the specific DataArray for this variable
                path, measure_name = _merge_zarr_skim(partialSkims, skims_ds[var_name], completed_failed_dict, timePeriods)
                if path is not None: # Check if processing wasn't skipped
                    processed_vars.add(var_name)
            except Exception as e:
                logger.error(f"Error merging Zarr variable {var_name}: {e}")

    logger.info(f"Completed processing {len(processed_vars)} skim variables.")

    # Step 4: Apply TNC-specific post-processing rules
    # This must happen *after* all base TNC and SOV skims (TRIPS, FAILURES, IWAIT, DDIST, TOTIVT, FAR, DIST, TIME)
    # have been merged in the steps above.
    _postprocess_tnc_skims_zarr(skims_ds, timePeriods, settings)


    # Step 5: Handle transit mode availability
    logger.info("Processing transit mode availability...")
    _handle_transit_mode_availability(skims_ds, timePeriods)


    # Step 6: Copy skims for unobserved modes (e.g., SOV -> HOV/SOVTOLL)
    # This should happen after the base skims (like SOV) are finalized.
    mapping = {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]}
    copy_skims_for_unobserved_modes(mapping, skims_ds)

    # Step 7: Save the updated Zarr dataset
    logger.info(f"Started writing updated zarr skims to {all_skims_path}")
    # Use mode='w' to overwrite with the updated data, consolidated=True for performance
    # Ensure zarr_version=2 for compatibility
    skims_ds.to_zarr(all_skims_path, mode='w', consolidated=True, zarr_version=2)
    logger.info("Completed writing zarr skims")

    # Close the datasets
    skims_ds.close()
    partialSkims.close()

    return current_skims_path

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
    try:
        skims = xr.open_zarr(all_skims_path, mode='a') # Open in append/write mode
    except Exception as e:
        logger.error(f"Failed to open skims file {all_skims_path} for trimming inaccessible ODs: {e}")
        return

    order = zone_order(settings, settings['start_year'])
    periods = settings["periods"]
    transit_paths_settings = settings.get('transit_paths', {}) # Use .get for safety

    if not transit_paths_settings:
         logger.warning("No 'transit_paths' defined in settings for trimming inaccessible ODs.")
         skims.close()
         return

    tp_to_idx = {p: i for i, p in enumerate(periods)}

    totalTrips = {tp: np.zeros((len(order), len(order))) for tp in periods}

    for var_name in skims.data_vars:
         parts = var_name.rsplit('__', 1)
         if len(parts) == 2:
              measure_part, tp = parts
              if tp in totalTrips and not measure_part.startswith('RH_'): # Exclude Ridehail as in original
                   if 'TRIPS__' in var_name:
                        tp_idx = tp_to_idx[tp]
                        # Need to add the 2D slice for this period to the 2D total trips array
                        if skims[var_name].ndim == 3 and skims[var_name].shape[-1] > tp_idx:
                             totalTrips[tp] += np.nan_to_num(skims[var_name].data[:, :, tp_idx])
                        elif skims[var_name].ndim == 2 and len(periods) == 1 and periods[0] == tp:
                              # Handle case where there's only one period and it's 2D
                              totalTrips[tp] += np.nan_to_num(skims[var_name].data[:, :])
                        else:
                            logger.warning(f"Skipping '{var_name}' for total trips calculation due to unexpected shape or period mismatch.")

    for path, metrics in transit_paths_settings.items():
        trip_name_zarr = f"{path}_TRIPS"
        fail_name_zarr = f"{path}_FAILURES"

        trip_da = skims.get(trip_name_zarr)
        fail_da = skims.get(fail_name_zarr)

        if trip_da is None or fail_da is None:
             logger.debug(f"Skipping trim for {path}: missing {trip_name_zarr} or {fail_name_zarr}")
             continue

        for tpIdx, period in enumerate(periods):
            if period not in totalTrips:
                 logger.warning(f"Skipping trim for {path} in {period}: total trip data missing for this period.")
                 continue

            completedAllTripsByOandD = totalTrips[period].sum(axis=0) + totalTrips[period].sum(axis=1)

            # Get 2D slices for this period
            if trip_da.ndim == 3 and trip_da.shape[-1] > tpIdx:
                 completedTransitTrips = np.nan_to_num(trip_da.data[:, :, tpIdx])
                 failedTransitTrips = np.nan_to_num(fail_da.data[:, :, tpIdx])
            elif trip_da.ndim == 2 and len(periods) == 1 and periods[0] == period:
                 completedTransitTrips = np.nan_to_num(trip_da.data[:, :])
                 failedTransitTrips = np.nan_to_num(fail_da.data[:, :])
            else:
                 logger.warning(f"Skipping trim for {path} in {period}: trip data has unexpected shape or period mismatch.")
                 continue


            completedTransitTripsByOandD = completedTransitTrips.sum(axis=0) + completedTransitTrips.sum(axis=1)
            failedTransitTripsByOandD = failedTransitTrips.sum(axis=0) + failedTransitTrips.sum(axis=1)

            toDelete = np.squeeze((completedAllTripsByOandD > 1000) & (failedTransitTripsByOandD > 200) & (completedTransitTripsByOandD == 0))

            num_zones_to_delete = np.sum(toDelete)
            if num_zones_to_delete > 0:
                logger.info(f"Trimming {path} service for {num_zones_to_delete} zones in {period} because no completed transit trips were observed for these zones (conditions met).")
                for metric in metrics:
                    name_zarr = f"{path}_{metric}" # Zarr key format
                    metric_da = skims.get(name_zarr)

                    if metric_da is not None:
                        # Ensure the DataArray has enough dimensions for the period
                        if metric_da.ndim == 3 and metric_da.shape[-1] > tpIdx:
                             # Get the 2D slice for this period
                             arr_slice = metric_da.data[:, :, tpIdx]
                             # Apply the deletion mask. This sets values to 0 where the origin OR destination zone is in `toDelete`.
                             # This corresponds to setting the row AND the column corresponding to the zones in `toDelete` to 0.
                             arr_slice[toDelete[:, None] | toDelete[None, :]] = 0.0
                             # Update the data in the Zarr array
                             metric_da.data[:, :, tpIdx] = arr_slice
                             logger.debug(f"Trimmed {name_zarr} for {period} for {num_zones_to_delete} zones.")
                        elif metric_da.ndim == 2 and len(periods) == 1 and periods[0] == period:
                             # Handle 2D case (single period)
                             arr_slice = metric_da.data[:, :]
                             arr_slice[toDelete[:, None] | toDelete[None, :]] = 0.0
                             metric_da.data[:, :] = arr_slice
                             logger.debug(f"Trimmed {name_zarr} (2D) for {period} for {num_zones_to_delete} zones.")
                        else:
                             logger.warning(f"Skipping trim for metric {name_zarr} in {period} due to unexpected shape or period mismatch.")
                    else:
                        logger.debug(f"Skipping trim for metric {name_zarr} in {period}: variable not found in skims.")

    # Save the updated Zarr dataset after trimming
    logger.info(f"Started writing zarr skims after trimming inaccessible ODs to {all_skims_path}")
    skims.to_zarr(all_skims_path, mode='w', consolidated=True, zarr_version=2) # Use mode='w' to overwrite
    logger.info("Completed writing zarr skims after trimming.")
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
    cur_skims['completedRequests'] = cur_skims['observations'] * (1. - cur_skims['unmatchedRequestsPercent'] / 100.)
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
