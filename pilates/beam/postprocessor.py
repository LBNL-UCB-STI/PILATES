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

if True:
    # Configure the logger to print to stdout
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set to DEBUG for even more output

    # Create a handler for stdout
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Make sure the logger propagates messages to the root logger
    logger.propagate = True
else:
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


def copy_skims_for_unobserved_modes(mapping, skims, mats):
    for fromMode, toModes in mapping.items():
        relevantSkimKeys = [key for key in mats if
                            key.startswith(fromMode + "_") and not ("TOLL" in key)]
        for skimKey in relevantSkimKeys:
            for toMode in toModes:
                toKey = skimKey.replace(fromMode + "_", toMode + "_")
                skims[toKey][:] = skims[skimKey][:]
                print("Copying values from {0} to {1}".format(skimKey, toKey))


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
                completed_failed_dict[mode][0][:, :, tpIdx] = partialSkims[key][:]
            elif key.endswith(f"_FAILURES__{tp}"):
                mode = key.rsplit('_', 3)[0]  # Extract mode from key (e.g., WLK_TRN_WLK)
                if mode not in completed_failed_dict:
                    completed_failed_dict[mode] = [np.zeros(out_array_shape, dtype=np.float32),
                                                   np.zeros(out_array_shape, dtype=np.float32)]
                completed_failed_dict[mode][1][:, :, tpIdx] = partialSkims[key][:]

    return completed_failed_dict


def _transform_measure(input_vals, completed, failed, measure):
    """
    Transforms skim values based on measure type and trip completion statistics.

    Parameters:
    -----------
    input_vals : ndarray
        The new observed values
    completed : ndarray
        Count of completed trips for each OD pair for current time period
    failed : ndarray
        Count of failed trips for each OD pair for current time period
    measure : str
        The measure name (e.g., "IWAIT", "TOTIVT")

    Returns:
    --------
    tuple (mask, vals)
        mask: Boolean array indicating which cells to update
        vals: Values to assign to those cells
    """
    # Basic mask - where there are completed trips and valid input values
    valid = ~np.isnan(input_vals)
    completed_mask = (completed > 0)
    basic_mask = valid & completed_mask

    # Handle measures that need scaling by 100
    if measure in ["IWAIT", "XWAIT", "WACC", "WAUX", "WEGR", "DTIM", "DDIST", "FERRYIVT"]:
        return basic_mask, input_vals[basic_mask] * 100.0

    # Handle travel time measures with penalty logic
    elif measure in ["TOTIVT", "IVT"]:
        # Create masks for different conditions
        to_cancel = (failed > 3) & (failed > (6 * completed))
        to_penalize = (failed > completed) & ~to_cancel & valid & completed_mask
        to_allow = valid & completed_mask & ~to_cancel & ~to_penalize

        # Prepare values based on condition
        result_vals = np.zeros_like(input_vals)

        # Apply penalty where needed
        if to_penalize.any():
            penalty_factor = (failed + 1) / (completed + 1)
            indices = np.where(to_penalize)
            result_vals[indices] = input_vals[indices] * penalty_factor[indices]

        # Use regular values where allowed (with scaling)
        if to_allow.any():
            indices = np.where(to_allow)
            result_vals[indices] = input_vals[indices] * 100.0

        # Combined mask for cells to update
        update_mask = to_penalize | to_allow

        # Also return the to_cancel mask for use in post-processing
        return update_mask, result_vals[update_mask], to_cancel

    # Handle non-TOLL measures
    elif not measure.endswith("TOLL"):
        return basic_mask, input_vals[basic_mask]

    # Default case (for TOLL measures)
    return np.zeros_like(input_vals, dtype=bool), np.array([])


def _merge_zarr_skim(partialSkims, skims, completed_failed_dict, timePeriods):
    source_skims = partialSkims.list_matrices()
    # Extract the path from the skims name
    path = skims.name.rsplit('_', maxsplit=1)[0]

    # Get the completed and failed trips for the current mode
    completed, failed = completed_failed_dict.get(path, (None, None))

    if completed is None or failed is None:
        logger.info(f"No input skim for mode {path}")
        return path, (completed, failed)

    # Get the measure name from the skims name
    measure_name = skims.name.rsplit('_', maxsplit=1)[1]

    # Construct the key for the partial skims
    partial_key = f"{path}_{measure_name}__"

    # Store cancel masks for KEYIVT post-processing
    cancel_masks = {}

    for tpIdx, tp in enumerate(timePeriods):
        partial_key_with_tp = f"{partial_key}{tp}"
        if partial_key_with_tp in partialSkims:
            input_vals = partialSkims[partial_key_with_tp][:]

            # Apply transformations directly to the skims array
            if measure_name in ["TOTIVT", "IVT"]:
                # For these measures, we need a bit more preprocessing
                mask, vals, to_cancel = _transform_measure(input_vals, completed[:, :, tpIdx], failed[:, :, tpIdx],
                                                           measure_name)

                # Log cancellations if needed (simplified)
                if to_cancel.sum() > 0:
                    logger.info(
                        f"Marking {to_cancel.sum()} {path} trips completely impossible in {tp}. "
                        f"There were {completed[:, :, tpIdx][to_cancel].sum()} completed trips but "
                        f"{failed[:, :, tpIdx][to_cancel].sum()} failed trips in these ODs."
                    )

                # Update values directly with mask
                if mask.any():
                    skims.values[:, :, tpIdx][mask] = vals

                # Zero out canceled ODs directly
                if to_cancel.any():
                    skims.values[:, :, tpIdx][to_cancel] = 0

                # Store to_cancel mask for later KEYIVT processing
                cancel_masks[tp] = to_cancel

            else:
                # For other measures, just apply mask directly
                mask, vals = _transform_measure(input_vals, completed[:, :, tpIdx], failed[:, :, tpIdx], measure_name)
                if mask.any():
                    skims.values[:, :, tpIdx][mask] = vals

            # Handle SOV special case immediately
            if path.startswith('SOV_') and not measure_name.endswith("TOLL"):
                for sub in ['SOVTOLL_', 'HOV2_', 'HOV2TOLL_', 'HOV3_', 'HOV3TOLL_']:
                    new_key = f"{path}_{measure_name.replace('SOV_', sub)}__{tp}"
                    if new_key in skims:
                        # Direct masked update
                        completed_mask = completed[:, :, tpIdx] > 0
                        if completed_mask.any():
                            skims[new_key].values[:, :, tpIdx][completed_mask] = input_vals[completed_mask]

                        # Simplify logging to avoid any calculations on the skims
                        logger.info(
                            f"Updated HOV skims for {path}: added values for {completed_mask.sum()} valid OD pairs in {tp}")

    # Handle KEYIVT post-processing for this skim
    if measure_name in ["TOTIVT", "IVT"]:
        for tp, to_cancel in cancel_masks.items():
            tp_idx = timePeriods.index(tp)
            keyivt_path = f"{path}_KEYIVT__{tp}"

            if keyivt_path in skims:
                # Compute masks for different operations
                if keyivt_path in partialSkims:
                    keyivt_vals = partialSkims[keyivt_path][:]
                    valid = ~np.isnan(keyivt_vals)
                    to_penalize = (failed[:, :, tp_idx] > completed[:, :, tp_idx]) & ~to_cancel
                    to_allow = ~to_cancel & ~to_penalize & valid & (completed[:, :, tp_idx] > 0)

                    # Direct masked updates
                    if to_allow.any():
                        skims[keyivt_path].values[:, :, tp_idx][to_allow] = keyivt_vals[to_allow] * 100.0
                        logger.info(f"Updated {to_allow.sum()} KEYIVT values for {path} in {tp}")

                # Zero out canceled ODs directly
                if to_cancel.any():
                    skims[keyivt_path].values[:, :, tp_idx][to_cancel] = 0
                    logger.info(f"Zeroed out {to_cancel.sum()} KEYIVT values for {path} in {tp}")

    if measure_name in ["TIME", "TOTIVT"]:
        # Create a clean, tabular format
        logger.info(f"{'-' * 60}")
        logger.info(f"Summary for {path}_{measure_name}:")
        logger.info(f"{'-' * 60}")
        logger.info(f"{'Period':<6} | {'Completed':<10} | {'Failed':<10} | {'Success Rate':<12}")
        logger.info(f"{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 12}")

        for tpIdx, tp in enumerate(timePeriods):
            # Calculate total completed and failed trips for this time period
            completed_count = np.sum(completed[:, :, tpIdx] > 0)
            failed_count = np.sum(failed[:, :, tpIdx] > 0)

            # Calculate success rate
            total = completed_count + failed_count
            success_rate = (completed_count / total) * 100 if total > 0 else 0

            # Format as aligned columns with separators
            logger.info(f"{tp:<6} | {completed_count:<10} | {failed_count:<10} | {success_rate:>6.1f}%")

        logger.info(f"{'-' * 60}")

    return path, (completed, failed)


def _handle_transit_mode_availability(skims, timePeriods):
    """
    Handles transit mode availability by checking specific transit modes.

    For OD pairs with 50+ successful transit trips, any specific transit mode
    with 0 successful trips will be marked as unavailable (TOTIVT = 0).
    Only marks OD pairs where TOTIVT is currently > 0 or np.nan.
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

    # Create a summary table header
    logger.info(f"{'=' * 80}")
    logger.info(f"Transit Mode Availability Analysis")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Period':<6} | {'Mode':<12} | {'ODs w/50+ Transit':<18} | {'ODs w/0 Mode Trips':<18} | {'Changed':<10} | {'% Changed':<10}")
    logger.info(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 10}")

    # Process each time period
    for tp in timePeriods:
        # Check if general transit trip counts exist
        general_trips_key = f"{general_transit_path}_TRIPS__{tp}"
        if general_trips_key not in skims:
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {'NO DATA':<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}")
            continue

        # Find OD pairs with at least 50 successful transit trips
        general_transit_trips = skims[general_trips_key].values[:, :, 0]  # Assuming it's a 3D array with one time slice
        mask_significant_transit = general_transit_trips >= 50

        if not np.any(mask_significant_transit):
            logger.info(f"{tp:<6} | {'ALL MODES':<12} | {0:<18} | {0:<18} | {0:<10} | {0:<10.1f}")
            continue

        significant_count = np.sum(mask_significant_transit)

        # Check each specific transit mode
        for mode in specific_transit_modes:
            # Get trip counts for this mode
            mode_trips_key = f"{mode}_TRIPS__{tp}"

            if mode_trips_key not in skims:
                logger.info(
                    f"{tp:<6} | {mode:<12} | {significant_count:<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}")
                continue

            mode_trips = skims[mode_trips_key].values[:, :, 0]  # Assuming it's a 3D array with one time slice

            # Find OD pairs where general transit has 50+ trips but this mode has 0
            mask_mode_unused = mask_significant_transit & (mode_trips == 0)

            unused_count = np.sum(mask_mode_unused)

            # Mark this mode as unavailable for these OD pairs by setting TOTIVT = 0
            totivt_key = f"{mode}_TOTIVT"
            changed_count = 0

            if totivt_key in skims and unused_count > 0:
                # Check dimension of the data
                if len(skims[totivt_key].dims) == 3:
                    # Get time period index
                    tp_idx = timePeriods.index(tp)

                    # Get current TOTIVT values
                    current_totivt = skims[totivt_key].values[:, :, tp_idx]

                    # Only mark as unavailable if current TOTIVT is > 0 or np.nan
                    mask_to_change = mask_mode_unused & ((current_totivt > 0) | np.isnan(current_totivt))
                    changed_count = np.sum(mask_to_change)

                    # Update values
                    if changed_count > 0:
                        skims[totivt_key].values[:, :, tp_idx][mask_to_change] = 0
                else:
                    # If it's 2D (just one time period), apply directly
                    current_totivt = skims[totivt_key].values

                    # Only mark as unavailable if current TOTIVT is > 0 or np.nan
                    mask_to_change = mask_mode_unused & ((current_totivt > 0) | np.isnan(current_totivt))
                    changed_count = np.sum(mask_to_change)

                    # Update values
                    if changed_count > 0:
                        skims[totivt_key].values[mask_to_change] = 0

            percent_changed = (changed_count / significant_count) * 100 if significant_count > 0 else 0

            logger.info(
                f"{tp:<6} | {mode:<12} | {significant_count:<18} | {unused_count:<18} | {changed_count:<10} | {percent_changed:<10.1f}")

    logger.info(f"{'=' * 80}")


def merge_current_zarr_od_skims(all_skims_path, previous_skims_path, beam_output_dir, settings):
    # Set parallel to False explicitly
    parallel = False

    skims = xr.open_zarr(all_skims_path)
    current_skims_path = find_produced_od_skims(beam_output_dir, "omx")
    partialSkims = omx.open_file(current_skims_path, mode='r')
    partialSkimKeys = pd.Series(partialSkims.list_matrices())
    partialSkimDataKeys = partialSkimKeys.loc[
        ~(partialSkimKeys.str.contains("TRIPS") | partialSkimKeys.str.contains("FAILURES"))]
    iterable = [(
        path + '_' + measure, vals[3].to_list()) for (path, measure), vals
        in
        partialSkimDataKeys.str.rsplit('_', n=3, expand=True).groupby([0, 1])]
    timePeriods = settings["periods"]
    completed_failed_dict = _accumulate_completed_failed_trips(partialSkims, timePeriods)

    # Process all skims sequentially
    logger.info(f"Processing {len(iterable)} skim groups sequentially")
    processed_count = 0

    for path, tps in iterable:
        if path in skims:
            try:
                result_path, _ = _merge_zarr_skim(partialSkims, skims[path], completed_failed_dict, timePeriods)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error merging skim for mode {path}: {e}")

    # After all skims are processed, handle transit mode availability
    logger.info("Processing transit mode availability...")
    _handle_transit_mode_availability(skims, timePeriods)

    logger.info(f"Completed processing {processed_count} skims")

    mapping = {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]}
    copy_skims_for_unobserved_modes(mapping, skims, skims.keys())

    skims.close()
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
