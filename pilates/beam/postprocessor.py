import logging
import os

import numpy as np
import openmatrix as omx
import pandas as pd
import typing

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

def rename_beam_output_directory(settings, year, replanning_iteration_number=0):
    beam_output_dir = settings['beam_local_output_folder']
    iteration_output_directory, _ = find_latest_beam_iteration(beam_output_dir)
    beam_run_output_dir = os.path.join(*iteration_output_directory.split(os.sep)[:-2])
    new_iteration_output_directory = os.path.join(beam_output_dir, settings['region'],
                                                  "year-{0}-iteration-{1}".format(year, replanning_iteration_number))
    if os.path.exists(new_iteration_output_directory):
        os.rename(new_iteration_output_directory, find_not_taken_dir_name(new_iteration_output_directory))
    os.rename(beam_run_output_dir, new_iteration_output_directory)


def find_produced_od_skims(beam_output_dir, suffix="csv.gz"):
    iteration_dir, it_num = find_latest_beam_iteration(beam_output_dir)
    if iteration_dir is None:
        return None
    od_skims_path = os.path.join(iteration_dir, "{0}.activitySimODSkims_current.{1}".format(it_num, suffix))
    logger.info("expecting skims at {0}".format(od_skims_path))
    if not os.path.exists(od_skims_path):
        return None
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
    complete_key = '_'.join([path, 'TRIPS', '', timePeriod])
    failed_key = '_'.join([path, 'FAILURES', '', timePeriod])
    completed, failed = None, None
    if complete_key in inputMats.keys():
        completed = np.array(inputMats[complete_key]).copy()
        if '_'.join([path, 'TOTIVT', '', timePeriod]) in inputMats.keys():
            shouldNotBeZero = (completed > 0) & (np.array(inputMats['_'.join([path, 'TOTIVT', '', timePeriod])]) == 0)
            if shouldNotBeZero.any():
                logger.warning(
                    "In BEAM outputs for {0} in {1} we have {2} completed trips with "
                    "time = 0".format(path, timePeriod, shouldNotBeZero.sum()))
                completed[shouldNotBeZero] = 0
        failed = np.array(inputMats[failed_key])
        logger.info("Adding {0} valid trips and {1} impossible trips to skim {2}, where {3} had existed before".format(
            np.nan_to_num(completed).sum(),
            np.nan_to_num(failed).sum(),
            complete_key,
            np.nan_to_num(np.array(outputMats[complete_key])).sum()))
        try:
            logger.info("Of the {0} completed trips, {1} were to a previously unobserved "
                        "OD".format(np.nan_to_num(completed).sum(),
                                    np.nan_to_num(completed[outputMats[complete_key][:] == 0]).sum()))
        except:
            pass
        toPenalize = np.array([0])
        toCancel = np.array([0])
        for measure in measures:
            inputKey = '_'.join([path, measure, '', timePeriod])
            if path in ["WALK", "BIKE"]:
                if measure == "DIST":
                    outputKey = path + "DIST"
                else:
                    outputKey = '_'.join([path, measure])
            else:
                outputKey = inputKey
            if (outputKey in outputMats) and (inputKey in inputMats):
                if measure == "TRIPS":
                    outputMats[outputKey][completed > 0] += completed[completed > 0]
                elif measure == "FAILURES":
                    outputMats[outputKey][failed > 0] += failed[failed > 0]
                elif measure == "DIST":
                    outputMats[outputKey][completed > 0] = 0.5 * (
                            outputMats[outputKey][completed > 0] + inputMats[inputKey][completed > 0])
                elif measure in ["IWAIT", "XWAIT", "WACC", "WAUX", "WEGR", "DTIM", "DDIST", "FERRYIVT"]:
                    # NOTE: remember the mtc asim implementation has scaled units for these variables
                    valid = ~np.isnan(inputMats[inputKey][:])
                    outputMats[outputKey][(completed > 0) & valid] = inputMats[inputKey][
                                                                         (completed > 0) & valid] * 100.0
                elif measure in ["TOTIVT", "IVT"]:

                    inputKeyKEYIVT = '_'.join([path, 'KEYIVT', '', timePeriod])
                    outputKeyKEYIVT = inputKeyKEYIVT
                    if (inputKeyKEYIVT in inputMats.keys()) & (outputKeyKEYIVT in outputMats.keys()):
                        additionalFilter = (outputMats[outputKeyKEYIVT][:] > 0)
                    else:
                        additionalFilter = False
                    outputTravelTime = np.array(outputMats[outputKey])
                    toCancel = (failed > 3) & (failed > (6 * completed))
                    previouslyNonZero = ((outputTravelTime > 0) | additionalFilter) & toCancel
                    # save this for later so it doesn't get overwritten
                    toPenalize = (failed > completed) & ~toCancel & ((outputTravelTime > 0) | additionalFilter)
                    if toCancel.sum() > 0:
                        logger.info(
                            "Marking {0} {1} trips completely impossible in {2}. There were {3} completed trips but {4}"
                            " failed trips in these ODs. Previously, {5} were nonzero".format(
                                toCancel.sum(), path, timePeriod, completed[toCancel].sum(), failed[toCancel].sum(),
                                previouslyNonZero.sum()))
                        logger.info("There are now {0} observed ODs, {1} impossible ODs, and {2} default ODs".format(
                            ((completed > 0) & (outputTravelTime > 0)).sum(),
                            (outputTravelTime == 0).sum(),
                            ((completed == 0) & (outputTravelTime > 0)).sum()
                        ))
                    toAllow = ~toCancel & ~toPenalize & ~np.isnan(inputMats[inputKey][:])
                    outputMats[outputKey][toAllow] = inputMats[inputKey][toAllow] * 100
                    # outputMats[outputKey][toCancel] = 0.0
                    if (inputKeyKEYIVT in inputMats.keys()) & (outputKeyKEYIVT in outputMats.keys()):
                        # outputMats[outputKeyKEYIVT][toCancel] = 0.0
                        outputMats[outputKeyKEYIVT][toAllow] = inputMats[inputKeyKEYIVT][toAllow] * 100
                elif ~measure.endswith("TOLL"):  # hack to avoid overwriting initial tolls
                    outputMats[outputKey][completed > 0] = inputMats[inputKey][completed > 0]
                    if path.startswith('SOV_'):
                        for sub in ['SOVTOLL_', 'HOV2_', 'HOV2TOLL_', 'HOV3_', 'HOV3TOLL_']:
                            newKey = '_'.join([path, measure.replace('SOV_', sub), '', timePeriod])
                            outputMats[newKey][completed > 0] = inputMats[inputKey][completed > 0]
                            logger.info("Adding {0} valid trips and {1} impossible trips to skim {2}".format(
                                np.nan_to_num(completed).sum(),
                                np.nan_to_num(failed).sum(),
                                newKey))
                badVals = np.sum(np.isnan(outputMats[outputKey][:]))
                if badVals > 0:
                    logger.warning("Total number of {0} skim values are NaN for skim {1}".format(badVals, outputKey))
            elif outputKey in outputMats:
                logger.warning("Target skims are missing key {0}".format(outputKey))
            else:
                logger.warning("BEAM skims are missing key {0}".format(outputKey))

        if toCancel.sum() > 0:
            for measure in measures:
                if measure not in ["TRIPS, FAILURES"]:
                    key = '_'.join([path, measure, '', timePeriod])
                    try:
                        outputMats[key][toCancel] = 0.0
                    except:
                        logger.warning(
                            "Tried to cancel {0} trips for key {1} but couldn't find key".format(toCancel.sum(), key))

        if ("TOTIVT" in measures) & ("IWAIT" in measures) & ("KEYIVT" in measures):
            if toPenalize.sum() > 0:
                inputKey = '_'.join([path, 'IWAIT', '', timePeriod])
                outputMats[inputKey][toPenalize] = inputMats[inputKey][toPenalize] * (failed[toPenalize] + 1) / (
                        completed[toPenalize] + 1)
    # outputSkim.close()
    # inputSkim.close()
    else:
        logger.info(
            "No input skim for mode {0} and time period {1}, with key {2}".format(path, timePeriod, complete_key))
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


def copy_skims_for_unobserved_modes(mapping, skims):
    for fromMode, toModes in mapping.items():
        relevantSkimKeys = [key for key in skims.list_matrices() if key.startswith(fromMode + "_") & ~("TOLL" in key)]
        for skimKey in relevantSkimKeys:
            for toMode in toModes:
                toKey = skimKey.replace(fromMode + "_", toMode + "_")
                skims[toKey][:] = skims[skimKey][:]
                print("Copying values from {0} to {1}".format(skimKey, toKey))


def merge_current_omx_od_skims(all_skims_path, beam_output_dir, settings):
    skims = omx.open_file(all_skims_path, 'a')
    current_skims_path = find_produced_od_skims(beam_output_dir, "omx")
    if current_skims_path is None:
        return None
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

    discover_impossible_ods(result, skims)
    mapping = {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]}
    copy_skims_for_unobserved_modes(mapping, skims)

    order = zone_order(settings, settings['start_year'])
    zone_id = np.arange(1, len(order) + 1)

    # Generint offset
    skims.create_mapping('zone_id', zone_id, overwrite=True)

    skims.close()
    partialSkims.close()
    return current_skims_path


def trim_inaccessible_ods(settings):
    all_skims_path = os.path.join(settings['asim_local_input_folder'], "skims.omx")
    order = zone_order(settings, settings['start_year'])
    skims = omx.open_file(all_skims_path, "a")
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


def discover_impossible_ods(result, skims):
    # return (path, timePeriod), (completed, failed)
    allMats = skims.list_matrices()
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
        failed = {(a, b): c for (a, b), (c, d) in result if
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

    cur_skims = pd.read_csv(current_skims_path, dtype=schema, index_col=index_columns, na_values=["∞"])
    if os.path.exists(all_skims_path):
        all_skims = pd.read_csv(all_skims_path, dtype=schema, index_col=index_columns, na_values=["∞"])
    else:
        all_skims = cur_skims.copy()
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


    cur_skims = pd.read_csv(current_skims_path, dtype=rawInputSchema, na_values=["∞"])
    cur_skims['timePeriod'] = cur_skims['hour'].apply(hourToTimeBin)
    cur_skims.rename(columns={'tazId': 'origin'}, inplace=True)
    cur_skims['completedRequests'] = cur_skims['observations'] * (1. - cur_skims['unmatchedRequestsPercent'] / 100.)
    cur_skims = cur_skims.groupby(['timePeriod', 'reservationType', 'origin']).apply(aggregateInTimePeriod)
    if os.path.exists(all_skims_path):
        all_skims = pd.read_csv(all_skims_path, dtype=aggregatedInput, na_values=["∞"])
        all_skims.set_index(index_columns, drop=True, inplace=True)
    else:
        all_skims = cur_skims.copy()
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

def read_beam_omx_skims_to_pd_series(path_to_skims: str, measure: str) -> typing.Optional[pd.Series]:
    skims = omx.open_file(path_to_skims, 'r')
    mapping_path = path_to_skims + '.mapping'
    if os.path.exists(mapping_path):
        zone_ids = pd.read_csv(mapping_path, dtype=str)['zone_id']
    else:
        mapping_exists = 'zone_id' in skims.list_mappings()
        if not mapping_exists:
            logger.warning(f"No mapping 'zone_id' in skim file {path_to_skims}")
            return None
        zone_ids = list(skims.mapping('zone_id').keys())
    travel_time_mins = np.array(skims[measure])
    skims.close()
    index = pd.Index(zone_ids, name="origin", dtype=str)
    columns = pd.Index(zone_ids, name="destination", dtype=str)
    return pd.DataFrame(travel_time_mins, index=index, columns=columns).stack()
