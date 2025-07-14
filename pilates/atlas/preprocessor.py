import logging
import os
import shutil
from pathlib import Path
import glob
from typing import Tuple

import numpy as np
import pandas as pd

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


def _get_usim_datastore_fname(settings, io, year=None):
    # reference: asim postprocessor
    if io == "output":
        datastore_name = settings["usim_formattable_output_file_name"].format(year=year)
    elif io == "input":
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        usim_base_fname = settings["usim_formattable_input_file_name"]
        datastore_name = usim_base_fname.format(region_id=region_id)

    return datastore_name


class AtlasPreprocessor(GenericPreprocessor):
    """
    ATLAS-specific preprocessor that consolidates all preprocessing steps for the ATLAS vehicle ownership model.
    This includes extracting UrbanSim outputs, formatting them as ATLAS inputs, and (optionally) calculating accessibility
    using BEAM skims. All provenance tracking for input files is handled here.
    """

    def __init__(self):
        super().__init__()
        self.required_input_data = ["usim_data"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
        provenance_tracker,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        ATLAS does not require copying input files to a mutable location at initialization.
        Returns empty RecordStores.
        """
        return RecordStore(), RecordStore()

    def preprocess(
        self,
        state,
        workspace,
        provenance_tracker: FileProvenanceTracker,
    ) -> RecordStore:
        """
        Prepares all data needed to run ATLAS, including extracting UrbanSim outputs
        and formatting them as ATLAS inputs. Handles provenance tracking.

        Steps:
        1. Record all input files (UrbanSim HDF5, BEAM skims if needed) for provenance.
        2. Start the model run in provenance.
        3. Extract UrbanSim HDF5 tables and write them as CSVs for ATLAS.
        4. If enabled, compute accessibility using BEAM skims.
        5. Complete the model run in provenance and return an empty RecordStore (no outputs at this stage).
        """
        logger.info("[AtlasPreprocessor] Starting preprocessing for ATLAS.")
        settings = state.full_settings
        # set where to find urbansim output
        urbansim_output_path = workspace.get_usim_mutable_data_dir()
        if state.is_start_year():
            urbansim_output_fname = _get_usim_datastore_fname(settings, io="input")
        else:
            urbansim_output_fname = _get_usim_datastore_fname(
                settings, io="output", year=state.forecast_year
            )
        urbansim_output = os.path.join(urbansim_output_path, urbansim_output_fname)

        # set where to put atlas csv inputs (processed from urbansim outputs)
        atlas_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(),
            "year{}".format(state.year),
        )

        # if atlas input path does not exist, create one
        if not os.path.exists(atlas_input_path):
            os.makedirs(atlas_input_path)
            logger.info(f"[AtlasPreprocessor] ATLAS Input Path Created for Year {state.year}: {atlas_input_path}")

        if state.year != state.start_year:
            old_input_path = os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year{}".format(state.start_year),
            )
            for f in glob.glob(os.path.join(old_input_path, "*.RData")):
                if os.path.exists(os.path.join(atlas_input_path, Path(f).name)):
                    logger.info(
                        f"[AtlasPreprocessor] Not copying file {f} to atlas input {os.path.join(atlas_input_path, Path(f).name)} because it exists"
                    )
                else:
                    logger.info(
                        f"[AtlasPreprocessor] Moving file {f} to atlas input for year {state.year}"
                    )
                    shutil.copyfile(f, os.path.join(atlas_input_path, Path(f).name))

        # --- Record all input files before starting model run ---
        input_records = []

        # Record UrbanSim HDF5 as input
        if os.path.exists(urbansim_output):
            logger.info(f"[AtlasPreprocessor] Recording UrbanSim HDF5 as input: {urbansim_output}")
            input_records.append(
                provenance_tracker.record_input_file(
                    "atlas_preprocessor",
                    urbansim_output,
                    description="UrbanSim output for Atlas input preparation",
                )
            )
        else:
            logger.warning(f"[AtlasPreprocessor] UrbanSim output file not found: {urbansim_output}")

        # Record BEAM skims as input if needed
        beamac = settings.get("atlas_beamac", 0)
        if beamac > 0:
            beam_output_dir = workspace.get_beam_output_dir()
            expected_beam_skims_path = os.path.join(
                beam_output_dir, settings["skims_fname"]
            )
            if os.path.exists(expected_beam_skims_path):
                logger.info(f"[AtlasPreprocessor] Recording BEAM skims as input: {expected_beam_skims_path}")
                input_records.append(
                    provenance_tracker.record_input_file(
                        "atlas_preprocessor",
                        expected_beam_skims_path,
                        description="BEAM skims for Atlas accessibility calculation",
                    )
                )
            else:
                logger.warning(f"[AtlasPreprocessor] BEAM skims file not found: {expected_beam_skims_path}")

        # Now start the model run, passing all input records
        model_run_hash = provenance_tracker.start_model_run(
            "atlas_preprocessor",
            state.current_year,
            description="ATLAS preprocessing",
            inputs=RecordStore(recordList=[r for r in input_records if r is not None]),
        )
        logger.info(f"[AtlasPreprocessor] Started provenance model run for ATLAS preprocessing: {model_run_hash}")

        # read urbansim h5 outputs
        with pd.HDFStore(urbansim_output, mode="r") as data:
            if not state.is_start_year():
                try:
                    # prepare households atlas input
                    households = data["/{}/households".format(state.year)]
                    households.to_csv("{}/households.csv".format(atlas_input_path))

                    # prepare blocks atlas input
                    blocks = data["/{}/blocks".format(state.year)]
                    blocks.to_csv("{}/blocks.csv".format(atlas_input_path))

                    # prepare persons atlas input
                    persons = data["/{}/persons".format(state.year)]
                    persons.to_csv("{}/persons.csv".format(atlas_input_path))

                    # prepare dead persons atlas input (RIP)
                    persons = data["/{}/graveyard".format(state.year)]
                    persons.to_csv("{}/grave.csv".format(atlas_input_path))

                    # prepare residential unit atlas input
                    residential_units = data["/{}/residential_units".format(state.year)]
                    residential_units.to_csv("{}/residential.csv".format(atlas_input_path))

                    # prepare jobs atlas input
                    jobs = data["/{}/jobs".format(state.year)]
                    jobs.to_csv("{}/jobs.csv".format(atlas_input_path))

                    logger.info(
                        f"[AtlasPreprocessor] Prepared ATLAS Year {state.year} input from UrbanSim output."
                    )

                except Exception as e:
                    logger.error(
                        f"[AtlasPreprocessor] UrbanSim Year {state.year} Output Was Not Loaded Correctly by ATLAS: {e}"
                    )

            else:
                try:
                    # prepare households atlas input
                    households = data["/households"]
                    households.to_csv("{}/households.csv".format(atlas_input_path))

                    # prepare blocks atlas input
                    blocks = data["/blocks"]
                    blocks.to_csv("{}/blocks.csv".format(atlas_input_path))

                    # prepare persons atlas input
                    persons = data["/persons"]
                    persons.to_csv("{}/persons.csv".format(atlas_input_path))

                    # prepare residential unit atlas input
                    residential_units = data["/residential_units"]
                    residential_units.to_csv("{}/residential.csv".format(atlas_input_path))

                    # prepare jobs atlas input
                    jobs = data["/jobs"]
                    jobs.to_csv("{}/jobs.csv".format(atlas_input_path))

                    logger.info(
                        f"[AtlasPreprocessor] Prepared ATLAS Year {state.year} input from UrbanSim output."
                    )

                except Exception as e:
                    logger.error(
                        f"[AtlasPreprocessor] UrbanSim Year {state.year} Output Was Not Loaded Correctly by ATLAS: {e}"
                    )

        # --- Accessibility calculation (BEAM skims) ---
        if beamac > 0:
            logger.info("[AtlasPreprocessor] Calculating accessibility using BEAM skims for ATLAS.")
            path_list = [
                "WLK_COM_WLK",
                "WLK_EXP_WLK",
                "WLK_HVY_WLK",
                "WLK_LOC_WLK",
                "WLK_LRF_WLK",
            ]
            measure_list = ["WACC", "IWAIT", "XWAIT", "TOTIVT", "WEGR"]
            # compute_accessibility expects (path_list, measure_list, settings, year)
            compute_accessibility(
                path_list,
                measure_list,
                settings,
                state.forecast_year,
            )
            logger.info("[AtlasPreprocessor] Accessibility calculation complete.")

        output_records = RecordStore()

        provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=output_records.all_records()
        )
        logger.info(f"[AtlasPreprocessor] Completed provenance model run for ATLAS preprocessing: {model_run_hash}")
        return RecordStore(recordList=output_records.all_records())


def compute_accessibility(path_list, measure_list, settings, year, threshold=500):
    # set where to put atlas csv inputs (processed from urbansim outputs)
    atlas_input_path = settings["atlas_host_input_folder"] + "/year{}".format(year)

    # for each OD, compute minimum time taken by public transit
    # inf means no public transit available; unit = minute
    ODmatrix = _get_time_ODmatrix(settings, path_list, measure_list, threshold)

    # assign values = 1 if time taken by public transit <= 30min; 0 if not
    ODmatrix = ODmatrix <= 30

    # read and format geoid_to_zoneid mapping list
    mapping = pd.read_csv(
        "pilates/utils/data/{}/beam/geoid_to_zone.csv".format(settings["region"])
    )
    mapping.index = mapping["GEOID"]
    mapping = mapping["zone_id"].to_dict()

    # read OD matrix size (i.e., range of zone_id)
    zone_count = ODmatrix.shape[0]

    # read in jobs data (keep low_memory=False to solve dtypeerror)
    jobs = pd.read_csv(
        "{}/year{}/jobs.csv".format(settings["atlas_host_input_folder"], year),
        low_memory=False,
    )

    # map jobs geoid to zone id in OD matrix
    jobs["zone_id"] = jobs["block_id"].map(mapping)

    # count number of jobs for each block_id
    jobs_vector = (
        jobs.groupby("block_id")
        .agg({"job_id": "size", "zone_id": "max"})
        .rename(columns={"job_id": "access_sum"})
    )

    # average # of jobs per block for each taz
    jobs_vector = jobs_vector.groupby("zone_id").agg({"access_sum": "mean"})

    # make sure every zone id has a row in jobs_vector
    jobs_vector = jobs_vector.reindex(list(range(1, zone_count + 1)), fill_value=0)

    # multiply OD matrix (o*d) with jobs vector (d*1)
    # to get number of jobs accessible by public transit within 30min
    accessibility = np.matmul(ODmatrix, jobs_vector)
    accessibility.index.name = "zone_id"
    accessibility.index = accessibility.index + 1

    # # calculate taz-level zscore
    # accessibility['access_zscore'] = (accessibility['access_sum'] - accessibility['access_sum'].mean())/accessibility['access_sum'].std()

    # # write taz-level accessibility data
    # accessibility.to_csv('{}/accessibility_{}_taz.csv'.format(atlas_input_path, year))

    # read in taz_to_tract conversion matrix (1454*1588)
    taz_to_tract = pd.read_csv(
        "{}/taz_to_tract_{}.csv".format(
            settings["atlas_host_input_folder"], settings["region"]
        ),
        index_col=0,
    )

    # convert taz- to tract-level accessibility data
    accessibility_tract = np.matmul(
        np.transpose(accessibility), np.array(taz_to_tract.values)
    )
    accessibility_tract.columns = taz_to_tract.columns
    accessibility_tract = accessibility_tract.transpose()

    # calculate tract-level zscore
    accessibility_tract["access_zscore"] = (
        accessibility_tract["access_sum"] - accessibility_tract["access_sum"].mean()
    ) / accessibility_tract["access_sum"].std()

    # format before writing
    accessibility_tract.index.name = "tract"
    accessibility_tract["urban_cbsa"] = 1  ## all sfbay tracts belong to cbsa

    # write tract-level accessibility data
    accessibility_tract.to_csv(
        "{}/accessibility_{}_tract.csv".format(atlas_input_path, year)
    )


def _get_time_ODmatrix(settings, path_list, measure_list, threshold):
    # open skims file
    import openmatrix as omx
    skims_dir = settings["asim_local_mutable_data_folder"]
    skims = omx.open_file(os.path.join(skims_dir, "skims.omx"), mode="r")

    # find the path with minimum time for each o-d
    ODmatrix = np.ones(skims.shape()) * np.inf

    for path in path_list:
        tmp_path = np.zeros(skims.shape())

        # sum total time taken for each specific path
        for measure in measure_list:
            tmp_measure = np.zeros(skims.shape())

            # extract data from skims.omx
            key = "{}_{}__AM".format(path, measure)
            try:
                tmp_measure = np.array(skims[key])
            except:
                tmp_measure = np.zeros(skims.shape())
                # logger.error('{} not found in skims'.format(key))

            # sum up time taken for each path
            tmp_path = tmp_path + tmp_measure

        # filter out paths with unreasonable TOTIVT (no available transit)
        tmp_path[tmp_path <= threshold] = 1e6

        # find the path with minimum total time taken
        ODmatrix = np.minimum(ODmatrix, tmp_path)

        # divide by 100 to get minute values before returning
    ODmatrix = ODmatrix / 100

    return ODmatrix
