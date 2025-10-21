import logging
import os
import shutil
from pathlib import Path
import glob
from typing import Tuple, Optional

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

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)
        self.required_input_data = ["usim_data"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy ATLAS input files from the production directory to the run's mutable input directory,
        recording provenance for both the source and the copied files.
        """
        input_records = []
        output_records = []
        source_dir = "pilates/atlas/atlas_input"
        start_year = settings["start_year"]
        year_specific_output_dir = os.path.join(output_dir, f"year{start_year}")

        logger.info(f"[AtlasPreprocessor] Copying files from {source_dir} to {year_specific_output_dir}")

        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if "readme" in filename.lower():
                    continue

                source_path = os.path.join(root, filename)
                relative_path = os.path.relpath(source_path, source_dir)
                dest_path = os.path.join(year_specific_output_dir, relative_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(source_path, dest_path)

                short_name = os.path.splitext(filename)[0]

                input_rec = self.provenance_tracker.record_input_file(
                    self.model_name,
                    source_path,
                    description=f"ATLAS input file: {filename}",
                    short_name=short_name,
                )
                output_rec = self.provenance_tracker.record_output_file(
                    self.model_name,
                    dest_path,
                    description=f"Mutable ATLAS input file: {filename}",
                    short_name=short_name,
                )
                input_records.append(input_rec)
                output_records.append(output_rec)

        logger.info(f"[AtlasPreprocessor] Finished copying {len(output_records)} files.")
        return RecordStore(recordList=input_records), RecordStore(recordList=output_records)

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Prepares all data needed to run ATLAS, including extracting UrbanSim outputs
        and formatting them as ATLAS inputs. Handles provenance tracking.

        Steps:
        1. Record all input files (UrbanSim HDF5, BEAM skims if needed) for provenance.
        2. Start the model run in provenance.
        3. Extract UrbanSim HDF5 tables and write them as CSVs for ATLAS.
        4. If enabled, compute accessibility using BEAM skims.
        5. Complete the model run in provenance and return a RecordStore of all input/output files.
        """
        logger.info("[AtlasPreprocessor] Starting preprocessing for ATLAS.")
        settings = self.state.full_settings
        urbansim_output_path = workspace.get_usim_mutable_data_dir()
        if self.state.is_start_year():
            urbansim_output_fname = _get_usim_datastore_fname(settings, io="input")
        else:
            urbansim_output_fname = _get_usim_datastore_fname(
                settings, io="output", year=self.state.forecast_year
            )
        urbansim_output = os.path.join(urbansim_output_path, urbansim_output_fname)

        atlas_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(),
            "year{}".format(self.state.year),
        )

        if not os.path.exists(atlas_input_path):
            os.makedirs(atlas_input_path)
            logger.info(
                f"[AtlasPreprocessor] ATLAS Input Path Created for Year {self.state.year}: {atlas_input_path}"
            )

        if self.state.year != self.state.start_year:
            old_input_path = os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year{}".format(self.state.start_year),
            )
            for f in glob.glob(os.path.join(old_input_path, "*.RData")):
                if os.path.exists(os.path.join(atlas_input_path, Path(f).name)):
                    logger.info(
                        f"[AtlasPreprocessor] Not copying file {f} to atlas input {os.path.join(atlas_input_path, Path(f).name)} because it exists"
                    )
                else:
                    logger.info(
                        f"[AtlasPreprocessor] Moving file {f} to atlas input for year {self.state.year}"
                    )
                    shutil.copyfile(f, os.path.join(atlas_input_path, Path(f).name))

        # --- Record all input files before starting model run ---
        input_records = []

        # Record UrbanSim HDF5 as input
        h5_file_record = None
        if os.path.exists(urbansim_output):
            logger.info(
                f"[AtlasPreprocessor] Recording UrbanSim HDF5 container as input: {urbansim_output}"
            )
            # Record the H5 container file
            h5_file_record = self.provenance_tracker.record_h5_input_container(
                "atlas_preprocessor",
                urbansim_output,
                description="UrbanSim output HDF5 container for Atlas input preparation",
                short_name="usim_h5_container",
            )
            if h5_file_record:
                input_records.append(h5_file_record)
        else:
            logger.warning(
                f"[AtlasPreprocessor] UrbanSim output file not found: {urbansim_output}"
            )

        # Record BEAM skims as input if needed
        beamac = settings.get("atlas_beamac", 0)
        if beamac > 0:
            beam_output_dir = workspace.get_beam_output_dir()
            expected_beam_skims_path = os.path.join(
                beam_output_dir, settings["skims_fname"]
            )
            if os.path.exists(expected_beam_skims_path):
                logger.info(
                    f"[AtlasPreprocessor] Recording BEAM skims as input: {expected_beam_skims_path}"
                )
                input_records.append(
                    self.provenance_tracker.record_input_file(
                        "atlas_preprocessor",
                        expected_beam_skims_path,
                        description="BEAM skims for Atlas accessibility calculation",
                        short_name="beam_skims_input",
                    )
                )
            else:
                logger.warning(
                    f"[AtlasPreprocessor] BEAM skims file not found: {expected_beam_skims_path}"
                )
        else:
            # FIX ATLAS ISSUE 3: Track .RData accessibility files when NOT using BEAM skims
            logger.info(
                "[AtlasPreprocessor] atlas_beamac=0, looking for .RData accessibility files"
            )
            # Look for .RData files in the year-specific input directory
            year_input_dir = os.path.join(
                workspace.get_atlas_mutable_input_dir(), f"year{self.state.year}"
            )
            if os.path.exists(year_input_dir):
                rdata_files = glob.glob(os.path.join(year_input_dir, "*.RData"))
                for rdata_file in rdata_files:
                    if "access" in os.path.basename(rdata_file).lower():
                        logger.info(
                            f"[AtlasPreprocessor] Recording accessibility .RData file: {rdata_file}"
                        )
                        input_records.append(
                            self.provenance_tracker.record_input_file(
                                "atlas_preprocessor",
                                rdata_file,
                                description="ATLAS accessibility data (RData)",
                                short_name="atlas_rdata_accessibility",
                            )
                        )

        # Now start the model run, passing all input records
        model_run_hash = self.provenance_tracker.start_model_run(
            "atlas_preprocessor",
            self.state.current_year,
            description="ATLAS preprocessing",
            inputs=RecordStore(recordList=[r for r in input_records if r is not None]),
        )
        logger.info(
            f"[AtlasPreprocessor] Started provenance model run for ATLAS preprocessing: {model_run_hash}"
        )

        # --- Write ATLAS input CSVs and record as outputs ---
        output_records = []
        table_records = []

        if not h5_file_record:
            logger.error("[AtlasPreprocessor] Cannot process HDF5 tables, container record not found.")
            return RecordStore()

        with pd.HDFStore(urbansim_output, mode="r") as data:
            
            def process_table(table_name_in_h5, output_csv_name, output_short_name, output_description):
                try:
                    table_data = data[table_name_in_h5]
                    output_csv_path = f"{atlas_input_path}/{output_csv_name}.csv"
                    table_data.to_csv(output_csv_path)

                    # Record the H5 table as a specific input
                    table_record = self.provenance_tracker.record_h5_table_input(
                        model_name="atlas_preprocessor",
                        h5_container_record=h5_file_record,
                        table_name=table_name_in_h5,
                        description=f"Source table {table_name_in_h5} for {output_csv_name}",
                        short_name=f"{output_short_name}_h5_table",
                        model_run_id=model_run_hash,
                    )
                    table_records.append(table_record)

                    # Record the output CSV, linking it to the H5 table record
                    csv_record = self.provenance_tracker.record_output_file_with_inputs(
                        model_name="atlas_preprocessor",
                        file_path=output_csv_path,
                        input_records=[table_record],
                        year=self.state.year,
                        description=output_description,
                        short_name=output_short_name,
                        model_run_id=model_run_hash,
                        state=self.state,
                    )
                    output_records.append(csv_record)
                except KeyError:
                    logger.warning(f"[AtlasPreprocessor] Table '{table_name_in_h5}' not found in HDF5 file.")
                except Exception as e:
                    logger.error(f"[AtlasPreprocessor] Error processing table {table_name_in_h5}: {e}")

            year_prefix = f"/{self.state.year}" if not self.state.is_start_year() else ""
            
            process_table(f"{year_prefix}/households", "households", "atlas_households_csv", "ATLAS households input CSV")
            process_table(f"{year_prefix}/blocks", "blocks", "atlas_blocks_csv", "ATLAS blocks input CSV")
            process_table(f"{year_prefix}/persons", "persons", "atlas_persons_csv", "ATLAS persons input CSV")
            if not self.state.is_start_year():
                 process_table(f"{year_prefix}/graveyard", "grave", "atlas_grave_csv", "ATLAS graveyard input CSV")
            process_table(f"{year_prefix}/residential_units", "residential", "atlas_residential_csv", "ATLAS residential units input CSV")
            process_table(f"{year_prefix}/jobs", "jobs", "atlas_jobs_csv", "ATLAS jobs input CSV")

            logger.info(
                f"[AtlasPreprocessor] Prepared ATLAS Year {self.state.year} input from UrbanSim output."
            )

        # --- Accessibility calculation (BEAM skims) ---
        if beamac > 0:
            logger.info(
                "[AtlasPreprocessor] Calculating accessibility using BEAM skims for ATLAS."
            )
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
                self.state.forecast_year,
            )
            logger.info("[AtlasPreprocessor] Accessibility calculation complete.")

            # Record accessibility output file
            accessibility_csv = "{}/accessibility_{}_tract.csv".format(
                atlas_input_path, self.state.forecast_year
            )
            if os.path.exists(accessibility_csv):
                output_records.append(
                    self.provenance_tracker.record_output_file(
                        "atlas_preprocessor",
                        accessibility_csv,
                        description="ATLAS accessibility tract-level CSV",
                        short_name="atlas_accessibility_csv",
                        model_run_id=model_run_hash,
                    )
                )

        self.provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=output_records
        )
        logger.info(
            f"[AtlasPreprocessor] Completed provenance model run for ATLAS preprocessing: {model_run_hash}"
        )
        return RecordStore(recordList=output_records)


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
