from __future__ import annotations

import logging
import os
import shutil
import glob
from typing import Tuple, Optional, TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

import numpy as np
import pandas as pd

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord, sanitize_artifact_key
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def _get_usim_datastore_fname(settings, io, year=None):
    # reference: asim postprocessor
    if io == "output":
        datastore_name = get_setting(settings, "urbansim.output_file_template").format(
            year=year
        )
    elif io == "input":
        region = get_setting(settings, "run.region")
        region_id = get_setting(
            settings, "urbansim.region_mappings.region_to_region_id"
        )[region]
        usim_base_fname = get_setting(settings, "urbansim.input_file_template")
        datastore_name = usim_base_fname.format(region_id=region_id)

    return datastore_name


class AtlasPreprocessor(GenericPreprocessor):
    """
    ATLAS-specific preprocessor that consolidates all preprocessing steps for the ATLAS vehicle ownership model.
    This includes extracting UrbanSim outputs, formatting them as ATLAS inputs, and (optionally) calculating accessibility
    using BEAM skims.
    """

    @staticmethod
    def expected_inputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this preprocessor expects from the workflow.
        """
        usim_input_fname = _get_usim_datastore_fname(
            settings,
            io="input" if state.is_start_year() else "output",
            year=state.forecast_year,
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "atlas_mutable_input_dir": workspace.get_atlas_mutable_input_dir(),
            "usim_datastore_h5": (
                usim_input_path if os.path.exists(usim_input_path) else None
            ),
        }

    @staticmethod
    def expected_outputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this preprocessor produces.

        Notes
        -----
        Output keys
            - ``atlas_mutable_input_dir``: ATLAS mutable input directory with
              prepared configs and data.
        Related docs
            - See `pilates/atlas/inputs.py` for the corresponding input
              descriptions used by ATLAS and downstream models.
        """
        return {
            "atlas_mutable_input_dir": workspace.get_atlas_mutable_input_dir(),
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)
        self.required_input_data = ["usim_datastore_h5"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy ATLAS input files from the production directory to the run's mutable input directory.
        """
        input_records = []
        output_records = []
        source_dir = get_setting(settings, "atlas.host_input_folder", "pilates/atlas/atlas_input")
        scenario = get_setting(settings, "atlas.scenario")
        adscen = get_setting(settings, "atlas.adscen")
        scenario_key = str(scenario).lower() if scenario else None
        vehicle_type_mapping_by_scenario = {
            "baseline": "vehicle_type_mapping_baseline.csv",
            "ess_cons": "vehicle_type_mapping_ESS_const_220_price.csv",
            "zev_mandate": "vehicle_type_mapping_evMandForced2.csv",
        }
        selected_vehicle_type_mapping = None
        if scenario_key:
            selected_vehicle_type_mapping = vehicle_type_mapping_by_scenario.get(
                scenario_key
            )
            if selected_vehicle_type_mapping is None:
                logger.warning(
                    "[AtlasPreprocessor] Unknown atlas.scenario=%s; "
                    "vehicle_type_mapping inputs will not be filtered.",
                    scenario,
                )
        if adscen and scenario and adscen != scenario:
            logger.warning(
                "[AtlasPreprocessor] atlas.adscen=%s differs from atlas.scenario=%s; "
                "using scenario for input selection.",
                adscen,
                scenario,
            )
        logger.info(
            f"[AtlasPreprocessor] Copying files from {source_dir} to {output_dir}"
        )

        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if "readme" in filename.lower():
                    continue

                source_path = os.path.join(root, filename)
                relative_path = os.path.relpath(source_path, source_dir)
                if scenario:
                    rel_parts = relative_path.split(os.sep)
                    if rel_parts[0] == "adopt":
                        if len(rel_parts) < 2 or rel_parts[1] != scenario:
                            continue
                if selected_vehicle_type_mapping:
                    if filename.startswith("vehicle_type_mapping_"):
                        if filename != selected_vehicle_type_mapping:
                            continue

                dest_path = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(source_path, dest_path)

                rel_no_ext = os.path.splitext(relative_path)[0]
                rel_key = rel_no_ext.replace(os.sep, "/")
                short_name = sanitize_artifact_key(rel_key) or rel_key

                input_records.append(
                    FileRecord(
                        file_path=source_path,
                        description=f"ATLAS input file: {filename}",
                        short_name=short_name,
                    )
                )
                output_records.append(
                    FileRecord(
                        file_path=dest_path,
                        description=f"Mutable ATLAS input file: {filename}",
                        short_name=short_name,
                    )
                )

        logger.info(
            f"[AtlasPreprocessor] Finished copying {len(output_records)} files."
        )
        return RecordStore(recordList=input_records), RecordStore(
            recordList=output_records
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Prepares all data needed to run ATLAS, including extracting UrbanSim outputs
        and formatting them as ATLAS inputs. Handles provenance tracking.

        Steps:
        1. Collect required input paths (UrbanSim HDF5, BEAM skims if needed).
        2. Extract UrbanSim HDF5 tables and write them as CSVs for ATLAS.
        3. If enabled, compute accessibility using BEAM skims.
        """
        logger.info("[AtlasPreprocessor] Starting preprocessing for ATLAS.")
        settings = self.state.full_settings

        # --- Ensure global ATLAS input files are present for every year ---
        # Source for global files (e.g., cpi.csv, RData files)
        global_source_dir = "pilates/atlas/atlas_input"
        # Destination for global files in the current run's mutable directory
        current_atlas_mutable_input_root = workspace.get_atlas_mutable_input_dir()

        # Copy global CSV files
        for f in glob.glob(os.path.join(global_source_dir, "*.csv")):
            dest_path = os.path.realpath(
                os.path.join(current_atlas_mutable_input_root, os.path.basename(f))
            )
            if not os.path.exists(dest_path):
                shutil.copy(f, dest_path)
                logger.info(
                    f"[AtlasPreprocessor] Copied global CSV file: {f} to {dest_path}"
                )
            else:
                logger.debug(
                    f"[AtlasPreprocessor] Global CSV file already exists: {dest_path}"
                )

        # Copy global RData files
        for f in glob.glob(os.path.join(global_source_dir, "*.RData")):
            dest_path = os.path.realpath(
                os.path.join(current_atlas_mutable_input_root, os.path.basename(f))
            )
            if not os.path.exists(dest_path):
                shutil.copy(f, dest_path)
                logger.info(
                    f"[AtlasPreprocessor] Copied global RData file: {f} to {dest_path}"
                )
            else:
                logger.debug(
                    f"[AtlasPreprocessor] Global RData file already exists: {dest_path}"
                )

        # --- End Global File Handling ---

        # --- Restart Logic for year-specific data ---
        if self.state.run_info_path and os.path.exists(self.state.run_info_path):
            # This is a restarted run
            previous_run_dir = os.path.dirname(self.state.run_info_path)
            logger.info(
                f"[AtlasPreprocessor] Restarted run detected. Using previous run's output path from {previous_run_dir}"
            )

            # 1. Copy base year atlas inputs from previous run
            old_base_year_input_path = os.path.join(
                previous_run_dir, "atlas", "atlas_input", f"year{self.state.start_year}"
            )
            new_base_year_input_path = os.path.join(
                workspace.get_atlas_mutable_input_dir(), f"year{self.state.start_year}"
            )
            if os.path.exists(old_base_year_input_path) and not os.path.exists(
                new_base_year_input_path
            ):
                logger.info(
                    f"[AtlasPreprocessor] Copying base year ATLAS inputs from previous run: {old_base_year_input_path}"
                )
                shutil.copytree(
                    old_base_year_input_path,
                    new_base_year_input_path,
                    dirs_exist_ok=True,
                    symlinks=True,
                )

            # 2. Set path for UrbanSim output
            urbansim_output_path = os.path.join(previous_run_dir, "urbansim", "data")
        else:
            # This is a fresh run
            urbansim_output_path = workspace.get_usim_mutable_data_dir()

        if self.state.is_start_year():
            urbansim_output_fname = _get_usim_datastore_fname(settings, io="input")
        else:
            urbansim_output_fname = _get_usim_datastore_fname(
                settings, io="output", year=self.state.main_forecast_year
            )
        urbansim_output = os.path.join(urbansim_output_path, urbansim_output_fname)

        atlas_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(),
            "year{}".format(self.state.year),
        )

        # --- Record all input files before processing ---
        input_records: List[FileRecord] = []

        # Record UrbanSim HDF5 as input
        h5_file_record = None
        if os.path.exists(urbansim_output):
            logger.info(
                f"[AtlasPreprocessor] Recording UrbanSim HDF5 container as input: {urbansim_output}"
            )
            h5_file_record = FileRecord(
                file_path=urbansim_output,
                description="UrbanSim output HDF5 container for Atlas input preparation",
                short_name="usim_h5_container",
            )
            input_records.append(h5_file_record)
        else:
            logger.warning(
                f"[AtlasPreprocessor] UrbanSim output file not found: {urbansim_output}"
            )

        # Record BEAM skims as input if needed
        beamac = settings.atlas.beamac
        if beamac > 0:
            beam_output_dir = workspace.get_beam_output_dir()
            expected_beam_skims_path = os.path.join(
                beam_output_dir, settings.shared.skims.fname
            )
            if os.path.exists(expected_beam_skims_path):
                logger.info(
                    f"[AtlasPreprocessor] Recording BEAM skims as input: {expected_beam_skims_path}"
                )
                input_records.append(
                    FileRecord(
                        file_path=expected_beam_skims_path,
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
            # Look for .RData files in the root input directory
            year_input_dir = workspace.get_atlas_mutable_input_dir()
            if os.path.exists(year_input_dir):
                rdata_files = glob.glob(os.path.join(year_input_dir, "*.RData"))
                for rdata_file in rdata_files:
                    if "access" in os.path.basename(rdata_file).lower():
                        logger.info(
                            f"[AtlasPreprocessor] Recording accessibility .RData file: {rdata_file}"
                        )
                        input_records.append(
                            FileRecord(
                                file_path=rdata_file,
                                description="ATLAS accessibility data (RData)",
                                short_name="atlas_rdata_accessibility",
                            )
                        )

        # --- Write ATLAS input CSVs and record as outputs ---
        output_records = []
        if not h5_file_record:
            logger.error(
                "[AtlasPreprocessor] Cannot process HDF5 tables, container record not found."
            )
            return RecordStore()

        with pd.HDFStore(urbansim_output, mode="r") as data:

            def process_table(
                table_name_in_h5, output_csv_name, output_short_name, output_description
            ):
                try:
                    table_data = data[table_name_in_h5]
                    output_csv_path = f"{atlas_input_path}/{output_csv_name}.csv"
                    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                    table_data.to_csv(output_csv_path)

                    output_records.append(
                        FileRecord(
                            file_path=output_csv_path,
                            year=self.state.year,
                            description=output_description,
                            short_name=output_short_name,
                        )
                    )
                except KeyError:
                    logger.warning(
                        f"[AtlasPreprocessor] Table '{table_name_in_h5}' not found in HDF5 file."
                    )
                except Exception as e:
                    logger.error(
                        f"[AtlasPreprocessor] Error processing table {table_name_in_h5}: {e}"
                    )

            year_prefix = (
                f"/{self.state.year}" if not self.state.is_start_year() else ""
            )

            process_table(
                f"{year_prefix}/households",
                "households",
                "atlas_households_csv",
                "ATLAS households input CSV",
            )
            process_table(
                f"{year_prefix}/blocks",
                "blocks",
                "atlas_blocks_csv",
                "ATLAS blocks input CSV",
            )
            process_table(
                f"{year_prefix}/persons",
                "persons",
                "atlas_persons_csv",
                "ATLAS persons input CSV",
            )
            if not self.state.is_start_year():
                process_table(
                    f"{year_prefix}/graveyard",
                    "grave",
                    "atlas_grave_csv",
                    "ATLAS graveyard input CSV",
                )
            process_table(
                f"{year_prefix}/residential_units",
                "residential",
                "atlas_residential_csv",
                "ATLAS residential units input CSV",
            )
            process_table(
                f"{year_prefix}/jobs", "jobs", "atlas_jobs_csv", "ATLAS jobs input CSV"
            )

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
                workspace,
            )
            logger.info("[AtlasPreprocessor] Accessibility calculation complete.")

            # Record accessibility output file
            accessibility_csv = "{}/accessibility_{}_tract.csv".format(
                atlas_input_path, self.state.forecast_year
            )
            if os.path.exists(accessibility_csv):
                output_records.append(
                    FileRecord(
                        file_path=accessibility_csv,
                        description="ATLAS accessibility tract-level CSV",
                        short_name="atlas_accessibility_csv",
                    )
                )

        logger.info("[AtlasPreprocessor] ATLAS preprocessing complete.")
        return RecordStore(recordList=output_records)


def compute_accessibility(
    path_list, measure_list, settings, year, workspace, threshold=500
):
    # set where to put atlas csv inputs (processed from urbansim outputs)
    atlas_input_path = os.path.join(
        workspace.get_atlas_mutable_input_dir(), f"year{year}"
    )
    os.makedirs(atlas_input_path, exist_ok=True)

    # --- Get Canonical Zone Information ---
    from pilates.utils.zone_utils import (
        load_canonical_zones,
        get_block_to_zone_mapping,
    )

    canonical_zones_df = load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.values

    # for each OD, compute minimum time taken by public transit
    # inf means no public transit available; unit = minute
    ODmatrix_df = _get_time_ODmatrix(
        settings, path_list, measure_list, threshold, workspace, canonical_order
    )

    # assign values = 1 if time taken by public transit <= 30min; 0 if not
    ODmatrix = ODmatrix_df <= 30

    # read and format geoid_to_zoneid mapping list
    mapping = get_block_to_zone_mapping(settings, year, workspace)

    # read OD matrix size (i.e., range of zone_id)
    zone_count = ODmatrix.shape[0]

    # read in jobs data (keep low_memory=False to solve dtypeerror)
    jobs = pd.read_csv(
        "{}/jobs.csv".format(atlas_input_path),
        low_memory=False,
    )

    # map jobs geoid to zone id in OD matrix
    jobs["zone_id"] = jobs["block_id"].astype(str).map(mapping)

    # Drop jobs that couldn't be mapped to a zone
    jobs.dropna(subset=["zone_id"], inplace=True)

    # count number of jobs for each block_id
    jobs_vector = (
        jobs.groupby("block_id")
        .agg({"job_id": "size", "zone_id": "max"})
        .rename(columns={"job_id": "access_sum"})
    )

    # average # of jobs per block for each taz
    jobs_vector = jobs_vector.groupby("zone_id").agg({"access_sum": "mean"})

    # make sure every zone id has a row in jobs_vector
    jobs_vector = jobs_vector.reindex(canonical_order, fill_value=0)

    # multiply OD matrix (o*d) with jobs vector (d*1)
    # to get number of jobs accessible by public transit within 30min
    accessibility = np.matmul(ODmatrix, jobs_vector)
    accessibility.index.name = "zone_id"

    # # calculate taz-level zscore
    # accessibility['access_zscore'] = (accessibility['access_sum'] - accessibility['access_sum'].mean())/accessibility['access_sum'].std()

    # # write taz-level accessibility data
    # accessibility.to_csv('{}/accessibility_{}_taz.csv'.format(atlas_input_path, year))

    # read in taz_to_tract conversion matrix (1454*1588)
    taz_to_tract = pd.read_csv(
        "{}/taz_to_tract_{}.csv".format(
            settings.atlas.host_input_folder, settings.run.region
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


def _get_time_ODmatrix(
    settings, path_list, measure_list, threshold, workspace, canonical_order
):
    # open skims file
    import openmatrix as omx

    skims_dir = workspace.get_asim_mutable_data_dir()
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

    skims.close()
    return pd.DataFrame(ODmatrix, index=canonical_order, columns=canonical_order)
