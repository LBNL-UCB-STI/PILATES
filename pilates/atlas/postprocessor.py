from typing import Any, Optional
import logging
import os

import numpy as np
import pandas as pd
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def atlas_update_h5_vehicle(
    settings: dict,
    output_year: int,
    h5_file_path: str,
    household_v_csv_path: str,
):
    """Update the UrbanSim HDF5 file with vehicle ownership data from ATLAS.

    Reads vehicle ownership data from the given CSV file and updates the 'cars'
    and 'hh_cars' columns in the 'households' table within the specified HDF5 file.

    Args:
        settings (dict): The simulation settings.
        output_year (int): The forecast year being processed.
        h5_file_path (str): The absolute path to the UrbanSim HDF5 file to update.
        household_v_csv_path (str): The absolute path to the ATLAS householdv CSV file.
    """
    if not os.path.exists(h5_file_path) or not os.path.exists(household_v_csv_path):
        logger.error(
            f"[AtlasPostprocessor] Missing input files for H5 update. H5: {h5_file_path}, CSV: {household_v_csv_path}"
        )
        return

    logger.info(f"ATLAS is updating urbansim outputs for Year {output_year}")

    # Read and format ATLAS vehicle ownership output
    df = pd.read_csv(household_v_csv_path)
    df = (
        df.rename(columns={"nvehicles": "cars"})
        .set_index("household_id")
        .sort_index(ascending=True)
    )
    df["hh_cars"] = pd.cut(
        df["cars"], bins=[-0.5, 0.5, 1.5, np.inf], labels=["none", "one", "two or more"]
    )

    logger.info(f"Writing updated household vehicle info to h5 file {h5_file_path}")

    # Read original h5 files and update
    with pd.HDFStore(h5_file_path, mode="r+") as h5:
        warm_start = settings.get("warm_start", False)
        key = f"/{output_year}/households" if not warm_start else "households"

        try:
            olddf = h5[key]
        except KeyError:
            logger.error(f"Table '{key}' not found in HDF5 file: {h5_file_path}")
            return

        olddf.index = olddf.index.astype(int)
        olddf = olddf.reindex(df.index.astype(int))

        if olddf.shape[0] != df.shape[0]:
            logger.error("ATLAS household_id mismatch found - NOT updating h5 datastore")
        else:
            olddf["cars"] = df["cars"].values
            olddf["hh_cars"] = df["hh_cars"].values
            for col in olddf.columns:
                if olddf[col].dtype.name == "category":
                    logger.info(f"Converting column {col} from category to str")
                    olddf[col] = olddf[col].astype(str)
            h5[key] = olddf
            logger.info(f"ATLAS update h5 datastore table {key} - done")


def atlas_add_vehileTypeId(
    settings: dict,
    output_year: int,
    vehicles_csv_path: str,
    output_vehicles2_csv_path: str,
):
    """Add a 'vehicleTypeId' column to the ATLAS vehicles CSV.

    Reads the main ATLAS vehicles output, creates a composite 'vehicleTypeId'
    column for BEAM, and writes the result to a new 'vehicles2_{year}.csv' file.

    Args:
        settings (dict): The simulation settings.
        output_year (int): The forecast year being processed.
        vehicles_csv_path (str): Path to the input ATLAS vehicles CSV file.
        output_vehicles2_csv_path (str): Path to write the output CSV file to.
    """
    if not os.path.exists(vehicles_csv_path):
        logger.error(
            f"[AtlasPostprocessor] Missing input file for vehicleTypeId addition: {vehicles_csv_path}"
        )
        return

    # Read original ATLAS output "vehicles_*.csv" as dataframe
    df = pd.read_csv(vehicles_csv_path)

    # ATLAS:v1.0.6 can generate continuous modelyear
    df["modelyear"] = df["modelyear"].astype(int)

    # Add "vehicleTypeId" column in dataframe for BEAM
    # For prior-2015-model vehicles, vehicleTypeId is *_*_2015
    df["vehicleTypeId"] = (
        df[["bodytype", "pred_power", "modelyear"]].astype(str).agg("_".join, axis=1)
    )
    df.loc[df["modelyear"] < 2015, "vehicleTypeId"] = (
        df.loc[df["modelyear"] < 2015, ["bodytype", "pred_power"]]
        .astype(str)
        .agg("_".join, axis=1)
        + "_2015"
    )

    # Write to a new file vehicles2_*.csv
    df.to_csv(output_vehicles2_csv_path, index=False)


def get_usim_datastore_fname(settings, io, year=None):
    """Construct the UrbanSim datastore filename based on settings.

    Args:
        settings (dict): The simulation settings.
        io (str): The direction of I/O, either 'input' or 'output'.
        year (int, optional): The simulation year. Required if io is 'output'.

    Returns:
        str: The formatted UrbanSim datastore filename.
    """
    if io == "output":
        datastore_name = get_setting(settings, "urbansim.output_file_template").format(year=year)
    elif io == "input":
        region = get_setting(settings, "run.region")
        region_id = settings["region_to_region_id"][region]
        usim_base_fname = get_setting(settings, "urbansim.input_file_template")
        datastore_name = usim_base_fname.format(region_id=region_id)
    else:
        raise ValueError(
            f"Invalid io parameter: {io}. Must be either 'input' or 'output'"
        )

    return datastore_name


class AtlasPostprocessor(GenericPostprocessor):
    """
    ATLAS-specific postprocessor that consolidates all postprocessing steps for the ATLAS vehicle ownership model.
    This includes updating UrbanSim HDF5 with new vehicle ownership and adding vehicleTypeId to ATLAS vehicle outputs.
    All provenance tracking for output files should be handled here.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)

    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        runInfo: Optional[ModelRunInfo] = None,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocess ATLAS outputs: update UrbanSim HDF5 with new vehicle ownership,
        and add vehicleTypeId to ATLAS vehicle outputs. Handles provenance tracking.

        Args:
            raw_outputs (RecordStore): The raw outputs from the ATLAS model run.
            workspace (Workspace): The workspace object for path management.
            runInfo (Optional[ModelRunInfo]): Metadata about the model run. Not used by this processor.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: A RecordStore containing records for the generated output files.
        """
        logger.info(
            "[AtlasPostprocessor] Starting postprocessing for ATLAS for year %s",
            self.state.current_year,
        )

        if model_run_hash is None:
            model_run_hash = self.provenance_tracker.start_model_run(
                "atlas_postprocessor",
                self.state.current_year,
                self.state.current_inner_iter,
                description="Post-processing ATLAS outputs",
                inputs=raw_outputs,
            )

        settings = self.state.full_settings
        output_year = self.state.forecast_year
        output_records = []

        # --- HDF5 Update and Provenance ---
        usim_h5_path = workspace.get_usim_mutable_data_dir()
        usim_h5_fname = get_usim_datastore_fname(settings, io="output", year=output_year)
        usim_h5_file = os.path.join(usim_h5_path, usim_h5_fname)
        atlas_hh_file = os.path.join(workspace.get_atlas_output_dir(), f"householdv_{output_year}.csv")

        if os.path.exists(usim_h5_file) and os.path.exists(atlas_hh_file):
            # 1. Record H5 container and householdv.csv as inputs
            h5_container_input_record = self.provenance_tracker.record_h5_input_container(
                "atlas_postprocessor",
                usim_h5_file,
                description=f"UrbanSim HDF5 before ATLAS update for year {output_year}",
                short_name="usim_h5_before_update",
                model_run_id=model_run_hash,
            )
            atlas_hh_input_record = self.provenance_tracker.record_input_file(
                "atlas_postprocessor",
                atlas_hh_file,
                description=f"ATLAS household vehicle counts for year {output_year}",
                short_name="atlas_householdv_input",
                model_run_id=model_run_hash,
            )

            # 2. Define the table to be updated
            table_name = f"/{output_year}/households" if not settings.get("warm_start") else "households"

            # 3. Record the source table as an input
            source_table_record = self.provenance_tracker.record_h5_table_input(
                "atlas_postprocessor",
                h5_container_record=h5_container_input_record,
                table_name=table_name,
                description=f"Source households table before update",
                short_name="households_table_before_update",
                model_run_id=model_run_hash,
            )

            # 4. Perform the update
            atlas_update_h5_vehicle(settings, output_year, usim_h5_file, atlas_hh_file)
            logger.info("[AtlasPostprocessor] Updated UrbanSim HDF5 with new vehicle ownership.")

            # 5. Record the updated table as an output
            updated_table_record = self.provenance_tracker.record_h5_table_output(
                "atlas_postprocessor",
                h5_container_record=h5_container_input_record, # The container path is the same
                table_name=table_name,
                input_records=[source_table_record, atlas_hh_input_record],
                description=f"Updated households table after ATLAS run",
                short_name="households_table_after_update",
                model_run_id=model_run_hash,
            )
            output_records.append(updated_table_record)

        # --- vehicleTypeId addition and Provenance ---
        atlas_veh_file = os.path.join(workspace.get_atlas_output_dir(), f"vehicles_{output_year}.csv")
        atlas_veh2_file = os.path.join(workspace.get_atlas_output_dir(), f"vehicles2_{output_year}.csv")

        if os.path.exists(atlas_veh_file):
            atlas_add_vehileTypeId(settings, output_year, atlas_veh_file, atlas_veh2_file)
            logger.info("[AtlasPostprocessor] Added vehicleTypeId to ATLAS vehicle outputs.")

            atlas_veh_input_record = self.provenance_tracker.record_input_file(
                "atlas_postprocessor",
                atlas_veh_file,
                description=f"ATLAS vehicles CSV before vehicleTypeId addition",
                short_name="atlas_vehicles_input",
                model_run_id=model_run_hash,
            )

            if os.path.exists(atlas_veh2_file):
                atlas_veh2_output_record = self.provenance_tracker.record_output_file_with_inputs(
                    "atlas_postprocessor",
                    atlas_veh2_file,
                    input_records=[atlas_veh_input_record],
                    year=output_year,
                    description=f"ATLAS vehicles2 CSV with vehicleTypeId",
                    short_name="atlas_vehicles2_output",
                    model_run_id=model_run_hash,
                )
                output_records.append(atlas_veh2_output_record)

        self.provenance_tracker.complete_model_run(
            model_run_hash, status="completed", output_records=output_records
        )
        logger.info(
            "[AtlasPostprocessor] Completed provenance model run for ATLAS postprocessing: %s",
            model_run_hash,
        )
        return RecordStore(recordList=output_records)
