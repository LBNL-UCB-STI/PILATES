from typing import Any
import logging
import os

import numpy as np
import pandas as pd
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


def atlas_update_h5_vehicle(
    settings, output_year, state: WorkflowState, warm_start=False
):
    # use atlas outputs in year provided and update "cars" & "hh_cars"
    # columns in urbansim h5 files
    logger.info("ATLAS is updating urbansim outputs for Year {}".format(output_year))

    # read and format atlas vehicle ownership output
    atlas_output_path = os.path.join(
        state.full_path, settings["atlas_host_output_folder"]
    )  # 'pilates/atlas/atlas_output'  #
    fname = "householdv_{}.csv".format(output_year)
    df = pd.read_csv(os.path.join(atlas_output_path, fname))
    df = (
        df.rename(columns={"nvehicles": "cars"})
        .set_index("household_id")
        .sort_index(ascending=True)
    )
    df["hh_cars"] = pd.cut(
        df["cars"], bins=[-0.5, 0.5, 1.5, np.inf], labels=["none", "one", "two or more"]
    )

    # set which h5 file to update
    h5path = os.path.join(state.full_path, settings["usim_local_mutable_data_folder"])
    if warm_start:
        h5fname = get_usim_datastore_fname(settings, io="input")
    else:
        h5fname = get_usim_datastore_fname(settings, io="output", year=output_year)

    logger.info("Writing updated household vehicle info to h5 file {0}".format(h5fname))

    # read original h5 files
    with pd.HDFStore(os.path.join(h5path, h5fname), mode="r+") as h5:

        # if in main loop, update "model_data_*.h5", which has three layers ({$year}/households/cars)
        if not warm_start:
            key = "/{}/households".format(output_year)
        # if in warm start, update "custom_mpo_***.h5", which has two layers (households/cars)
        else:
            key = "households"

        olddf = h5[key]
        olddf.index = olddf.index.astype(int)
        olddf = olddf.reindex(df.index.astype(int))

        if olddf.shape[0] != df.shape[0]:
            logger.error("ATLAS household_id mismatch found - NOT update h5 datastore")
        else:
            olddf["cars"] = df["cars"].values
            olddf["hh_cars"] = df["hh_cars"].values
            for col in olddf.columns:
                if olddf[col].dtype.name == "category":
                    logger.info(
                        "Converting column {0} from category to str".format(col)
                    )
                    olddf[col] = olddf[col].astype(str)
            h5[key] = olddf
            logger.info("ATLAS update h5 datastore table {0} - done".format(key))


def atlas_add_vehileTypeId(settings, output_year, state):
    # add a "vehicleTypeId" column in atlas output vehicles_{$year}.csv,
    # write as vehicles2_{$year}.csv
    # which will be read by beam preprocessor
    # vehicleTypeId = conc "bodytype"-"vintage_category"-"pred_power"

    atlas_output_path = os.path.join(
        state.full_path, settings["atlas_host_output_folder"]
    )
    fname = "vehicles_{}.csv".format(output_year)

    # read original atlas output "vehicles_*.csv" as dataframe
    df = pd.read_csv(os.path.join(atlas_output_path, fname))

    # atlas:v1.0.6 can generate continuous modelyear
    df["modelyear"] = df["modelyear"].astype(int)

    # add "vehicleTypeId" column in dataframe for BEAM
    # for prior-2015-model vehicles, vehicleTypeId is *_*_2015
    df["vehicleTypeId"] = (
        df[["bodytype", "pred_power", "modelyear"]].astype(str).agg("_".join, axis=1)
    )
    df.loc[df["modelyear"] < 2015, "vehicleTypeId"] = (
        df.loc[df["modelyear"] < 2015, ["bodytype", "pred_power"]]
        .astype(str)
        .agg("_".join, axis=1)
        + "_2015"
    )

    # write to a new file vehicles2_*.csv
    # because original file cannot be overwritten (root-owned)
    # may revise later
    df.to_csv(
        os.path.join(atlas_output_path, "vehicles2_{}.csv".format(output_year)),
        index=False,
    )


def get_usim_datastore_fname(settings, io, year=None):
    if io == "output":
        datastore_name = settings["usim_formattable_output_file_name"].format(year=year)
    elif io == "input":
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        usim_base_fname = settings["usim_formattable_input_file_name"]
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

    def postprocess(
        self,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
        model_run_hash: str,
    ) -> RecordStore:
        """
        Postprocess ATLAS outputs: update UrbanSim HDF5 with new vehicle ownership,
        and add vehicleTypeId to ATLAS vehicle outputs. Handles provenance tracking.

        Steps:
        1. Start the model run in provenance (should already be started by caller).
        2. Update UrbanSim HDF5 with new vehicle ownership (atlas_update_h5_vehicle).
        3. Add vehicleTypeId to ATLAS vehicle outputs (atlas_add_vehileTypeId).
        4. Complete the model run in provenance and return an empty RecordStore (no outputs at this stage).
        """
        logger.info(
            "[AtlasPostprocessor] Starting postprocessing for ATLAS for year %s",
            state.current_year,
        )
        settings = state.full_settings
        output_year = state.forecast_year

        # --- Record input files ---
        # UrbanSim HDF5 file (input)
        usim_h5_path = os.path.join(
            state.full_path, settings["usim_local_mutable_data_folder"]
        )
        usim_h5_fname = get_usim_datastore_fname(
            settings, io="output", year=output_year
        )
        usim_h5_file = os.path.join(usim_h5_path, usim_h5_fname)
        usim_input_record = None
        if os.path.exists(usim_h5_file):
            usim_input_record = provenance_tracker.record_input_file(
                "atlas_postprocessor",
                usim_h5_file,
                description=f"UrbanSim HDF5 before ATLAS vehicle update for year {output_year}",
                model_run_id=model_run_hash,
            )

        # ATLAS output CSV (input)
        atlas_output_path = os.path.join(
            state.full_path, settings["atlas_host_output_folder"]
        )
        atlas_veh_file = os.path.join(atlas_output_path, f"vehicles_{output_year}.csv")
        atlas_veh_input_record = None
        if os.path.exists(atlas_veh_file):
            atlas_veh_input_record = provenance_tracker.record_input_file(
                "atlas_postprocessor",
                atlas_veh_file,
                description=f"ATLAS vehicles CSV before vehicleTypeId for year {output_year}",
                model_run_id=model_run_hash,
            )

        # --- Perform postprocessing steps ---
        atlas_update_h5_vehicle(settings, output_year, state)
        logger.info(
            "[AtlasPostprocessor] Updated UrbanSim HDF5 with new vehicle ownership for year %s",
            output_year,
        )
        atlas_add_vehileTypeId(settings, output_year, state)
        logger.info(
            "[AtlasPostprocessor] Added vehicleTypeId to ATLAS vehicle outputs for year %s",
            output_year,
        )

        # --- Record output files ---
        # UrbanSim HDF5 file (output)
        usim_output_record = None
        if os.path.exists(usim_h5_file):
            usim_output_record = provenance_tracker.record_output_file(
                "atlas_postprocessor",
                usim_h5_file,
                description=f"UrbanSim HDF5 after ATLAS vehicle update for year {output_year}",
                model_run_id=model_run_hash,
            )

        # ATLAS vehicles2 CSV (output)
        atlas_veh2_file = os.path.join(
            atlas_output_path, f"vehicles2_{output_year}.csv"
        )
        atlas_veh2_output_record = None
        if os.path.exists(atlas_veh2_file):
            atlas_veh2_output_record = provenance_tracker.record_output_file(
                "atlas_postprocessor",
                atlas_veh2_file,
                description=f"ATLAS vehicles2 CSV with vehicleTypeId for year {output_year}",
                model_run_id=model_run_hash,
            )

        # Collect all input and output records
        input_records = [r for r in [usim_input_record, atlas_veh_input_record] if r]
        output_records = [
            r for r in [usim_output_record, atlas_veh2_output_record] if r
        ]

        provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=output_records
        )
        logger.info(
            "[AtlasPostprocessor] Completed provenance model run for ATLAS postprocessing: %s",
            model_run_hash,
        )
        return RecordStore(recordList=output_records)
