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
    def postprocess(
        self,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
        model_run_hash: str,
    ) -> RecordStore:
        settings = state.full_settings
        output_year = state.forecast_year

        model_run_hash = provenance_tracker.start_model_run(
            "atlas_postprocessor",
            state.current_year,
            description="ATLAS postprocessing",
        )

        atlas_update_h5_vehicle(settings, output_year, state)
        atlas_add_vehileTypeId(settings, output_year, state)

        input_records = workspace.input_data.get("atlas", RecordStore())
        output_records = RecordStore()

        provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=output_records.all_records()
        )
        return RecordStore(recordList=output_records.all_records())
