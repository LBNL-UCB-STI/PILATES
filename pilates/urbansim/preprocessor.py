from typing import Tuple
import os
import shutil
import logging
import pandas as pd
import h5py
import openmatrix as omx
import numpy as np

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from pilates.utils.geog import geoid_to_zone_map

logger = logging.getLogger(__name__)

skim_dtypes = {
    "timePeriod": str,
    "pathType": str,
    "origin": int,
    "destination": int,
    "TIME_minutes": float,
    "TOTIVT_IVT_minutes": float,
    "VTOLL_FAR": float,
    "DIST_meters": float,
    "WACC_minutes": float,
    "WAUX_minutes": float,
    "WEGR_minutes": float,
    "DTIM_minutes": float,
    "DDIST_meters": float,
    "KEYIVT_minutes": float,
    "FERRYIVT_minutes": float,
    "BOARDS": float,
    "DEBUG_TEXT": str,
}


def _load_raw_skims(settings, asim_data_dir, skim_format):
    skims_fname = settings.get("skims_fname", False)

    try:
        if skim_format == "beam":
            if skims_fname.endswith("csv"):
                path_to_skims = os.path.join(
                    settings["beam_local_output_folder"], skims_fname
                )
                # load skims from disk or url
                skims = pd.read_csv(path_to_skims, dtype=skim_dtypes)
                skims = skims.loc[
                    (skims["pathType"] == "SOV") & (skims["timePeriod"] == "AM")
                ]
                skims = skims[
                    ["origin", "destination", "TOTIVT_IVT_minutes", "DIST_meters"]
                ]
                skims = skims.rename(
                    columns={
                        "origin": "from_zone_id",
                        "destination": "to_zone_id",
                        "TOTIVT_IVT_minutes": "SOV_AM_IVT_mins",
                    }
                )
            elif skims_fname.endswith("omx"):
                skims_fname = "skims.omx"
                mutable_skims_location = os.path.join(asim_data_dir, skims_fname)
                skims = omx.open_file(mutable_skims_location, "r")
                zone_ids = skims.mapping("zone_id").keys()
                index = pd.Index(zone_ids, name="from_zone_id", dtype=str)
                columns = pd.Index(zone_ids, name="to_zone_id", dtype=str)
                travel_time_mins = np.array(skims["SOV_TIME__AM"])
                out = (
                    pd.DataFrame(travel_time_mins, index=index, columns=columns)
                    .stack()
                    .rename("SOV_AM_IVT_mins")
                )
                skims.close()
                return out.to_frame()
            else:
                raise NotImplementedError(
                    "Invalid skim format {0}".format(skims_fname.split(".")[-1])
                )
        elif skim_format == "polaris":
            path_to_skims = os.path.join(
                settings["polaris_local_data_folder"], skims_fname
            )
            f = h5py.File(path_to_skims, "r")
            ivtt_8_9 = pd.DataFrame(list(f["auto_skims"]["t4"]["ivtt"]))
            cost_8_9 = pd.DataFrame(list(f["auto_skims"]["t4"]["cost"]))
            f.close()
            ivtt_8_9 = pd.DataFrame(ivtt_8_9.stack(), columns=["auto_ivtt_8_9_am"])
            cost_8_9 = pd.DataFrame(cost_8_9.stack(), columns=["auto_cost_8_9_am"])
            skims = ivtt_8_9.join(cost_8_9)
            skims.index.names = ["from_zone_id", "to_zone_id"]
            skims = skims.reset_index()

    except KeyError:
        raise KeyError("Couldn't find input skims named {0}".format(skims_fname))

    logger.info("Converting skims to UrbanSim data format.")
    skims["from_zone_id"] = skims["from_zone_id"].astype("str")
    skims["to_zone_id"] = skims["to_zone_id"].astype("str")

    # for GEOID/FIPS-based skims, we have to convert the zone IDs
    if settings["skims_zone_type"] in ["block", "block_group"]:
        mapping = geoid_to_zone_map(settings)
        for col in ["from_zone_id", "to_zone_id"]:
            skims[col] = skims[col].map(mapping)

    skims = skims.set_index(["from_zone_id", "to_zone_id"])

    return skims


class UrbansimPreprocessor(GenericPreprocessor):
    def __init__(self):
        super().__init__()
        self.required_input_data = ["usim_data_reference"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
        provenance_tracker,
    ) -> Tuple[RecordStore, RecordStore]:
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        year_specific_model_data_fname = settings.get(
            "usim_formattable_input_file_name_year", ""
        ).format(region_id=region_id, start_year=settings["start_year"])
        model_data_fname = settings["usim_formattable_input_file_name"].format(
            region_id=region_id
        )
        data_dir = settings["usim_local_data_input_folder"]
        if os.path.exists(os.path.join(data_dir, year_specific_model_data_fname)) & (
            settings.get("usim_formattable_input_file_name_year") is not None
        ):
            src = os.path.join(data_dir, year_specific_model_data_fname)
        else:
            src = os.path.join(data_dir, model_data_fname)
        dest = os.path.join(output_dir, model_data_fname)

        logger.info("Copying input urbansim data from {0} to {1}".format(src, dest))
        if os.path.exists(src):
            shutil.copyfile(src, dest)
        else:
            # Create an empty HDF5 file if the source does not exist
            with pd.HDFStore(dest, "w"):
                pass
            logger.warning(
                f"Source UrbanSim HDF5 file not found at {src}. Created empty HDF5 at {dest}."
            )
        inputs = [
            provenance_tracker.record_input_file(
                "urbansim",
                src,
                description="Reference urbanSim model data",
                short_name="usim_data_reference",
            )
        ]
        outputs = [
            provenance_tracker.record_output_file(
                "urbansim", dest, description="UrbanSim model data", short_name="usim_data"
            )
        ]
        other_data_fnames = {
            "hsize_ct_{0}.csv".format(region_id): "hh_size",
            "income_rates_{0}.csv".format(region_id): "income_rates",
            "relmap_{0}.csv".format(region_id): "relmap",
            "schools_2010.csv": "schools",
            "blocks_school_districts_2010.csv": "school_districts",
        }
        for fname, short_name in other_data_fnames.items():
            src = os.path.join(data_dir, fname)
            dest = os.path.join(output_dir, fname)
            if os.path.exists(src):
                logger.info("Copying input urbansim file from {0} to {1}".format(src, dest))
                shutil.copyfile(src, dest)
                inputs.append(
                    provenance_tracker.record_input_file(
                        "urbansim",
                        src,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
                outputs.append(
                    provenance_tracker.record_output_file(
                        "urbansim",
                        dest,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
        return RecordStore(recordList=inputs), RecordStore(recordList=outputs)

    def preprocess(
        self,
        state,
        workspace,
        provenance_tracker,
    ) -> RecordStore:
        # For now, just return the input data as the preprocessed data
        input_records = workspace.input_data.get("urbansim", RecordStore())
        return input_records
