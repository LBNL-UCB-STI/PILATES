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
from pilates.utils.provenance import FileProvenanceTracker

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


def _load_raw_skims(settings, asim_data_dir, usim_data_dir, skim_format):
    skims_fname = settings.get("skims_fname", False)

    try:
        if skim_format == "beam":
            if skims_fname.endswith("csv"):
                raise NotImplementedError("DEMOS requires skims in omx format, not csv")
                # path_to_skims = os.path.join(
                #     settings["beam_local_output_folder"], skims_fname
                # )
                # # load skims from disk or url
                # skims = pd.read_csv(path_to_skims, dtype=skim_dtypes)
                # skims = skims.loc[
                #     (skims["pathType"] == "SOV") & (skims["timePeriod"] == "AM")
                # ]
                # skims = skims[
                #     ["origin", "destination", "TOTIVT_IVT_minutes", "DIST_meters"]
                # ]
                # skims = skims.rename(
                #     columns={
                #         "origin": "from_zone_id",
                #         "destination": "to_zone_id",
                #         "TOTIVT_IVT_minutes": "SOV_AM_IVT_mins",
                #     }
                # )
            elif skims_fname.endswith("omx"):
                skims_fname = "skims.omx"
                mutable_skims_location = os.path.join(asim_data_dir, skims_fname)
                input_skims_location = "pilates/urbansim/data/skims_mpo_{0}.omx".format(settings['region_id'])
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(mutable_skims_location, input_skims_location))
                shutil.copyfile(mutable_skims_location, input_skims_location)
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
            elif skims_fname.endswith('zarr'):
                beam_output_dir = settings['beam_local_output_folder']
                skims_fname = settings['skims_fname']
                mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
                region_id = settings['region_to_region_id'][settings['region']]
                input_skims_location = os.path.join(usim_data_dir, "skims_mpo_{0}.zarr".format(region_id))
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(mutable_skims_location, input_skims_location))

                # Copy zarr directory (it's a directory, not a single file)
                if os.path.exists(input_skims_location):
                    shutil.rmtree(input_skims_location)
                shutil.copytree(mutable_skims_location, input_skims_location)

                ds = xr.open_zarr(mutable_skims_location)

                # Get zone IDs
                zone_ids = ds.coords['otaz'].values
                zone_ids = [str(z) for z in zone_ids]

                # Get AM time period data
                am_idx = list(ds.coords['time_period'].values).index('AM')
                travel_time_data = ds['SOV_TIME'].isel(time_period=am_idx).values

                # Create DataFrame
                index = pd.Index(zone_ids, name="from_zone_id", dtype=str)
                columns = pd.Index(zone_ids, name="to_zone_id", dtype=str)
                travel_time_mins = np.array(travel_time_data)
                out = pd.DataFrame(travel_time_mins, index=index, columns=columns).stack().rename('SOV_AM_IVT_mins')

                return out.to_frame()
            else:
                raise NotImplementedError(
                    "Invalid skim format {0}".format(skims_fname.split(".")[-1])
                )
        elif skim_format == "polaris":
            raise NotImplementedError("DEMOS requires skims in omx format, not polaris")
            # path_to_skims = os.path.join(
            #     settings["polaris_local_data_folder"], skims_fname
            # )
            # f = h5py.File(path_to_skims, "r")
            # ivtt_8_9 = pd.DataFrame(list(f["auto_skims"]["t4"]["ivtt"]))
            # cost_8_9 = pd.DataFrame(list(f["auto_skims"]["t4"]["cost"]))
            # f.close()
            # ivtt_8_9 = pd.DataFrame(ivtt_8_9.stack(), columns=["auto_ivtt_8_9_am"])
            # cost_8_9 = pd.DataFrame(cost_8_9.stack(), columns=["auto_cost_8_9_am"])
            # skims = ivtt_8_9.join(cost_8_9)
            # skims.index.names = ["from_zone_id", "to_zone_id"]
            # skims = skims.reset_index()

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
    """
    UrbanSim-specific preprocessor that consolidates all preprocessing steps for the UrbanSim land use model.
    This includes copying input files to a mutable location, recording provenance, and preparing any additional
    data needed for the UrbanSim run.
    """

    def __init__(self, model_name: str, state: "WorkflowState", provenance_tracker: FileProvenanceTracker):
        super().__init__(model_name, state, provenance_tracker)
        self.required_input_data = ["usim_data_reference"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy UrbanSim input files from production to mutable location,
        recording provenance for inputs and outputs.

        Returns:
            Tuple[RecordStore, RecordStore]: (inputs, outputs) as RecordStores
        """
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        year_specific_model_data_fname = settings.get(
            "usim_formattable_input_file_name_year", ""
        ).format(region_id=region_id, start_year=settings["start_year"])
        model_data_fname = settings["usim_formattable_input_file_name"].format(
            region_id=region_id
        )
        data_dir = settings["usim_local_data_input_folder"]
        if os.path.exists(os.path.join(data_dir, year_specific_model_data_fname)) and (
            settings.get("usim_formattable_input_file_name_year") is not None
        ):
            src = os.path.join(data_dir, year_specific_model_data_fname)
        else:
            src = os.path.join(data_dir, model_data_fname)
        dest = os.path.join(output_dir, model_data_fname)

        logger.info(
            f"[UrbansimPreprocessor] Copying input urbansim data from {src} to {dest}"
        )
        if os.path.exists(src):
            shutil.copyfile(src, dest)
        else:
            # Create an empty HDF5 file if the source does not exist
            with pd.HDFStore(dest, "w"):
                pass
            logger.warning(
                f"[UrbansimPreprocessor] Source UrbanSim HDF5 file not found at {src}. Created empty HDF5 at {dest}."
            )
        inputs = [
            self.provenance_tracker.record_input_file(
                "urbansim",
                src,
                description="Reference urbanSim model data",
                short_name="usim_data_reference",
            )
        ]
        outputs = [
            self.provenance_tracker.record_output_file(
                "urbansim",
                dest,
                description="UrbanSim model data",
                short_name="usim_data",
            )
        ]
        other_data_fnames = {
            f"hsize_ct_{region_id}.csv": "hh_size",
            f"income_rates_{region_id}.csv": "income_rates",
            f"relmap_{region_id}.csv": "relmap",
            "schools_2010.csv": "schools",
            "blocks_school_districts_2010.csv": "school_districts",
        }
        for fname, short_name in other_data_fnames.items():
            src = os.path.join(data_dir, fname)
            dest = os.path.join(output_dir, fname)
            if os.path.exists(src):
                logger.info(
                    f"[UrbansimPreprocessor] Copying input urbansim file from {src} to {dest}"
                )
                shutil.copyfile(src, dest)
                inputs.append(
                    self.provenance_tracker.record_input_file(
                        "urbansim",
                        src,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
                outputs.append(
                    self.provenance_tracker.record_output_file(
                        "urbansim",
                        dest,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
        logger.info(
            "[UrbansimPreprocessor] Finished copying UrbanSim input files and recording provenance."
        )
        return RecordStore(recordList=inputs), RecordStore(recordList=outputs)

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess UrbanSim data.
        Returns the input RecordStore from the workspace.
        """
        logger.info("[UrbansimPreprocessor] Preprocessing for UrbanSim.")
        input_records = workspace.input_data.get("urbansim", RecordStore())

        # Start preprocessor run tracking
        model_run_hash = self.provenance_tracker.start_model_run(
            "urbansim_preprocessor",
            self.state.current_year,
            self.state.current_inner_iter,
            description="Preprocessing UrbanSim inputs",
            inputs=input_records,
        )

        processed_records = []
        
        try:
            # Record provenance for ActivitySim skims if available
            settings = self.state.full_settings
            if "skims_fname" in settings:
                skims_path = os.path.join(
                    workspace.get_asim_mutable_data_dir(),
                    settings["skims_fname"],
                )
                if os.path.exists(skims_path):
                    skims_record = self.provenance_tracker.record_input_file(
                        "urbansim_preprocessor",
                        skims_path,
                        description="ActivitySim skims for UrbanSim input",
                        short_name="asim_skims",
                        model_run_id=model_run_hash,
                        state=self.state,
                    )
                    if skims_record:
                        processed_records.append(skims_record)
                        logger.info(f"Recorded skims input: {skims_path}")
                else:
                    logger.warning(f"Skims file not found: {skims_path}")

            # TODO: Add other UrbanSim-specific preprocessing logic here
            # This might include:
            # - Loading and validating UrbanSim input data
            # - Processing skims in different formats (omx, zarr, csv)
            # - Preparing data transformations needed for the model
            
            # For now, add any existing input records to processed records
            for record in input_records.all_records():
                if record.file_path and os.path.exists(record.file_path):
                    processed_records.append(record)

        except Exception as e:
            logger.error(f"Error during UrbanSim preprocessing: {e}")
            self.provenance_tracker.complete_model_run(model_run_hash, status="failed")
            raise

        # Complete preprocessor tracking
        self.provenance_tracker.complete_model_run(
            model_run_hash, status="completed", output_records=processed_records
        )

        return RecordStore(recordList=processed_records)
