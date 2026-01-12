from __future__ import annotations

from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any
import os
import shutil
import logging
import pandas as pd
import openmatrix as omx
import numpy as np

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.path_utils import find_project_root

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

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


def _load_raw_skims(settings, asim_data_dir, usim_data_dir, skim_format, workspace):
    """
    Load raw skims and format for UrbanSim, ensuring canonical zone order.
    """
    from pilates.utils.zone_utils import load_canonical_zones

    # Get the authoritative, sorted list of zone IDs
    canonical_zones_df = load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.values

    skims_fname = settings.shared.skims.fname

    try:
        if skim_format == "beam":
            if skims_fname.endswith("csv"):
                raise NotImplementedError("DEMOS requires skims in omx format, not csv")

            elif skims_fname.endswith("omx"):
                skims_fname = "skims.omx"
                mutable_skims_location = os.path.join(asim_data_dir, skims_fname)
                # This copy seems redundant if the file is already in the right place,
                # but preserving original logic for now.
                region_id = settings.urbansim.region_mappings["region_to_region_id"][
                    settings.run.region
                ]
                input_skims_location = "pilates/urbansim/data/skims_mpo_{0}.omx".format(
                    region_id
                )
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(
                        mutable_skims_location, input_skims_location
                    )
                )
                shutil.copyfile(mutable_skims_location, input_skims_location)

                with omx.open_file(mutable_skims_location, "r") as skims:
                    # Use the canonical order for the DataFrame index and columns
                    index = pd.Index(canonical_order, name="from_zone_id", dtype=str)
                    columns = pd.Index(canonical_order, name="to_zone_id", dtype=str)

                    # The skim matrix data is assumed to be in the canonical order
                    # as enforced by the ActivitySim preprocessor.
                    travel_time_mins = np.array(skims["SOV_TIME__AM"])
                    out = (
                        pd.DataFrame(travel_time_mins, index=index, columns=columns)
                        .stack()
                        .rename("SOV_AM_IVT_mins")
                    )
                    return out.to_frame()

            elif skims_fname.endswith("zarr"):
                beam_output_dir = settings.beam.local_output_folder
                skims_fname = settings.shared.skims.fname
                mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
                region_id = settings.urbansim.region_mappings["region_to_region_id"][
                    settings.run.region
                ]
                input_skims_location = os.path.join(
                    usim_data_dir, "skims_mpo_{0}.zarr".format(region_id)
                )
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(
                        mutable_skims_location, input_skims_location
                    )
                )

                if os.path.exists(input_skims_location):
                    shutil.rmtree(input_skims_location)
                shutil.copytree(mutable_skims_location, input_skims_location)

                import xarray as xr

                with xr.open_zarr(mutable_skims_location) as ds:
                    # Use the canonical order for the DataFrame index and columns
                    index = pd.Index(canonical_order, name="from_zone_id", dtype=str)
                    columns = pd.Index(canonical_order, name="to_zone_id", dtype=str)

                    # Get AM time period data
                    am_idx = list(ds.coords["time_period"].values).index("AM")
                    travel_time_data = ds["SOV_TIME"].isel(time_period=am_idx).values

                    travel_time_mins = np.array(travel_time_data)
                    out = (
                        pd.DataFrame(travel_time_mins, index=index, columns=columns)
                        .stack()
                        .rename("SOV_AM_IVT_mins")
                    )
                    return out.to_frame()
            else:
                raise NotImplementedError(
                    "Invalid skim format {0}".format(skims_fname.split(".")[-1])
                )
        elif skim_format == "polaris":
            raise NotImplementedError("DEMOS requires skims in omx format, not polaris")

    except KeyError:
        raise KeyError("Couldn't find input skims named {0}".format(skims_fname))

    raise NotImplementedError("Invalid skims format {0}".format(skim_format))


class UrbansimPreprocessor(GenericPreprocessor):
    """
    UrbanSim-specific preprocessor that consolidates all preprocessing steps for the UrbanSim land use model.
    This includes copying input files to a mutable location and preparing any additional
    data needed for the UrbanSim run.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this preprocessor expects from the workflow.
        """
        project_root = find_project_root(start_path=os.path.dirname(__file__))
        if not project_root:
            project_root = os.path.realpath(os.getcwd())
        source_dir = (
            settings.urbansim.local_data_input_folder
            if os.path.isabs(settings.urbansim.local_data_input_folder)
            else os.path.join(project_root, settings.urbansim.local_data_input_folder)
        )
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "usim_source_data_dir": source_dir,
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": (
                usim_input_path if os.path.exists(usim_input_path) else None
            ),
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this preprocessor produces.

        Notes
        -----
        Output keys
            - ``usim_mutable_data_dir``: UrbanSim mutable data directory populated
              with inputs for the runner.
            - ``usim_datastore_h5``: Base-year UrbanSim datastore staged into the
              mutable directory.
        Related docs
            - See `pilates/urbansim/inputs.py` for the corresponding input
              descriptions used by UrbanSim and downstream models.
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": usim_input_path,
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)
        self.required_input_data = ["usim_data_reference"]

    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy UrbanSim input files from production to mutable location.

        Returns:
            Tuple[RecordStore, RecordStore]: (inputs, outputs) as RecordStores
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]

        # Get the input file templates
        year_template = settings.urbansim.input_file_template_year
        if year_template:
            year_specific_model_data_fname = year_template.format(
                region_id=region_id, start_year=settings.run.start_year
            )
        else:
            year_specific_model_data_fname = ""

        base_template = settings.urbansim.input_file_template
        if base_template:
            model_data_fname = base_template.format(region_id=region_id)
        else:
            model_data_fname = ""
        project_root = find_project_root(start_path=os.path.dirname(__file__))
        if not project_root:
            project_root = os.path.realpath(os.getcwd())
            logger.warning(
                "[NOT IDEAL] Could not locate PILATES project root via markers; falling back to cwd='%s'. "
                "This is error-prone in production and may affect inputs:// vs workspace:// URI labeling.",
                project_root,
            )
        data_dir = (
            settings.urbansim.local_data_input_folder
            if os.path.isabs(settings.urbansim.local_data_input_folder)
            else os.path.join(project_root, settings.urbansim.local_data_input_folder)
        )

        # Validate we have a filename
        if not model_data_fname:
            raise ValueError(
                "UrbanSim input file template is not configured. "
                "Please set 'urbansim.input_file_template' or 'usim_formattable_input_file_name' in settings."
            )

        if os.path.exists(os.path.join(data_dir, year_specific_model_data_fname)) and (
            settings.urbansim.input_file_template_year is not None
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
            FileRecord(
                file_path=src,
                description="Reference urbanSim model data",
                short_name="usim_data_reference",
            )
        ]
        outputs = [
            FileRecord(
                file_path=dest,
                description="UrbanSim model data",
                short_name="usim_data",
            )
        ]

        beam_inputs_root = (
            settings.beam.local_input_folder
            if os.path.isabs(settings.beam.local_input_folder)
            else os.path.join(project_root, settings.beam.local_input_folder)
        )
        skims_src = os.path.abspath(
            os.path.join(
                beam_inputs_root, settings.run.region, settings.shared.skims.fname
            )
        )
        skims_target = os.path.join(output_dir, "skims_mpo_{0}.omx".format(region_id))

        inputs.append(
            FileRecord(
                file_path=skims_src,
                short_name="omx_skims",
                description="Raw BEAM OD skims",
            )
        )
        shutil.copyfile(skims_src, skims_target)

        outputs.append(
            FileRecord(
                file_path=skims_target,
                short_name="omx_skims",
                description="Raw BEAM OD skims for USim",
            )
        )

        logger.info(
            f"[UrbansimPreprocessor] Copying input beam skims data from {skims_src} to {skims_target}"
        )

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
                    FileRecord(
                        file_path=src,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
                outputs.append(
                    FileRecord(
                        file_path=dest,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
        logger.info("[UrbansimPreprocessor] Finished copying UrbanSim input files.")
        return RecordStore(recordList=inputs), RecordStore(recordList=outputs)

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess UrbanSim data, including copying necessary skims.
        """
        logger.info("[UrbansimPreprocessor] Preprocessing for UrbanSim.")
        settings = self.state.full_settings

        # Ensure the mutable data directory exists, especially on restarts when Initialization is skipped.
        usim_mutable_data_dir = workspace.get_usim_mutable_data_dir()
        os.makedirs(usim_mutable_data_dir, exist_ok=True)

        input_records = workspace.input_data.get("urbansim", RecordStore())

        processed_records = RecordStore()

        try:
            # Generate the block-to-zone mapping for UrbanSim
            logger.info("Generating block-to-zone mapping for UrbanSim.")
            from pilates.utils.zone_utils import get_block_to_zone_mapping

            mapping = get_block_to_zone_mapping(
                settings, self.state.start_year, workspace
            )

            # Save the mapping to a CSV file in the mutable data directory
            geoid_to_zone_fname = "geoid_to_zone.csv"
            geoid_to_zone_path = os.path.join(
                workspace.get_usim_mutable_data_dir(), geoid_to_zone_fname
            )
            (
                pd.DataFrame.from_dict(mapping, orient="index", columns=["zone_id"])
                .rename_axis("GEOID")
                .to_csv(geoid_to_zone_path)
            )

            mapping_output_rec = FileRecord(
                file_path=geoid_to_zone_path,
                description="Block to zone mapping for UrbanSim input",
                short_name="geoid_to_zone",
            )
            processed_records.add_record(mapping_output_rec)

            # If not the first iteration, check if BEAM is enabled and copy updated skims
            if (
                self.state.current_year > settings.run.start_year
                or self.state.iteration > 0
            ):
                if settings.run.models.travel == "beam":
                    logger.info(
                        "Updating skims from BEAM mutable output for subsequent iteration."
                    )
                    if self.state.run_info_path and os.path.exists(
                        self.state.run_info_path
                    ):
                        logger.info(
                            f"[UrbansimPreprocessor] Restarted run detected. Using previous run's output path from {self.state.run_info_path}"
                        )
                        previous_run_dir = os.path.dirname(self.state.run_info_path)
                        beam_mutable_data_dir = os.path.join(
                            previous_run_dir, "beam", "input"
                        )
                    else:
                        beam_mutable_data_dir = workspace.get_beam_mutable_data_dir()

                    source_skims_path = os.path.join(
                        beam_mutable_data_dir,
                        settings.run.region,
                        settings.shared.skims.fname,
                    )

                    region_id = settings.urbansim.region_mappings[
                        "region_to_region_id"
                    ][settings.run.region]
                    dest_skims_fname = f"skims_mpo_{region_id}.omx"
                    dest_skims_path = os.path.join(
                        workspace.get_usim_mutable_data_dir(), dest_skims_fname
                    )

                    if os.path.exists(source_skims_path):
                        logger.info(
                            f"Copying skims from {source_skims_path} to {dest_skims_path}"
                        )
                        shutil.copy(source_skims_path, dest_skims_path)

                        skims_output_rec = FileRecord(
                            file_path=dest_skims_path,
                            description="Copied updated skims for UrbanSim consumption",
                            short_name="usim_skims_input_updated",
                        )
                        if skims_output_rec:
                            processed_records.add_record(skims_output_rec)
                    else:
                        logger.warning(
                            f"Skims file not found at source: {source_skims_path}"
                        )

            # Pass through any existing input records
            for record in input_records.all_records():
                if record.file_path and os.path.exists(record.file_path):
                    processed_records.add_record(record)

        except Exception as e:
            logger.error(f"Error during UrbanSim preprocessing: {e}")
            raise

        return processed_records
