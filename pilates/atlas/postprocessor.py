from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pilates.atlas.outputs import AtlasPostprocessOutputs, AtlasRunOutputs
from pilates.config import PilatesConfig
from pilates.workspace import Workspace
from pilates.utils.coupler_helpers import enqueue_archive_copy
from workflow_state import WorkflowState
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.workflows.artifact_keys import USIM_H5_UPDATED

logger = logging.getLogger(__name__)


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


def get_usim_datastore_fname(settings: PilatesConfig, io, year=None):
    """Construct the UrbanSim datastore filename based on settings.

    Args:
        settings (dict): The simulation settings.
        io (str): The direction of I/O, either 'input' or 'output'.
        year (int, optional): The simulation year. Required if io is 'output'.

    Returns:
        str: The formatted UrbanSim datastore filename.
    """
    if io == "output":
        datastore_name = settings.urbansim.output_file_template.format(year=year)
    elif io == "input":
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_base_fname = settings.urbansim.input_file_template
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
    Produces updated UrbanSim inputs and ATLAS vehicle outputs.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this postprocessor expects from the workflow.
        """
        usim_output_fname = get_usim_datastore_fname(
            settings, io="output", year=state.forecast_year
        )
        usim_output_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_output_fname
        )
        atlas_output_dir = workspace.get_atlas_output_dir()
        return {
            "atlas_output_dir": (
                atlas_output_dir if os.path.exists(atlas_output_dir) else None
            ),
            "usim_datastore_h5": (
                usim_output_path if os.path.exists(usim_output_path) else None
            ),
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this postprocessor produces.

        Notes
        -----
        Output keys
            - ``atlas_output_dir``: ATLAS output directory after postprocessing.
            - ``usim_datastore_h5``: Updated UrbanSim datastore (H5) emitted
              for subsequent model stages.
        Related docs
            - See `pilates/atlas/inputs.py` for the corresponding input
              descriptions used by ATLAS and downstream models.
        """
        usim_output_fname = get_usim_datastore_fname(
            settings, io="output", year=state.forecast_year
        )
        usim_output_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_output_fname
        )
        return {
            "atlas_output_dir": workspace.get_atlas_output_dir(),
            "usim_datastore_h5": usim_output_path,
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)

    def _postprocess(
        self,
        raw_outputs: AtlasRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> AtlasPostprocessOutputs:
        """
        Postprocess ATLAS outputs: update UrbanSim HDF5 with new vehicle ownership,
        and add vehicleTypeId to ATLAS vehicle outputs. Handles provenance tracking.

        Args:
            raw_outputs (AtlasRunOutputs): The raw outputs from the ATLAS model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            AtlasPostprocessOutputs: Typed outputs for the generated files.
        """
        logger.info(
            "[AtlasPostprocessor] Starting postprocessing for ATLAS for year %s",
            self.state.current_year,
        )

        settings = self.state.full_settings
        output_year = self.state.forecast_year
        output_paths: Dict[str, Path] = {}
        atlas_hh_path = raw_outputs.raw_outputs.get(f"householdv_{output_year}")
        atlas_veh_path = raw_outputs.raw_outputs.get(f"vehicles_{output_year}")
        if atlas_hh_path is None or not atlas_hh_path.exists():
            raise RuntimeError(
                "ATLAS postprocess requires the current-year householdv CSV from "
                "AtlasRunOutputs"
            )
        if atlas_veh_path is None or not atlas_veh_path.exists():
            raise RuntimeError(
                "ATLAS postprocess requires the current-year vehicles CSV from "
                "AtlasRunOutputs"
            )

        # --- HDF5 Update and Provenance ---
        usim_h5_path = workspace.get_usim_mutable_data_dir()
        usim_h5_fname = get_usim_datastore_fname(
            settings, io="output", year=output_year
        )
        usim_h5_file = os.path.join(usim_h5_path, usim_h5_fname)
        if not os.path.exists(usim_h5_file):
            raise RuntimeError(
                "ATLAS postprocess requires the current-year UrbanSim datastore H5"
            )

        # Perform the update
        self.atlas_update_h5_vehicle(
            settings, output_year, usim_h5_file, str(atlas_hh_path)
        )
        logger.info(
            "[AtlasPostprocessor] Updated UrbanSim HDF5 with new vehicle ownership."
        )
        updated_usim_h5 = Path(usim_h5_file)
        output_paths[USIM_H5_UPDATED] = Path(usim_h5_file)

        # --- vehicleTypeId addition and Provenance ---
        atlas_veh2_file = os.path.join(
            workspace.get_atlas_output_dir(), f"vehicles2_{output_year}.csv"
        )

        atlas_add_vehileTypeId(
            settings, output_year, str(atlas_veh_path), atlas_veh2_file
        )
        logger.info(
            "[AtlasPostprocessor] Added vehicleTypeId to ATLAS vehicle outputs."
        )
        if not os.path.exists(atlas_veh2_file):
            raise RuntimeError(
                "ATLAS postprocess did not produce vehicles2 output for year "
                f"{output_year}"
            )
        output_paths["atlas_vehicles2_output"] = Path(atlas_veh2_file)

        # Keep ATLAS subyear intermediates durable for restart and subyear chaining.
        atlas_input_root = workspace.get_atlas_mutable_input_dir()
        atlas_year_input_dir = os.path.join(atlas_input_root, f"year{output_year}")
        enqueue_archive_copy(
            key=f"atlas_input_year_dir_{output_year}",
            path=atlas_year_input_dir,
        )
        for base_dir in (
            atlas_year_input_dir,
            atlas_input_root,
            workspace.get_atlas_output_dir(),
        ):
            for filename in ("vehicles_output.RData", "households_output.RData"):
                enqueue_archive_copy(
                    key=f"atlas_rdata_{output_year}",
                    path=os.path.join(base_dir, filename),
                )

        return AtlasPostprocessOutputs(
            atlas_output_dir=Path(workspace.get_atlas_output_dir()),
            usim_datastore_h5=updated_usim_h5,
            processed_outputs=output_paths,
        )

    def postprocess(
        self,
        raw_outputs: AtlasRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> AtlasPostprocessOutputs:
        if not isinstance(raw_outputs, AtlasRunOutputs):
            raise TypeError("AtlasPostprocessor.postprocess expects AtlasRunOutputs")
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(raw_outputs, workspace, model_run_hash)

    def atlas_update_h5_vehicle(
        self,
        settings: PilatesConfig,
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
            df["cars"],
            bins=[-0.5, 0.5, 1.5, np.inf],
            labels=["none", "one", "two or more"],
        )

        logger.info(f"Writing updated household vehicle info to h5 file {h5_file_path}")

        # Read original h5 files and update
        with pd.HDFStore(h5_file_path, mode="r+") as h5:
            # The sub-state's `is_start_year` method correctly determines if this is a warm start context
            key = (
                "households"
                if self.state.is_start_year()
                else f"/{output_year}/households"
            )

            try:
                olddf = h5[key]
            except KeyError:
                logger.error(f"Table '{key}' not found in HDF5 file: {h5_file_path}")
                return

            olddf.index = olddf.index.astype(int)
            olddf = olddf.reindex(df.index.astype(int))

            if olddf.shape[0] != df.shape[0]:
                logger.error(
                    "ATLAS household_id mismatch found - NOT updating h5 datastore"
                )
            else:
                olddf["cars"] = df["cars"].values
                olddf["hh_cars"] = df["hh_cars"].values
                for col in olddf.columns:
                    if olddf[col].dtype.name == "category":
                        logger.info(f"Converting column {col} from category to str")
                        olddf[col] = olddf[col].astype(str)
                h5[key] = olddf
                logger.info(f"ATLAS update h5 datastore table {key} - done")
