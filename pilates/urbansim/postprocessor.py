import os
import logging
from typing import Optional, Dict, Any

import pandas as pd

from pilates.config import PilatesConfig
from pilates.utils.io import read_datastore
from pilates.utils.coupler_helpers import enqueue_archive_copy
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.workspace import Workspace


logger = logging.getLogger(__name__)


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


def create_next_iter_usim_data(
    settings,
    forecast_year,
    mutable_data_dir,
):
    """Merge UrbanSim outputs with previous inputs to create the input for the next iteration."""
    # Define paths
    input_datastore_name = get_usim_datastore_fname(settings, io="input")
    input_store_path = os.path.join(mutable_data_dir, input_datastore_name)
    archive_fname = f"input_data_for_{forecast_year}_outputs.h5"
    archived_input_store_path = input_store_path.replace(
        input_datastore_name, archive_fname
    )

    if not os.path.exists(input_store_path):
        logger.error(
            f"Input store path {input_store_path} does not exist. Cannot create next iteration data."
        )
        return

    # Load the UrbanSim output H5 from the current run.
    output_store, table_prefix_year = read_datastore(
        settings, forecast_year, mutable_data_dir=mutable_data_dir
    )
    output_store_path = output_store._path
    output_store.close()

    # Archive the original input file.
    logger.info(f"Archiving previous iteration's inputs to {archive_fname}")
    os.rename(input_store_path, archived_input_store_path)

    # Merge and create new input file table by table.
    logger.info("Merging results back into new UrbanSim input store!")

    # Create an empty HDF5 file for the merged output.
    with pd.HDFStore(str(input_store_path), "w") as store:
        pass

    output_store = pd.HDFStore(str(output_store_path), "r")
    with pd.HDFStore(
        str(archived_input_store_path), "r"
    ) as archived_store, pd.HDFStore(str(input_store_path), "w") as new_store:

        processed_tables = set()

        # Copy tables from the current run's output
        for h5_key in output_store.keys():
            table_name = h5_key.split("/")[-1]
            if os.path.join("/", table_prefix_year, table_name) == h5_key:
                # Copy data and record output table
                new_store[table_name] = output_store[h5_key]
                processed_tables.add(table_name)

        # Copy missing tables from the archived (original) input store
        for h5_key in archived_store.keys():
            table_name = h5_key.split("/")[-1]
            if table_name not in processed_tables:
                logger.info(
                    f"Copying '{table_name}' from archived input to new input store!"
                )
                # Copy data and record output table
                new_store[table_name] = archived_store[h5_key]

        if set(new_store.keys()) != set(archived_store.keys()):
            logger.warning(
                "Mismatch in tables between archived input and new input store after merging."
            )
            logger.info(
                f"New tables: {set(new_store.keys())}. Archived tables: {set(archived_store.keys())}."
            )

    output_store.close()

    logger.info(f"Prepared merged input H5 for next iteration: {input_store_path}")

    # Ensure restart-critical UrbanSim H5s are copied to archive as soon as they
    # are produced/updated.
    enqueue_archive_copy(
        key=f"usim_year_output_h5_{forecast_year}",
        path=output_store_path,
    )
    enqueue_archive_copy(
        key=f"usim_input_archive_{forecast_year}",
        path=archived_input_store_path,
    )
    enqueue_archive_copy(
        key=f"usim_input_merged_{forecast_year}",
        path=input_store_path,
    )

    return [
        FileRecord(
            file_path=archived_input_store_path,
            description=f"Archived UrbanSim input H5 for year {forecast_year}",
            short_name=f"usim_input_archive_{forecast_year}",
            year=forecast_year,
        ),
        FileRecord(
            file_path=input_store_path,
            description=(
                f"Merged UrbanSim input H5 for next iteration "
                f"(from year {forecast_year} outputs)"
            ),
            short_name=f"usim_input_merged_{forecast_year}",
            year=forecast_year,
        ),
    ]


class UrbansimPostprocessor(GenericPostprocessor):
    """
    UrbanSim-specific postprocessor that consolidates all postprocessing steps for the UrbanSim land use model.
    This class is responsible for any postprocessing needed after the UrbanSim run, such as updating outputs,
    provenance, or preparing data for downstream models.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this postprocessor expects from the workflow.
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        usim_output_fname = settings.urbansim.output_file_template.format(
            year=state.forecast_year
        )
        usim_output_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_output_fname
        )
        return {
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": (
                usim_input_path if os.path.exists(usim_input_path) else None
            ),
            "usim_output_h5": (
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
            - ``usim_datastore_h5``: UrbanSim input datastore updated for the
              next model stage (H5).
        Related docs
            - See `pilates/urbansim/inputs.py` for the corresponding input
              descriptions used by UrbanSim and downstream models.
        """
        usim_input_fname = get_usim_datastore_fname(settings, io="input")
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "usim_datastore_h5": usim_input_path,
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)

    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocess UrbanSim outputs.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        logger.info(
            "[UrbansimPostprocessor] Postprocessing UrbanSim outputs for year %s",
            self.state.current_year,
        )

        settings = self.state.full_settings
        processed_records = []

        try:
            if settings.run.models.land_use == "urbansim":
                output_records = create_next_iter_usim_data(
                    settings,
                    self.state.forecast_year,
                    workspace.get_usim_mutable_data_dir(),
                )
                if output_records:
                    processed_records.extend(output_records)
                logger.info("Prepared UrbanSim data for next iteration")
            else:
                logger.info("Urbansim model is not activated, skipping postprocessing.")

        except Exception as e:
            logger.error(f"Error during UrbanSim postprocessing: {e}")
            raise

        return RecordStore(recordList=processed_records)
