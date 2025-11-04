import os
import logging
from typing import Optional

import pandas as pd
from pilates.utils.io import read_datastore
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.utils.provenance import FileProvenanceTracker
from pilates.workspace import Workspace
from pilates.utils.settings_helper import get as get_setting


logger = logging.getLogger(__name__)


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


def create_next_iter_usim_data(
    settings,
    forecast_year,
    mutable_data_dir,
    provenance_tracker,
    model_run_hash,
):
    """Merge UrbanSim outputs with previous inputs to create the input for the next iteration."""
    # Define paths
    input_datastore_name = get_usim_datastore_fname(settings, io="input")
    input_store_path = os.path.join(mutable_data_dir, input_datastore_name)
    archive_fname = f"input_data_for_{forecast_year}_outputs.h5"
    archived_input_store_path = input_store_path.replace(input_datastore_name, archive_fname)

    if not os.path.exists(input_store_path):
        logger.error(f"Input store path {input_store_path} does not exist. Cannot create next iteration data.")
        return

    # --- Provenance Step 1: Record initial inputs ---
    # Record the original input H5 that will be archived
    source_input_container = provenance_tracker.record_h5_input_container(
        "urbansim_postprocessor",
        input_store_path,
        description=f"UrbanSim input H5 before archiving for year {forecast_year}",
        short_name=f"usim_input_pre_archive_{forecast_year}",
        model_run_id=model_run_hash,
    )

    # Record the UrbanSim output H5 from the current run
    output_store, table_prefix_year = read_datastore(settings, forecast_year, mutable_data_dir=mutable_data_dir)
    output_store_path = output_store._path
    output_store.close()
    output_container = provenance_tracker.record_h5_input_container(
        "urbansim_postprocessor",
        output_store_path,
        description=f"UrbanSim output H5 for year {forecast_year}",
        short_name=f"usim_output_{forecast_year}",
        model_run_id=model_run_hash,
    )

    # --- File Operation: Archive the original input file ---
    logger.info(f"Archiving previous iteration's inputs to {archive_fname}")
    os.rename(input_store_path, archived_input_store_path)

    # --- Provenance Step 2: Record the archived file as an intermediate output ---
    archived_container = provenance_tracker.record_h5_output_container(
        "urbansim_postprocessor",
        archived_input_store_path,
        input_records=[source_input_container],
        description=f"Archived UrbanSim input H5 for year {forecast_year}",
        short_name=f"usim_input_archive_{forecast_year}",
        model_run_id=model_run_hash,
    )

    # --- Merge and Provenance Step 3: Create new input file table by table ---
    logger.info("Merging results back into new UrbanSim input store!")
    final_table_records = []

    # Create an empty HDF5 file so that provenance can be recorded.
    with pd.HDFStore(str(input_store_path), "w") as store:
        pass

    # --- Provenance Step 4: Record the final merged HDF5 container ---
    final_output_container = provenance_tracker.record_h5_output_container(
        "urbansim_postprocessor",
        input_store_path,
        input_records=[archived_container, output_container],
        table_records=final_table_records,
        year=forecast_year,
        description=f"Merged UrbanSim input H5 for next iteration (from year {forecast_year} outputs)",
        short_name=f"usim_input_merged_{forecast_year}",
        model_run_id=model_run_hash,
    )

    output_store = pd.HDFStore(str(output_store_path), "r")
    with pd.HDFStore(str(archived_input_store_path), "r") as archived_store, \
         pd.HDFStore(str(input_store_path), "w") as new_store:

        processed_tables = set()

        # Copy tables from the current run's output
        for h5_key in output_store.keys():
            table_name = h5_key.split('/')[-1]
            if os.path.join("/", table_prefix_year, table_name) == h5_key:
                # Record source table from output_store
                source_table = provenance_tracker.record_h5_table_input(
                    "urbansim_postprocessor", output_container, h5_key, model_run_id=model_run_hash
                )
                # Copy data and record output table
                new_store[table_name] = output_store[h5_key]
                output_table = provenance_tracker.record_h5_table_output(
                    "urbansim_postprocessor", final_output_container, table_name, [source_table], model_run_id=model_run_hash
                )
                final_table_records.append(output_table)
                processed_tables.add(table_name)

        # Copy missing tables from the archived (original) input store
        for h5_key in archived_store.keys():
            table_name = h5_key.split('/')[-1]
            if table_name not in processed_tables:
                logger.info(f"Copying '{table_name}' from archived input to new input store!")
                # Record source table from archived_store
                source_table = provenance_tracker.record_h5_table_input(
                    "urbansim_postprocessor", archived_container, h5_key, model_run_id=model_run_hash
                )
                # Copy data and record output table
                new_store[table_name] = archived_store[h5_key]
                output_table = provenance_tracker.record_h5_table_output(
                    "urbansim_postprocessor", final_output_container, table_name, [source_table], model_run_id=model_run_hash
                )
                final_table_records.append(output_table)

        if set(new_store.keys()) != set(archived_store.keys()):
            logger.warning("Mismatch in tables between archived input and new input store after merging.")
            logger.info(f"New tables: {set(new_store.keys())}. Archived tables: {set(archived_store.keys())}.")

    output_store.close()

    final_output_container.table_record_ids = [tr.unique_id for tr in final_table_records]
    provenance_tracker.run_info.file_records[final_output_container.unique_id] = final_output_container
    logger.info(f"Recorded merged input H5 for provenance: {input_store_path}")


class UrbansimPostprocessor(GenericPostprocessor):
    """
    UrbanSim-specific postprocessor that consolidates all postprocessing steps for the UrbanSim land use model.
    This class is responsible for any postprocessing needed after the UrbanSim run, such as updating outputs,
    provenance, or preparing data for downstream models.
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
        Postprocess UrbanSim outputs.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            runInfo (Optional[ModelRunInfo]): Metadata about the model run. Not used by this processor.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        logger.info(
            "[UrbansimPostprocessor] Postprocessing UrbanSim outputs for year %s",
            self.state.current_year,
        )

        if model_run_hash is None:
            model_run_hash = self.provenance_tracker.start_model_run(
                "urbansim_postprocessor",
                self.state.current_year,
                self.state.current_inner_iter,
                description="Post-processing UrbanSim outputs",
                inputs=raw_outputs,
            )

        settings = self.state.full_settings
        processed_records = []

        try:
            if get_setting(settings, "run.models.land_use") == "urbansim":
                create_next_iter_usim_data(
                    settings,
                    self.state.forecast_year,
                    workspace.get_usim_mutable_data_dir(),
                    provenance_tracker=self.provenance_tracker,
                    model_run_hash=model_run_hash,
                )
                logger.info("Prepared UrbanSim data for next iteration")
            else:
                logger.info("Urbansim model is not activated, skipping postprocessing.")

        except Exception as e:
            logger.error(f"Error during UrbanSim postprocessing: {e}")
            self.provenance_tracker.complete_model_run(model_run_hash, status="failed")
            raise

        # Complete postprocessor tracking
        # Note: The output records are created within create_next_iter_usim_data
        self.provenance_tracker.complete_model_run(
            model_run_hash, status="completed", output_records=[]
        )

        return RecordStore(recordList=processed_records)
