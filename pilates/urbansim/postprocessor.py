import os
import logging
import pandas as pd
from pilates.utils.io import read_datastore
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace


logger = logging.getLogger(__name__)


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


def create_next_iter_usim_data(
    settings,
    year,
    forecast_year,
    full_path,
    provenance_tracker=None,
    model_run_hash=None,
):
    data_dir = settings["usim_local_mutable_data_folder"]

    # Move UrbanSim input store (e.g. custom_mpo_193482435_model_data.h5)
    # to archive (e.g. input_data_for_2015_outputs.h5) because otherwise
    # it will be overwritten in the next step.
    input_datastore_name = get_usim_datastore_fname(settings, io="input")
    input_store_path = os.path.join(data_dir, input_datastore_name)

    # First check if urbansim model is activated
    if settings.get("land_use_model") == "urbansim":
        if os.path.exists(input_store_path):
            archive_fname = "input_data_for_{0}_outputs.h5".format(forecast_year)
            logger.info(
                "Moving urbansim inputs from the previous iteration to {0}".format(
                    archive_fname
                )
            )
            new_input_store_path = input_store_path.replace(
                input_datastore_name, archive_fname
            )

            # FIX ISSUE 1: Record the input H5 BEFORE it's renamed to archive
            if provenance_tracker:
                provenance_tracker.record_input_file(
                    "urbansim_postprocessor",
                    input_store_path,
                    description=f"UrbanSim input H5 before archiving to year {forecast_year}",
                    short_name=f"usim_input_pre_archive_{forecast_year}",
                    model_run_id=model_run_hash,
                )

            os.rename(input_store_path, new_input_store_path)

            # FIX ISSUE 1: Record the archived file as an output
            if provenance_tracker:
                provenance_tracker.record_output_file(
                    "urbansim_postprocessor",
                    new_input_store_path,
                    year=forecast_year,
                    description=f"Archived UrbanSim input H5 for year {forecast_year} outputs",
                    short_name=f"usim_input_archive_{forecast_year}",
                    model_run_id=model_run_hash,
                    source_file_paths=[input_store_path],
                )
            og_input_store = pd.HDFStore(str(new_input_store_path))
            new_input_store = pd.HDFStore(str(input_store_path))
            assert len(new_input_store.keys()) == 0
            updated_tables = []

            # load last iter output data
            # output_datastore_name = _get_usim_datastore_fname(settings, 'output', forecast_year)
            # output_store_path = os.path.join(data_dir, output_datastore_name)

            # copy usim outputs into new input data store
            logger.info("Merging results back into UrbanSim and storing as .h5!")
            output_store, table_prefix_year = read_datastore(
                settings, forecast_year, mutable_data_dir=full_path
            )

            # Track the UrbanSim output H5 as an input to this merge process
            output_store_path = output_store._path
            if provenance_tracker and os.path.exists(output_store_path):
                provenance_tracker.record_input_file(
                    "urbansim_postprocessor",
                    output_store_path,
                    description=f"UrbanSim output H5 for year {forecast_year}",
                    short_name=f"usim_output_{forecast_year}",
                    model_run_id=model_run_hash,
                )

            for h5_key in output_store.keys():
                table_name = h5_key.split("/")[-1]
                if os.path.join("/", table_prefix_year, table_name) == h5_key:
                    updated_tables.append(table_name)
                    new_input_store[table_name] = output_store[h5_key]

            # copy missing tables from original usim inputs into new input data store
            for h5_key in og_input_store.keys():
                table_name = h5_key.split("/")[-1]
                if table_name not in updated_tables:
                    logger.info(
                        "Copying {0} input table to output store!".format(table_name)
                    )
                    new_input_store[table_name] = og_input_store[h5_key]

            assert new_input_store.keys() == og_input_store.keys()
            og_input_store.close()
            new_input_store.close()
            output_store.close()

            # FIX ISSUE 2: Record the newly created input H5 as an output
            # This file will be used as input for the NEXT UrbanSim run
            if provenance_tracker:
                provenance_tracker.record_output_file(
                    "urbansim_postprocessor",
                    input_store_path,
                    year=forecast_year,
                    description=f"Merged UrbanSim input H5 for next iteration (from year {forecast_year} outputs)",
                    short_name=f"usim_input_merged_{forecast_year}",
                    model_run_id=model_run_hash,
                    source_file_paths=[new_input_store_path, output_store_path],
                    updated_children=updated_tables,
                )
                logger.info(
                    f"Recorded merged input H5 for provenance: {input_store_path}"
                )
        else:
            logger.error(f"Input store path {input_store_path} does not exist")
            return  # or handle this case appropriately
    else:
        # If urbansim is not activated, just log an info message and continue
        logger.info("Urbansim model is not activated, skipping input store path check")


class UrbansimPostprocessor(GenericPostprocessor):
    """
    UrbanSim-specific postprocessor that consolidates all postprocessing steps for the UrbanSim land use model.
    This class is responsible for any postprocessing needed after the UrbanSim run, such as updating outputs,
    provenance, or preparing data for downstream models.
    """

    def postprocess(
        self,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        workspace: Workspace,
        model_run_hash: str = None,
    ) -> RecordStore:
        """
        Postprocess UrbanSim outputs.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            runInfo (ModelRunInfo): Metadata or information about the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (str): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        logger.info(
            "[UrbansimPostprocessor] Postprocessing UrbanSim outputs for year %s",
            self.state.current_year,
        )

        # Start postprocessor tracking if no hash provided
        if model_run_hash is None:
            model_run_hash = self.provenance_tracker.start_model_run(
                "urbansim_postprocessor",
                self.state.current_year,
                self.state.current_inner_iter,
                description="Post-processing UrbanSim outputs",
                inputs=raw_outputs,
            )

        processed_records = []
        settings = self.state.full_settings

        try:
            # Process each raw output file
            for record in raw_outputs.all_records():
                if record.file_path and os.path.exists(record.file_path):
                    logger.info(f"Processing UrbanSim output: {record.file_path}")

                    # TODO: Add specific UrbanSim postprocessing logic here
                    # This might include:
                    # - Validating output file structure
                    # - Converting formats if needed
                    # - Creating summary statistics
                    # - Preparing data for next model iteration

                    # For now, just record the processed file
                    processed_record = self.provenance_tracker.record_output_file(
                        "urbansim_postprocessor",
                        record.file_path,
                        year=self.state.forecast_year,
                        description=f"Processed UrbanSim output: {record.description}",
                        short_name=f"{record.short_name}_processed",
                        model_run_id=model_run_hash,
                        state=self.state,
                        source_file_paths=[record.file_path],
                    )

                    if processed_record:
                        processed_records.append(processed_record)
                else:
                    logger.warning(f"Raw output file not found: {record.file_path}")

            # Handle data preparation for next iteration if needed
            if settings.get("land_use_model") == "urbansim":
                try:
                    # Prepare data for next iteration with provenance tracking
                    create_next_iter_usim_data(
                        settings,
                        self.state.current_year,
                        self.state.forecast_year,
                        workspace.get_usim_mutable_data_dir(),
                        provenance_tracker=self.provenance_tracker,
                        model_run_hash=model_run_hash,
                    )
                    logger.info("Prepared UrbanSim data for next iteration")
                except Exception as e:
                    logger.error(f"Error preparing next iteration data: {e}")
                    # Don't fail the entire postprocessing for this

        except Exception as e:
            logger.error(f"Error during UrbanSim postprocessing: {e}")
            self.provenance_tracker.complete_model_run(model_run_hash, status="failed")
            raise

        # Complete postprocessor tracking
        self.provenance_tracker.complete_model_run(
            model_run_hash, status="completed", output_records=processed_records
        )

        return RecordStore(recordList=processed_records)
