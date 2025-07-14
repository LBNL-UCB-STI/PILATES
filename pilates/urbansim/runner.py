from typing import Tuple
import os
import logging
import sys
import shutil

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class UrbansimRunner(GenericRunner):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.required_input_files = [
            "urbansim_h5",
            "hh_size",
            "income_rates",
            "relmap",
            "schools",
            "school_districts",
        ]

    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: "FileProvenanceTracker",
    ) -> Tuple[RecordStore, ModelRunInfo]:
        logger.info("Running UrbanSim model")

        settings = state.full_settings
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        usim_docker_vols = {
            workspace.get_usim_mutable_data_dir(): {
                "bind": settings["usim_client_data_folder"],
                "mode": "rw",
            }
        }
        forecast_year = state.forecast_year
        usim_cmd = settings["usim_formattable_command"].format(
            region_id, state.year, forecast_year, settings["land_use_freq"], settings["travel_model"]
        )

        client = None
        if settings.get("container_manager") == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")

        # Start UrbanSim run and get model_run_hash
        usim_run_hash = provenance_tracker.start_model_run(
            self.model_name,
            state.current_year,
            state.current_inner_iter,
            description="UrbanSim run",
            inputs=store,
        )

        success = self.run_container(
            client=client,
            settings=settings,
            image=settings[f"{settings['container_manager']}_images"][self.model_name],
            volumes=usim_docker_vols,
            command=usim_cmd,
            model_name=self.model_name,
            working_dir=settings["usim_client_base_folder"],
        )

        if not success:
            logger.error("UrbanSim run failed.")
            provenance_tracker.complete_model_run(usim_run_hash, status="failed")
            sys.exit(1)

        # After successful run, record outputs
        usim_output_store_name = settings["usim_formattable_output_file_name"].format(year=forecast_year)
        usim_datastore_fpath = os.path.join(workspace.get_usim_mutable_data_dir(), usim_output_store_name)

        output_records = []
        if os.path.exists(usim_datastore_fpath):
            output_rec = provenance_tracker.record_output_file(
                self.model_name,
                usim_datastore_fpath,
                year=forecast_year,
                description="UrbanSim forecast output data",
                model_run_id=usim_run_hash,
            )
            if output_rec:
                output_records.append(output_rec)
        else:
            logger.error(f"UrbanSim output file not found at {usim_datastore_fpath}")

        provenance_tracker.complete_model_run(
            usim_run_hash, status="completed" if success else "failed", output_records=output_records
        )

        return RecordStore(recordList=output_records), provenance_tracker.run_info.model_runs.get(usim_run_hash)
