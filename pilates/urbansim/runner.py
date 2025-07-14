from typing import Tuple
from abc import ABC
import logging
import os

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


class UrbansimRunner(GenericRunner):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.required_input_files = ["usim_data"]

    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        settings = state.full_settings
        forecast_year = state.forecast_year

        usim_output_store_name = settings["usim_formattable_output_file_name"].format(year=forecast_year)
        usim_datastore_fpath = os.path.join(workspace.get_usim_mutable_data_dir(), usim_output_store_name)

        output_records = []
        if os.path.exists(usim_datastore_fpath):
            output_rec = provenance_tracker.record_output_file(
                self.model_name,
                usim_datastore_fpath,
                year=forecast_year,
                description="UrbanSim forecast output data",
                model_run_id=None,  # To be filled in later
            )
            if output_rec:
                output_records.append(output_rec)
        else:
            logger.error(f"UrbanSim output file not found at {usim_datastore_fpath}")

        # ...

        return RecordStore(), None
