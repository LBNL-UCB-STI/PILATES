import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import RecordStore
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


class GenericPreprocessor(ABC, Model):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() and copy_data_to_mutable_location() methods.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)  # new
        self.required_input_data: list[str] = []

    @abstractmethod
    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir: str,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy initial data into a mutable location and record the input and output files as provenance.
        Returns a tuple (input_record_store, output_record_store).
        """
        raise NotImplementedError

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess input data for the model.
        """
        if (
            self.state.current_major_stage == self.major_stage
            and self.state.sub_stage_progress in ["runner", "postprocessor"]
        ):
            logger.info("Skipping preprocessor, loading outputs from provenance.")
            preprocessor_run = self.provenance_tracker.get_latest_completed_model_run(
                f"{self.model_name}_preprocessor",
                self.state.current_year,
                self.state.current_inner_iter,
            )
            if preprocessor_run:
                return RecordStore.from_file_records(
                    preprocessor_run.output_record_hashes,
                    self.provenance_tracker.run_info.file_records,
                )
            else:
                logger.warning(
                    "Could not find completed preprocessor run in provenance, re-running preprocessor."
                )
                self.state.set_sub_stage_progress("preprocessor")
                return self._preprocess(workspace, previous_records)
        else:
            self.state.set_sub_stage_progress("preprocessor")
            return self._preprocess(workspace, previous_records)

    @abstractmethod
    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess input data for the model.
        """
        raise NotImplementedError
