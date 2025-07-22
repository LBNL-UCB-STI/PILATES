from abc import ABC, abstractmethod
from typing import Any, Tuple

from pilates.generic.model import Model
from pilates.generic.records import RecordStore
from pilates.utils.provenance import FileProvenanceTracker


class GenericPreprocessor(ABC, Model):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() and copy_data_to_mutable_location() methods.
    """

    def __init__(self, model_name: str, state: "WorkflowState", provenance_tracker: FileProvenanceTracker):
        super().__init__(model_name, state, provenance_tracker)
        self.required_input_data: list[str] = []

    @abstractmethod
    def copy_data_to_mutable_location(
        self,
        settings: dict,
        output_dir: str,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy initial data into a mutable location and record the input and output files as provenance.
        Returns a tuple (input_record_store, output_record_store).
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess input data for the model.
        """
        raise NotImplementedError
