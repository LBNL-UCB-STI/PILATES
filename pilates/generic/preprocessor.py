from abc import ABC, abstractmethod
from typing import Any, Tuple
from pilates.generic.records import RecordStore
from pilates.utils.provenance import FileProvenanceTracker


class GenericPreprocessor(ABC):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() and copy_data_to_mutable_location() methods.
    """

    def __init__(self):
        self.required_input_data: list[str] = []

    @abstractmethod
    def copy_data_to_mutable_location(
        self,
        settings: dict,
        output_dir: str,
        provenance_tracker: FileProvenanceTracker,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy initial data into a mutable location and record the input and output files as provenance.
        Returns a tuple (input_record_store, output_record_store).
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(
        self,
        state: "WorkflowState",
        workspace: "Workspace",
        provenance_tracker: FileProvenanceTracker,
    ) -> RecordStore:
        """
        Preprocess input data for the model.
        """
        raise NotImplementedError
