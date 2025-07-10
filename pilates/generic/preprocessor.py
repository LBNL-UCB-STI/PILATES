from abc import ABC, abstractmethod
from typing import Any
from pilates.generic.records import RecordStore

# from workflow_state import WorkflowState
# from pilates.utils.provenance import FileProvenanceTracker


class GenericPreprocessor(ABC):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() method.
    """

    def __init__(self):
        self.required_input_data: list[str] = []
        pass

    @abstractmethod
    def preprocess(
        self,
        state: "WorkflowState",
        workspace: "Workspace",
        provenance_tracker: "FileProvenanceTracker",
    ) -> RecordStore:
        """
        Preprocess input data for the model.

        Args:
            state (WorkflowState): The workflow state or context object.
            workspace (Workspace): The workspace containing input data.
            provenance_tracker (FileProvenanceTracker): Tracker for file provenance.
            model_run_hash (str): The unique hash for this preprocessor run.

        Returns:
            RecordStore: Preprocessed input data for the model.
        """
        pass
