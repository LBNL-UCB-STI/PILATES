from abc import ABC, abstractmethod
from typing import Any
from pilates.generic.records import RecordStore
from workflow_state import WorkflowState


class GenericPreprocessor(ABC):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def preprocess(
        self,
        state: WorkflowState,
        workspace: "Workspace",
        provenance_tracker: "FileProvenanceTracker",
        model_run_hash: str,
    ) -> RecordStore:
        """
        Preprocess input data for the model.

        Args:
            state: The workflow state or context object.
            workspace: The workspace containing input data.
            provenance_tracker: Tracker for file provenance.
            model_run_hash: The unique hash for this preprocessor run.

        Returns:
            RecordStore: Preprocessed input data for the model.
        """
        pass
