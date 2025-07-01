from abc import ABC, abstractmethod
from typing import Any
from pilates.generic.records import RecordStore
from pilates.utils.provenance import FileProvenanceTracker
from workflow_state import WorkflowState


class GenericPreprocessor(ABC):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() method.
    """

    def __init__(self, provenanceTracker: FileProvenanceTracker):
        self.provenanceTracker = provenanceTracker

    @classmethod
    @abstractmethod
    def preprocess(cls, state: WorkflowState) -> RecordStore:
        """
        Preprocess input data for the model.

        Args:
            state: The workflow state or context object.

        Returns:
            RecordStore: Preprocessed input data for the model.
        """
        pass
