from abc import ABC, abstractmethod
from typing import Any, Tuple
from pilates.generic.records import RecordStore, ModelRunInfo
from workflow_state import WorkflowState


class GenericRunner(ABC):
    """
    Abstract base class for all model runners.
    Subclasses should implement the run() method.
    """

    @classmethod
    @abstractmethod
    def run(cls, inputs: RecordStore, state: WorkflowState) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Run the model.

        Args:
            inputs (RecordStore): The preprocessed input data for the model.
            state: The workflow state or context object.

        Returns:
            Tuple[RecordStore, ModelRunInfo]: raw_outputs and runInfo.
        """
        pass
