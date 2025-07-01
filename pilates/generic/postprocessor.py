from abc import ABC, abstractmethod
from typing import Any
from pilates.generic.records import RecordStore, ModelRunInfo
from workflow_state import WorkflowState


class GenericPostprocessor(ABC):
    """
    Abstract base class for all model postprocessors.
    Subclasses should implement the postprocess() method.
    """

    @classmethod
    @abstractmethod
    def postprocess(
        cls,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        state: WorkflowState
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            runInfo (ModelRunInfo): Metadata or information about the model run.
            state: The workflow state or context object.

        Returns:
            RecordStore: Postprocessed output data.
        """
        pass
