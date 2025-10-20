from abc import ABC, abstractmethod
from typing import Optional
from pilates.generic.model import Model
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker


class GenericPostprocessor(ABC, Model):
    """
    Abstract base class for all model postprocessors.
    Subclasses should implement the postprocess() method.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
    ):
        super().__init__(model_name, state, provenance_tracker)
        self.required_input_data: list[str] = []

    @abstractmethod
    def postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        runInfo: Optional[ModelRunInfo] = None,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            runInfo (Optional[ModelRunInfo]): Metadata about the model run.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        pass
