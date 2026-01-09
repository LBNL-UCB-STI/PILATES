from abc import ABC, abstractmethod
from typing import Optional
from pilates.generic.model import Model, provenance_logging
from pilates.generic.records import RecordStore
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
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)  # new
        self.required_input_data: list[str] = []

    def postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(raw_outputs, workspace, model_run_hash)

    @abstractmethod
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Subclasses should apply the @provenance_logging decorator to their implementation.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        pass
