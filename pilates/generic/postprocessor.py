from abc import ABC, abstractmethod
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

    def __init__(self, model_name: str, state: "WorkflowState", provenance_tracker: FileProvenanceTracker):
        super().__init__(model_name, state, provenance_tracker)

    @abstractmethod
    def postprocess(
        self,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        workspace: Workspace,
        model_run_hash: str,
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            runInfo (ModelRunInfo): Metadata or information about the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (str): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        pass
