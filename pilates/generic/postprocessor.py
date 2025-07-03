from abc import ABC, abstractmethod
from typing import Any
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

class GenericPostprocessor(ABC):
    """
    Abstract base class for all model postprocessors.
    Subclasses should implement the postprocess() method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def postprocess(
        self,
        raw_outputs: RecordStore,
        runInfo: ModelRunInfo,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
        model_run_hash: str,
    ) -> RecordStore:
        """
        Postprocess the output data for the model.

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            runInfo (ModelRunInfo): Metadata or information about the model run.
            state (WorkflowState): The workflow state or context object.
            workspace (Workspace): The workspace object for path management.
            provenance_tracker (FileProvenanceTracker): The provenance tracker.
            model_run_hash (str): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
        """
        pass
