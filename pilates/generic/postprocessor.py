from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from pilates.generic.model import Model
from pilates.generic.records import RecordStore

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


class GenericPostprocessor(ABC, Model):
    """Base class for postprocessors that consume and emit ``RecordStore`` data."""

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)
        self.required_input_data: list[str] = []

    def postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: "Workspace",
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """Run postprocessing on runner outputs and return a ``RecordStore``."""
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(raw_outputs, workspace, model_run_hash)

    @abstractmethod
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: "Workspace",
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """Implement postprocessing and return a ``RecordStore`` of outputs."""
        raise NotImplementedError
