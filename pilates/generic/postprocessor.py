from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TYPE_CHECKING, TypeVar

from pilates.generic.model import Model

if TYPE_CHECKING:
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


PostprocessInputsT = TypeVar("PostprocessInputsT")
PostprocessOutputsT = TypeVar("PostprocessOutputsT")


class GenericPostprocessor(
    Model, ABC, Generic[PostprocessInputsT, PostprocessOutputsT]
):
    """Base class for postprocessors with model-specific input and output types."""

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_data: list[str] = []

    def postprocess(
        self,
        raw_outputs: PostprocessInputsT,
        workspace: "Workspace",
        model_run_hash: Optional[str] = None,
    ) -> PostprocessOutputsT:
        """Run postprocessing on runner outputs and return final model outputs."""
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(raw_outputs, workspace, model_run_hash)

    @abstractmethod
    def _postprocess(
        self,
        raw_outputs: PostprocessInputsT,
        workspace: "Workspace",
        model_run_hash: Optional[str] = None,
    ) -> PostprocessOutputsT:
        """Implement postprocessing and return model-specific outputs."""
        raise NotImplementedError
