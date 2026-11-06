from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from pilates.workflows.state import WorkflowState
    from pilates.workspace import Workspace


PreprocessInputsT = TypeVar("PreprocessInputsT")
PreprocessOutputsT = TypeVar("PreprocessOutputsT")


class GenericPreprocessor(ABC, Generic[PreprocessInputsT, PreprocessOutputsT]):
    """Base class for preprocessors with model-specific staged outputs."""

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        self.model_name = model_name
        self.state = state

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[PreprocessInputsT] = None,
    ) -> PreprocessOutputsT:
        """Run preprocessing using optional prior records and return staged outputs."""
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(workspace, previous_records)

    @abstractmethod
    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[PreprocessInputsT],
    ) -> PreprocessOutputsT:
        """Implement preprocessing and return model-specific staged outputs."""
        raise NotImplementedError
