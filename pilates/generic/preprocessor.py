from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TYPE_CHECKING, TypeVar

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import RecordStore

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace


PreprocessInputsT = TypeVar("PreprocessInputsT")
PreprocessOutputsT = TypeVar("PreprocessOutputsT")


class GenericPreprocessor(Model, ABC, Generic[PreprocessInputsT, PreprocessOutputsT]):
    """Base class for preprocessors with model-specific staged outputs."""

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_data: list[str] = []

    @abstractmethod
    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir: str,
        workspace: Optional["Workspace"] = None,
    ) -> Tuple[RecordStore, RecordStore]:
        """Copy immutable seed inputs into the mutable workspace."""
        raise NotImplementedError

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
