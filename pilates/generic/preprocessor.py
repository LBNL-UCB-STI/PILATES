from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import RecordStore

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace


class GenericPreprocessor(ABC, Model):
    """Base class for preprocessors that stage and emit ``RecordStore`` data."""

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
    ) -> Tuple[RecordStore, RecordStore]:
        """Copy immutable seed inputs into the mutable workspace."""
        raise NotImplementedError

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
    ) -> RecordStore:
        """Run preprocessing using upstream ``RecordStore`` inputs."""
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(
            workspace,
            previous_records if previous_records is not None else RecordStore(),
        )

    @abstractmethod
    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore,
    ) -> RecordStore:
        """Implement preprocessing and return a ``RecordStore`` of outputs."""
        raise NotImplementedError
