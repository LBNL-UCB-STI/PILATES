from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import RecordStore

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


class GenericPreprocessor(ABC, Model):
    """
    Abstract base class for all model preprocessors.
    Subclasses should implement the preprocess() and copy_data_to_mutable_location() methods.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        super().__init__(model_name, state, major_stage)  # new
        self.required_input_data: list[str] = []

    @abstractmethod
    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir: str,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy initial data into a mutable location.
        Returns a tuple (input_record_store, output_record_store).
        """
        raise NotImplementedError

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess input data for the model.
        """
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(workspace, previous_records)

    @abstractmethod
    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Preprocess input data for the model.

        Subclasses should return a RecordStore of outputs without provenance side effects.
        """
        raise NotImplementedError
