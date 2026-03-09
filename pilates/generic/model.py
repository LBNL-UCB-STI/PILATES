from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from workflow_state import WorkflowState


class Model:
    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        self.model_name = model_name
        self.state = state
