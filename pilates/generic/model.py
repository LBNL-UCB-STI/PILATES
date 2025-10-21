from typing import Optional

from pilates.utils.provenance import FileProvenanceTracker


class Model:
    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        self.model_name = model_name
        self.state = state
        self.provenance_tracker = provenance_tracker
        self.major_stage = major_stage  # new

    def update_state(self, state: "WorkflowState"):
        self.state = state
