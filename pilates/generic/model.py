from pilates.utils.provenance import FileProvenanceTracker


class Model:
    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
    ):
        self.model_name = model_name
        self.state = state
        self.provenance_tracker = provenance_tracker

    def update_state(self, state: "WorkflowState"):
        self.state = state
