import abc
import logging
import subprocess
from typing import Tuple

from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState

try:
    import docker
except ImportError:
    print("Warning: Unable to import Docker Module")


logger = logging.getLogger(__name__)


class UrbansimRunner(GenericRunner):

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.required_input_files = [
            "urbansim_h5",
            "hh_size",
            "income_rates",
            "relmap",
            "schools",
            "school_districts",
        ]


    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: "FileProvenanceTracker",
    ) -> Tuple[RecordStore, ModelRunInfo]:

        return RecordStore(), ModelRunInfo(model="activitysim", year=2010)