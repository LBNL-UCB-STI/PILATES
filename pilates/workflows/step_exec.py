import logging
from typing import Protocol

from pilates.activitysim import postprocessor as asim_post
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class Preprocessor(Protocol):
    """Protocol for preprocessors that emit a RecordStore."""

    def preprocess(
        self,
        workspace: Workspace,
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """Run preprocessing for the given workspace."""


class Runner(Protocol):
    """Protocol for runners that consume and emit RecordStores."""

    def run(self, input_store: RecordStore, workspace: Workspace) -> RecordStore:
        """Run the model using the provided inputs."""


class Postprocessor(Protocol):
    """Protocol for postprocessors that emit a RecordStore."""

    def postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: str | None = None,
    ) -> RecordStore:
        """Postprocess model outputs."""


def warm_start_activities(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    """
    Run ActivitySim warm-start to update UrbanSim inputs with long-term choices.

    Parameters
    ----------
    settings : PilatesConfig
        Simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace used to resolve paths.
    """
    factory = ModelFactory()
    activity_demand_model, _ = GenericRunner.get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif activity_demand_model == "activitysim":
        runner = factory.get_runner("activitysim", state)
        preprocessor = factory.get_preprocessor("activitysim", state)

        input_data = preprocessor.preprocess(workspace)
        runner.run(input_data, workspace)

        logger.info(
            "Appending warm start activities/choices to UrbanSim base year input data"
        )

        asim_post.update_usim_inputs_after_warm_start(
            settings,
            state,
            workspace,
            model_run_hash=None,
        )

    logger.info("Done!")
