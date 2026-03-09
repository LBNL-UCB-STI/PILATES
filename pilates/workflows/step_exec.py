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

    def preprocess(self, workspace: Workspace) -> RecordStore:
        """Run preprocessing for the given workspace."""


class Runner(Protocol):
    """Protocol for runners that consume and emit RecordStores."""

    def run(self, input_store: RecordStore, workspace: Workspace) -> RecordStore:
        """Run the model using the provided inputs."""


class Postprocessor(Protocol):
    """Protocol for postprocessors that emit a RecordStore."""

    def postprocess(
        self, raw_outputs: RecordStore, workspace: Workspace
    ) -> RecordStore:
        """Postprocess model outputs."""


def run_preprocessor(preprocessor: Preprocessor, workspace: Workspace) -> RecordStore:
    """
    Execute a preprocessor and return its RecordStore outputs.

    Parameters
    ----------
    preprocessor : Preprocessor
        Component providing a ``preprocess`` method.
    workspace : Workspace
        Workspace used to resolve inputs/outputs.

    Returns
    -------
    RecordStore
        Record store of preprocessor outputs.
    """
    return preprocessor.preprocess(workspace)


def run_runner(
    runner: Runner,
    input_store: RecordStore,
    workspace: Workspace,
) -> RecordStore:
    """
    Execute a runner with provided inputs and return its RecordStore outputs.

    Parameters
    ----------
    runner : Runner
        Component providing a ``run`` method.
    input_store : RecordStore
        Inputs prepared by preprocessing.
    workspace : Workspace
        Workspace used to resolve inputs/outputs.

    Returns
    -------
    RecordStore
        Record store of runner outputs.
    """
    return runner.run(input_store, workspace)


def run_postprocessor(
    postprocessor: Postprocessor,
    raw_outputs: RecordStore,
    workspace: Workspace,
) -> RecordStore:
    """
    Execute a postprocessor and return its RecordStore outputs.

    Parameters
    ----------
    postprocessor : Postprocessor
        Component providing a ``postprocess`` method.
    raw_outputs : RecordStore
        Raw outputs from a runner.
    workspace : Workspace
        Workspace used to resolve inputs/outputs.

    Returns
    -------
    RecordStore
        Record store of postprocessed outputs.
    """
    return postprocessor.postprocess(raw_outputs, workspace)


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
