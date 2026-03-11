import logging

from pilates.activitysim import postprocessor as asim_post
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


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
