import logging
import os
import sys
from typing import Optional, Protocol

from pilates.activitysim import postprocessor as asim_post
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore
from pilates.generic.runner import GenericRunner
from pilates.utils.formatting import formatted_print
from pilates.workspace import Workspace
from pilates.urbansim import postprocessor as usim_post
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

    def postprocess(self, raw_outputs: RecordStore, workspace: Workspace) -> RecordStore:
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


def forecast_land_use(
    settings: PilatesConfig,
    year: int,
    workflow_state: WorkflowState,
    workspace: Workspace,
) -> None:
    """
    High-level wrapper to start an UrbanSim (land use) run.

    Parameters
    ----------
    settings : PilatesConfig
        Simulation settings.
    year : int
        Current simulation year.
    workflow_state : WorkflowState
        Workflow state for the run.
    workspace : Workspace
        Workspace used to resolve paths.
    """
    land_use_model, _ = GenericRunner.get_model_and_image(settings, "land_use_model")

    run_land_use(
        year,
        workflow_state.forecast_year,
        land_use_model,
        workflow_state,
        workspace,
    )

    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=workflow_state.forecast_year
    )
    usim_datastore_fpath = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_output_store_name
    )

    if not os.path.exists(usim_datastore_fpath):
        logger.critical(
            "No UrbanSim output data found at %s. Run failed.",
            usim_datastore_fpath,
        )
        sys.exit(1)


def run_land_use(
    year: int,
    forecast_year: int,
    land_use_model: str,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    """
    Prepare inputs, run UrbanSim, and postprocess outputs for a land-use forecast.

    Parameters
    ----------
    year : int
        Base year for the land-use forecast.
    forecast_year : int
        Target forecast year.
    land_use_model : str
        Model key identifying the land-use backend.
    state : WorkflowState
        Workflow state for the run.
    workspace : Workspace
        Workspace used to resolve paths.
    """
    logger.info("Running land use")

    factory = ModelFactory()

    preprocessor, runner, postprocessor = factory.get_components(
        "urbansim", state, major_stage=WorkflowState.Stage.land_use
    )

    formatted_print(f"Preparing {year} input data for land use development simulation.")
    input_data = preprocessor.preprocess(workspace)
    formatted_print(
        f"Simulating land use development from {year} to {forecast_year} with {land_use_model}."
    )
    raw_outputs = runner.run(input_data, workspace)
    postprocessor.postprocess(raw_outputs, workspace)

    logger.info("Done!")


def run_activity_demand(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    input_store: Optional[RecordStore] = None,
) -> RecordStore:
    """
    Generate activity plans for the current year using the configured model.

    Parameters
    ----------
    settings : PilatesConfig
        Simulation settings.
    state : WorkflowState
        Workflow state for the run.
    workspace : Workspace
        Workspace used to resolve paths.
    input_store : RecordStore, optional
        Precomputed inputs; when omitted the preprocessor will be run.

    Returns
    -------
    RecordStore
        Postprocessed ActivitySim outputs.
    """
    factory = ModelFactory()
    activity_demand_model = settings.run.models.activity_demand

    if activity_demand_model == "polaris":
        logger.info("POLARIS module is not activated")
        return RecordStore()

    if activity_demand_model == "activitysim":
        preprocessor, runner, postprocessor = factory.get_components(
            "activitysim",
            state,
            major_stage=WorkflowState.Stage.activity_demand,
        )

        input_data = input_store or preprocessor.preprocess(workspace)
        raw_outputs = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace)

        return processed_outputs

    logger.warning("Unknown activity demand model: %s", activity_demand_model)
    return RecordStore()


def run_traffic_assignment(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    activity_demand_outputs: Optional[RecordStore] = None,
    previous_beam_outputs: Optional[RecordStore] = None,
) -> RecordStore:
    """
    Run the configured traffic assignment (supply) model for the current year.

    Parameters
    ----------
    settings : PilatesConfig
        Simulation settings.
    state : WorkflowState
        Workflow state for the run.
    workspace : Workspace
        Workspace used to resolve paths.
    activity_demand_outputs : RecordStore, optional
        Activity demand outputs for the current iteration.
    previous_beam_outputs : RecordStore, optional
        Prior BEAM outputs for warm starts.

    Returns
    -------
    RecordStore
        Postprocessed BEAM outputs.
    """
    factory = ModelFactory()
    travel_model = settings.run.models.travel

    if travel_model == "polaris":
        logger.info("POLARIS module is not activated")
        return RecordStore()
    if travel_model == "beam":
        preprocessor, runner, postprocessor = factory.get_components(
            "beam",
            state,
            major_stage=WorkflowState.Stage.traffic_assignment,
        )

        combined_inputs = RecordStore()
        if activity_demand_outputs:
            combined_inputs += activity_demand_outputs
        if previous_beam_outputs:
            combined_inputs += previous_beam_outputs
        input_data = preprocessor.preprocess(workspace, combined_inputs)
        raw_outputs = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace)
        return processed_outputs

    logger.warning("Unknown travel model: %s", travel_model)
    return RecordStore()
