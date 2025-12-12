"""
run.py

Main entrypoint and workflow orchestrator for PILATES simulations.

This module:
- Parses settings and initializes workflow state.
- Initializes the Consist Tracker and Scenario Context.
- Executes the multi-stage simulation loop using the Scenario/Step API.
- Manages provenance for the critical "Data Initialization" step to link
  immutable inputs to the mutable workspace.
"""

import warnings
from datetime import datetime
import uuid
import os
import logging
import sys

# Consist Imports
from consist import Tracker, Artifact
from consist.models.run import Run

# Legacy/PILATES Imports
from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore, Record
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from pilates.generic.initialization import Initialization
from pilates.config.models import PilatesConfig
from pilates.utils.io import parse_args_and_settings
from pilates.postprocessing.postprocessor import process_event_file, copy_outputs_to_mep
from pilates.utils.consist_adapter import ConsistProvenanceTracker

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.activitysim import postprocessor as asim_post
from pilates.urbansim import postprocessor as usim_post

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def formatted_print(string, width=50, fill_char="#"):
    """
    Print a formatted banner for major workflow steps.
    """
    print("\n")
    if len(string) + 2 > width:
        width = len(string) + 4
    string = string.upper()
    print(fill_char * width)
    print("{:#^{width}}".format(" " + string + " ", width=width))
    print(fill_char * width, "\n")


def warm_start_activities(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    TODO: THIS IS BROKEN

    Run ActivitySim warm-start to update UrbanSim inputs with long-term choices.

    This function is typically executed only in the initial start year as part of
    the Land Use step when ActivitySim is used as the activity demand model.

    Sequence:
      - Instantiate ActivitySim preprocessor/runner via ModelFactory
      - Preprocess to prepare ActivitySim inputs
      - Run ActivitySim runner to produce outputs
      - Apply a specialized post-processing function that updates UrbanSim H5 input
        files with warm-started choices (workplace, school, auto ownership)
      - Record the postprocessor model run with the provenance tracker

    Args:
        settings (PilatesConfig): Parsed simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace object with file paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
    """
    factory = ModelFactory()
    activity_demand_model, _ = GenericRunner.get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        logger.info("POLARIS module is not activated due to missing polarisruntime library")

    elif activity_demand_model == "activitysim":
        runner = factory.get_runner("activitysim", state, provenance_tracker)
        preprocessor = factory.get_preprocessor("activitysim", state, provenance_tracker)

        # Preprocess
        inputData = preprocessor.preprocess(workspace)

        # Run ActivitySim
        _, runInfo = runner.run(inputData, workspace)

        logger.info("Appending warm start activities/choices to UrbanSim base year input data")

        asim_post.update_usim_inputs_after_warm_start(
            settings,
            state,
            runInfo,
            usim_data_dir=workspace.get_usim_mutable_data_dir(),
            warm_start_dir=workspace.get_asim_output_dir(),
            provenance_tracker=provenance_tracker,
            model_run_hash=None,
        )

    logger.info("Done!")


def forecast_land_use(
    settings: PilatesConfig,
    year: int,
    workflow_state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    High-level wrapper to start an UrbanSim (land use) run.
    """
    land_use_model, _ = GenericRunner.get_model_and_image(
        settings, "land_use_model"
    )

    run_land_use(
        year,
        workflow_state.forecast_year,
        land_use_model,
        workflow_state,
        workspace,
        provenance_tracker,
    )

    # Verify Output
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=workflow_state.forecast_year
    )
    usim_datastore_fpath = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_output_store_name
    )

    if not os.path.exists(usim_datastore_fpath):
        logger.critical(
            f"No UrbanSim output data found at {usim_datastore_fpath}. Run failed."
        )
        sys.exit(1)


def run_land_use(
    year,
    forecast_year,
    land_use_model,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    Prepare inputs, run UrbanSim, and postprocess outputs for a land-use forecast.

    Args:
        year (int): Current simulation year (start of forecast).
        forecast_year (int): Forecast target year.
        land_use_model (str): Land use model identifier (e.g., 'urbansim').
        state (WorkflowState): Workflow state.
        workspace (Workspace): Workspace instance for data paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
    """
    logger.info("Running land use")

    factory = ModelFactory()

    preprocessor = factory.get_preprocessor(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )
    runner = factory.get_runner(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )
    postprocessor = factory.get_postprocessor(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )

    # 1. PREPROCESS
    formatted_print(f"Preparing {year} input data for land use development simulation.")
    input_data = preprocessor.preprocess(workspace)

    # 2. RUN
    formatted_print(f"Simulating land use development from {year} to {forecast_year} with {land_use_model}.")
    raw_outputs, run_info = runner.run(input_data, workspace)

    # 3. POSTPROCESS
    postprocessor.postprocess(raw_outputs, workspace, run_info)

    logger.info("Done!")


def run_activity_demand(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
) -> RecordStore:
    """
    Generate activity plans for the current year using the configured activity demand model.

    Supports:
      - ActivitySim: full preprocess -> run -> postprocess sequence
      - Polaris: logged but not implemented here

    Args:
        settings (PilatesConfig): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance for file operations.
        provenance_tracker (OpenLineageTracker): Provenance tracker for model events.

    Returns:
        RecordStore: Processed outputs (file records) from the activity demand postprocessor.
    """
    factory = ModelFactory()
    activity_demand_model = settings.run.models.activity_demand

    if activity_demand_model == "polaris":
        logger.info("POLARIS module is not activated")
        return RecordStore()

    elif activity_demand_model == "activitysim":
        preprocessor = factory.get_preprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )
        runner = factory.get_runner(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )
        postprocessor = factory.get_postprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )

        input_data = preprocessor.preprocess(workspace)
        raw_outputs, run_info = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace, run_info)

        return processed_outputs

    else:
        logger.warning(f"Unknown activity demand model: {activity_demand_model}")
        return RecordStore()


def run_traffic_assignment(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
    activity_demand_outputs: RecordStore = None,
):
    """
    Run the configured traffic assignment (supply) model for the current year/iteration.

    Currently supports:
      - BEAM: obtains preprocessor, runner, postprocessor via ModelFactory and runs the
        preprocessor with the activity demand outputs as input.

    Args:
        settings (PilatesConfig): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
        activity_demand_outputs (RecordStore, optional): Processed activity demand outputs.
    """
    factory = ModelFactory()
    travel_model = settings.run.models.travel

    if travel_model == "polaris":
        logger.info("POLARIS module is not activated")
    elif travel_model == "beam":
        preprocessor = factory.get_preprocessor(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )
        runner = factory.get_runner(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )
        postprocessor = factory.get_postprocessor(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )

        input_data = preprocessor.preprocess(workspace, activity_demand_outputs)
        raw_outputs, run_info = runner.run(input_data, workspace)
        postprocessor.postprocess(raw_outputs, workspace, run_info)

    else:
        logger.warning(f"Unknown travel model: {travel_model}")


def main():
    """
    Main entrypoint refactored to use Consist Scenario API.
    """
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

    # 2. SETUP PATHS
    output_directory = settings.run.output_directory
    if not output_directory:
        raise ValueError("output_directory not found in config")
    output_path = os.path.realpath(os.path.expandvars(output_directory))

    if state.run_info_path:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        partial_run_name = settings.run.output_run_name
        run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting fresh run. Creating new output folder: {run_name}")

    full_run_dir = os.path.join(output_path, run_name)
    os.makedirs(full_run_dir, exist_ok=True)

    # 3. INITIALIZE CONSIST TRACKER
    # Mount Strategy:
    # - 'inputs': The project root. Source files resolve here.
    # - 'workspace': The mutable run dir. Destination files resolve here.
    project_root_abs = os.getcwd()

    logger.info(f"Initializing Consist Tracker in {full_run_dir}")
    tracker = Tracker(
        run_dir=full_run_dir,
        db_path=settings.shared.database.path if settings.shared.database.enabled else None,
        mounts={
            "inputs": project_root_abs,  # Immutable Source
            "workspace": full_run_dir  # Mutable Destination
        },
        project_root=project_root_abs
    )

    # 4. INITIALIZE WORKSPACE & ADAPTER
    # We pass the native 'tracker' to the adapter.
    # The adapter will detect active scenario steps and "attach" to them.
    adapter = ConsistProvenanceTracker(
        run_id="placeholder_id",  # Will be overwritten by attach mode
        output_path=full_run_dir,
        folder_name=run_name,
        tracker=tracker
    )

    workspace = Workspace(
        settings,
        output_path,
        folder_name=run_name,
        provenance_tracker=adapter,
    )
    state.file_loc = os.path.join(workspace.full_path, "run_state.yaml")


    # 5. START SCENARIO
    with tracker.scenario(
            name=run_name,
            config=settings.get_initialization_signature(),
            tags=["pilates_simulation"],
            model="pilates_orchestrator"
    ) as scenario:


        # 6. DATA INITIALIZATION STEP
        if not state.data_initialized:
            logger.info("Running Initialization Step (Copying mutable data)")

            # We use 'initialization' as the step name.
            # Initialization.py calls `adapter.start_model_run("initialization")` internally.
            # The Adapter will see this active step and ATTACH to it, rather than creating a new one.
            with scenario.step(
                "initialization",
                model="initialization",
                year=state.start_year,
                iteration=0,
            ):

                init_model = Initialization("initialization", state, provenance_tracker=adapter)

                # This performs the copy.
                # Source files -> recorded as inputs (inputs://...)
                # Dest files -> recorded as outputs (workspace://...)
                init_model.run(settings, workspace)

            state.set_data_initialized(True)
        else:
            logger.info("Restarting from a previous state. Skipping data initialization.")

        # 6. MAIN WORKFLOW LOOP
        for year in state:
            formatted_print(f"STARTING YEAR {year}")

            # A. LAND USE FORECASTING
            if state.should_run(WorkflowState.Stage.land_use):
                formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")

                step_name = f"urbansim_{year}"

                with scenario.step(step_name, model="urbansim", year=year, iteration=0):
                    if state.is_start_year() and settings.activitysim.warm_start_activities:
                        logger.info("[Main] Running warm start activities for ActivitySim.")
                        warm_start_activities(settings, state, workspace, adapter)

                    forecast_land_use(settings, year, state, workspace, adapter)

                state.complete_step(WorkflowState.Stage.land_use)

            # B. VEHICLE OWNERSHIP MODEL (ATLAS)
            if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                formatted_print(f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}")
                logger.info("[Main] Running ATLAS vehicle ownership model.")

                # ATLAS Logic extraction
                factory = ModelFactory()
                preprocessor = factory.get_preprocessor("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)
                runner = factory.get_runner("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)
                postprocessor = factory.get_postprocessor("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)

                warm_start_atlas = state.is_start_year()
                forecast = True
                yrs = [state.year] + [y + 2 for y in range(state.year, state.forecast_year, 2)] if forecast else [state.year]
                if not yrs and forecast: yrs = [state.forecast_year]

                # ATLAS Sub-loop
                for atlas_year in yrs:
                    # Create SubState
                    class AtlasSubState:
                        def __init__(self, parent_state, year):
                            self.__dict__ = parent_state.__dict__.copy()
                            self.year = year
                            self.current_year = year
                            self.forecast_year = year
                            self.main_forecast_year = parent_state.forecast_year
                            self.start_year = parent_state.start_year
                            self.full_settings = parent_state.full_settings
                            self.is_start_year = lambda: (year == parent_state.start_year)
                        def set_sub_stage_progress(self, sub_stage_progress):
                            state.set_sub_stage_progress(sub_stage_progress)

                    atlas_state = AtlasSubState(state, atlas_year)
                    step_name = f"atlas_{atlas_year}"

                    # Run ATLAS Step
                    with scenario.step(step_name, model="atlas", year=atlas_year, iteration=0):
                        # 1. Preprocess
                        preprocessor.update_state(atlas_state)
                        input_data = preprocessor.preprocess(workspace)

                        # 2. Run
                        runner.update_state(atlas_state)
                        try:
                            raw_outputs, run_info = runner.run(input_data, workspace)
                            if run_info and run_info.status == "completed":
                                # 3. Postprocess
                                postprocessor.update_state(atlas_state)
                                postprocessor.postprocess(raw_outputs, workspace, run_info)
                            else:
                                raise RuntimeError(f"AtlasRunner incomplete: {run_info.status if run_info else 'None'}")
                        except Exception as e:
                            logger.error(f"ATLAS failed for {atlas_year}: {e}")
                            sys.exit(1)

                state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

            # C. SUPPLY/DEMAND LOOP
            if state.should_run(WorkflowState.Stage.supply_demand_loop):
                total_iters = settings.run.supply_demand_iters

                for i in range(state.iteration, total_iters):
                    state.iteration = i
                    formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
                    activity_demand_outputs = None

                    # C1. ACTIVITY DEMAND
                    if state.should_run(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.activity_demand):
                        formatted_print("ACTIVITY DEMAND MODEL")
                        step_name = f"activitysim_{year}_iter{i}"

                        with scenario.step(step_name, model="activitysim", year=year, iteration=i):
                            activity_demand_outputs = run_activity_demand(settings, state, workspace, adapter)

                        state.complete_step(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.activity_demand)

                    # C2. TRAFFIC ASSIGNMENT
                    if state.should_run(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.traffic_assignment):
                        formatted_print("TRAFFIC ASSIGNMENT MODEL")
                        step_name = f"beam_{year}_iter{i}"

                        with scenario.step(step_name, model="beam", year=year, iteration=i):
                            run_traffic_assignment(settings, state, workspace, adapter, activity_demand_outputs)

                        state.complete_step(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.traffic_assignment)

                state.complete_step(WorkflowState.Stage.supply_demand_loop)

            # D. POST-PROCESSING
            if state.should_run(WorkflowState.Stage.postprocessing):
                formatted_print("POST-PROCESSING")
                with scenario.step(f"postprocessing_{year}", model="postprocessing", year=year):
                    if "postprocessing" in settings:
                        process_event_file(settings, state, workspace, tracker)
                        copy_outputs_to_mep(settings, state, workspace, tracker)
                state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")
    logger.info("[Main] Simulation complete.")


if __name__ == "__main__":
    main()
