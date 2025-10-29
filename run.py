"""
run.py

Main entrypoint and workflow orchestrator for PILATES simulations.

This module:
- Parses settings and initializes workflow state, workspace, and provenance tracking.
- Executes the multi-stage simulation loop across years and internal iterations:
  - Land use forecasting (UrbanSim)
  - Vehicle ownership (ATLAS)
  - Supply/Demand loop (ActivitySim \-> BEAM)
  - Post-processing and output copying
- Uses ModelFactory to obtain model-specific preprocessors, runners, and postprocessors.
- Records provenance via OpenLineageTracker.
"""

import warnings
from datetime import datetime
import uuid

import pandas as pd
from tables import HDF5ExtError

from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore

# Helper import for model/image lookup and docker client
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from pilates.utils.provenance import OpenLineageTracker
from pilates.generic.initialization import Initialization

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState


try:
    import docker
except ImportError:
    print("Warning: Unable to import Docker Module")

import os
import logging
import sys

from pilates.activitysim import postprocessor as asim_post
from pilates.urbansim import postprocessor as usim_post
from pilates.utils.io import parse_args_and_settings
from pilates.postprocessing.postprocessor import process_event_file, copy_outputs_to_mep

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_already_opened_in_write_mode(filename):
    """
    Check if a file is already opened in a mode that would prevent writing.

    This attempts to open the HDF5 file in append mode. If the underlying HDF5
    library reports the file is already in use, or a RuntimeError occurs while
    opening, we treat the file as locked/unavailable.

    Args:
        filename (str): Path to the HDF5 or datastore file.

    Returns:
        bool: True if the file is currently opened/locked for writing, False otherwise.
    """
    if os.path.exists(filename):
        try:
            f = pd.HDFStore(filename, "a")
            f.close()
        except HDF5ExtError:
            return True
        except RuntimeError as e:
            logger.warning(str(e))
            return True
    return False


def record_inputs_and_outputs(
    provenance_tracker, model, inputs=None, outputs=None, year=None, model_run_id=None
):
    """
    Record model inputs and outputs with the provenance tracker.

    This helper iterates over tuples of (file_path, description) and records
    them as inputs or outputs if the files exist. Missing files are logged as warnings.

    Args:
        provenance_tracker (OpenLineageTracker): Tracker used to log provenance events.
        model (str): Model name or identifier for provenance context.
        inputs (Iterable[Tuple[str, str]]): Iterable of (file_path, description) for inputs.
        outputs (Iterable[Tuple[str, str]]): Iterable of (file_path, description) for outputs.
        year (int, optional): Year associated with outputs (if any).
        model_run_id (str, optional): Identifier for the specific model run.
    """
    if inputs:
        for file_path, description in inputs:
            if os.path.exists(file_path):
                provenance_tracker.record_input_file(
                    model, file_path, description=description, model_run_id=model_run_id
                )
            else:
                logger.warning(f"Input file not found: {file_path}")

    if outputs:
        for file_path, description in outputs:
            if os.path.exists(file_path):
                provenance_tracker.record_output_file(
                    model,
                    file_path,
                    year=year,
                    description=description,
                    model_run_id=model_run_id,
                )
            else:
                logger.warning(f"Output file not found: {file_path}")


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


def find_latest_beam_iteration(beam_output_dir):
    """
    Find the latest BEAM iteration directory (if any).
    """
    iter_dirs = []
    for root, dirs, files in os.walk(beam_output_dir):
        for dir in dirs:
            if dir == "ITER":
                iter_dirs.append(os.path.join(root, dir))
    if iter_dirs:
        logger.info(f"Found BEAM iteration directories: {iter_dirs}")
    else:
        logger.info("No BEAM iteration directories found.")
    return iter_dirs


def warm_start_activities(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
):
    """
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
        settings (dict): Parsed simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace object with file paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
    """
    factory = ModelFactory()
    activity_demand_model, activity_demand_image = GenericRunner.get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif activity_demand_model == "activitysim":
        runner = factory.get_runner("activitysim", state, provenance_tracker)
        preprocessor = factory.get_preprocessor(
            "activitysim", state, provenance_tracker
        )

        # Preprocess
        inputData = preprocessor.preprocess(workspace)

        # Run ActivitySim
        rawOutputs, runInfo = runner.run(inputData, workspace)

        logger.info(
            "Appending warm start activities/choices to UrbanSim base year input data"
        )

        # Start a provenance run for the ActivitySim postprocessing step and call the
        # targeted warm-start updater (does not run the full ActivitySim postprocessor).
        post_run_hash = provenance_tracker.start_model_run(
            "activitysim_postprocessor",
            state.current_year,
            state.current_inner_iter,
            description="Post-processing for ActivitySim warm start",
        )
        asim_post.update_usim_inputs_after_warm_start(
            settings,
            state,
            runInfo,
            usim_data_dir=workspace.get_usim_mutable_data_dir(),
            warm_start_dir=workspace.get_asim_output_dir(),
            provenance_tracker=provenance_tracker,
            model_run_hash=post_run_hash,
        )
        provenance_tracker.complete_model_run(post_run_hash)

    logger.info("Done!")

    return


def forecast_land_use(
    settings,
    year,
    workflow_state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
):
    """
    High-level wrapper to start an UrbanSim (land use) run and handle post-run checks.

    It starts a model run in the provenance tracker, delegates actual execution to
    `run_land_use`, and verifies UrbanSim outputs exist after completion.

    Args:
        settings (dict): Simulation settings.
        year (int): Current simulation year.
        workflow_state (WorkflowState): Current workflow state instance.
        workspace (Workspace): Workspace for file operations.
        provenance_tracker (OpenLineageTracker): Provenance tracker.
    """
    land_use_model, land_use_image = GenericRunner.get_model_and_image(
        settings, "land_use_model"
    )

    # Start UrbanSim run and get model_run_hash
    usim_run_hash = provenance_tracker.start_model_run(
        land_use_model,
        workflow_state.current_year,
        workflow_state.current_inner_iter,
        description="UrbanSim run",
    )

    run_land_use(
        settings,
        year,
        workflow_state.forecast_year,
        land_use_model,
        workflow_state,
        workspace,
        provenance_tracker,
        usim_run_hash,
    )

    # Record UrbanSim run completion
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=workflow_state.forecast_year
    )
    usim_datastore_fpath = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_output_store_name
    )

    if os.path.exists(usim_datastore_fpath):
        # The postprocessor now handles its own completion
        pass
    else:
        # No output found: mark run as failed in provenance and exit early.
        logger.critical(
            "No UrbanSim output data found at {0}. It probably did not finish successfully.".format(
                usim_datastore_fpath
            )
        )
        provenance_tracker.complete_model_run(usim_run_hash, status="failed")
        sys.exit(1)


def run_land_use(
    settings,
    year,
    forecast_year,
    land_use_model,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
    model_run_hash: str,
):
    """
    Prepare inputs, run UrbanSim, and postprocess outputs for a land-use forecast.

    Args:
        settings (dict): Settings for the models and run environment.
        year (int): Current simulation year (start of forecast).
        forecast_year (int): Forecast target year.
        land_use_model (str): Land use model identifier (e.g., 'urbansim').
        state (WorkflowState): Workflow state.
        workspace (Workspace): Workspace instance for data paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
        model_run_hash (str): Provenance model run identifier created before calling this function.
    """
    logger.info("Running land use")

    factory = ModelFactory()

    # Obtain model components for the land use major stage
    preprocessor = factory.get_preprocessor("urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use)
    runner = factory.get_runner("urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use)
    postprocessor = factory.get_postprocessor("urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use)

    # 2. PREPARE URBANSIM DATA
    print_str = f"Preparing {year} input data for land use development simulation."
    formatted_print(print_str)

    input_data = preprocessor.preprocess(workspace)

    # 3. RUN URBANSIM
    print_str = f"Simulating land use development from {year} to {forecast_year} with {land_use_model}."
    formatted_print(print_str)


    raw_outputs, run_info = runner.run(input_data, workspace)

    # 4. POSTPROCESS URBANSIM OUTPUTS
    # Postprocessor will handle writing outputs and provenance via the provided model_run_hash.
    postprocessor.postprocess(raw_outputs, workspace, run_info, model_run_hash=model_run_hash)

    logger.info("Done!")

    return


def run_activity_demand(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
) -> RecordStore:
    """
    Generate activity plans for the current year using the configured activity demand model.

    Supports:
      - ActivitySim: full preprocess -> run -> postprocess sequence
      - Polaris: logged but not implemented here

    Args:
        settings (dict): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance for file operations.
        provenance_tracker (OpenLineageTracker): Provenance tracker for model events.

    Returns:
        RecordStore: Processed outputs (file records) from the activity demand postprocessor.
    """
    factory = ModelFactory()
    activity_demand_model = settings.get("activity_demand_model")

    if activity_demand_model == "polaris":
        # POLARIS integration placeholder
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )
        return RecordStore()
    elif activity_demand_model == "activitysim":
        preprocessor = factory.get_preprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand
        )
        runner = factory.get_runner("activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand)
        postprocessor = factory.get_postprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand
        )

        # Preprocess -> Run -> Postprocess flow
        input_data = preprocessor.preprocess(workspace)
        raw_outputs, run_info = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace, run_info)

        return processed_outputs

    else:
        logger.warning(
            f"Unknown or disabled activity demand model: {activity_demand_model}"
        )
        return RecordStore()


def run_traffic_assignment(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
    activity_demand_outputs: RecordStore = None,
):
    """
    Run the configured traffic assignment (supply) model for the current year/iteration.

    Currently supports:
      - BEAM: obtains preprocessor, runner, postprocessor via ModelFactory and runs the
        preprocessor with the activity demand outputs as input.

    Args:
        settings (dict): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
        activity_demand_outputs (RecordStore, optional): Processed activity demand outputs.
    """
    factory = ModelFactory()
    travel_model = settings.get("travel_model")

    if travel_model == "polaris":
        # run_polaris(state, settings)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )
    elif travel_model == "beam":
        preprocessor = factory.get_preprocessor("beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment)
        runner = factory.get_runner("beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment)
        postprocessor = factory.get_postprocessor("beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment)

        input_data = preprocessor.preprocess(workspace, activity_demand_outputs)
        raw_outputs, run_info = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace, run_info)

    else:
        logger.warning(f"Unknown or disabled travel model: {travel_model}")


def main():
    """
    Main function to execute the full simulation workflow.

    Responsibilities:
      - Parse settings and initialize WorkflowState
      - Initialize provenance tracker and workspace
      - Run an initialization job to copy mutable data and record provenance
      - Optionally initialize Docker client
      - Iterate over years and major workflow stages:
        Land use \-> ATLAS vehicle ownership \-> Supply/Demand loop (ActivitySim/BEAM) \-> Postprocessing
      - Persist/log completion status via provenance tracker and workspace
    """
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

    # Set up provenance tracking and workspace
    output_path = os.path.expandvars(settings.get("output_directory"))
    partial_run_name = settings.get("output_run_name", "pilates-run")
    run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_id = str(uuid.uuid4())

    provenance_tracker = OpenLineageTracker(run_id, output_path, folder_name=run_name)
    provenance_tracker.initialize_from_settings(settings)

    workspace = Workspace(
        settings,
        output_path,
        folder_name=run_name,
        provenance_tracker=provenance_tracker,
    )
    state.file_loc = os.path.join(workspace.full_path, "run_state.yaml")
    state.set_run_info_path(provenance_tracker.run_info_path)

    # Perform initialization (data copies and provenance recording)
    if not state.data_initialized:
        initialization = Initialization("initialization", state, provenance_tracker)
        initialization.run(settings, workspace)
        state.set_data_initialized(True)
    else:
        logger.info("Restarting from a previous state. Skipping data initialization.")

    # 2. MAIN WORKFLOW LOOP: iterate years and stages according to WorkflowState
    for year in state:
        formatted_print(f"STARTING YEAR {year}")

        # A. LAND USE FORECASTING
        if state.should_run(WorkflowState.Stage.land_use):
            formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")
            if state.is_start_year() and settings.get("warm_start_activities"):
                logger.info("[Main] Running warm start activities for ActivitySim.")
                warm_start_activities(
                    settings, state, workspace, provenance_tracker
                )
            forecast_land_use(
                settings, year, state, workspace, provenance_tracker
            )
            state.complete_step(WorkflowState.Stage.land_use)

        # B. VEHICLE OWNERSHIP MODEL (ATLAS)
        if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
            formatted_print(
                f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}"
            )
            logger.info("[Main] Running ATLAS vehicle ownership model.")

            # Use ModelFactory for all model/image lookups
            factory = ModelFactory()
            preprocessor = factory.get_preprocessor("atlas", state, provenance_tracker, major_stage=WorkflowState.Stage.vehicle_ownership_model)
            runner = factory.get_runner("atlas", state, provenance_tracker, major_stage=WorkflowState.Stage.vehicle_ownership_model)
            postprocessor = factory.get_postprocessor(
                "atlas", state, provenance_tracker, major_stage=WorkflowState.Stage.vehicle_ownership_model
            )

            # Determine if this is a warm start for ATLAS
            warm_start_atlas = state.is_start_year()
            forecast = True  # Always forecast for main loop

            # Multi-year loop logic (as in run_atlas)
            if forecast:
                # For forecast, run for all years between state.year and state.forecast_year, step 2
                yrs = [y + 2 for y in range(state.year, state.forecast_year, 2)]
                if not yrs:
                    yrs = [state.forecast_year]
            else:
                yrs = [state.year]

            for atlas_year in yrs:
                # Create a lightweight sub-state object for this ATLAS sub-run. It copies
                # the parent state's attributes and overrides the year/fiscal context used
                # by preprocessors/runners/postprocessors.
                class AtlasSubState:
                    def __init__(self, parent_state, year):
                        # Shallow copy parent state dict to inherit attributes
                        self.__dict__ = parent_state.__dict__.copy()
                        self.year = year
                        self.current_year = year
                        self.forecast_year = year
                        self.start_year = parent_state.start_year
                        self.full_settings = parent_state.full_settings
                        # Provide an is_start_year method consistent with the parent
                        self.is_start_year = lambda: (year == parent_state.start_year)

                    def set_sub_stage_progress(self, sub_stage_progress):
                        state.set_sub_stage_progress(sub_stage_progress)

                atlas_state = AtlasSubState(state, atlas_year)

                logger.info(
                    f"[run.py] [ATLAS] Preprocessing for year {atlas_year} (is_start_year={atlas_state.is_start_year()})"
                )
                # Preprocess
                preprocessor.update_state(atlas_state)
                input_data = preprocessor.preprocess(workspace)
                logger.info(
                    f"[run.py] [ATLAS] Preprocessing complete for year {atlas_year}"
                )

                logger.info(
                    f"[run.py] [ATLAS] Running AtlasRunner for year {atlas_year}"
                )
                # Run
                runner.update_state(atlas_state)
                raw_outputs, run_info = runner.run(input_data, workspace)
                logger.info(
                    f"[run.py] [ATLAS] AtlasRunner complete for year {atlas_year}"
                )

                logger.info(f"[run.py] [ATLAS] Postprocessing for year {atlas_year}")
                # Postprocess
                post_run_hash = provenance_tracker.start_model_run(
                    "atlas_postprocessor",
                    atlas_state.current_year,
                    description="ATLAS postprocessing",
                )
                postprocessor.update_state(atlas_state)
                processed_outputs = postprocessor.postprocess(
                    raw_outputs,
                    workspace,
                    run_info,
                    post_run_hash,
                )
                provenance_tracker.complete_model_run(
                    post_run_hash, output_records=processed_outputs.all_records()
                )
                logger.info(
                    f"[run.py] [ATLAS] Postprocessing complete for year {atlas_year}"
                )

            logger.info(
                "[run.py] [ATLAS] All ATLAS years complete for this major step."
            )
            state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

        # C. SUPPLY/DEMAND LOOP
        if state.should_run(WorkflowState.Stage.supply_demand_loop):
            total_iters = settings.get("supply_demand_iters", 1)
            for i in range(state.iteration, total_iters):
                state.iteration = i
                formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
                activity_demand_outputs = None

                # C1. ACTIVITY DEMAND
                if state.should_run(
                    WorkflowState.Stage.supply_demand_loop,
                    i,
                    WorkflowState.Stage.activity_demand,
                ):
                    formatted_print("ACTIVITY DEMAND MODEL")
                    logger.info("[Main] Running ActivitySim activity demand model.")
                    activity_demand_outputs = run_activity_demand(
                        settings, state, workspace, provenance_tracker
                    )
                    state.complete_step(
                        WorkflowState.Stage.supply_demand_loop,
                        i,
                        WorkflowState.Stage.activity_demand,
                    )

                # C2. TRAFFIC ASSIGNMENT
                if state.should_run(
                    WorkflowState.Stage.supply_demand_loop,
                    i,
                    WorkflowState.Stage.traffic_assignment,
                ):
                    formatted_print("TRAFFIC ASSIGNMENT MODEL")
                    logger.info("[Main] Running BEAM traffic assignment model.")
                    run_traffic_assignment(
                        settings,
                        state,
                        workspace,
                        provenance_tracker,
                        activity_demand_outputs,
                    )
                    state.complete_step(
                        WorkflowState.Stage.supply_demand_loop,
                        i,
                        WorkflowState.Stage.traffic_assignment,
                    )

            # After all iterations, complete the major stage
            state.complete_step(WorkflowState.Stage.supply_demand_loop)

        # D. POST-PROCESSING
        if state.should_run(WorkflowState.Stage.postprocessing):
            formatted_print("POST-PROCESSING")
            logger.info("[Main] Running post-processing steps.")
            if settings.get("mep_metrics_to_create"):
                process_event_file(settings, state, workspace, provenance_tracker)
            if settings.get("mep_output_dir"):
                copy_outputs_to_mep(settings, state, workspace, provenance_tracker)
            state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")
    logger.info("[Main] Simulation complete.")


if __name__ == "__main__":
    main()
