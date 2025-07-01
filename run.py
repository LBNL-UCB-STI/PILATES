import argparse
import random
import warnings
from datetime import datetime

import pandas as pd
import tables
from tables import HDF5ExtError

from pilates.generic.model_factory import ModelFactory

# Helper import for model/image lookup and docker client
from pilates.generic.runner import GenericRunner

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

import shutil
import subprocess
import multiprocessing
import psutil

try:
    import docker
except ImportError:
    print("Warning: Unable to import Docker Module")

import os
import logging
import sys
import glob
from pathlib import Path

from pilates.activitysim import preprocessor as asim_pre
from pilates.activitysim import postprocessor as asim_post
from pilates.urbansim import preprocessor as usim_pre
from pilates.urbansim import postprocessor as usim_post
from pilates.beam import preprocessor as beam_pre
from pilates.beam import postprocessor as beam_post
from pilates.atlas import preprocessor as atlas_pre  ##
from pilates.atlas import postprocessor as atlas_post  ##
from pilates.utils.io import parse_args_and_settings
from pilates.postprocessing.postprocessor import process_event_file, copy_outputs_to_mep

# from pilates.polaris.travel_model import run_polaris

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_already_opened_in_write_mode(filename):
    """Check if a file is already opened in write mode."""
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


def record_inputs_and_outputs(state, model, inputs=None, outputs=None, year=None, model_run_id=None):
    """Helper function to record inputs and outputs for provenance tracking."""
    if inputs:
        for file_path, description in inputs:
            if os.path.exists(file_path):
                state.record_input_file(model, file_path, description=description, model_run_id=model_run_id)
            else:
                logger.warning(f"Input file not found: {file_path}")

    if outputs:
        for file_path, description in outputs:
            if os.path.exists(file_path):
                state.record_output_file(
                    model, file_path, year=year, description=description, model_run_id=model_run_id
                )
            else:
                logger.warning(f"Output file not found: {file_path}")


def formatted_print(string, width=50, fill_char="#"):
    print("\n")
    if len(string) + 2 > width:
        width = len(string) + 4
    string = string.upper()
    print(fill_char * width)
    print("{:#^{width}}".format(" " + string + " ", width=width))
    print(fill_char * width, "\n")


def find_latest_beam_iteration(beam_output_dir):
    iter_dirs = []
    for root, dirs, files in os.walk(beam_output_dir):
        for dir in dirs:
            if dir == "ITER":
                iter_dirs += os.path.join(root, dir)
    print(iter_dirs)





def get_usim_docker_vols(settings, output_dir=None):
    usim_remote_data_folder = settings["usim_client_data_folder"]
    if output_dir is None:
        output_dir = settings[
            "usim_local_data_input_folder"
        ]  # This seems wrong, should be mutable output dir
        logger.warning(
            "get_usim_docker_vols called without output_dir, using usim_local_data_input_folder. Check logic."
        )
    usim_local_mutable_data_folder = os.path.abspath(output_dir)
    usim_docker_vols = {
        usim_local_mutable_data_folder: {"bind": usim_remote_data_folder, "mode": "rw"}
    }
    return usim_docker_vols


def get_usim_cmd(settings, year, forecast_year):
    region = settings["region"]
    region_id = settings["region_to_region_id"][region]
    land_use_freq = settings["land_use_freq"]
    skims_source = settings["travel_model"]
    formattable_usim_cmd = settings["usim_formattable_command"]
    usim_cmd = formattable_usim_cmd.format(
        region_id, year, forecast_year, land_use_freq, skims_source
    )
    return usim_cmd


## Atlas vehicle ownership model volume mount defintion, equivalent to
## docker run -v atlas_host_input_folder:atlas_container_input_folder
def get_atlas_docker_vols(settings, working_dir=None):
    if working_dir is None:
        # This case might happen if output_directory is not set in settings
        atlas_host_input_folder = os.path.abspath(settings["atlas_host_input_folder"])
        atlas_host_output_folder = os.path.abspath(settings["atlas_host_output_folder"])
    else:
        # Use the mutable input/output folders within the run directory
        atlas_host_input_folder = os.path.abspath(
            os.path.join(working_dir, settings["atlas_host_mutable_input_folder"])
        )
        atlas_host_output_folder = os.path.abspath(
            os.path.join(working_dir, settings["atlas_host_output_folder"])
        )

    atlas_container_input_folder = os.path.abspath(
        settings["atlas_container_input_folder"]
    )
    atlas_container_output_folder = os.path.abspath(
        settings["atlas_container_output_folder"]
    )
    atlas_docker_vols = {
        atlas_host_input_folder: {  ## source location, aka "local"
            "bind": atlas_container_input_folder,  ## destination loc, aka "remote", "client"
            "mode": "rw",
        },
        atlas_host_output_folder: {"bind": atlas_container_output_folder, "mode": "rw"},
    }
    return atlas_docker_vols


## For Atlas container command
def get_atlas_cmd(
    settings,
    freq,
    output_year,
    npe,
    nsample,
    beamac,
    mod,
    adscen,
    rebfactor,
    taxfactor,
    discIncent,
):
    basedir = settings.get("basedir", "/")
    codedir = settings.get("codedir", "/")
    formattable_atlas_cmd = settings["atlas_formattable_command"]
    atlas_cmd = formattable_atlas_cmd.format(
        freq,
        output_year,
        npe,
        nsample,
        basedir,
        codedir,
        beamac,
        mod,
        adscen,
        rebfactor,
        taxfactor,
        discIncent,
    )
    return atlas_cmd


def warm_start_activities(settings, state: WorkflowState, client):
    """
    Run activity demand models to update UrbanSim inputs with long-term
    choices it needs: workplace location, school location, and
    auto ownership.
    This runs only in the first year as part of the Land Use stage.
    """
    # Use ModelFactory for all model/image lookups
    factory = ModelFactory()
    activity_demand_model, activity_demand_image = GenericRunner.get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        # run_polaris(None, settings, warm_start=True)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif activity_demand_model == "activitysim":
        runner = factory.get_runner("activitysim")
        preprocessor = factory.get_preprocessor("activitysim")

        inputData = preprocessor.preprocess(state)
        rawOutputs, runInfo = runner.run(inputData, state)

        logger.info(
            "Appending warm start activities/choices to UrbanSim base year input data"
        )

        # The post-processing step for warm start is just updating the UrbanSim H5 file.
        # We call the specific function for this, not the full postprocessor.
        asim_post.update_usim_inputs_after_warm_start(
            settings,
            state,
            runInfo,
            usim_data_dir=os.path.join(
                state.full_path, settings["usim_local_mutable_data_folder"]
            ),
            warm_start_dir=os.path.join(
                state.full_path, settings["asim_local_output_folder"]
            ),
        )

    logger.info("Done!")

    return


def forecast_land_use(
    settings, year, workflow_state: WorkflowState, client, container_manager
):
    land_use_model, land_use_image = GenericRunner.get_model_and_image(settings, "land_use_model")

    # Record UrbanSim run start
    usim_run_index = workflow_state.record_model_init(
        land_use_model, year=workflow_state.forecast_year
    )

    run_land_use(settings, year, workflow_state, client)

    # Record UrbanSim run completion
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=workflow_state.forecast_year
    )
    output_dir = os.path.join(
        workflow_state.full_path, settings["usim_local_mutable_data_folder"]
    )
    usim_datastore_fpath = os.path.join(output_dir, usim_output_store_name)

    if os.path.exists(usim_datastore_fpath):
        # workflow_state.record_model_completion(usim_run_index, status="completed")
        # Record UrbanSim output file
        workflow_state.record_output_file(
            land_use_model,
            usim_datastore_fpath,
            year=workflow_state.forecast_year,
            description="UrbanSim forecast output data",
            model_run_id=usim_run_index
        )
    else:
        logger.critical(
            "No UrbanSim output data found at {0}. It probably did not finish successfully.".format(
                usim_datastore_fpath
            )
        )
        workflow_state.record_model_completion(usim_run_index, status="failed")
        sys.exit(1)


def run_land_use(settings, year, workflow_state: WorkflowState, client):
    logger.info("Running land use")

    # 1. PARSE SETTINGS
    output_dir = os.path.join(
        workflow_state.full_path, settings["usim_local_mutable_data_folder"]
    )
    os.makedirs(output_dir, exist_ok=True)
    land_use_model, land_use_image = GenericRunner.get_model_and_image(settings, "land_use_model")
    usim_docker_vols = get_usim_docker_vols(
        settings, output_dir
    )  # Pass the mutable output_dir
    forecast_year = workflow_state.forecast_year
    usim_cmd = get_usim_cmd(settings, year, forecast_year)

    # 2. PREPARE URBANSIM DATA
    print_str = "Preparing {0} input data for land use development simulation.".format(
        year
    )
    formatted_print(print_str)

    # Record inputs to UrbanSim preprocessing (skims from ASim output)
    asim_output_dir = os.path.join(
        workflow_state.full_path, settings["asim_local_mutable_data_folder"]
    )
    asim_skims_path = os.path.join(
        asim_output_dir, settings["skims_fname"]
    )  # Assuming skims are here

    # usim_run_index is not defined in this function, so do not pass model_run_id here
    workflow_state.record_input_file(
        land_use_model,
        asim_skims_path,
        description="ActivitySim skims for UrbanSim input"
    )

    usim_pre.add_skims_to_model_data(settings, output_dir, asim_output_dir)

    usim_data_path = os.path.join(
        settings[
            "usim_local_data_input_folder"
        ],  # This path seems inconsistent with mutable output_dir
        settings["usim_formattable_input_file_name"].format(
            region_id=settings["region_to_region_id"][settings["region"]]
        ),
    )
    # The actual input used by the container is in output_dir, not settings['usim_local_data_input_folder']
    # Let's record the input file path *within the run directory* that the container will use
    usim_input_in_run_dir = os.path.join(
        output_dir, usim_post.get_usim_datastore_fname(settings, io="input")
    )
    workflow_state.record_input_file(
        land_use_model,
        usim_input_in_run_dir,
        description="UrbanSim input data for forecast"
    )

    if is_already_opened_in_write_mode(
        usim_data_path
    ):  # This check might need adjustment if using mutable copy
        logger.warning(
            "Closing h5 files {0} because they were left open. You should really "
            "figure out where this happened".format(tables.file._open_files.filenames)
        )
        tables.file._open_files.close_all()

    # 3. RUN URBANSIM
    print_str = "Simulating land use development from {0} " "to {1} with {2}.".format(
        year, forecast_year, land_use_model
    )
    formatted_print(print_str)
    usim_hash = workflow_state.record_model_start()

    # run_container call is now wrapped in forecast_land_use for provenance tracking
    GenericRunner.run_container(
        client=client,
        settings=settings,
        image=land_use_image,
        volumes=usim_docker_vols,
        command=usim_cmd,
        model_name=land_use_model,
        working_dir=settings["usim_client_base_folder"],
    )
    logger.info("Done!")
    workflow_state.record_model_completion(usim_hash, status="completed")

    return


## Atlas: evolve household vehicle ownership
def run_atlas(
    settings,
    state: WorkflowState,
    client,
    warm_start_atlas,
    forecast=False,
    atlas_run_count=1,
):
    # warm_start: warm_start_atlas = True, output_year = year = start_year
    # asim_no_usim: warm_start_atlas = True, output_year = year (should  = start_year)
    # normal: warm_start_atlas = False, output_year = forecast_year

    if forecast:
        yr = state.forecast_year
    else:
        yr = state.start_year

    # 1. PARSE SETTINGS
    vehicle_ownership_model, atlas_image = GenericRunner.get_model_and_image(
        settings, "vehicle_ownership_model"
    )
    freq = settings.get("vehicle_ownership_freq", False)
    npe = settings.get("atlas_num_processes", False)
    nsample = settings.get("atlas_sample_size", False)
    beamac = settings.get("atlas_beamac", 0)
    mod = settings.get("atlas_mod", 1)
    adscen = settings.get("atlas_adscen", False)
    rebfactor = settings.get("atlas_rebfactor", 0)
    taxfactor = settings.get("atlas_taxfactor", 0)
    discIncent = settings.get("atlas_discIncent", 0)
    atlas_docker_vols = get_atlas_docker_vols(settings, state.full_path)
    atlas_cmd = get_atlas_cmd(
        settings,
        freq,
        yr,
        npe,
        nsample,
        beamac,
        mod,
        adscen,
        rebfactor,
        taxfactor,
        discIncent,
    )
    docker_stdout = settings.get("docker_stdout", False)

    # 2. PREPARE ATLAS DATA
    if warm_start_atlas:
        print_str = "Preparing input data for warm start vehicle ownership simulation for {0}.".format(
            yr
        )
    else:
        print_str = (
            "Preparing input data for vehicle ownership simulation for {0}.".format(yr)
        )
    formatted_print(print_str)

    # prepare atlas inputs from urbansim h5 output
    # preprocessed csv input files saved in "atlas/atlas_inputs/year{}/"
    if forecast:
        yrs = [y + 2 for y in range(state.year, yr, 2)]
    else:
        yrs = [yr]

    # Record inputs to Atlas preprocessing
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=state.forecast_year
    )
    usim_output_store_path = os.path.join(
        state.full_path,
        settings["usim_local_mutable_data_folder"],
        usim_output_store_name,
    )
    state.record_input_file(
        vehicle_ownership_model,
        usim_output_store_path,
        description="UrbanSim output for Atlas input preparation",
    )

    for yr_it in yrs:
        atlas_pre.prepare_atlas_inputs(
            settings, yr_it, state, warm_start=warm_start_atlas
        )
        # Record outputs from Atlas preprocessing (the CSV files)
        atlas_input_csv_dir = os.path.join(
            state.full_path,
            settings["atlas_host_mutable_input_folder"],
            f"atlas_inputs/year{yr_it}",
        )
        if os.path.exists(atlas_input_csv_dir):
            for file in os.listdir(atlas_input_csv_dir):
                if file.endswith(".csv"):
                    state.record_output_file(
                        vehicle_ownership_model,
                        os.path.join(atlas_input_csv_dir, file),
                        year=yr_it,
                        description="Atlas preprocessed input CSV"
                    )

    # calculate accessibility if beamac != 0
    if beamac > 0:
        # Record inputs to accessibility calculation (BEAM skims)
        beam_output_dir = os.path.join(
            state.full_path, settings["beam_local_output_folder"]
        )
        # Again, finding the exact skim file path is complex, record the expected location
        expected_beam_skims_path = os.path.join(
            beam_output_dir, settings["skims_fname"]
        )
        state.record_input_file(
            vehicle_ownership_model,
            expected_beam_skims_path,
            description="BEAM skims for Atlas accessibility calculation",
        )

        path_list = [
            "WLK_COM_WLK",
            "WLK_EXP_WLK",
            "WLK_HVY_WLK",
            "WLK_LOC_WLK",
            "WLK_LRF_WLK",
        ]
        measure_list = ["WACC", "IWAIT", "XWAIT", "TOTIVT", "WEGR"]
        atlas_pre.compute_accessibility(
            path_list, measure_list, settings, state.forecast_year
        )
        # Record outputs from accessibility calculation (accessibility CSV)
        atlas_acc_output_path = os.path.join(
            state.full_path,
            settings["atlas_host_mutable_input_folder"],
            f"atlas_inputs/year{state.forecast_year}",
            "accessibility.csv",
        )
        state.record_output_file(
            vehicle_ownership_model,
            atlas_acc_output_path,
            year=state.forecast_year,
            description="Atlas accessibility output CSV"
        )

    # 3. RUN ATLAS via docker container client
    print_str = (
        "Simulating vehicle ownership for {0} "
        "with frequency {1}, npe {2} nsample {3} beamac {4}".format(
            yr, freq, npe, nsample, beamac
        )
    )
    formatted_print(print_str)

    # Record Atlas run start
    atlas_run_index = state.record_model_start(
        vehicle_ownership_model, year=yr, iteration=atlas_run_count
    )

    success = GenericRunner.run_container(
        client=client,
        settings=settings,
        image=atlas_image,
        volumes=atlas_docker_vols,
        command=atlas_cmd,
        model_name=vehicle_ownership_model,
        working_dir="/",
    )

    # Record Atlas run completion
    state.record_model_completion(
        atlas_run_index, status="completed" if success else "failed"
    )
    if not success:
        logger.error("Atlas run failed.")
        # Don't exit immediately here, let run_atlas_auto handle retries

    # 4. ATLAS OUTPUT -> UPDATE USIM OUTPUT CARS & HH_CARS
    atlas_output_path = os.path.join(
        state.full_path, settings["atlas_host_output_folder"]
    )
    atlas_post.update_usim_for_atlas(settings, yr, state, warm_start=warm_start_atlas)

    # Record outputs from Atlas postprocessing (updated UrbanSim H5 file)
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=yr
    )
    usim_output_store_path = os.path.join(
        state.full_path,
        settings["usim_local_mutable_data_folder"],
        usim_output_store_name,
    )
    state.record_output_file(
        vehicle_ownership_model,
        usim_output_store_path,
        year=yr,
        description="UrbanSim data updated with Atlas vehicle ownership",
        model_run_id=atlas_run_index
    )

    return success


def run_atlas_auto(settings, state: WorkflowState, client, warm_start_atlas, forecast):
    """
    Run Atlas with automatic retries on failure.
    """
    max_retries = settings.get("atlas_max_retries", 3)
    for i in range(max_retries):
        success = run_atlas(
            settings,
            state,
            client,
            warm_start_atlas,
            forecast=forecast,
            atlas_run_count=i + 1,
        )
        if success:
            return
        else:
            logger.warning(
                f"Atlas run failed on attempt {i + 1}. Retrying... ({max_retries - i - 1} retries left)"
            )
    logger.error(f"Atlas failed after {max_retries} attempts. Exiting.")
    sys.exit(1)


def run_activity_demand(settings, state: WorkflowState, client):
    """
    Generate activity plans for the current year.
    """
    factory = ModelFactory()
    activity_demand_model = settings.get("activity_demand_model")

    if activity_demand_model == "polaris":
        # run_polaris(state, settings)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )
    elif activity_demand_model == "activitysim":
        preprocessor = factory.get_preprocessor("activitysim")
        runner = factory.get_runner("activitysim")
        postprocessor = factory.get_postprocessor("activitysim")

        # Preprocess
        input_data = preprocessor.preprocess(state)

        # Run
        raw_outputs, run_info = runner.run(input_data, state)

        # Postprocess
        processed_outputs = postprocessor.postprocess(raw_outputs, run_info, state)

    else:
        logger.warning(
            f"Unknown or disabled activity demand model: {activity_demand_model}"
        )


def run_traffic_assignment(settings, state: WorkflowState, client):
    """
    Run traffic assignment for the current year.
    """
    factory = ModelFactory()
    travel_model = settings.get("travel_model")

    if travel_model == "polaris":
        # run_polaris(state, settings)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )
    elif travel_model == "beam":
        preprocessor = factory.get_preprocessor("beam")
        runner = factory.get_runner("beam")
        postprocessor = factory.get_postprocessor("beam")

        # Preprocess
        input_data = preprocessor.preprocess(state)

        # Run
        raw_outputs, run_info = runner.run(input_data, state)

        # Postprocess
        processed_outputs = postprocessor.postprocess(raw_outputs, run_info, state)

    else:
        logger.warning(f"Unknown or disabled travel model: {travel_model}")


def main():
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

    # Initialize Docker/Singularity client if needed
    client = None
    if settings.get("container_manager") == "docker":
        try:
            client = GenericRunner.initialize_docker_client(settings)
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            # If Docker is required and fails, we should probably exit.
            # For now, we allow continuing for Singularity/stub cases.

    # 2. MAIN WORKFLOW LOOP
    for year in state:
        formatted_print(f"STARTING YEAR {year}")

        # A. LAND USE FORECASTING
        if state.should_run(WorkflowState.Stage.land_use):
            formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")
            if state.is_start_year() and settings.get("warm_start_activities"):
                warm_start_activities(settings, state, client)
            forecast_land_use(
                settings, year, state, client, settings["container_manager"]
            )
            state.complete_step(WorkflowState.Stage.land_use)

        # B. VEHICLE OWNERSHIP MODEL (ATLAS)
        if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
            formatted_print(f"VEHICLE OWNERSHIP MODEL FOR YEAR {state.forecast_year}")
            run_atlas_auto(
                settings,
                state,
                client,
                warm_start_atlas=state.is_start_year(),
                forecast=True,
            )
            state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

        # C. SUPPLY/DEMAND LOOP
        if state.should_run(WorkflowState.Stage.supply_demand_loop):
            total_iters = settings.get("supply_demand_iters", 1)
            for i in range(state.iteration, total_iters):
                state.iteration = i
                formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")

                # C1. ACTIVITY DEMAND
                if state.should_run(
                    WorkflowState.Stage.supply_demand_loop,
                    i,
                    WorkflowState.Stage.activity_demand,
                ):
                    formatted_print("ACTIVITY DEMAND MODEL")
                    run_activity_demand(settings, state, client)
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
                    run_traffic_assignment(settings, state, client)
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
            if settings.get("mep_metrics_to_create"):
                process_event_file(settings, state)
            if settings.get("mep_output_dir"):
                copy_outputs_to_mep(settings, state)
            state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")


if __name__ == "__main__":
    main()
