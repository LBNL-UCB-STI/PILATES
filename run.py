import argparse
import random
import warnings
from datetime import datetime

import pandas as pd
import tables
from tables import HDF5ExtError

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
    level=logging.INFO,
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


def record_inputs_and_outputs(state, model, inputs=None, outputs=None, year=None):
    """Helper function to record inputs and outputs for provenance tracking."""
    if inputs:
        for file_path, description in inputs:
            if os.path.exists(file_path):
                state.record_input_file(model, file_path, description=description)
            else:
                logger.warning(f"Input file not found: {file_path}")

    if outputs:
        for file_path, description in outputs:
            if os.path.exists(file_path):
                state.record_output_file(
                    model, file_path, year=year, description=description
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


def get_base_asim_cmd(settings, household_sample_size=None, num_processes=None):
    formattable_asim_cmd = settings["asim_formattable_command"]
    if not household_sample_size:
        household_sample_size = settings.get("household_sample_size", 0)
    num_processes = num_processes or settings.get(
        "num_processes", multiprocessing.cpu_count() - 1
    )
    chunk_size = settings.get("chunk_size", 0)  # default no chunking
    base_asim_cmd = formattable_asim_cmd.format(
        household_sample_size, num_processes, chunk_size
    )
    return base_asim_cmd


def get_asim_additional_args(settings, asim_docker_vols, compile):
    additional_args = []
    if settings.get("file_format", "parquet") == "parquet":
        additional_args.append("--persist-sharrow-cache")
        for local, d in asim_docker_vols.items():
            if "data" in d["bind"]:
                additional_args.append("-d")
                additional_args.append('"{0}"'.format(d["bind"]))
            elif "output" in d["bind"]:
                additional_args.append("-o")
                additional_args.append('"{0}"'.format(d["bind"]))
            elif "compile" in d["bind"]:
                if compile:
                    additional_args.append("-c")
                    additional_args.append('"{0}"'.format(d["bind"]))
            elif "configs" in d["bind"]:
                additional_args.append("-c")
                additional_args.append('"{0}"'.format(d["bind"]))
    return additional_args


def get_asim_docker_vols(settings, working_dir=None):
    region = settings["region"]
    asim_subdir = settings["region_to_asim_subdir"][region]
    asim_remote_workdir = os.path.join("/activitysim", asim_subdir)
    if working_dir is not None:
        asim_local_mutable_data_folder = os.path.abspath(
            os.path.join(working_dir, settings["asim_local_mutable_data_folder"])
        )
        asim_local_output_folder = os.path.abspath(
            os.path.join(working_dir, settings["asim_local_output_folder"])
        )
        asim_local_configs_folder = os.path.abspath(
            os.path.join(
                working_dir,
                settings["asim_local_mutable_configs_folder"],
                settings.get("asim_main_configs_dir", "configs"),
            )
        )
        asim_local_configs_compile_folder = os.path.abspath(
            os.path.join(
                working_dir,
                settings["asim_local_mutable_configs_folder"],
                "configs_sh_compile",
            )
        )
    else:
        asim_local_mutable_data_folder = os.path.abspath(
            settings["asim_local_mutable_data_folder"]
        )
        asim_local_output_folder = os.path.abspath(settings["asim_local_output_folder"])
        asim_local_configs_folder = os.path.abspath(
            os.path.join(settings["asim_local_configs_folder"], region, "configs")
        )
        asim_local_configs_compile_folder = os.path.abspath(
            os.path.join(
                settings["asim_local_configs_folder"], region, "configs_sh_compile"
            )
        )
    asim_remote_input_folder = os.path.join(asim_remote_workdir, "data")
    asim_remote_output_folder = os.path.join(asim_remote_workdir, "output")
    asim_remote_configs_folder = os.path.join(asim_remote_workdir, "configs")
    asim_remote_configs_compile_folder = os.path.join(
        asim_remote_workdir, "configs_sh_compile"
    )
    asim_docker_vols = {
        asim_local_mutable_data_folder: {
            "bind": asim_remote_input_folder,
            "mode": "rw",
        },
        asim_local_output_folder: {"bind": asim_remote_output_folder, "mode": "rw"},
        asim_local_configs_compile_folder: {
            "bind": asim_remote_configs_compile_folder,
            "mode": "rw",
        },
        asim_local_configs_folder: {"bind": asim_remote_configs_folder, "mode": "rw"},
    }
    return asim_docker_vols


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
    activity_demand_model, activity_demand_image = get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        # run_polaris(None, settings, warm_start=True)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif activity_demand_model == "activitysim":
        # 1. PARSE SETTINGS
        land_use_model = settings["land_use_model"]
        travel_model = settings["travel_model"]
        region = settings["region"]
        asim_subdir = settings["region_to_asim_subdir"][region]
        asim_workdir = os.path.join("/activitysim", asim_subdir)
        asim_docker_vols = get_asim_docker_vols(settings, state.full_path)
        base_asim_cmd = get_base_asim_cmd(settings)

        print_str = "Initializing {0} warm start sequence".format(activity_demand_model)
        formatted_print(print_str)

        # 2. CREATE DATA FROM BASE YEAR SKIMS AND URBANSIM INPUTS

        # data tables
        logger.info(
            "Creating {0} input data from {1} outputs".format(
                activity_demand_model, land_use_model
            ).upper()
        )

        # Record inputs and outputs for ActivitySim warm start preprocessing
        usim_output_store_name = usim_post.get_usim_datastore_fname(
            settings, io="output", year=state.forecast_year
        )
        usim_output_store_path = os.path.join(
            state.full_path,
            settings["usim_local_mutable_data_folder"],
            usim_output_store_name,
        )
        expected_beam_skims_path = os.path.join(
            state.full_path,
            settings["beam_local_output_folder"],
            settings["skims_fname"],
        )

        record_inputs_and_outputs(
            state,
            activity_demand_model,
            inputs=[
                (usim_output_store_path, "UrbanSim output for warm start"),
                (expected_beam_skims_path, "BEAM skims for warm start"),
            ],
        )

        if not os.path.exists(
            os.path.join(
                state.full_path, settings["asim_local_mutable_data_folder"], "skims.omx"
            )
        ):
            asim_pre.create_skims_from_beam(settings, state, overwrite=False)

        asim_pre.create_asim_data_from_h5(settings, state, warm_start=True)

        asim_warm_start_persons_path = os.path.join(
            state.full_path,
            settings["asim_local_output_folder"],
            "warm_start_persons.csv",
        )
        asim_warm_start_households_path = os.path.join(
            state.full_path,
            settings["asim_local_output_folder"],
            "warm_start_households.csv",
        )

        record_inputs_and_outputs(
            state,
            activity_demand_model,
            outputs=[
                (asim_warm_start_persons_path, "ActivitySim warm start persons output"),
                (
                    asim_warm_start_households_path,
                    "ActivitySim warm start households output",
                ),
            ],
            year=state.current_year,
        )

        # 3. RUN ACTIVITYSIM IN WARM START MODE
        logger.info(
            "Running {0} in warm start mode".format(activity_demand_model).upper()
        )
        ws_asim_cmd = base_asim_cmd + " -w"  # warm start flag
        additional_args = get_asim_additional_args(
            settings, asim_docker_vols, False
        )  # No compile in warm start run

        # Record ActivitySim warm start run start
        asim_run_index = state.record_model_start(
            activity_demand_model, year=state.current_year, iteration=0
        )

        success = run_container(
            client,
            settings,
            activity_demand_image,
            asim_docker_vols,
            ws_asim_cmd,
            working_dir=asim_workdir,
            model_name=activity_demand_model,
            args=additional_args,
        )

        # Record ActivitySim warm start run completion
        state.record_model_completion(
            asim_run_index, status="completed" if success else "failed"
        )
        if not success:
            logger.error("ActivitySim warm start run failed.")
            sys.exit(1)  # Exit if warm start fails

        # 4. UPDATE URBANSIM BASE YEAR INPUT DATA
        logger.info(
            ("Appending warm start activities/choices to " " {0} base year input data")
            .format(land_use_model)
            .upper()
        )

        # Record inputs to UrbanSim update postprocessing
        state.record_input_file(
            land_use_model,
            asim_warm_start_persons_path,
            description="ActivitySim warm start persons output for USim update",
        )
        state.record_input_file(
            land_use_model,
            asim_warm_start_households_path,
            description="ActivitySim warm start households output for USim update",
        )
        usim_input_store_path = os.path.join(
            state.full_path,
            settings["usim_local_mutable_data_folder"],
            usim_post.get_usim_datastore_fname(settings, io="input"),
        )
        state.record_input_file(
            land_use_model,
            usim_input_store_path,
            description="UrbanSim input data before warm start update",
        )

        asim_post.update_usim_inputs_after_warm_start(
            settings,
            usim_data_dir=os.path.join(
                state.full_path, settings["usim_local_mutable_data_folder"]
            ),
            warm_start_dir=os.path.join(
                state.full_path, settings["asim_local_output_folder"]
            ),
        )

        # Record outputs from UrbanSim update postprocessing
        state.record_output_file(
            land_use_model,
            usim_input_store_path,
            year=state.current_year,
            description="UrbanSim input data after warm start update",
        )

    logger.info("Done!")

    return


def forecast_land_use(
    settings, year, workflow_state: WorkflowState, client, container_manager
):
    land_use_model, land_use_image = get_model_and_image(settings, "land_use_model")

    # Record UrbanSim run start
    usim_run_index = workflow_state.record_model_start(
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
        workflow_state.record_model_completion(usim_run_index, status="completed")
        # Record UrbanSim output file
        workflow_state.record_output_file(
            land_use_model,
            usim_datastore_fpath,
            year=workflow_state.forecast_year,
            description="UrbanSim forecast output data",
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
    land_use_model, land_use_image = get_model_and_image(settings, "land_use_model")
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
    workflow_state.record_input_file(
        land_use_model,
        asim_skims_path,
        description="ActivitySim skims for UrbanSim input",
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
        description="UrbanSim input data for forecast",
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

    # run_container call is now wrapped in forecast_land_use for provenance tracking
    run_container(
        client,
        settings,
        land_use_image,
        usim_docker_vols,
        usim_cmd,
        working_dir=settings["usim_client_base_folder"],
        model_name=land_use_model,
    )
    logger.info("Done!")

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
    vehicle_ownership_model, atlas_image = get_model_and_image(
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
                        description="Atlas preprocessed input CSV",
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
            description="Atlas accessibility output CSV",
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

    success = run_container(
        client,
        settings,
        atlas_image,
        atlas_docker_vols,
        atlas_cmd,
        working_dir="/",
        model_name=vehicle_ownership_model,
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
    atlas_vehicle_output_file = os.path.join(
        atlas_output_path, f"vehicles_{yr}.csv"
    )  # Assuming CSV output before gz
    # Record Atlas output file (before it's potentially gzipped or moved)
    state.record_output_file(
        vehicle_ownership_model,
        atlas_vehicle_output_file,
        year=yr,
        description="Atlas raw vehicle ownership output CSV",
    )

    # Record inputs to Atlas postprocessing (Atlas output CSV and UrbanSim output H5)
    state.record_input_file(
        vehicle_ownership_model,
        atlas_vehicle_output_file,
        description="Atlas vehicle ownership output for USim update",
    )
    state.record_input_file(
        vehicle_ownership_model,
        usim_output_store_path,
        description="UrbanSim output data for Atlas update",
    )  # Re-using path from earlier

    atlas_post.atlas_update_h5_vehicle(settings, yr, state, warm_start=warm_start_atlas)

    # Record outputs from Atlas postprocessing (Updated UrbanSim output H5)
    state.record_output_file(
        vehicle_ownership_model,
        usim_output_store_path,
        year=state.forecast_year,
        description="UrbanSim output data after Atlas update",
    )

    # 5. ATLAS OUTPUT -> ADD A VEHICLETYPEID COL FOR BEAM
    # Record inputs to BEAM vehicle input preparation (Atlas output CSV and UrbanSim output H5)
    state.record_input_file(
        vehicle_ownership_model,
        atlas_vehicle_output_file,
        description="Atlas vehicle ownership output for BEAM vehicle input",
    )
    state.record_input_file(
        vehicle_ownership_model,
        usim_output_store_path,
        description="UrbanSim output data for BEAM vehicle input",
    )  # Re-using path

    atlas_post.atlas_add_vehileTypeId(settings, yr, state)
    atlas_post.build_beam_vehicles_input(settings, yr, state)

    # Record outputs from BEAM vehicle input preparation (BEAM vehicles.csv.gz)
    beam_scenario_folder = os.path.join(
        state.full_path,
        settings["beam_local_mutable_data_folder"],
        settings["region"],
        settings["beam_scenario_folder"],
    )
    beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")
    state.record_output_file(
        vehicle_ownership_model,
        beam_vehicles_path,
        year=yr,
        description="BEAM vehicles input file generated from Atlas/UrbanSim",
    )

    logger.info("Atlas Done!")

    return


## Atlas: evolve household vehicle ownership
# run_atlas_auto is a run_atlas upgraded version, which will run_atlas again if
# outputs are not generated. This is mainly for preventing crash due to parellel
# computiing errors that can be resolved by a simple resubmission
def run_atlas_auto(
    settings, state: WorkflowState, client, warm_start_atlas, forecast=False
):
    if forecast:
        yr = state.forecast_year
    else:
        yr = state.start_year
    atlas_output_path = os.path.join(
        state.full_path, settings["atlas_host_output_folder"]
    )
    fname = "vehicles_{}.csv".format(yr)  # Check for the CSV before gzip
    gz_fname = "vehicles_{}.csv.gz".format(yr)

    # Check if the final expected output (gzipped) exists for warm start
    if warm_start_atlas and os.path.exists(os.path.join(atlas_output_path, gz_fname)):
        logger.info(
            f"Running in warm start mode but warm started file {gz_fname} for year {yr} already exist. Assuming we can skip this "
            "step and move on to the forecast year."
        )
        # If skipping, ensure the output is recorded if it wasn't already
        state.record_output_file(
            settings.get("vehicle_ownership_model", "atlas"),
            os.path.join(atlas_output_path, gz_fname),
            year=yr,
            description="Atlas gzipped vehicle ownership output (skipped run)",
        )
        return

    # run atlas
    atlas_run_count = 1
    success = False
    while atlas_run_count <= 3 and not success:
        if atlas_run_count > 1:
            logger.warning("ATLAS RUN #{} RE-LAUNCHING".format(atlas_run_count))
        try:
            run_atlas(
                settings, state, client, warm_start_atlas, forecast, atlas_run_count
            )
            # Check if the expected output file was created by run_atlas
            if os.path.exists(os.path.join(atlas_output_path, fname)) or os.path.exists(
                os.path.join(atlas_output_path, gz_fname)
            ):
                success = True
                logger.info(
                    "ATLAS RUN #{} COMPLETED SUCCESSFULLY".format(atlas_run_count)
                )
            else:
                logger.error(
                    "ATLAS RUN #{} FAILED: Expected output file {} not found.".format(
                        atlas_run_count, fname
                    )
                )
        except Exception as e:
            logger.error(
                "ATLAS RUN #{} FAILED with exception: {}".format(atlas_run_count, e)
            )

        if not success:
            atlas_run_count += 1

    if not success:
        logger.critical(
            f"ATLAS failed after {atlas_run_count-1} retries. Expected output file {fname} not found."
        )
        sys.exit(1)


def generate_activity_plans(
    settings, year, state: WorkflowState, client, resume_after=None, warm_start=False
):
    """
    Parameters

    year : int
        Start year for the simulation iteration.
    forecast_year : int
        Simulation year for which activities are generated. If `forecast_year`
        is the start year of the whole simulation, then we are probably
        generating warm start activities based on the base year input data in
        order to generate "warm start" skims.

    Note: For the main supply-demand loop (not warm start), this should read
    data for `state.current_year` (the year the LU model just output)
    """

    activity_demand_model, activity_demand_image = get_model_and_image(
        settings, "activity_demand_model"
    )

    if settings.get("regenerate_seed", True):
        new_seed = random.randint(0, int(1e9))
        logger.info("Re-seeding asim with new seed {0}".format(new_seed))
        try:
            # Record input (config file) before modification
            asim_config_path = os.path.join(
                state.full_path,
                settings["asim_local_mutable_configs_folder"],
                settings.get("asim_main_configs_dir", "configs"),
                "settings.yaml",
            )  # Assuming settings.yaml
            state.record_input_file(
                activity_demand_model,
                asim_config_path,
                description="ActivitySim settings file before seed update",
            )

            asim_pre.update_asim_config(
                settings, state.full_path, "random_seed", new_seed
            )

            # Record output (modified config file)
            state.record_output_file(
                activity_demand_model,
                asim_config_path,
                description="ActivitySim settings file after seed update",
            )

        except FileNotFoundError:
            logger.error(
                "Error updating random seed in ASim config. Please check your settings."
            )

    if activity_demand_model == "polaris":
        # run_polaris(state.forecast_year, settings, warm_start=True)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif activity_demand_model == "activitysim":

        # 1. PARSE SETTINGS

        region = settings["region"]
        asim_subdir = settings["region_to_asim_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)
        asim_docker_vols = get_asim_docker_vols(settings, state.full_path)

        docker_stdout = settings.get("docker_stdout", False)

        overwrite_skims_arg = False  # Logic seems to keep this False after warm start

        # 2. PREPROCESS DATA FOR ACTIVITY DEMAND MODEL
        print_str = "Creating {0} input data from {1} outputs".format(
            activity_demand_model,
            settings.get("land_use_model", "UrbanSim Inputs if Land Use Disabled"),
        )
        formatted_print(print_str)

        # Record inputs and outputs for ActivitySim preprocessing
        usim_output_store_name = usim_post.get_usim_datastore_fname(
            settings, io="output", year=state.forecast_year
        )
        usim_output_store_path = os.path.join(
            state.full_path,
            settings["usim_local_mutable_data_folder"],
            usim_output_store_name,
        )
        expected_beam_skims_path = os.path.join(
            state.full_path,
            settings["beam_local_output_folder"],
            settings["skims_fname"],
        )

        record_inputs_and_outputs(
            state,
            activity_demand_model,
            inputs=[
                (
                    usim_output_store_path,
                    "UrbanSim output for ActivitySim input preparation",
                ),
                (
                    expected_beam_skims_path,
                    "BEAM skims for ActivitySim input preparation",
                ),
            ],
        )

        asim_pre.create_skims_from_beam(
            settings, state=state, overwrite=overwrite_skims_arg
        )
        asim_pre.create_asim_data_from_h5(settings, state=state, warm_start=warm_start)

        # Record outputs from ActivitySim preprocessing
        asim_mutable_data_dir = os.path.join(
            state.full_path, settings["asim_local_mutable_data_folder"]
        )
        asim_input_tables = settings["asim_output_tables"]["tables"]
        outputs = [
            (
                os.path.join(
                    asim_mutable_data_dir,
                    f"{settings['asim_output_tables']['prefix']}{table_name}.csv",
                ),
                f"ActivitySim preprocessed input table: {table_name}",
            )
            for table_name in asim_input_tables
        ]
        outputs.append(
            (
                os.path.join(asim_mutable_data_dir, settings["skims_fname"]),
                "ActivitySim skims input file",
            )
        )

        record_inputs_and_outputs(
            state, activity_demand_model, outputs=outputs, year=state.forecast_year
        )

        # 3. GENERATE ACTIVITY PLANS
        print_str = "Generating activity plans for the year " "{0} with {1}".format(
            state.forecast_year, activity_demand_model
        )

        # Record ActivitySim run start (Compilation if needed)
        asim_compile_run_index = -1
        if not state.asim_compiled:
            asim_cmd = get_base_asim_cmd(
                settings, household_sample_size=2500, num_processes=1
            )
            if resume_after:
                asim_cmd += " -r {0}".format(resume_after)

            additional_args = get_asim_additional_args(settings, asim_docker_vols, True)

            asim_compile_run_index = state.record_model_start(
                activity_demand_model,
                year=state.forecast_year,
                iteration=state.current_inner_iter,
                description="ActivitySim Compilation Run",
            )

            success = run_container(
                client,
                settings,
                activity_demand_image,
                working_dir=asim_workdir,
                volumes=asim_docker_vols,
                command=asim_cmd,
                args=additional_args,
                model_name=activity_demand_model,
            )

            state.record_model_completion(
                asim_compile_run_index, status="completed" if success else "failed"
            )

            logger.info("ASIM Compilation success: {0}".format(success))
            if not success:
                raise RuntimeError("ASim Compilation failed")
            state.compile_asim()  # Update state to mark as compiled

        # Record ActivitySim run start (Main run)
        asim_main_run_index = state.record_model_start(
            activity_demand_model,
            year=state.forecast_year,
            iteration=state.current_inner_iter,
            description="ActivitySim Main Run",
        )

        asim_cmd = get_base_asim_cmd(settings)
        if resume_after:
            asim_cmd += " -r {0}".format(resume_after)
            print_str += ". Picking up after {0}".format(resume_after)
        formatted_print(print_str)

        additional_args = get_asim_additional_args(settings, asim_docker_vols, False)

        success = run_container(
            client,
            settings,
            activity_demand_image,
            working_dir=asim_workdir,
            volumes=asim_docker_vols,
            command=asim_cmd,
            args=additional_args,
            model_name=activity_demand_model,
        )

        # Record ActivitySim run completion (Main run)
        state.record_model_completion(
            asim_main_run_index, status="completed" if success else "failed"
        )
        if not success:
            logger.error("ActivitySim main run failed.")
            # Decide how to handle failure - maybe exit or allow workflow to continue with failed status?
            # For now, let's exit to prevent cascading errors.
            sys.exit(1)

        # Record outputs from ActivitySim run (final outputs)
        asim_output_dir = os.path.join(
            state.full_path, settings["asim_local_output_folder"]
        )
        asim_output_tables_settings = settings["asim_output_tables"]
        prefix = asim_output_tables_settings["prefix"]
        output_tables = asim_output_tables_settings["tables"]
        file_format = settings.get("file_format", "parquet")

        for table_name in output_tables:
            if file_format == "csv":
                file_name = f"{prefix}{table_name}.csv"
                file_path = os.path.join(
                    asim_output_dir, settings["asim_local_output_folder"], file_name
                )  # This path seems wrong, should be asim_output_dir directly?
                # Corrected path assumption:
                file_path = os.path.join(asim_output_dir, file_name)
                if os.path.exists(file_path):
                    state.record_output_file(
                        activity_demand_model,
                        file_path,
                        year=state.forecast_year,
                        description=f"ActivitySim final output table: {table_name} (CSV)",
                    )
            elif file_format == "parquet":
                # Parquet outputs are typically in a 'final_pipeline' subdir
                file_path = os.path.join(
                    asim_output_dir, "final_pipeline", table_name, "final.parquet"
                )
                if os.path.exists(file_path):
                    state.record_output_file(
                        activity_demand_model,
                        file_path,
                        year=state.forecast_year,
                        description=f"ActivitySim final output table: {table_name} (Parquet)",
                    )
            # Add other formats if needed

        # 4. COPY ACTIVITY DEMAND OUTPUTS --> LAND USE INPUTS
        if (settings.get("land_use_model") is not None) and (not warm_start):
            land_use_model = settings.get("land_use_model", "UrbanSim")
            print_str = "Generating {0} {1} input data from " "{2} outputs".format(
                state.forecast_year, land_use_model, activity_demand_model
            )
            formatted_print(print_str)

            # Record inputs to UrbanSim input creation postprocessing (ASim outputs and previous USim data)
            # ASim outputs recorded above
            usim_input_store_path_prev = os.path.join(
                state.full_path,
                settings["usim_local_mutable_data_folder"],
                usim_post.get_usim_datastore_fname(settings, io="input"),
            )
            state.record_input_file(
                land_use_model,
                usim_input_store_path_prev,
                description="UrbanSim input data from previous iteration",
            )
            usim_output_store_path_prev = os.path.join(
                state.full_path,
                settings["usim_local_mutable_data_folder"],
                usim_post.get_usim_datastore_fname(
                    settings, io="output", year=state.forecast_year
                ),
            )
            state.record_input_file(
                land_use_model,
                usim_output_store_path_prev,
                description="UrbanSim output data from current year",
            )

            asim_post.create_next_iter_inputs(settings, year, state)

            # Record outputs from UrbanSim input creation postprocessing (New USim input H5)
            usim_input_store_path_next = os.path.join(
                state.full_path,
                settings["usim_local_mutable_data_folder"],
                usim_post.get_usim_datastore_fname(settings, io="input"),
            )
            state.record_output_file(
                land_use_model,
                usim_input_store_path_next,
                year=state.forecast_year + settings.get("land_use_freq", 1),
                description="UrbanSim input data for next iteration",
            )

    logger.info("Done!")

    return


def run_traffic_assignment(
    settings, state: WorkflowState, client, iteration_number=0
):  # Use iteration_number for BEAM
    """
    This step will run the traffic simulation platform and
    generate new skims with updated congested travel times.
    """
    logger.info("===== STARTING TRAFFIC ASSIGNMENT =====")
    travel_model, travel_model_image = get_model_and_image(settings, "travel_model")
    logger.info(f"Travel model: {travel_model}, Image: {travel_model_image}")

    if travel_model == "polaris":
        # run_polaris(state.forecast_year, settings, warm_start=False)
        logger.info(
            "POLARIS module is not activated due to missing polarisruntime library"
        )

    elif travel_model == "beam":
        # 1. PARSE SETTINGS
        beam_config = settings["beam_config"]
        year = state.current_year  # Use year from state
        region = settings["region"]
        path_to_beam_config = "/app/input/{0}/{1}".format(region, beam_config)
        run_path = state.full_path
        beam_local_mutable_data_folder = os.path.join(
            run_path, settings["beam_local_mutable_data_folder"]
        )
        abs_beam_input = os.path.abspath(str(beam_local_mutable_data_folder))
        logger.info(
            f"Absolute path to BEAM input: {abs_beam_input} -> Container: /app/input (rw)"
        )

        beam_local_output_folder = os.path.join(
            run_path, settings["beam_local_output_folder"]
        )
        abs_beam_output = os.path.abspath(str(beam_local_output_folder))
        logger.info(
            f"Absolute path to BEAM output: {abs_beam_output} -> Container: /app/output (rw)"
        )

        activity_demand_model = settings.get("activity_demand_model", False)
        logger.info(f"Activity demand model: {activity_demand_model}")

        docker_stdout = settings["docker_stdout"]
        skims_fname = settings["skims_fname"]
        origin_skims_fname = settings["origin_skims_fname"]
        beam_memory = settings.get(
            "beam_memory",
            str(int(psutil.virtual_memory().total / (1024.0**3)) - 2) + "g",
        )
        logger.info(f"BEAM memory allocation: {beam_memory}")

        # remember the last produced skims in order to detect that
        # beam didn't work properly during this run
        if skims_fname.endswith(".csv.gz"):
            skimFormat = "csv.gz"
        elif skims_fname.endswith(".omx"):
            skimFormat = "omx"
        else:
            logger.error("Invalid skim format {0}".format(skims_fname))
            sys.exit(1)  # Exit on invalid skim format

        previous_od_skims = beam_post.find_produced_od_skims(
            beam_local_output_folder, skimFormat
        )
        previous_origin_skims = beam_post.find_produced_origin_skims(
            beam_local_output_folder
        )
        if previous_origin_skims:
            logger.info(f"Found skims from the previous BEAM run: {previous_od_skims}")

        # 2. COPY ACTIVITY DEMAND OUTPUTS --> TRAFFIC ASSIGNMENT INPUTS
        if settings["traffic_assignment_enabled"]:
            print_str = "Generating {0} {1} input data from " "{2} outputs".format(
                year, travel_model, activity_demand_model
            )
            formatted_print(print_str)
            logger.info("Copying plans from ActivitySim to BEAM")

            # Record inputs for BEAM input preparation
            asim_output_data_dir = os.path.join(
                state.full_path, settings["asim_local_output_folder"]
            )
            file_format = settings.get("file_format", "parquet")
            asim_plans_path = (
                os.path.join(
                    asim_output_data_dir,
                    "final_pipeline",
                    "beam_plans",
                    "final.parquet",
                )
                if file_format == "parquet"
                else os.path.join(asim_output_data_dir, "final_plans.csv")
            )
            asim_households_path = (
                os.path.join(
                    asim_output_data_dir,
                    "final_pipeline",
                    "households",
                    "final.parquet",
                )
                if file_format == "parquet"
                else os.path.join(asim_output_data_dir, "final_households.csv")
            )
            asim_persons_path = (
                os.path.join(
                    asim_output_data_dir, "final_pipeline", "persons", "final.parquet"
                )
                if file_format == "parquet"
                else os.path.join(asim_output_data_dir, "final_persons.csv")
            )
            atlas_vehicles_file = os.path.join(
                state.full_path,
                settings["atlas_host_output_folder"],
                f"vehicles_{state.forecast_year}.csv.gz",
            )

            inputs = [
                (asim_plans_path, "ActivitySim plans for BEAM input"),
                (asim_households_path, "ActivitySim households for BEAM input"),
                (asim_persons_path, "ActivitySim persons for BEAM input"),
            ]

            if settings.get("vehicle_ownership_model_enabled"):
                inputs.append(
                    (atlas_vehicles_file, "Atlas vehicles output for BEAM input")
                )

            record_inputs_and_outputs(state, travel_model, inputs=inputs)

            beam_pre.copy_plans_from_asim(settings, state, iteration_number)

            # Record outputs from BEAM input preparation
            beam_scenario_folder = os.path.join(
                state.full_path,
                settings["beam_local_mutable_data_folder"],
                settings["region"],
                settings["beam_scenario_folder"],
            )
            beam_plans_path = os.path.join(beam_scenario_folder, f"plans.{file_format}")
            beam_households_path = os.path.join(
                beam_scenario_folder, f"households.{file_format}"
            )
            beam_persons_path = os.path.join(
                beam_scenario_folder, f"persons.{file_format}"
            )
            beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")

            outputs = [
                (beam_plans_path, "BEAM scenario plans file"),
                (beam_households_path, "BEAM scenario households file"),
                (beam_persons_path, "BEAM scenario persons file"),
                (beam_vehicles_path, "BEAM scenario vehicles file"),
            ]

            record_inputs_and_outputs(
                state, travel_model, outputs=outputs, year=state.forecast_year
            )

        # 3. RUN BEAM
        logger.info(
            "Starting BEAM container, input: %s, output: %s, config: %s",
            abs_beam_input,
            abs_beam_output,
            beam_config,
        )

        # Check if the beam config file exists (the one copied to the run directory)
        expected_config_path_in_run_dir = os.path.join(
            abs_beam_input, region, beam_config
        )
        if os.path.exists(expected_config_path_in_run_dir):
            logger.info(
                f"BEAM config file exists at host path: {expected_config_path_in_run_dir}"
            )
            # Record the config file as an input to the BEAM run itself
            state.record_input_file(
                travel_model,
                expected_config_path_in_run_dir,
                description="BEAM configuration file used for run",
            )
        else:
            logger.warning(
                f"BEAM config file NOT FOUND at expected host path: {expected_config_path_in_run_dir}"
            )
            # Decide how to handle this - maybe exit? For now, just warn.

        # Record BEAM run start
        beam_run_index = state.record_model_start(
            travel_model, year=state.forecast_year, iteration=iteration_number
        )

        success = run_container(
            client,
            settings,
            travel_model_image,
            volumes={
                abs_beam_input: {"bind": "/app/input", "mode": "rw"},
                abs_beam_output: {"bind": "/app/output", "mode": "rw"},
            },
            environment={"JAVA_OPTS": ("-Xmx{0}".format(beam_memory))},
            working_dir="/app",
            command="--config={0}".format(path_to_beam_config),
            model_name="beam",
        )

        # Record BEAM run completion
        state.record_model_completion(
            beam_run_index, status="completed" if success else "failed"
        )
        if not success:
            logger.error("BEAM run failed.")
            sys.exit(1)  # Exit if BEAM fails

        # 4. POSTPROCESS
        path_to_mutable_od_skims = os.path.join(abs_beam_output, skims_fname)
        path_to_origin_skims = os.path.join(abs_beam_output, origin_skims_fname)
        logger.info(f"Path to mutable OD skims: {path_to_mutable_od_skims}")
        logger.info(f"Path to origin skims: {path_to_origin_skims}")

        # Record raw BEAM output skims before merging/processing
        if os.path.exists(path_to_mutable_od_skims):
            state.record_output_file(
                travel_model,
                path_to_mutable_od_skims,
                year=state.forecast_year,
                description="BEAM raw OD skims output",
            )
        if os.path.exists(path_to_origin_skims):
            state.record_output_file(
                travel_model,
                path_to_origin_skims,
                year=state.forecast_year,
                description="BEAM raw origin skims output",
            )

        if skimFormat == "csv.gz":
            logger.info("Processing CSV.GZ format skims")
            current_od_skims = beam_post.merge_current_od_skims(
                path_to_mutable_od_skims, previous_od_skims, beam_local_output_folder
            )
            if (
                current_od_skims == previous_od_skims and iteration_number > 0
            ):  # Check for failure in non-first iter
                logger.error(
                    "BEAM hasn't produced the new skims at {0} for some reason. "
                    "Please check beamLog.out for errors in the directory {1}".format(
                        current_od_skims, abs_beam_output
                    )
                )
                sys.exit(1)

            beam_post.merge_current_origin_skims(
                path_to_origin_skims, previous_origin_skims, beam_local_output_folder
            )

            # Record outputs from BEAM postprocessing (merged skims)
            # The merged skims overwrite the input skims for ASim, so record the ASim skims path as output
            asim_mutable_data_dir = os.path.join(
                state.full_path, settings["asim_local_mutable_data_folder"]
            )
            asim_skims_path = os.path.join(
                asim_mutable_data_dir, settings["skims_fname"]
            )
            if os.path.exists(asim_skims_path):
                state.record_output_file(
                    travel_model,
                    asim_skims_path,
                    year=state.forecast_year,
                    description="ActivitySim skims input file updated by BEAM postprocessing",
                )

        else:  # OMX or Zarr
            logger.info(f"Processing {skimFormat} format skims")

            # Check if ActivitySim is enabled - only proceed with ActivitySim integration if it's enabled
            asim_enabled = (
                activity_demand_model and activity_demand_model == "activitysim"
            )
            beam_asim_ridehail_measure_map = settings["beam_asim_ridehail_measure_map"]

            if asim_enabled:
                asim_data_dir = (
                    os.path.join(
                        state.full_path, settings["asim_local_output_folder"], "cache"
                    )
                    if settings["file_format"] == "parquet"
                    else os.path.join(
                        state.full_path, settings["asim_local_mutable_data_folder"]
                    )
                )
                asim_skims_path = (
                    os.path.join(asim_data_dir, "skims.zarr")
                    if settings["file_format"] == "parquet"
                    else os.path.join(asim_data_dir, "skims.omx")
                )

                # Record inputs to BEAM postprocessing (ASim skims and BEAM raw skims)
                if os.path.exists(asim_skims_path):
                    state.record_input_file(
                        travel_model,
                        asim_skims_path,
                        description="ActivitySim skims input file before BEAM update",
                    )
                # BEAM raw skims recorded above

                if settings["file_format"] == "parquet":
                    current_od_skims = beam_post.merge_current_zarr_od_skims(
                        asim_skims_path, beam_local_output_folder, settings
                    )
                    logger.warning(
                        "RIDEHAIL SKIM MERGING NOT YET IMPLEMENTED FOR PARQUET FILES"
                    )
                else:  # OMX
                    current_od_skims = beam_post.merge_current_omx_od_skims(
                        asim_skims_path,
                        previous_od_skims,
                        beam_local_output_folder,
                        settings,
                    )
                    beam_post.merge_current_omx_origin_skims(
                        asim_skims_path,
                        previous_origin_skims,
                        beam_local_output_folder,
                        beam_asim_ridehail_measure_map,
                    )

                logger.info(f"ActivitySim data directory: {asim_data_dir}")
                logger.info(f"ActivitySim skims path: {asim_skims_path}")

                # Record outputs from BEAM postprocessing (updated ASim skims)
                if os.path.exists(asim_skims_path):
                    state.record_output_file(
                        travel_model,
                        asim_skims_path,
                        year=state.forecast_year,
                        description=f"ActivitySim skims input file updated by BEAM postprocessing ({skimFormat})",
                    )

                if current_od_skims == previous_od_skims and iteration_number > 0:
                    logger.error(
                        "BEAM hasn't produced the new skims at {0} for some reason. "
                        "Please check beamLog.out for errors in the directory {1}".format(
                            current_od_skims, abs_beam_output
                        )
                    )
                    sys.exit(1)
            else:
                logger.info(
                    "ActivitySim is not enabled, skipping skim merging for ActivitySim"
                )
                # Still check if BEAM produced skims even if ASim is off
                current_od_skims = beam_post.find_produced_od_skims(
                    beam_local_output_folder, skimFormat
                )
                if (
                    current_od_skims == previous_od_skims and iteration_number > 0
                ):  # Check for failure in non-first iter
                    logger.error(
                        "BEAM hasn't produced the new skims at {0} for some reason. "
                        "Please check beamLog.out for errors in the directory {1}".format(
                            current_od_skims, abs_beam_output
                        )
                    )
                    sys.exit(1)  # Exit if BEAM failed to produce skims

        logger.info(
            f"Renaming BEAM output directory for year {year}, iteration {iteration_number}"
        )
        beam_post.rename_beam_output_directory(
            abs_beam_output, settings, year, iteration_number
        )
        # The renamed directory contains the raw BEAM outputs, which were already recorded above.

        logger.info("===== COMPLETED TRAFFIC ASSIGNMENT =====")

    return


def initialize_docker_client(settings):
    land_use_model = settings.get("land_use_model", False)
    vehicle_ownership_model = settings.get("vehicle_ownership_model", False)  ## ATLAS
    activity_demand_model = settings.get("activity_demand_model", False)
    travel_model = settings.get("travel_model", False)
    models = [
        land_use_model,
        vehicle_ownership_model,
        activity_demand_model,
        travel_model,
    ]
    image_names = settings["docker_images"]
    pull_latest = settings.get("pull_latest", False)

    client = docker.from_env()
    if pull_latest:
        logger.info("Pulling from docker...")
        for model in models:
            if model:
                image = image_names.get(model)  # Use .get for safety
                if image is not None:
                    print("Pulling latest image for {0}".format(image))
                    try:
                        client.images.pull(image)
                    except docker.errors.ImageNotFound:
                        logger.error(f"Docker image {image} not found.")
                    except Exception as e:
                        logger.error(f"Error pulling docker image {image}: {e}")

    return client


def postprocess_all(settings, state: WorkflowState):
    logger.info("===== STARTING POSTPROCESSING =====")
    # Record Postprocessing stage start
    postprocess_run_index = state.record_model_start(
        "postprocessing", year=state.current_year
    )

    beam_output_dir = settings["beam_local_output_folder"]
    region = settings["region"]
    # Need to find the actual output directories created by BEAM runs within this run's output folder
    run_beam_output_base = os.path.join(state.full_path, beam_output_dir)
    output_path_pattern = os.path.join(run_beam_output_base, region, "year*-iter*")
    outputDirs = glob.glob(output_path_pattern)
    yearsAndIters = [
        (
            os.path.basename(loc).split("-year")[-1].split("-iter")[0],
            os.path.basename(loc).split("-iter")[-1],
        )
        for loc in outputDirs
    ]

    # Process event files for the latest iteration of each year found
    processed_years = set()
    for year_str, iter_str in sorted(
        yearsAndIters, key=lambda x: (int(x[0]), int(x[1]))
    ):  # Process in order
        year = int(year_str)
        iter = int(iter_str)
        if year not in processed_years:
            logger.info(f"Processing event file for year {year}, iteration {iter}")
            try:
                # Record inputs to event file processing (BEAM event file)
                # Assuming event file is named events.csv.gz inside the iteration folder
                beam_iter_output_dir = os.path.join(
                    run_beam_output_base, region, f"year{year}-iter{iter}"
                )
                beam_event_file = os.path.join(beam_iter_output_dir, "events.csv.gz")
                if os.path.exists(beam_event_file):
                    state.record_input_file(
                        "postprocessing",
                        beam_event_file,
                        description=f"BEAM events file (Year {year}, Iter {iter})",
                    )
                else:
                    logger.warning(
                        f"BEAM event file not found for Year {year}, Iter {iter} at {beam_event_file}"
                    )

                process_event_file(
                    settings, year, iter, base_output_dir=run_beam_output_base
                )

                # Record outputs from event file processing (processed CSVs)
                processed_output_dir = os.path.join(
                    run_beam_output_base, region, f"year{year}-iter{iter}", "processed"
                )
                if os.path.exists(processed_output_dir):
                    for file in os.listdir(processed_output_dir):
                        if file.endswith(".csv"):
                            state.record_output_file(
                                "postprocessing",
                                os.path.join(processed_output_dir, file),
                                year=year,
                                description=f"Processed BEAM output file: {file} (Year {year}, Iter {iter})",
                            )

                processed_years.add(year)  # Mark year as processed (only latest iter)
            except Exception as e:
                logger.error(
                    f"Error processing event file for year {year}, iteration {iter}: {e}"
                )
                # Decide how to handle postprocessing errors - continue or exit?
                # For now, log and continue.

    # Copy outputs to MEP if enabled
    if settings.get("copy_outputs_to_mep", False):
        logger.info("Copying outputs to MEP")
        try:
            copy_outputs_to_mep(settings, state.full_path)
            # Recording outputs copied to MEP is complex as destination is external.
            # Could record the source files that were copied.
            logger.info("Finished copying outputs to MEP")
        except Exception as e:
            logger.error(f"Error copying outputs to MEP: {e}")

    # Record Postprocessing stage completion
    state.record_model_completion(
        postprocess_run_index, status="completed"
    )  # Assuming completed even if some steps failed

    logger.info("===== COMPLETED POSTPROCESSING =====")


def to_singularity_volumes(volumes):
    bindings = [
        f"{local_folder}:{binding['bind']}:{binding['mode']}"
        for local_folder, binding in volumes.items()
    ]
    result_str = ",".join(bindings)
    return result_str


def to_singularity_env(env):
    bindings = [f"{env_var}={value}" for env_var, value in env.items()]
    result_str = ",".join(bindings)
    return '"' + result_str + '"'


def run_container(
    client,
    settings: dict,
    image: str,
    volumes: dict,
    command: str,
    working_dir=None,
    environment=None,
    args=None,
    model_name=None,
) -> bool:
    """
    Executes container using docker or singularity
    :param client: the docker client. If it's provided then docker is used, otherwise singularity is used
    :param settings: settings to get docker configuration
    :param image: the image to run
    :param volumes: a dictionary describing volume binding
    :param command: the command to run
    :param working_dir: the working directory inside the container. It's not necessary for docker because
    docker file may have an instruction WORKDIR. In this case that directory is used. Singularity don't take this
    instruction into account and the container working dir is the host working dir. Because of that most of the time
     singularity requires working dir. One can get the work dir from a docker image by looking at the Dockerfile
     (or image layers at the docker hub) and find the last WORKDIR instruction or by issuing a command:
      docker run -it --entrypoint /bin/bash ghcr.io/lbnl-science-it/atlas:v1.0.7 -c "env | grep PWD"
    :param environment: a dictionary that contains environment variables that needs to be set to the container
    :param args: additional arguments to the command
    :param model_name: The name of the model being run (for stubbing)
    :return: True if the container/stub ran successfully (exit code 0), False otherwise.
    """
    if client:  # Docker client is available
        docker_stdout = settings.get("docker_stdout", False)
        logger.info("Running docker container: %s, command: %s", image, command)
        run_kwargs = {
            "volumes": volumes,
            "command": command,
            "stdout": docker_stdout,
            "stderr": True,  # Always capture stderr
            "detach": True,  # Run in detached mode to stream logs
        }
        if working_dir:
            run_kwargs["working_dir"] = working_dir
        if environment:
            run_kwargs["environment"] = environment
        if args:
            # Append args to command string for docker
            full_command = command + " " + " ".join(args)
            run_kwargs["command"] = full_command
            logger.info("Full docker command: %s", full_command)

        container = None
        try:
            container = client.containers.run(image, **run_kwargs)
            # Stream logs
            for line in container.logs(stream=True, stderr=True, stdout=docker_stdout):
                # Decode bytes and print
                try:
                    print(line.decode("utf-8").strip())
                except UnicodeDecodeError:
                    print(line.strip())  # Print raw bytes if decoding fails

            # Wait for container to finish and get exit code
            result = container.wait()
            exit_code = result["StatusCode"]
            logger.info(
                f"Docker container {image} finished with exit code: {exit_code}"
            )
            return exit_code == 0

        except docker.errors.ImageNotFound:
            logger.error(f"Docker image not found: {image}")
            return False
        except docker.errors.APIError as e:
            logger.error(f"Docker API error running container {image}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running docker container {image}: {e}")
            return False
        finally:
            if container:
                try:
                    container.remove()
                    logger.debug(f"Removed container {container.id}")
                except docker.errors.APIError as e:
                    logger.warning(f"Could not remove container {container.id}: {e}")

    else:  # Singularity
        for local_folder in volumes:
            # Ensure local directories exist for singularity binds
            os.makedirs(local_folder, exist_ok=True)

        singularity_volumes = to_singularity_volumes(volumes)
        # Construct the singularity command
        proc = (
            ["singularity", "run", "--cleanenv", "--writable-tmpfs"]
            + (["--env", to_singularity_env(environment)] if environment else [])
            + (["--pwd", working_dir] if working_dir else [])
            + ["-B", singularity_volumes, image]
            + (args if args else [])
            + command.split()
        )  # Split command string into list

        logger.info("Running singularity command: %s", " ".join(proc))

        # Check if using stubs
        if settings.get("use_stubs"):
            # Pass the full command string as config_name to the stub
            stub_cmd = [
                "python",
                os.path.join(
                    os.path.dirname(__file__), "tests", "stubs", "run_stub.py"
                ),  # Use absolute path for stub
                "--model_name",
                model_name,
                "--cwd",
                os.getcwd(),  # Pass current working directory of run.py
                "--config_name",
                command,  # Pass the original command string
            ]
            logger.info(
                f"Using stub for {model_name} ({image}). Running stub command: {' '.join(stub_cmd)}"
            )
            try:
                result = subprocess.run(
                    stub_cmd, check=True, capture_output=True, text=True
                )
                print("Stub stdout:\n", result.stdout)
                print("Stub stderr:\n", result.stderr)
                logger.info(f"Stub for {model_name} finished successfully.")
                return True  # Stub success is check=True
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Stub for {model_name} failed with exit code {e.returncode}."
                )
                print("Stub stdout:\n", e.stdout)
                print("Stub stderr:\n", e.stderr)
                return False
            except FileNotFoundError:
                logger.error(f"Stub script not found at {stub_cmd[2]}.")
                return False
            except Exception as e:
                logger.error(f"Unexpected error running stub for {model_name}: {e}")
                return False

        else:  # Run actual singularity container
            try:
                # Use subprocess.run to execute singularity command
                result = subprocess.run(
                    proc, check=False
                )  # Don't raise exception on non-zero exit code
                logger.info(
                    f"Singularity container {image} finished with exit code: {result.returncode}"
                )
                return result.returncode == 0
            except FileNotFoundError:
                logger.error(
                    f"Singularity command not found. Is Singularity installed and in your PATH?"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error running singularity container {image}: {e}"
                )
                return False


def get_model_and_image(settings: dict, model_type: str):
    manager = settings.get("container_manager")
    if manager == "docker":
        image_names = settings.get("docker_images", {})
    elif manager == "singularity":
        image_names = settings.get("singularity_images", {})
    else:
        raise ValueError(
            "Container Manager not specified (container_manager param in settings.yaml)"
        )

    model_name = settings.get(model_type)
    if not model_name:
        # If model type is optional (e.g., vehicle_ownership_model), return None
        optional_models = ["vehicle_ownership_model"]  # Add other optional models here
        if model_type in optional_models:
            return None, None
        else:
            raise ValueError(f"No model {model_type} specified in settings.")

    image_name = image_names.get(model_name)
    if not image_name:
        raise ValueError(
            f"No {manager} image specified for model '{model_name}' (model type: {model_type}). Check settings['{manager}_images']."
        )

    return model_name, image_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Path to the config file to be used", action="store_true"
    )  # This action='store_true' seems incorrect for a path
    parser.add_argument(
        "-s",
        "--state",
        help="Path to the current_state file to be used",
        action="store_true",
    )  # This action='store_true' seems incorrect for a path

    # Corrected argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Path to the config file to be used", type=str
    )
    parser.add_argument(
        "-s", "--state", help="Path to the current_state file to be used", type=str
    )

    logger = logging.getLogger(__name__)

    logger = logging.getLogger(__name__)

    logger.info("Preparing runtime environment...")

    #########################################
    #  PREPARE PILATES RUNTIME ENVIRONMENT  #
    #########################################

    # load args and settings
    settings = parse_args_and_settings()

    logger.info(
        "Using config file {}".format(settings.get("settings_file", "N/A"))
    )  # Use .get for safety

    # parse scenario settings
    start_year = settings["start_year"]
    end_year = settings["end_year"]
    travel_model = settings.get("travel_model", False)
    formatted_print("RUNNING PILATES FROM {0} TO {1}".format(start_year, end_year))
    travel_model_freq = settings.get("travel_model_freq", 1)
    warm_start_skims = settings.get("warm_start_skims", False)  # Use .get with default
    warm_start_activities_enabled = settings.get(
        "warm_start_activities", False
    )  # Use .get with default
    static_skims = settings.get("static_skims", False)  # Use .get with default
    land_use_enabled = settings.get("land_use_enabled", False)  # Use .get with default
    land_use_freq = settings.get("land_use_freq", 1)  # Use .get with default
    vehicle_ownership_model_enabled = settings.get(
        "vehicle_ownership_model_enabled", False
    )  # Atlas, Use .get with default
    activity_demand_enabled = settings.get(
        "activity_demand_enabled", False
    )  # Use .get with default
    traffic_assignment_enabled = settings.get(
        "traffic_assignment_enabled", False
    )  # Use .get with default
    replanning_enabled = settings.get(
        "replanning_enabled", False
    )  # Use .get with default
    container_manager = settings.get("container_manager")  # Use .get

    state = WorkflowState.from_settings(settings)
    working_dir = (
        state.full_path
    )  # This is the run-specific output directory if specified, else cwd

    if not land_use_enabled:
        print("LAND USE MODEL DISABLED")
    if not activity_demand_enabled:
        print("ACTIVITY DEMAND MODEL DISABLED")
    if not traffic_assignment_enabled:
        print("TRAFFIC ASSIGNMENT MODEL DISABLED")
    if not vehicle_ownership_model_enabled:
        print("VEHICLE OWNERSHIP MODEL DISABLED")

    if traffic_assignment_enabled:
        try:
            # Update BEAM config in the mutable run directory
            beam_config_path_in_run_dir = os.path.join(
                state.full_path,
                settings["beam_local_mutable_data_folder"],
                settings["region"],
                settings["beam_config"],
            )
            if os.path.exists(beam_config_path_in_run_dir):
                beam_pre.update_beam_config(settings, state.full_path, "beam_sample")
                # Record the modified config file
                state.record_output_file(
                    settings.get("travel_model", "beam"),
                    beam_config_path_in_run_dir,
                    description="BEAM config file updated with beam_sample",
                )
            else:
                logger.warning(
                    f"BEAM config file not found at {beam_config_path_in_run_dir}. Cannot update beam_sample."
                )

        except FileNotFoundError:
            logger.error("Failed to update beam_config (FileNotFoundError).")
        except Exception as e:
            logger.error(f"An error occurred while updating beam_config: {e}")

    if warm_start_skims:
        formatted_print('"WARM START SKIMS" MODE ENABLED')
        logger.info("Generating activity plans for the base year only.")
    elif static_skims:
        formatted_print('"STATIC SKIMS" MODE ENABLED')
        logger.info("Using the same set of skims for every iteration.")

    # start docker client
    client = None  # Initialize client to None
    if container_manager == "docker":
        try:
            client = initialize_docker_client(settings)
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            # Decide how to handle failure - maybe exit?
            # For now, log and continue, assuming Singularity might be used or stubs.
            # If no client and no singularity, container runs will fail later.

    #################################
    #  RUN THE SIMULATION WORKFLOW  #
    #################################

    # The WorkflowState iterator handles advancing years and stopping
    for year in state:
        logger.info(f"=== Starting workflow for year {year} ===")
        logger.info(
            f"Current state: major_stage={state.current_major_stage.name if state.current_major_stage else None}, inner_iter={state.current_inner_iter}, sub_stage={state.current_sub_stage.name if state.current_sub_stage else None}"
        )
        logger.info(
            f"Enabled stages: {[s.name for s in state.enabled_individual_stages]}"
        )

        # 1. FORECAST LAND USE
        if state.should_run(WorkflowState.Stage.land_use):
            # Skip if land use model is not enabled
            if not land_use_enabled:
                logger.info("Skipping land use stage: land use model not enabled")
                state.complete_step(WorkflowState.Stage.land_use)
            else:
                # hack: make sure that the usim datastore isn't open
                # This check might need adjustment if using mutable copy in run directory
                usim_data_path_check = os.path.join(
                    state.full_path,
                    settings["usim_local_mutable_data_folder"],
                    usim_post.get_usim_datastore_fname(settings, io="input"),
                )
                if is_already_opened_in_write_mode(usim_data_path_check):
                    logger.warning(
                        "Closing h5 files {0} because they were left open. You should really "
                        "figure out where this happened".format(
                            tables.file._open_files.filenames
                        )
                    )
                    tables.file._open_files.close_all()

                # 1a. IF START YEAR, WARM START MANDATORY ACTIVITIES
                if (state.is_start_year()) and warm_start_activities_enabled:
                    # IF ATLAS ENABLED, UPDATE USIM INPUT H5
                    if vehicle_ownership_model_enabled:
                        # run_atlas_auto handles its own provenance tracking internally
                        run_atlas_auto(settings, state, client, warm_start_atlas=True)
                    # warm_start_activities handles its own provenance tracking internally
                    warm_start_activities(settings, state, client)

                # 1b. RUN LAND USE SIMULATION
                # forecast_land_use handles its own provenance tracking internally
                forecast_land_use(settings, year, state, client, container_manager)
                state.complete_step(WorkflowState.Stage.land_use)

        # 2. RUN ATLAS (HOUSEHOLD VEHICLE OWNERSHIP) - Outside supply-demand loop
        # This stage runs *after* Land Use for the current year, but before the Supply-Demand loop
        if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
            if not vehicle_ownership_model_enabled:
                logger.info("Skipping vehicle ownership model: not enabled in settings")
                state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
            elif (
                state.forecast_year > 2017
            ):  # Check if forecast year is valid for Atlas
                # run_atlas_auto handles its own provenance tracking internally
                # It also handles warm_start vs forecast logic internally
                run_atlas_auto(
                    settings, state, client, warm_start_atlas=False, forecast=True
                )
            else:
                logger.info(
                    f"Skipping atlas in year {state.year} because forecast year {state.forecast_year} is not > 2017"
                )

            # Complete the vehicle ownership model stage
            state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

        # 3. SUPPLY-DEMAND LOOP - Run multiple iterations for the same year
        # The should_run check for supply_demand_loop is handled by the iterator logic
        # and the complete_step logic for substages.
        # The loop itself is managed by the WorkflowState's iteration count.
        while state.should_run(
            WorkflowState.Stage.supply_demand_loop, state.current_inner_iter
        ):
            iteration = state.current_inner_iter
            logger.info(
                f"Starting supply-demand iteration {iteration + 1} of {settings.get('supply_demand_iters', 1)} for year {year}"
            )

            # Iterate through substages within the loop
            # Get the sequence of substages starting from the current one
            substage_sequence = state.get_sub_stages_from(state.current_sub_stage)

            for substage in substage_sequence:
                if state.should_run(
                    WorkflowState.Stage.supply_demand_loop, iteration, substage
                ):
                    if substage == WorkflowState.Stage.activity_demand:
                        activity_demand_model = settings.get(
                            "activity_demand_model", None
                        )
                        if activity_demand_model and activity_demand_enabled:
                            logger.info(
                                f"Starting activity demand iteration {iteration + 1} for year {year}"
                            )
                            # generate_activity_plans handles its own provenance tracking internally
                            generate_activity_plans(
                                settings,
                                year,
                                state,
                                client,
                                warm_start=warm_start_skims or not land_use_enabled,
                            )
                        else:
                            logger.info(
                                "Skipping activity demand generation: activity demand model not enabled"
                            )

                    elif (
                        substage
                        == WorkflowState.Stage.activity_demand_directly_from_land_use
                    ):
                        land_use_model = settings.get("land_use_model", False)
                        if (
                            not settings.get("land_use_enabled", False)
                            or not land_use_model
                        ):
                            logger.info(
                                "Skipping direct activity generation from land use: land use model not enabled"
                            )
                        else:
                            logger.info(
                                f"Starting direct activity generation from land use: {land_use_model}"
                            )
                            # Record inputs to direct activity generation (UrbanSim output H5)
                            usim_output_store_name = usim_post.get_usim_datastore_fname(
                                settings, io="output", year=state.forecast_year
                            )
                            usim_output_store_path = os.path.join(
                                state.full_path,
                                settings["usim_local_mutable_data_folder"],
                                usim_output_store_name,
                            )
                            state.record_input_file(
                                land_use_model,
                                usim_output_store_path,
                                description="UrbanSim output for direct activity generation",
                            )

                            usim_post.create_next_iter_usim_data(
                                settings, year, state.forecast_year, state.full_path
                            )

                            # Record outputs from direct activity generation (New USim input H5)
                            usim_input_store_path_next = os.path.join(
                                state.full_path,
                                settings["usim_local_mutable_data_folder"],
                                usim_post.get_usim_datastore_fname(
                                    settings, io="input"
                                ),
                            )
                            state.record_output_file(
                                land_use_model,
                                usim_input_store_path_next,
                                year=state.forecast_year
                                + settings.get("land_use_freq", 1),
                                description="UrbanSim input data for next iteration (from direct activity)",
                            )

                    elif substage == WorkflowState.Stage.traffic_assignment:
                        if settings["discard_plans_every_year"]:
                            # Update BEAM config in the mutable run directory
                            beam_config_path_in_run_dir = os.path.join(
                                state.full_path,
                                settings["beam_local_mutable_data_folder"],
                                settings["region"],
                                settings["beam_config"],
                            )
                            if os.path.exists(beam_config_path_in_run_dir):
                                beam_pre.update_beam_config(
                                    settings, state.full_path, "max_plans_memory", 0
                                )
                                state.record_output_file(
                                    settings.get("travel_model", "beam"),
                                    beam_config_path_in_run_dir,
                                    description=f"BEAM config file updated with max_plans_memory=0 (Iter {iteration+1})",
                                )
                            else:
                                logger.warning(
                                    f"BEAM config file not found at {beam_config_path_in_run_dir}. Cannot update max_plans_memory."
                                )

                        else:
                            # Update BEAM config in the mutable run directory
                            beam_config_path_in_run_dir = os.path.join(
                                state.full_path,
                                settings["beam_local_mutable_data_folder"],
                                settings["region"],
                                settings["beam_config"],
                            )
                            if os.path.exists(beam_config_path_in_run_dir):
                                beam_pre.update_beam_config(
                                    settings, state.full_path, "max_plans_memory"
                                )  # Update with default/settings value
                                state.record_output_file(
                                    settings.get("travel_model", "beam"),
                                    beam_config_path_in_run_dir,
                                    description=f"BEAM config file updated with max_plans_memory (Iter {iteration+1})",
                                )
                            else:
                                logger.warning(
                                    f"BEAM config file not found at {beam_config_path_in_run_dir}. Cannot update max_plans_memory."
                                )

                        if vehicle_ownership_model_enabled:
                            # Record inputs to BEAM vehicle copy (Atlas vehicles)
                            atlas_output_path = os.path.join(
                                state.full_path, settings["atlas_host_output_folder"]
                            )
                            atlas_vehicles_file = os.path.join(
                                atlas_output_path,
                                f"vehicles_{state.forecast_year}.csv.gz",
                            )  # Assuming gzipped output
                            if os.path.exists(atlas_vehicles_file):
                                state.record_input_file(
                                    settings.get("travel_model", "beam"),
                                    atlas_vehicles_file,
                                    description=f"Atlas vehicles output for BEAM input (Iter {iteration+1})",
                                )
                            else:
                                logger.warning(
                                    f"Atlas vehicles file not found at {atlas_vehicles_file}. Cannot copy to BEAM."
                                )

                            beam_pre.copy_vehicles_from_atlas(settings, state)

                            # Record outputs from BEAM vehicle copy (BEAM vehicles.csv.gz)
                            beam_scenario_folder = os.path.join(
                                state.full_path,
                                settings["beam_local_mutable_data_folder"],
                                settings["region"],
                                settings["beam_scenario_folder"],
                            )
                            beam_vehicles_path = os.path.join(
                                beam_scenario_folder, "vehicles.csv.gz"
                            )
                            if os.path.exists(beam_vehicles_path):
                                state.record_output_file(
                                    settings.get("travel_model", "beam"),
                                    beam_vehicles_path,
                                    year=state.forecast_year,
                                    description=f"BEAM vehicles input file updated from Atlas (Iter {iteration+1})",
                                )

                        logger.info(
                            f"Running traffic assignment for iteration {iteration + 1} of year {year}"
                        )
                        # run_traffic_assignment handles its own provenance tracking internally
                        run_traffic_assignment(settings, state, client, iteration)

                    # Complete the current substage
                    state.complete_step(
                        WorkflowState.Stage.supply_demand_loop, iteration, substage
                    )

            # After iterating through all substages, the complete_step for the last substage
            # will automatically advance the iteration or the major stage.
            # The while loop condition will then be re-evaluated.

        # 4. POST-PROCESSING
        # This stage runs after the supply-demand loop for the year is complete
        if state.should_run(WorkflowState.Stage.postprocessing):
            # postprocess_all handles its own provenance tracking internally
            postprocess_all(settings, state)
            state.complete_step(WorkflowState.Stage.postprocessing)

    logger.info("Workflow finished.")


if __name__ == "__main__":
    main()
