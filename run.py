import random
import warnings

import pandas as pd
import tables
from tables import HDF5ExtError

from pilates.activitysim.preprocessor import copy_beam_geoms

warnings.simplefilter(action='ignore', category=FutureWarning)
from workflow_state import WorkflowState

import shutil
import subprocess
import multiprocessing
import psutil

try:
    import docker
except ImportError:
    print('Warning: Unable to import Docker Module')

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
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_and_init_data():
    usim_path = os.path.abspath('pilates/urbansim/data')
    if os.path.isdir(Path(usim_path) / 'backup'):
        clean_data(usim_path, '*.h5')
        clean_data(usim_path, '*.txt')
        init_data(usim_path, '*.h5')

    polaris_path = os.path.abspath('pilates/polaris/data')
    if os.path.isdir(Path(polaris_path) / 'backup'):
        clean_data(polaris_path, '*.hdf5')
        init_data(polaris_path, '*.hdf5')


def clean_data(path, wildcard):
    search_path = Path(path) / wildcard
    filelist = glob.glob(str(search_path))
    for filepath in filelist:
        try:
            os.remove(filepath)
        except:
            logger.error("Error whie deleting file : {0}".format(filepath))


def is_already_opened_in_write_mode(filename):
    if os.path.exists(filename):
        try:
            f = pd.HDFStore(filename, 'a')
            f.close()
        except HDF5ExtError:
            return True
        except RuntimeError as e:
            logger.warning(str(e))
            return True
    return False


def init_data(dest, wildcard):
    backup_dir = Path(dest) / 'backup' / wildcard
    for filepath in glob.glob(str(backup_dir)):
        shutil.copy(filepath, dest)


def formatted_print(string, width=50, fill_char='#'):
    print('\n')
    if len(string) + 2 > width:
        width = len(string) + 4
    string = string.upper()
    print(fill_char * width)
    print('{:#^{width}}'.format(' ' + string + ' ', width=width))
    print(fill_char * width, '\n')


def find_latest_beam_iteration(beam_output_dir):
    iter_dirs = []
    for root, dirs, files in os.walk(beam_output_dir):
        for dir in dirs:
            if dir == "ITER":
                iter_dirs += os.path.join(root, dir)
    print(iter_dirs)


def setup_beam_skims(settings):
    region = settings['region']
    region_id = settings['region_to_region_id'][region]
    beam_input_dir = settings['beam_local_input_folder']
    beam_output_dir = settings['beam_local_output_folder']
    skims_fname = settings['skims_fname']
    origin_skims_fname = settings['origin_skims_fname']
    beam_geoms_fname = settings['beam_geoms_fname']
    beam_router_directory = settings['beam_router_directory']
    asim_geoms_location = os.path.join(settings['asim_local_input_folder'], beam_geoms_fname)

    input_skims_location = os.path.join(beam_input_dir, region, skims_fname)
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)

    beam_geoms_location = os.path.join(beam_input_dir, region, beam_router_directory, beam_geoms_fname)

    # TODO: Handle exception when these dont exist

    if os.path.exists(input_skims_location):
        logger.info("Copying input skims from {0} to {1}".format(
            input_skims_location,
            mutable_skims_location))
        shutil.copyfile(input_skims_location, mutable_skims_location)
    else:
        if os.path.exists(mutable_skims_location):
            logger.info("No input skims at {0}. Proceeding with defaults at {1}".format(
                input_skims_location,
                mutable_skims_location))
        else:
            logger.info("No default skims found anywhere. We will generate defaults instead")

    input_skims_location = os.path.join(beam_input_dir, region, origin_skims_fname)
    mutable_skims_location = os.path.join(beam_output_dir, origin_skims_fname)

    if os.path.exists(input_skims_location):
        logger.info("Copying input origin skims from {0} to {1}".format(
            input_skims_location,
            mutable_skims_location))
        shutil.copyfile(input_skims_location, mutable_skims_location)
    else:
        if os.path.exists(mutable_skims_location):
            logger.info("No input skims at {0}. Proceeding with defaults at {1}".format(
                input_skims_location,
                mutable_skims_location))
        else:
            logger.info("No default input skims found anywhere. We will generate defaults instead")

    logger.info("Copying beam zone geoms from {0} to {1}".format(
        beam_geoms_location,
        asim_geoms_location))

    copy_beam_geoms(settings, beam_geoms_location, asim_geoms_location)


def get_base_asim_cmd(settings, household_sample_size=None):
    formattable_asim_cmd = settings['asim_formattable_command']
    if not household_sample_size:
        household_sample_size = settings.get('household_sample_size', 0)
    num_processes = settings.get('num_processes', multiprocessing.cpu_count() - 1)
    chunk_size = settings.get('chunk_size', 0)  # default no chunking
    base_asim_cmd = formattable_asim_cmd.format(
        household_sample_size, num_processes, chunk_size)
    return base_asim_cmd


def get_asim_docker_vols(settings):
    region = settings['region']
    asim_subdir = settings['region_to_asim_subdir'][region]
    asim_remote_workdir = os.path.join('/activitysim', asim_subdir)
    asim_local_input_folder = os.path.abspath(
        settings['asim_local_input_folder'])
    asim_local_output_folder = os.path.abspath(
        settings['asim_local_output_folder'])
    asim_local_configs_folder = os.path.abspath(
        os.path.join(settings['asim_local_configs_folder'], region))
    asim_remote_input_folder = os.path.join(
        asim_remote_workdir, 'data')
    asim_remote_output_folder = os.path.join(
        asim_remote_workdir, 'output')
    asim_remote_configs_folder = os.path.join(
        asim_remote_workdir, 'configs')
    asim_docker_vols = {
        asim_local_input_folder: {
            'bind': asim_remote_input_folder,
            'mode': 'rw'},
        asim_local_output_folder: {
            'bind': asim_remote_output_folder,
            'mode': 'rw'},
        asim_local_configs_folder: {
            'bind': asim_remote_configs_folder,
            'mode': 'rw'}}
    return asim_docker_vols


def get_usim_docker_vols(settings):
    usim_remote_data_folder = settings['usim_client_data_folder']
    usim_local_data_folder = os.path.abspath(
        settings['usim_local_data_folder'])
    usim_docker_vols = {
        usim_local_data_folder: {
            'bind': usim_remote_data_folder,
            'mode': 'rw'}}
    return usim_docker_vols


def get_usim_cmd(settings, year, forecast_year):
    region = settings['region']
    region_id = settings['region_to_region_id'][region]
    land_use_freq = settings['land_use_freq']
    skims_source = settings['travel_model']
    formattable_usim_cmd = settings['usim_formattable_command']
    usim_cmd = formattable_usim_cmd.format(
        region_id, year, forecast_year, land_use_freq, skims_source)
    return usim_cmd


## Atlas vehicle ownership model volume mount defintion, equivalent to
## docker run -v atlas_host_input_folder:atlas_container_input_folder
def get_atlas_docker_vols(settings):
    atlas_host_input_folder = os.path.abspath(settings['atlas_host_input_folder'])
    atlas_host_output_folder = os.path.abspath(settings['atlas_host_output_folder'])
    atlas_container_input_folder = os.path.abspath(settings['atlas_container_input_folder'])
    atlas_container_output_folder = os.path.abspath(settings['atlas_container_output_folder'])
    atlas_docker_vols = {
        atlas_host_input_folder: {  ## source location, aka "local"
            'bind': atlas_container_input_folder,  ## destination loc, aka "remote", "client"
            'mode': 'rw'},
        atlas_host_output_folder: {
            'bind': atlas_container_output_folder,
            'mode': 'rw'}}
    return atlas_docker_vols


## For Atlas container command
def get_atlas_cmd(settings, freq, output_year, npe, nsample, beamac):
    basedir = settings.get('basedir', '/')
    codedir = settings.get('codedir', '/')
    formattable_atlas_cmd = settings['atlas_formattable_command']
    atlas_cmd = formattable_atlas_cmd.format(freq, output_year, npe, nsample, basedir, codedir, beamac)
    return atlas_cmd


def warm_start_activities(settings, year, client):
    """
    Run activity demand models to update UrbanSim inputs with long-term
    choices it needs: workplace location, school location, and
    auto ownership.
    """
    activity_demand_model, activity_demand_image = get_model_and_image(settings, 'activity_demand_model')

    if activity_demand_model == 'polaris':
        run_polaris(None, settings, warm_start=True)

    elif activity_demand_model == 'activitysim':
        # 1. PARSE SETTINGS
        land_use_model = settings['land_use_model']
        travel_model = settings['travel_model']
        region = settings['region']
        asim_subdir = settings['region_to_asim_subdir'][region]
        asim_workdir = os.path.join('/activitysim', asim_subdir)
        asim_docker_vols = get_asim_docker_vols(settings)
        base_asim_cmd = get_base_asim_cmd(settings)

        print_str = "Initializing {0} warm start sequence".format(
            activity_demand_model)
        formatted_print(print_str)

        # 2. CREATE DATA FROM BASE YEAR SKIMS AND URBANSIM INPUTS

        # # skims ## now moved to warm start atlas stage
        # logger.info("Creating {0} skims from {1}".format(
        #     activity_demand_model,
        #     travel_model).upper())
        # asim_pre.create_skims_from_beam(settings, year)

        # data tables
        logger.info("Creating {0} input data from {1} outputs".format(
            activity_demand_model,
            land_use_model).upper())
        if not os.path.exists(os.path.join(settings['asim_local_input_folder'], 'skims.omx')):
            asim_pre.create_skims_from_beam(settings, year, overwrite=False)
        asim_pre.create_asim_data_from_h5(settings, year, warm_start=True)

        # 3. RUN ACTIVITYSIM IN WARM START MODE
        logger.info("Running {0} in warm start mode".format(
            activity_demand_model).upper())
        ws_asim_cmd = base_asim_cmd + ' -w'  # warm start flag

        run_container(client, settings, activity_demand_image, asim_docker_vols, ws_asim_cmd, working_dir=asim_workdir)

        # 4. UPDATE URBANSIM BASE YEAR INPUT DATA
        logger.info((
                        "Appending warm start activities/choices to "
                        " {0} base year input data").format(land_use_model).upper())
        asim_post.update_usim_inputs_after_warm_start(settings)
    logger.info('Done!')

    return


def forecast_land_use(settings, year, forecast_year, client, container_manager):
    run_land_use(settings, year, forecast_year, client)

    # check for outputs, exit if none
    usim_local_data_folder = settings['usim_local_data_folder']
    usim_output_store = settings['usim_formattable_output_file_name'].format(
        year=forecast_year)
    usim_datastore_fpath = os.path.join(usim_local_data_folder, usim_output_store)
    if not os.path.exists(usim_datastore_fpath):
        logger.critical(
            "No UrbanSim output data found. It probably did not finish successfully.")
        sys.exit(1)


def run_land_use(settings, year, forecast_year, client):
    logger.info("Running land use")

    # 1. PARSE SETTINGS
    land_use_model, land_use_image = get_model_and_image(settings, "land_use_model")
    usim_docker_vols = get_usim_docker_vols(settings)
    usim_cmd = get_usim_cmd(settings, year, forecast_year)

    # 2. PREPARE URBANSIM DATA
    print_str = (
        "Preparing {0} input data for land use development simulation.".format(
            year))
    formatted_print(print_str)
    usim_pre.add_skims_to_model_data(settings)

    if is_already_opened_in_write_mode(usim_data_path):
        logger.warning(
            "Closing h5 files {0} because they were left open. You should really "
            "figure out where this happened".format(tables.file._open_files.filenames))
        tables.file._open_files.close_all()

    # 3. RUN URBANSIM
    print_str = (
        "Simulating land use development from {0} "
        "to {1} with {2}.".format(
            year, forecast_year, land_use_model))
    formatted_print(print_str)
    run_container(client, settings, land_use_image, usim_docker_vols, usim_cmd,
                  working_dir=settings['usim_client_base_folder'])
    logger.info('Done!')

    return


## Atlas: evolve household vehicle ownership
def run_atlas(settings, output_year, client, warm_start_atlas, atlas_run_count=1):
    # warm_start: warm_start_atlas = True, output_year = year = start_year
    # asim_no_usim: warm_start_atlas = True, output_year = year (should  = start_year)
    # normal: warm_start_atlas = False, output_year = forecast_year

    # 1. PARSE SETTINGS
    vehicle_ownership_model, atlas_image = get_model_and_image(settings, "vehicle_ownership_model")
    freq = settings.get('vehicle_ownership_freq', False)
    npe = settings.get('atlas_num_processes', False)
    nsample = settings.get('atlas_sample_size', False)
    beamac = settings.get('atlas_beamac', 0)
    atlas_docker_vols = get_atlas_docker_vols(settings)
    atlas_cmd = get_atlas_cmd(settings, freq, output_year, npe, nsample, beamac)
    docker_stdout = settings.get('docker_stdout', False)
    activity_demand_model = settings['activity_demand_model']
    travel_model = settings['travel_model']

    # 2. PREPARE ATLAS DATA
    if warm_start_atlas:
        print_str = (
            "Preparing input data for warm start vehicle ownership simulation for {0}.".format(output_year))
    else:
        print_str = (
            "Preparing input data for vehicle ownership simulation for {0}.".format(output_year))
    formatted_print(print_str)

    # create skims.omx (lines moved from warm_start_activities)
    if warm_start_atlas == True & atlas_run_count == 1:
        logger.info("Creating {0} skims from {1}".format(
            activity_demand_model,
            travel_model).upper())
        asim_pre.create_skims_from_beam(settings, year)

    # prepare atlas inputs from urbansim h5 output
    # preprocessed csv input files saved in "atlas/atlas_inputs/year{}/"
    atlas_pre.prepare_atlas_inputs(settings, output_year, warm_start=warm_start_atlas)

    # calculate accessibility if atlas_beamac != 0
    if (beamac > 0):
        ## if No Driving
        path_list = ['WLK_COM_WLK', 'WLK_EXP_WLK', 'WLK_HVY_WLK', 'WLK_LOC_WLK', 'WLK_LRF_WLK']
        measure_list = ['WACC', 'IWAIT', 'XWAIT', 'TOTIVT', 'WEGR']
        ## if Allow Driving for access/egress
        # path_list = ['WLK_COM_WLK', 'WLK_EXP_WLK', 'WLK_HVY_WLK', 'WLK_LOC_WLK', 'WLK_LRF_WLK',
        #             'DRV_COM_DRV', 'DRV_EXP_DRV', 'DRV_HVY_DRV', 'DRV_LOC_DRV', 'DRV_LRF_DRV',
        #             'WLK_COM_DRV', 'WLK_EXP_DRV', 'WLK_HVY_DRV', 'WLK_LOC_DRV', 'WLK_LRF_DRV',
        #             'DRV_COM_WLK', 'DRV_EXP_WLK', 'DRV_HVY_WLK', 'DRV_LOC_WLK', 'DRV_LRF_WLK']
        # measure_list = ['WACC','IWAIT','XWAIT','TOTIVT','WEGR','DTIM']
        atlas_pre.compute_accessibility(path_list, measure_list, settings, output_year)

    # 3. RUN ATLAS via docker container client
    print_str = (
        "Simulating vehicle ownership for {0} "
        "with frequency {1}, npe {2} nsample {3} beamac {4}".format(
            output_year, freq, npe, nsample, beamac))
    formatted_print(print_str)
    run_container(client, settings, atlas_image, atlas_docker_vols, atlas_cmd, working_dir='/')

    # 4. ATLAS OUTPUT -> UPDATE USIM OUTPUT CARS & HH_CARS
    atlas_post.atlas_update_h5_vehicle(settings, output_year, warm_start=warm_start_atlas)

    # 5. ATLAS OUTPUT -> ADD A VEHICLETYPEID COL FOR BEAM
    atlas_post.atlas_add_vehileTypeId(settings, output_year)
    atlas_post.build_beam_vehicles_input(settings, output_year)

    logger.info('Atlas Done!')

    return


## Atlas: evolve household vehicle ownership
# run_atlas_auto is a run_atlas upgraded version, which will run_atlas again if
# outputs are not generated. This is mainly for preventing crash due to parellel
# computiing errors that can be resolved by a simple resubmission
def run_atlas_auto(settings, output_year, client, warm_start_atlas):
    # run atlas
    atlas_run_count = 1
    try:
        run_atlas(settings, output_year, client, warm_start_atlas, atlas_run_count)
    except:
        logger.error('ATLAS RUN #{} FAILED'.format(atlas_run_count))

    # rerun atlas if outputs not found and run count <= 3
    atlas_output_path = settings['atlas_host_output_folder']
    fname = 'vehicles_{}.csv'.format(output_year)
    while atlas_run_count < 3:
        atlas_run_count = atlas_run_count + 1
        if not os.path.exists(os.path.join(atlas_output_path, fname)):
            logger.error('LAST ATLAS RUN FAILED -> RE-LAUNCHING ATLAS RUN #{} BELOW'.format(atlas_run_count))
            try:
                run_atlas(settings, output_year, client, warm_start_atlas, atlas_run_count)
            except:
                logger.error('ATLAS RUN #{} FAILED'.format(atlas_run_count))

    return


def generate_activity_plans(
        settings, year, forecast_year, client,
        resume_after=None,
        warm_start=False,
        overwrite_skims=True,
        demand_model=None):
    """
    Parameters

    year : int
        Start year for the simulation iteration.
    forecast_year : int
        Simulation year for which activities are generated. If `forecast_year`
        is the start year of the whole simulation, then we are probably
        generating warm start activities based on the base year input data in
        order to generate "warm start" skims.
    """

    if settings.get('regenerate_seed', True):
        new_seed = random.randint(0, int(1e9))
        logger.info("Re-seeding asim with new seed {0}".format(new_seed))
        asim_pre.update_asim_config(settings, "random_seed", new_seed)

    activity_demand_model, activity_demand_image = get_model_and_image(settings, 'activity_demand_model')

    if activity_demand_model == 'polaris':
        run_polaris(forecast_year, settings, warm_start=True)

    elif activity_demand_model == 'activitysim':

        # 1. PARSE SETTINGS

        land_use_model = settings['land_use_model']
        region = settings['region']
        asim_subdir = settings['region_to_asim_subdir'][region]
        asim_workdir = os.path.join('/activitysim', asim_subdir)
        asim_docker_vols = get_asim_docker_vols(settings)
        asim_cmd = get_base_asim_cmd(settings)
        docker_stdout = settings.get('docker_stdout', False)

        # If this is the first iteration, skims should only exist because
        # they were created during the warm start activities step. The skims
        # haven't been updated since then so we don't need to re-create them.
        # if year == settings['start_year']:
        #     overwrite_skims = False
        overwrite_skims = False

        # 2. PREPROCESS DATA FOR ACTIVITY DEMAND MODEL
        print_str = "Creating {0} input data from {1} outputs".format(
            activity_demand_model,
            land_use_model)
        formatted_print(print_str)
        asim_pre.create_skims_from_beam(
            settings, year=forecast_year, overwrite=overwrite_skims)
        asim_pre.create_asim_data_from_h5(settings, year=forecast_year, warm_start=warm_start)

        # 3. GENERATE ACTIVITY PLANS
        print_str = (
            "Generating activity plans for the year "
            "{0} with {1}".format(
                forecast_year, activity_demand_model))
        if resume_after:
            asim_cmd += ' -r {0}'.format(resume_after)
            print_str += ". Picking up after {0}".format(resume_after)
        formatted_print(print_str)

        run_container(client, settings,
                      activity_demand_image,
                      working_dir=asim_workdir,
                      volumes=asim_docker_vols,
                      command=asim_cmd)

        # 4. COPY ACTIVITY DEMAND OUTPUTS --> LAND USE INPUTS
        # If generating activities for the base year (i.e. warm start),
        # then we don't want to overwrite urbansim input data. Otherwise
        # we want to set up urbansim for the next simulation iteration
        if (settings['land_use_enabled']) and (not warm_start):
            print_str = (
                "Generating {0} {1} input data from "
                "{2} outputs".format(
                    forecast_year, land_use_model, activity_demand_model))
            formatted_print(print_str)
            asim_post.create_next_iter_inputs(settings, year, forecast_year)

    logger.info('Done!')

    return


def run_traffic_assignment(
        settings, year, forecast_year, client, replanning_iteration_number=0):
    """
    This step will run the traffic simulation platform and
    generate new skims with updated congested travel times.
    """
    travel_model, travel_model_image = get_model_and_image(settings, 'travel_model')
    if travel_model == 'polaris':
        run_polaris(forecast_year, settings, warm_start=False)

    elif travel_model == 'beam':
        # 1. PARSE SETTINGS
        beam_config = settings['beam_config']
        region = settings['region']
        path_to_beam_config = '/app/input/{0}/{1}'.format(
            region, beam_config)
        beam_local_input_folder = settings['beam_local_input_folder']
        abs_beam_input = os.path.abspath(beam_local_input_folder)
        beam_local_output_folder = settings['beam_local_output_folder']
        abs_beam_output = os.path.abspath(beam_local_output_folder)
        activity_demand_model = settings.get('activity_demand_model', False)
        docker_stdout = settings['docker_stdout']
        skims_fname = settings['skims_fname']
        origin_skims_fname = settings['origin_skims_fname']
        beam_memory = settings.get('beam_memory', str(int(psutil.virtual_memory().total / (1024. ** 3)) - 2) + 'g')

        # remember the last produced skims in order to detect that
        # beam didn't work properly during this run
        if skims_fname.endswith(".csv.gz"):
            skimFormat = "csv.gz"
        elif skims_fname.endswith(".omx"):
            skimFormat = "omx"
        else:
            logger.error("Invalid skim format {0}".format(skims_fname))
        previous_od_skims = beam_post.find_produced_od_skims(beam_local_output_folder, skimFormat)
        previous_origin_skims = beam_post.find_produced_origin_skims(beam_local_output_folder)
        logger.info("Found skims from the previous beam run: %s", previous_od_skims)

        # 2. COPY ACTIVITY DEMAND OUTPUTS --> TRAFFIC ASSIGNMENT INPUTS
        if settings['traffic_assignment_enabled']:
            print_str = (
                "Generating {0} {1} input data from "
                "{2} outputs".format(
                    year, travel_model, activity_demand_model))
            formatted_print(print_str)
            beam_pre.copy_plans_from_asim(
                settings, year, replanning_iteration_number)

        # 3. RUN BEAM
        logger.info(
            "Starting beam container, input: %s, output: %s, config: %s",
            abs_beam_input, abs_beam_output, beam_config)
        run_container(
            client,
            settings,
            travel_model_image,
            volumes={
                abs_beam_input: {
                    'bind': '/app/input',
                    'mode': 'rw'},
                abs_beam_output: {
                    'bind': '/app/output',
                    'mode': 'rw'}},
            environment={
                'JAVA_OPTS': (
                    '-XX:+UnlockExperimentalVMOptions -Xmx{0}'.format(beam_memory))},
            working_dir='/app',
            command="--config={0}".format(path_to_beam_config)
        )

        # 4. POSTPROCESS
        path_to_mutable_od_skims = os.path.join(abs_beam_output, skims_fname)
        path_to_origin_skims = os.path.join(abs_beam_output, origin_skims_fname)

        if skimFormat == "csv.gz":
            current_od_skims = beam_post.merge_current_od_skims(
                path_to_mutable_od_skims, previous_od_skims, beam_local_output_folder)
            if current_od_skims == previous_od_skims:
                logger.error(
                    "BEAM hasn't produced the new skims at {0} for some reason. "
                    "Please check beamLog.out for errors in the directory {1}".format(current_od_skims, abs_beam_output)
                )
                sys.exit(1)

            beam_post.merge_current_origin_skims(
                path_to_origin_skims, previous_origin_skims, beam_local_output_folder)
        else:
            asim_data_dir = settings['asim_local_input_folder']
            asim_skims_path = os.path.join(asim_data_dir, 'skims.omx')
            current_od_skims = beam_post.merge_current_omx_od_skims(asim_skims_path, previous_od_skims,
                                                                    beam_local_output_folder, settings)
            if current_od_skims == previous_od_skims:
                logger.error(
                    "BEAM hasn't produced the new skims at {0} for some reason. "
                    "Please check beamLog.out for errors in the directory {1}".format(current_od_skims, abs_beam_output)
                )
                sys.exit(1)
            beam_asim_ridehail_measure_map = settings['beam_asim_ridehail_measure_map']
            beam_post.merge_current_omx_origin_skims(
                asim_skims_path, previous_origin_skims, beam_local_output_folder,
                beam_asim_ridehail_measure_map)
        beam_post.rename_beam_output_directory(settings, year, replanning_iteration_number)

    return


def initialize_docker_client(settings):
    land_use_model = settings.get('land_use_model', False)
    vehicle_ownership_model = settings.get('vehicle_ownership_model', False)  ## ATLAS
    activity_demand_model = settings.get('activity_demand_model', False)
    travel_model = settings.get('travel_model', False)
    models = [land_use_model, vehicle_ownership_model, activity_demand_model, travel_model]
    image_names = settings['docker_images']
    pull_latest = settings.get('pull_latest', False)

    client = docker.from_env()
    if pull_latest:
        logger.info("Pulling from docker...")
        for model in models:
            if model:
                image = image_names[model]
                if image is not None:
                    print('Pulling latest image for {0}'.format(image))
                    client.images.pull(image)

    return client


def initialize_asim_for_replanning(settings, forecast_year):
    replan_hh_samp_size = settings['replan_hh_samp_size']
    activity_demand_model, activity_demand_image = get_model_and_image(settings, 'activity_demand_model')
    region = settings['region']
    asim_subdir = settings['region_to_asim_subdir'][region]
    asim_workdir = os.path.join('/activitysim', asim_subdir)
    asim_docker_vols = get_asim_docker_vols(settings)
    base_asim_cmd = get_base_asim_cmd(settings, replan_hh_samp_size)
    docker_stdout = settings.get('docker_stdout', False)

    if replan_hh_samp_size > 0:
        print_str = (
            "Re-running ActivitySim on smaller sample size to "
            "prepare cache for re-planning with BEAM.")
        formatted_print(print_str)
        run_container(client, settings,
                      activity_demand_image, working_dir=asim_workdir,
                      volumes=asim_docker_vols,
                      command=base_asim_cmd)


def run_replanning_loop(settings, forecast_year):
    replan_iters = settings['replan_iters']
    replan_hh_samp_size = settings['replan_hh_samp_size']
    activity_demand_model, activity_demand_image = get_model_and_image(settings, 'activity_demand_model')
    region = settings['region']
    asim_subdir = settings['region_to_asim_subdir'][region]
    asim_workdir = os.path.join('/activitysim', asim_subdir)
    asim_docker_vols = get_asim_docker_vols(settings)
    base_asim_cmd = get_base_asim_cmd(settings, replan_hh_samp_size)
    docker_stdout = settings.get('docker_stdout', False)
    last_asim_step = settings['replan_after']

    for i in range(replan_iters):
        replanning_iteration_number = i + 1
        print_str = (
            'Replanning Iteration {0}'.format(replanning_iteration_number))
        formatted_print(print_str)

        if settings.get('regenerate_seed', True):
            new_seed = random.randint(0, int(1e9))
            logger.info("Re-seeding asim with new seed {0}".format(new_seed))
            asim_pre.update_asim_config(settings, "random_seed", new_seed)

        # a) format new skims for asim
        asim_pre.create_skims_from_beam(settings, forecast_year, overwrite=False)

        # b) replan with asim
        print_str = (
            "Replanning {0} households with ActivitySim".format(
                replan_hh_samp_size))
        formatted_print(print_str)
        run_container(
            client,
            settings,
            activity_demand_image, working_dir=asim_workdir,
            volumes=asim_docker_vols,
            command=base_asim_cmd + ' -r ' + last_asim_step)

        # e) run BEAM
        if replanning_iteration_number < replan_iters:
            beam_pre.update_beam_config(settings, 'beam_replanning_portion')
            beam_pre.update_beam_config(settings, 'max_plans_memory')
        else:
            beam_pre.update_beam_config(settings, 'beam_replanning_portion', 1.0)
        run_traffic_assignment(
            settings, year, forecast_year, client, replanning_iteration_number)

    return


def postprocess_all(settings):
    beam_output_dir = settings['beam_local_output_folder']
    region = settings['region']
    output_path = os.path.join(beam_output_dir, region, "year*")
    outputDirs = glob.glob(output_path)
    yearsAndIters = [(loc.split('-', 3)[-3], loc.split('-', 3)[-1]) for loc in outputDirs]
    yrs = dict()
    # Only do this for the latest available iteration in each year
    for year, iter in yearsAndIters:
        if year in yrs:
            if int(iter) > int(yrs[year]):
                yrs[year] = iter
        else:
            yrs[year] = iter
    for year, iter in yrs.items():
        process_event_file(settings, year, iter)


def to_singularity_volumes(volumes):
    bindings = [f"{local_folder}:{binding['bind']}:{binding['mode']}" for local_folder, binding in volumes.items()]
    result_str = ','.join(bindings)
    return result_str


def to_singularity_env(env):
    bindings = [f"{env_var}={value}" for env_var, value in env.items()]
    result_str = ','.join(bindings)
    return '"' + result_str + '"'


def run_container(client, settings: dict, image: str, volumes: dict, command: str,
                  working_dir=None, environment=None):
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
    """
    if client:
        docker_stdout = settings.get('docker_stdout', False)
        logger.info("Running docker container: %s, command: %s", image, command)
        run_kwargs = {
            'volumes': volumes,
            'command': command,
            'stdout': docker_stdout,
            'stderr': True,
            'detach': True
        }
        if working_dir:
            run_kwargs['working_dir'] = working_dir
        if environment:
            run_kwargs['environment'] = environment
        container = client.containers.run(image, **run_kwargs)
        for log in container.logs(
                stream=True, stderr=True, stdout=docker_stdout):
            print(log)
        container.remove()
        logger.info("Finished docker container: %s, command: %s", image, command)
    else:
        for local_folder in volumes:
            os.makedirs(local_folder, exist_ok=True)
        singularity_volumes = to_singularity_volumes(volumes)
        proc = ["singularity", "run", "--cleanenv", "--writable-tmpfs"] \
               + (["--env", to_singularity_env(environment)] if environment else []) \
               + (["--pwd", working_dir] if working_dir else []) \
               + ["-B", singularity_volumes, image] \
               + command.split()
        logger.info("Running command: %s", " ".join(proc))
        subprocess.run(proc)
        logger.info("Finished command: %s", " ".join(proc))


def get_model_and_image(settings: dict, model_type: str):
    manager = settings['container_manager']
    if manager == "docker":
        image_names = settings['docker_images']
    elif manager == "singularity":
        image_names = settings['singularity_images']
    else:
        raise ValueError("Container Manager not specified (container_manager param in settings.yaml)")
    model_name = settings.get(model_type)
    if not model_name:
        raise ValueError(f"No model {model_type} specified")
    image_name = image_names[model_name]
    if not model_name:
        raise ValueError(f"No {manager} image specified for model {model_name}")
    return model_name, image_name


if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    logger.info("Initializing data...")
    clean_and_init_data()

    logger.info("Preparing runtime environment...")

    #########################################
    #  PREPARE PILATES RUNTIME ENVIRONMENT  #
    #########################################

    # load args and settings
    settings = parse_args_and_settings()

    # parse scenario settings
    start_year = settings['start_year']
    end_year = settings['end_year']
    travel_model = settings.get('travel_model', False)
    formatted_print(
        'RUNNING PILATES FROM {0} TO {1}'.format(start_year, end_year))
    travel_model_freq = settings.get('travel_model_freq', 1)
    warm_start_skims = settings['warm_start_skims']
    warm_start_activities_enabled = settings['warm_start_activities']
    static_skims = settings['static_skims']
    land_use_enabled = settings['land_use_enabled']
    land_use_freq = settings['land_use_freq']
    vehicle_ownership_model_enabled = settings['vehicle_ownership_model_enabled']  # Atlas
    activity_demand_enabled = settings['activity_demand_enabled']
    traffic_assignment_enabled = settings['traffic_assignment_enabled']
    replanning_enabled = settings['replanning_enabled']
    container_manager = settings['container_manager']

    if not land_use_enabled:
        print("LAND USE MODEL DISABLED")
    if not activity_demand_enabled:
        print("ACTIVITY DEMAND MODEL DISABLED")
    if not traffic_assignment_enabled:
        print("TRAFFIC ASSIGNMENT MODEL DISABLED")

    if traffic_assignment_enabled:
        beam_pre.update_beam_config(settings, 'beam_sample')

    if warm_start_skims:
        formatted_print('"WARM START SKIMS" MODE ENABLED')
        logger.info('Generating activity plans for the base year only.')
    elif static_skims:
        formatted_print('"STATIC SKIMS" MODE ENABLED')
        logger.info('Using the same set of skims for every iteration.')

    if settings.get('travel_model') == 'beam':
        setup_beam_skims(settings)

    # start docker client
    if container_manager == 'docker':
        client = initialize_docker_client(settings)
    else:
        client = None

    #################################
    #  RUN THE SIMULATION WORKFLOW  #
    #################################
    state = WorkflowState.from_settings(settings)
    for year in state:
        # 1. FORECAST LAND USE
        if state.should_do(WorkflowState.Stage.land_use):
            # hack: make sure that the usim datastore isn't open
            usim_data_path = os.path.join(settings['usim_local_data_folder'],
                                          settings['usim_formattable_input_file_name'].format(
                                              region_id=settings['region_to_region_id'][settings['region']]))
            if is_already_opened_in_write_mode(usim_data_path):
                logger.warning(
                    "Closing h5 files {0} because they were left open. You should really "
                    "figure out where this happened".format(tables.file._open_files.filenames))
                tables.file._open_files.close_all()

            # 1a. IF START YEAR, WARM START MANDATORY ACTIVITIES
            if (state.is_start_year()) and warm_start_activities_enabled:
                # IF ATLAS ENABLED, UPDATE USIM INPUT H5
                if vehicle_ownership_model_enabled:
                    run_atlas_auto(settings, year, client, warm_start_atlas=True)
                warm_start_activities(settings, year, client)

            # 1b. RUN LAND USE SIMULATION
            forecast_land_use(settings, year, state.forecast_year, client, container_manager)
            state.complete(WorkflowState.Stage.land_use)

        # 2. RUN ATLAS (HOUSEHOLD VEHICLE OWNERSHIP)
        if state.should_do(WorkflowState.Stage.vehicle_ownership_model):
            # If the forecast year is the same as the base year of this
            # iteration, then land use forecasting has not been run. In this
            # case, atlas need to update urbansim *inputs* before activitysim
            # reads it in the next step.
            if state.forecast_year == year:
                run_atlas_auto(settings, year, client, warm_start_atlas=True)

            # If urbansim has been called, ATLAS will read, run, and update
            # vehicle ownership info in urbansim *outputs* h5 datastore.
            else:
                run_atlas_auto(settings, state.forecast_year, client, warm_start_atlas=False)
            state.complete(WorkflowState.Stage.vehicle_ownership_model)

        # 3. GENERATE ACTIVITIES
        if state.should_do(WorkflowState.Stage.activity_demand):
            # If the forecast year is the same as the base year of this
            # iteration, then land use forecasting has not been run. In this
            # case we have to read from the land use *inputs* because no
            # *outputs* have been generated yet. This is usually only the case
            # for generating "warm start" skims, so we treat it the same even
            # if the "warm_start_skims" setting was not set to True at runtime
            generate_activity_plans(
                settings, year, state.forecast_year, client, warm_start=warm_start_skims or not land_use_enabled)
            state.complete(WorkflowState.Stage.activity_demand)

            # 5. INITIALIZE ASIM LITE IF BEAM REPLANNING ENABLED
            # have to re-run asim all the way through on sample to shrink the
            # cache for use in re-planning, otherwise cache will use entire pop
        if state.should_do(WorkflowState.Stage.initialize_asim_for_replanning):
            initialize_asim_for_replanning(settings, state.forecast_year)
            state.complete(WorkflowState.Stage.initialize_asim_for_replanning)

        if state.should_do(WorkflowState.Stage.activity_demand_directly_from_land_use):
            # If not generating activities with a separate ABM (e.g.
            # ActivitySim), then we need to create the next iteration of land
            # use data directly from the last set of land use outputs.
            usim_post.create_next_iter_usim_data(settings, year, state.forecast_year)
            state.complete(WorkflowState.Stage.activity_demand_directly_from_land_use)

        # DO traffic assignment - but skip if using polaris as this is done along
        # with activity_demand generation
        if state.should_do(WorkflowState.Stage.traffic_assignment):

            # 4. RUN TRAFFIC ASSIGNMENT
            if settings['discard_plans_every_year']:
                beam_pre.update_beam_config(settings, 'max_plans_memory', 0)
            else:
                beam_pre.update_beam_config(settings, 'max_plans_memory')
            beam_pre.update_beam_config(settings, 'beam_replanning_portion', 1.0)
            if vehicle_ownership_model_enabled:
                beam_pre.copy_vehicles_from_atlas(settings, year)
            run_traffic_assignment(settings, year, state.forecast_year, client, -1)
            state.complete(WorkflowState.Stage.traffic_assignment)

        # 5. REPLAN
        if state.should_do(WorkflowState.Stage.traffic_assignment_replan):
            if replanning_enabled > 0:
                run_replanning_loop(settings, state.forecast_year)
                process_event_file(settings, year, settings['replan_iters'])
                copy_outputs_to_mep(settings, year, settings['replan_iters'])
            else:
                process_event_file(settings, year, -1)
                copy_outputs_to_mep(settings, year, -1)
            beam_post.trim_inaccessible_ods(settings)
            state.complete(WorkflowState.Stage.traffic_assignment_replan)

    logger.info("Finished")
