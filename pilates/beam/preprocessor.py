import os
import logging
import gzip
import shutil
import pandas as pd
import numpy as np
import glob

from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

beam_param_map = {'beam_sample': 'beam.agentsim.agentSampleSizeAsFractionOfPopulation',
                  'beam_replanning_portion': 'beam.agentsim.agents.plans.merge.fraction',
                  'max_plans_memory': 'beam.replanning.maxAgentPlanMemorySize'
                  }


def copy_data_to_mutable_location(settings, output_dir):
    beam_config_path = os.path.join(
        settings['beam_local_input_folder'],
        settings['region'])
    dest = os.path.join(output_dir, settings['region'])
    logger.info("Copying beam inputs from {0} to {1}".format(beam_config_path, dest))

    ## TODO: Update configs not to rely on beam.inputDirectory or update BEAM to have it be relative to code location and/or configurable

    # for file in os.listdir(beam_config_path):
    #     pathname = os.path.join(beam_config_path, file)
    #     if file.endswith(".conf"):
    #         shutil.copy(pathname, output_subdir)
    #     elif file.startswith("r5") & os.path.isdir(pathname):
    #         shutil.copytree(pathname, os.path.join(output_subdir, file))
    #     elif file == "urbansim":
    #         shutil.copytree(pathname, os.path.join(output_subdir, file))

    shutil.copytree(beam_config_path, dest)
    common_config_path = os.path.join(settings['beam_local_input_folder'], 'common')
    shutil.copytree(common_config_path, os.path.join(output_dir, 'common'))


def update_beam_config(settings, working_dir, param, valueOverride=None):
    if param in settings:
        config_header = beam_param_map[param]
        if valueOverride is None:
            config_value = settings[param]
        else:
            config_value = valueOverride
        beam_config_path = os.path.join(
            working_dir,
            settings['beam_local_mutable_data_folder'],
            settings['region'],
            settings['beam_config'])
        modified = False
        with open(beam_config_path, 'r') as file:
            data = file.readlines()
        with open(beam_config_path, 'w') as file:
            for line in data:
                if config_header in line:
                    if ~modified:
                        file.writelines(config_header + " = " + str(config_value) + "\n")
                    modified = True
                else:
                    file.writelines(line)
            if not modified:
                file.writelines("\n" + config_header + " = " + str(config_value) + "\n")
    else:
        logger.warning("Tried to modify parameter {0} but couldn't find it in settings.yaml".format(param))


def make_archive(source, destination):
    """
    From https://stackoverflow.com/questions/32640053/compressing-directory-using-shutil-make-archive-while-preserving-directory-str
    """
    base = os.path.basename(destination)
    name = base.split('.')[0]
    fmt = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, fmt, archive_from, archive_to)
    shutil.move('%s.%s' % (name, fmt), destination)


def copy_vehicles_from_atlas(settings, year):
    beam_scenario_folder = os.path.join(
        settings['beam_local_input_folder'],
        settings['region'],
        settings['beam_scenario_folder'])
    beam_vehicles_path = os.path.join(beam_scenario_folder, 'vehicles.csv.gz')
    atlas_output_data_dir = settings['atlas_host_output_folder']
    atlas_vehicle_file_loc = os.path.join(atlas_output_data_dir, "vehicles_{0}.csv.gz".format(year))
    if not os.path.exists(atlas_vehicle_file_loc):
        atlas_vehicle_file_loc = os.path.join(atlas_output_data_dir, "vehicles_{0}.csv.gz".format(year - 1))
    shutil.copy(atlas_vehicle_file_loc, beam_vehicles_path)


def copy_plans_from_asim(settings, state: WorkflowState, replanning_iteration_number=0):
    asim_output_data_dir = os.path.join(state.full_path, settings['asim_local_output_folder'])
    beam_scenario_folder = os.path.join(
        state.full_path,
        settings['beam_local_input_folder'],
        settings['region'],
        settings['beam_scenario_folder'])

    def copy_with_compression_asim_file_to_beam(asim_file_name, beam_file_name):
        asim_file_path = os.path.join(asim_output_data_dir, asim_file_name)
        beam_file_path = os.path.join(beam_scenario_folder, beam_file_name)
        logger.info("Copying asim file %s to beam input scenario file %s", asim_file_path, beam_file_path)

        if os.path.exists(asim_file_path):
            df = pd.read_csv(asim_file_path, dtype={"household_id": pd.Int64Dtype(),
                                                    "person_id": pd.Int64Dtype(),
                                                    "trip_id": pd.Int64Dtype(),
                                                    "cars": pd.Int64Dtype(),
                                                    "VEHICL": pd.Int64Dtype(),
                                                    "age": pd.Int64Dtype(),
                                                    "sex": pd.Int64Dtype()}
                             ).rename(columns={"VEHICL": "cars"}).rename(columns={"auto_ownership": "cars"})
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df.to_csv(beam_file_path, compression="gzip")
            # with open(asim_file_path, 'rb') as f_in, gzip.open(
            #         beam_file_path, 'wb') as f_out:
            #     f_out.writelines(f_in)

    def copy_with_compression_asim_file_to_asim_archive(file_path, file_name, year, replanning_iteration_number):
        iteration_folder_name = "year-{0}-iteration-{1}".format(year, replanning_iteration_number)
        iteration_folder_path = os.path.join(asim_output_data_dir, iteration_folder_name)
        if ~os.path.exists(os.path.abspath(iteration_folder_path)):
            os.makedirs(iteration_folder_path, exist_ok=True)
        input_file_path = os.path.join(file_path, file_name)
        target_file_path = os.path.join(iteration_folder_path, file_name)
        if target_file_path.endswith('.csv'):
            target_file_path += '.gz'
            if os.path.exists(file_path):
                with open(input_file_path, 'rb') as f_in, gzip.open(
                        target_file_path, 'wb') as f_out:
                    f_out.writelines(f_in)
        elif os.path.isdir(os.path.abspath(input_file_path)):
            make_archive(input_file_path, target_file_path + ".zip")
        else:
            shutil.copy(input_file_path, target_file_path)

    def merge_only_updated_households():
        asim_plans_path = os.path.join(asim_output_data_dir, 'final_plans.csv')
        asim_households_path = os.path.join(asim_output_data_dir, 'final_households.csv')
        asim_persons_path = os.path.join(asim_output_data_dir, 'final_persons.csv')
        beam_plans_path = os.path.join(beam_scenario_folder, 'plans.csv.gz')
        beam_households_path = os.path.join(beam_scenario_folder, 'households.csv.gz')
        beam_persons_path = os.path.join(beam_scenario_folder, 'persons.csv.gz')
        if os.path.exists(beam_plans_path):
            logger.info("Merging asim outputs with existing beam input scenario files")
            original_households = pd.read_csv(beam_households_path, dtype={"household_id": pd.Int64Dtype(),
                                                                           "cars": pd.Int64Dtype(),
                                                                           "auto_ownership": pd.Int64Dtype()}
                                              )
            updated_households = pd.read_csv(asim_households_path, dtype={"household_id": pd.Int64Dtype(),
                                                                          "VEHICL": pd.Int64Dtype(),
                                                                          "auto_ownership": pd.Int64Dtype()}
                                             ).rename(columns={"VEHICL": "cars"}).rename(
                columns={"auto_ownership": "cars"})
            updated_households = updated_households.loc[:, ~updated_households.columns.duplicated()].copy()
            original_persons = pd.read_csv(beam_persons_path, dtype={"household_id": pd.Int64Dtype(),
                                                                     "person_id": pd.Int64Dtype(),
                                                                     "age": pd.Int64Dtype(),
                                                                     "sex": pd.Int64Dtype()}
                                           )
            updated_persons = pd.read_csv(asim_persons_path, dtype={"household_id": pd.Int64Dtype(),
                                                                    "person_id": pd.Int64Dtype(),
                                                                    "age": pd.Int64Dtype(),
                                                                    "sex": pd.Int64Dtype()}
                                          )
            per_o = original_persons.person_id.unique()
            per_u = updated_persons.person_id.unique()
            overlap = np.in1d(per_u.astype(float), per_o.astype(float)).sum()
            logger.info("There were %s persons replanned out of %s originally, and %s of them existed before",
                        len(per_u), len(per_o), overlap)

            hh_o = (original_persons.household_id.unique())
            hh_u = (updated_persons.household_id.unique())
            overlap = np.in1d(hh_u.astype(float), hh_o.astype(float)).sum()
            logger.info("There were %s households replanned out of %s originally, and %s of them existed before",
                        len(hh_u), len(hh_o), overlap)

            persons_final = pd.concat([updated_persons, original_persons.loc[
                                                        ~original_persons.person_id.isin(per_u), :]])
            persons_final = persons_final.astype({"household_id": pd.Int64Dtype(),
                                                  "person_id": pd.Int64Dtype(),
                                                  "age": pd.Int64Dtype(),
                                                  "sex": pd.Int64Dtype()})
            persons_final.to_csv(beam_persons_path, index=False, compression='gzip')
            households_final = pd.concat([updated_households, original_households.loc[
                                                              ~original_households.household_id.isin(hh_u), :]])
            households_final = households_final.astype({"household_id": pd.Int64Dtype(),
                                                        "cars": pd.Int64Dtype()})
            households_final.to_csv(beam_households_path, index=False, compression='gzip')

            original_plans = pd.read_csv(beam_plans_path).rename(columns={'tripId': 'trip_id'})
            updated_plans = pd.read_csv(asim_plans_path)
            unchanged_plans = original_plans.loc[~original_plans.person_id.isin(per_u), :]
            logger.info("Adding %s new plan elements after and keeping %s from previous iteration",
                        len(updated_plans), len(unchanged_plans))
            plans_final = pd.concat([updated_plans, unchanged_plans])
            persons_with_plans = np.in1d(persons_final.person_id.unique().astype(float),
                                         plans_final.person_id.unique().astype(float)).sum()
            logger.info("Of %s persons, %s of them have plans", len(persons_final), persons_with_plans)
            plans_final.to_csv(beam_plans_path, compression='gzip', index=False)
        else:
            logger.info("No plans existed already so copying them directly. THIS IS BAD")
            pd.read_csv(asim_plans_path).to_csv(beam_plans_path, compression='gzip')

    if replanning_iteration_number < 0:
        copy_with_compression_asim_file_to_beam('final_plans.csv', 'plans.csv.gz')
        copy_with_compression_asim_file_to_beam('final_households.csv', 'households.csv.gz')
        copy_with_compression_asim_file_to_beam('final_persons.csv', 'persons.csv.gz')
        # copy_with_compression_asim_file_to_beam('final_land_use.csv', 'land_use.csv.gz')
        # copy_with_compression_asim_file_to_beam('final_tours.csv', 'tours.csv.gz')
        # copy_with_compression_asim_file_to_beam('final_trips.csv', 'trips.csv.gz')
        # copy_with_compression_asim_file_to_beam('final_joint_tour_participants.csv',
        #                                         'joint_tour_participants.csv.gz')
    else:
        merge_only_updated_households()

    if settings.get('final_asim_plans_folder', False):
        # This first one not currently necessary when asim-lite is replanning all households
        # copy_with_compression_asim_file_to_asim_archive(asim_output_data_dir, 'final_plans.csv', year,
        #                                                 replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(beam_scenario_folder, 'plans.csv.gz', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(beam_scenario_folder, 'households.csv.gz', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(beam_scenario_folder, 'persons.csv.gz', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(asim_output_data_dir, 'final_land_use.csv', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(asim_output_data_dir, 'final_tours.csv', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(asim_output_data_dir, 'final_trips.csv', state.forecast_year,
                                                        replanning_iteration_number)
        copy_with_compression_asim_file_to_asim_archive(asim_output_data_dir, 'trip_mode_choice', state.forecast_year,
                                                        replanning_iteration_number)

    return
