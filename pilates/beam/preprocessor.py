import os
import logging
import gzip
import shutil
import pandas as pd
import numpy as np
import glob

logger = logging.getLogger(__name__)

beam_param_map = {
    "beam_sample": "beam.agentsim.agentSampleSizeAsFractionOfPopulation",
    "beam_replanning_portion": "beam.agentsim.agents.plans.merge.fraction",
    "max_plans_memory": "beam.replanning.maxAgentPlanMemorySize",
    "skim_zone_geoid_col": "beam.agentsim.taz.tazIdFieldName",
}


def copy_data_to_mutable_location(settings, output_dir):
    """
    Copy BEAM input files for the current region from the production directory to the run's mutable input directory.
    """
    # Source: pilates/beam/production/[region]/*
    # Dest:   [output_dir]/*
    region = settings["region"]
    # This is the *absolute* path to pilates/beam/production/[region]
    beam_production_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "pilates",
            "beam",
            "production",
            region,
        )
    )
    dest = os.path.abspath(output_dir)
    logger.info("Copying BEAM production inputs from {0} to {1}".format(beam_production_path, dest))

    if not os.path.exists(beam_production_path):
        logger.error(f"BEAM production input directory does not exist: {beam_production_path}")
        return

    shutil.copytree(beam_production_path, dest, dirs_exist_ok=True)

    # Optionally copy 'common' configs if present
    common_config_path = os.path.join(os.path.dirname(beam_production_path), "common")
    if os.path.exists(common_config_path):
        shutil.copytree(common_config_path, os.path.join(dest, "common"), dirs_exist_ok=True)

    if "beam_skims_shapefile" in settings:
        logger.info(
            "Updating beam config to use zone id of {0}".format(
                settings["skim_zone_geoid_col"]
            )
        )
        update_beam_config(
            settings,
            os.path.split(os.path.split(os.path.split(output_dir)[0])[0])[
                0
            ],  # Sorry...
            "skim_zone_geoid_col",
            settings["skim_zone_geoid_col"],
        )


def update_beam_config(settings, working_dir, param, valueOverride=None):
    if param in settings:
        config_header = beam_param_map[param]
        if valueOverride is None:
            config_value = settings[param]
        else:
            config_value = valueOverride
        beam_config_path = os.path.join(
            working_dir,
            settings["beam_local_mutable_data_folder"],
            settings["region"],
            settings["beam_config"],
        )
        modified = False
        with open(beam_config_path, "r") as file:
            data = file.readlines()
        with open(beam_config_path, "w") as file:
            for line in data:
                if config_header in line:
                    if not modified:
                        file.writelines(
                            config_header + " = " + str(config_value) + "\n"
                        )
                    modified = True
                else:
                    file.writelines(line)
            if not modified:
                file.writelines("\n" + config_header + " = " + str(config_value) + "\n")
    else:
        logger.warning(
            "Tried to modify parameter {0} but couldn't find it in settings.yaml".format(
                param
            )
        )


def make_archive(source, destination):
    """
    From https://stackoverflow.com/questions/32640053/compressing-directory-using-shutil-make-archive-while-preserving-directory-str
    """
    base = os.path.basename(destination)
    name = base.split(".")[0]
    fmt = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, fmt, archive_from, archive_to)
    shutil.move("%s.%s" % (name, fmt), destination)


def copy_vehicles_from_atlas(settings, state):
    beam_scenario_folder = os.path.join(
        state.full_path,
        settings["beam_local_mutable_data_folder"],
        settings["region"],
        settings["beam_scenario_folder"],
    )
    beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")
    atlas_output_data_dir = os.path.join(
        state.full_path, settings["atlas_host_output_folder"]
    )
    atlas_vehicle_file_loc = os.path.join(
        atlas_output_data_dir, "vehicles_{0}.csv.gz".format(state.forecast_year)
    )
    if not os.path.exists(atlas_vehicle_file_loc):
        atlas_vehicle_file_loc = os.path.join(
            atlas_output_data_dir, "vehicles_{0}.csv.gz".format(state.forecast_year - 1)
        )
    logger.info(
        "Copying atlas vehicles file from {0} to {1}".format(
            atlas_vehicle_file_loc, beam_vehicles_path
        )
    )
    shutil.copy(atlas_vehicle_file_loc, beam_vehicles_path)


def copy_plans_from_asim(
    settings, state, replanning_iteration_number=0
):
    asim_output_data_dir = os.path.join(
        state.full_path, settings["asim_local_output_folder"]
    )
    beam_scenario_folder = os.path.join(
        state.full_path,
        settings["beam_local_mutable_data_folder"],
        settings["region"],
        settings["beam_scenario_folder"],
    )

    def locate_asim_file(file_name, fmt):
        if fmt == "csv":
            return os.path.join(asim_output_data_dir, "final_" + file_name + ".csv")
        elif fmt == "parquet":
            if file_name == "plans":
                a_n = "beam_plans"
            else:
                a_n = file_name
            return os.path.join(
                asim_output_data_dir, "final_pipeline", a_n, "final.parquet"
            )
        elif fmt is None:
            return os.path.join(asim_output_data_dir, file_name)

    def locate_beam_file(file_name, fmt):
        if fmt == "csv":
            return os.path.join(beam_scenario_folder, file_name + ".csv.gz")
        elif fmt == "parquet":
            return os.path.join(beam_scenario_folder, file_name + ".parquet")

    def copy_with_compression_asim_file_to_beam(
        asim_file_name, beam_file_name, file_format
    ):
        """
        TODO: Switch this to polars for better performance
        def copy_with_compression_asim_file_to_beam(asim_file_name, beam_file_name):
            import polars as pl
            asim_file_path = os.path.join(asim_output_data_dir, asim_file_name)
            beam_file_path = os.path.join(beam_scenario_folder, beam_file_name)
            logger.info("Copying asim file %s to beam input scenario file %s", asim_file_path, beam_file_path)

            if os.path.exists(asim_file_path):
                df = pl.scan_csv(asim_file_path).with_columns([
                    pl.col("household_id").cast(pl.Int64),
                    pl.col("person_id").cast(pl.Int64),
                    pl.col("trip_id").cast(pl.Int64),
                    pl.col("cars").cast(pl.Int64),
                    pl.col("VEHICL").cast(pl.Int64).alias("cars"),
                    pl.col("auto_ownership").cast(pl.Int64).alias("cars"),
                    pl.col("age").cast(pl.Int64),
                    pl.col("sex").cast(pl.Int64)
                ]).select(pl.col("*").exclude_duplicates()).collect()
                df.write_csv(beam_file_path, compression="gzip")
        """
        from workflow_state import WorkflowState  # Import WorkflowState to access the provenance tracker

        if state.provenance_tracker.is_git_repo(asim_output_data_dir):
            repo_name = os.path.basename(asim_output_data_dir)
            git_hash = state.provenance_tracker.get_git_hash(asim_output_data_dir)
            state.provenance_tracker.record_repo_input(
                "beam",
                asim_output_data_dir,
                description=f"ActivitySim output repository",
                git_hash=git_hash,
            )
        else:
            if file_format == "csv":
                asim_file_path = locate_asim_file(asim_file_name, file_format)
                beam_file_path = locate_beam_file(beam_file_name, file_format)
                logger.info(
                    "Copying asim file %s to beam input scenario file %s",
                    asim_file_path,
                    beam_file_path,
                )

                if os.path.exists(asim_file_path):
                    df = (
                        pd.read_csv(
                            asim_file_path,
                            dtype={
                                "household_id": pd.Int64Dtype(),
                                "person_id": pd.Int64Dtype(),
                                "trip_id": pd.Int64Dtype(),
                                "cars": pd.Int64Dtype(),
                                "VEHICL": pd.Int64Dtype(),
                                "age": pd.Int64Dtype(),
                                "sex": pd.Int64Dtype(),
                            },
                        )
                        .rename(columns={"VEHICL": "cars"})
                        .rename(columns={"auto_ownership": "cars"})
                    )
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                    df.to_csv(beam_file_path, compression="gzip")

                    # Record the copied file as an input to BEAM
                    state.record_input_file(
                        "beam",
                        beam_file_path,
                        description=f"Copied from ActivitySim output: {asim_file_name}",
                        source_file_paths=[asim_file_path],
                        source_run_id=state.run_id
                    )
                    # with open(asim_file_path, 'rb') as f_in, gzip.open(
                    #         beam_file_path, 'wb') as f_out:
                    #     f_out.writelines(f_in)
            elif file_format == "parquet":
                asim_file_path = locate_asim_file(asim_file_name, file_format)
                beam_file_path = locate_beam_file(beam_file_name, file_format)
                logger.info(
                    "Copying asim file %s to beam input scenario file %s",
                    asim_file_path,
                    beam_file_path,
                )
                df = (
                    pd.read_parquet(asim_file_path)
                    .rename(columns={"VEHICL": "cars"})
                    .rename(columns={"auto_ownership": "cars"})
                    .rename(columns={"tripId": "trip_id"})
                )
                if "household_id" in df.columns:
                    df = df.astype({"household_id": pd.Int64Dtype()})
                df.loc[:, ~df.columns.duplicated()].to_parquet(beam_file_path)

                # Record the copied file as an input to BEAM
                state.record_input_file(
                    "beam",
                    beam_file_path,
                    description=f"Copied from ActivitySim output: {asim_file_name}",
                )
                if "household_id" in df.columns:
                    df = df.astype({"household_id": pd.Int64Dtype()})
                df.loc[:, ~df.columns.duplicated()].to_parquet(beam_file_path)

    def copy_with_compression_asim_file_to_asim_archive(
        file_path, file_name, year, replanning_iteration_number, fmt
    ):
        iteration_folder_name = "year-{0}-iteration-{1}".format(
            year, replanning_iteration_number
        )
        iteration_folder_path = os.path.join(
            asim_output_data_dir, iteration_folder_name
        )
        if not os.path.exists(os.path.abspath(iteration_folder_path)):
            os.makedirs(iteration_folder_path, exist_ok=True)
        input_file_path = locate_asim_file(file_name, fmt)
        target_file_path = os.path.join(iteration_folder_path, file_name)
        if target_file_path.endswith(".csv"):
            target_file_path += ".gz"
            if os.path.exists(file_path):
                with open(input_file_path, "rb") as f_in, gzip.open(
                    target_file_path, "wb"
                ) as f_out:
                    f_out.writelines(f_in)
                # Record the archived file path
                state.record_output_file(
                    "activitysim",
                    target_file_path,
                    year=year,
                    description=f"Archived ActivitySim output: {file_name}",
                )
        elif os.path.isdir(os.path.abspath(input_file_path)):
            make_archive(input_file_path, target_file_path + ".zip")
            # Record the archived file path
            state.record_output_file(
                "activitysim",
                target_file_path + ".zip",
                year=year,
                description=f"Archived ActivitySim output: {file_name}",
            )
        else:
            shutil.copy(input_file_path, target_file_path + ".parquet")
            # Record the archived file path
            state.record_output_file(
                "activitysim",
                target_file_path + ".parquet",
                year=year,
                description=f"Archived ActivitySim output: {file_name}",
            )

    def merge_only_updated_households():
        asim_plans_path = locate_asim_file("plans", file_format)
        asim_households_path = locate_asim_file("households", file_format)
        asim_persons_path = locate_asim_file("persons", file_format)
        beam_plans_path = locate_beam_file("plans", file_format)
        beam_households_path = locate_beam_file("households", file_format)
        beam_persons_path = locate_beam_file("persons", file_format)
        if os.path.exists(beam_plans_path):
            logger.info("Merging asim outputs with existing beam input scenario files")
            if file_format == "csv":
                original_households = pd.read_csv(
                    beam_households_path,
                    dtype={
                        "household_id": pd.Int64Dtype(),
                        "cars": pd.Int64Dtype(),
                        "auto_ownership": pd.Int64Dtype(),
                    },
                )
                updated_households = (
                    pd.read_csv(
                        asim_households_path,
                        dtype={
                            "household_id": pd.Int64Dtype(),
                            "VEHICL": pd.Int64Dtype(),
                            "auto_ownership": pd.Int64Dtype(),
                        },
                    )
                    .rename(columns={"VEHICL": "cars"})
                    .rename(columns={"auto_ownership": "cars"})
                )
                updated_households = updated_households.loc[
                    :, ~updated_households.columns.duplicated()
                ].copy()
                original_persons = pd.read_csv(
                    beam_persons_path,
                    dtype={
                        "household_id": pd.Int64Dtype(),
                        "person_id": pd.Int64Dtype(),
                        "age": pd.Int64Dtype(),
                        "sex": pd.Int64Dtype(),
                    },
                )
                updated_persons = pd.read_csv(
                    asim_persons_path,
                    dtype={
                        "household_id": pd.Int64Dtype(),
                        "person_id": pd.Int64Dtype(),
                        "age": pd.Int64Dtype(),
                        "sex": pd.Int64Dtype(),
                    },
                )
                original_plans = pd.read_csv(beam_plans_path).rename(
                    columns={"tripId": "trip_id"}
                )
                updated_plans = pd.read_csv(asim_plans_path)
            elif file_format == "parquet":
                original_households = pd.read_parquet(beam_households_path)
                updated_households = pd.read_parquet(asim_households_path)
                original_persons = pd.read_parquet(beam_persons_path)
                updated_persons = pd.read_parquet(asim_persons_path)
                original_plans = pd.read_parquet(beam_plans_path).rename(
                    columns={"tripId": "trip_id"}
                )
                updated_plans = pd.read_parquet(asim_plans_path)
            else:
                raise NotImplementedError
            if "person_id" in original_persons.columns:
                original_persons.set_index("person_id", inplace=True, drop=True)
                logger.warning("Setting index to person_id in original persons")
            if "person_id" in updated_persons.columns:
                updated_persons.set_index("person_id", inplace=True, drop=True)
                logger.warning("Setting index to person_id in updated persons")
            if "household_id" in original_households.columns:
                original_households.set_index("household_id", inplace=True, drop=True)
                logger.warning("Setting index to household_id in original households")
            if "household_id" in updated_households.columns:
                updated_households.set_index("household_id", inplace=True, drop=True)
                logger.warning("Setting index to household_id in updated households")
            per_o = original_persons.index.unique()
            per_u = updated_persons.index.unique()
            overlap = np.in1d(per_u.astype(float), per_o.astype(float)).sum()
            logger.info(
                "There were %s persons replanned out of %s originally, and %s of them existed before",
                len(per_u),
                len(per_o),
                overlap,
            )

            hh_o = original_households.index.unique()
            hh_u = updated_households.index.unique()
            overlap = np.in1d(hh_u.astype(float), hh_o.astype(float)).sum()
            logger.info(
                "There were %s households replanned out of %s originally, and %s of them existed before",
                len(hh_u),
                len(hh_o),
                overlap,
            )

            persons_final = pd.concat(
                [
                    updated_persons,
                    original_persons.loc[~original_persons.index.isin(per_u), :],
                ]
            )
            persons_final = persons_final.astype(
                {
                    "household_id": pd.Int64Dtype(),
                    "person_id": pd.Int64Dtype(),
                    "age": pd.Int64Dtype(),
                    "sex": pd.Int64Dtype(),
                }
            )

            households_final = pd.concat(
                [
                    updated_households,
                    original_households.loc[
                        ~original_households.index.isin(hh_u), :
                    ],
                ]
            )
            households_final = households_final.astype(
                {"household_id": pd.Int64Dtype(), "cars": pd.Int64Dtype()}
            )

            unchanged_plans = original_plans.loc[
                ~original_plans.person_id.isin(per_u), :
            ]
            logger.info(
                "Adding %s new plan elements after and keeping %s from previous iteration",
                len(updated_plans),
                len(unchanged_plans),
            )
            plans_final = pd.concat([updated_plans, unchanged_plans])
            persons_with_plans = np.in1d(
                persons_final.person_id.unique().astype(float),
                plans_final.person_id.unique().astype(float),
            ).sum()
            logger.info(
                "Of %s persons, %s of them have plans",
                len(persons_final),
                persons_with_plans,
            )
            if file_format == "csv":
                persons_final.to_csv(beam_persons_path, index=False, compression="gzip")
                households_final.to_csv(
                    beam_households_path, index=False, compression="gzip"
                )
                plans_final.to_csv(beam_plans_path, compression="gzip", index=False)
            else:
                persons_final.to_parquet(beam_persons_path, index=False)
                households_final.to_parquet(beam_households_path, index=False)
                plans_final.to_parquet(beam_plans_path, index=False)
        else:
            logger.info(
                "No plans existed already so copying them directly. THIS IS BAD"
            )
            pd.read_csv(asim_plans_path).to_csv(beam_plans_path, compression="gzip")

    if settings.get("copy_plans_from_asim_outputs", True):
        logging.info(
            "You have chosen to use final ASIM plans. Will attempt to read files from:"
        )
        logging.info(f"- Beam scenario folder: {beam_scenario_folder}")
        logging.info(f"- ASIM output data directory: {asim_output_data_dir}")
        file_format = settings.get("file_format", "parquet")
        if replanning_iteration_number <= 0:
            copy_with_compression_asim_file_to_beam("plans", "plans", file_format)
            copy_with_compression_asim_file_to_beam(
                "households", "households", file_format
            )
            copy_with_compression_asim_file_to_beam("persons", "persons", file_format)
        else:
            merge_only_updated_households()

        # Files from beam_scenario_folder
        if os.path.exists(beam_scenario_folder):
            file_format = settings.get("file_format", "parquet")
            try:
                copy_with_compression_asim_file_to_asim_archive(
                    beam_scenario_folder,
                    "plans",
                    state.year,
                    replanning_iteration_number,
                    file_format,
                )
                copy_with_compression_asim_file_to_asim_archive(
                    beam_scenario_folder,
                    "households",
                    state.year,
                    replanning_iteration_number,
                    file_format,
                )
                copy_with_compression_asim_file_to_asim_archive(
                    beam_scenario_folder,
                    "persons",
                    state.year,
                    replanning_iteration_number,
                    file_format,
                )
            except:
                logger.error("Error copying asim files to asim archive")
        else:
            logging.warning(
                f"Warning: Directory {beam_scenario_folder} does not exist. Cannot copy beam scenario files."
            )

        # Files from asim_output_data_dir
        if os.path.exists(asim_output_data_dir):
            copy_with_compression_asim_file_to_asim_archive(
                asim_output_data_dir,
                "land_use",
                state.year,
                replanning_iteration_number,
                file_format,
            )
            copy_with_compression_asim_file_to_asim_archive(
                asim_output_data_dir,
                "tours",
                state.year,
                replanning_iteration_number,
                file_format,
            )
            copy_with_compression_asim_file_to_asim_archive(
                asim_output_data_dir,
                "trips",
                state.year,
                replanning_iteration_number,
                file_format,
            )
            try:
                copy_with_compression_asim_file_to_asim_archive(
                    asim_output_data_dir,
                    "trip_mode_choice",
                    state.year,
                    replanning_iteration_number,
                    None,
                )
            except FileNotFoundError:
                logger.info("Note: trip_mode_choice utilities not found. Skipping.")

        else:
            logging.warning(
                f"Assuming that asim plans have already been generated and are stored in the beam input data"
            )

    return
