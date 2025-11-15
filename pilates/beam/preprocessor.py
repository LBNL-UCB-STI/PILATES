import gzip
import logging
import os
import shutil
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, Record, FileRecord
from pilates.utils.io import locate_beam_file
from pilates.utils.provenance import find_project_root, FileProvenanceTracker
from workflow_state import WorkflowState
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)

beam_param_map = {
    "beam_sample": "beam.agentsim.agentSampleSizeAsFractionOfPopulation",
    "beam_replanning_portion": "beam.agentsim.agents.plans.merge.fraction",
    "max_plans_memory": "beam.replanning.maxAgentPlanMemorySize",
    "beam.agentsim.taz.filePath": "beam.agentsim.taz.filePath",
    "beam.agentsim.taz.tazIdFieldName": "beam.agentsim.taz.tazIdFieldName",
}

BEAM_PYDANTIC_PATH_MAP = {
    "beam_sample": "beam.sample",
    "beam_replanning_portion": "beam.replanning_portion",
    "max_plans_memory": "beam.max_plans_memory",
    "skim_zone_geoid_col": "beam.skim_zone_geoid_col",
}


def prepare_beam_zone_shapefile(
    settings, workspace, provenance_tracker, model_run_hash
):
    """
    Creates a sorted zone shapefile for BEAM from the canonical zone definitions
    and updates the BEAM config to use it.
    """
    logger.info("--- Preparing BEAM Zone Shapefile ---")
    try:
        # 1. Load the authoritative, sorted canonical zone geometries
        from pilates.utils.zone_utils import load_canonical_zones

        canonical_zones_gdf = load_canonical_zones(settings, workspace)

        # 2. Define the path for the new, sorted shapefile
        region = settings.run.region
        beam_mutable_folder = workspace.get_beam_mutable_data_dir()
        output_shapefile_name = "canonical_zones_sorted.geojson"
        output_shapefile_path = os.path.join(
            beam_mutable_folder, region, "shape", output_shapefile_name
        )
        os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)

        # 3. Save the sorted GeoDataFrame as the new shapefile for BEAM
        logger.info(
            f"Saving sorted canonical zones to '{output_shapefile_path}' for BEAM."
        )
        canonical_zones_gdf.to_file(output_shapefile_path, driver='GeoJSON')

        # Record the new shapefile as an output
        provenance_tracker.record_output_file(
            "beam_preprocessor",
            output_shapefile_path,
            short_name="beam_zone_shapefile_sorted",
            description="BEAM zone shapefile created from sorted canonical zones.",
            model_run_id=model_run_hash,
        )

        # 4. Update the BEAM configuration to use the new shapefile and correct ID column
        logger.info("Updating BEAM configuration to use the new sorted shapefile.")

        # Update path to the new shapefile
        # The path in the BEAM config is relative to the BEAM input folder
        relative_shapefile_path = os.path.join("shape", output_shapefile_name)
        update_beam_config(
            settings,
            workspace.full_path,
            "beam.agentsim.taz.filePath",
            valueOverride='${beam.inputDirectory}'+'/"{0}"'.format(relative_shapefile_path),  # BEAM config needs quotes for paths
        )

        # Update the TAZ ID column name
        canonical_id_col = settings.shared.geography.zones.canonical_id_col
        update_beam_config(
            settings,
            workspace.full_path,
            "beam.agentsim.taz.tazIdFieldName",
            valueOverride=canonical_id_col,
        )

        logger.info("--- BEAM Zone Shapefile Preparation Complete ---")

    except Exception as e:
        logger.error(
            f"An error occurred during BEAM shapefile preparation: {e}", exc_info=True
        )
        raise


def copy_data_to_mutable_location(
    settings, output_dir, provenance_tracker: FileProvenanceTracker
) -> Tuple[RecordStore, RecordStore]:
    """
    Copy BEAM input files for the current region from the production directory to the run's mutable input directory.
    """
    input_records = []
    output_records = []
    region = settings.run.region
    # Find the project root by searching upwards for 'pilates' or '.git'
    pilates_root = find_project_root()
    if pilates_root is None:
        pilates_root = os.path.realpath(os.getcwd())
    beam_production_path = os.path.abspath(
        os.path.join(
            pilates_root,
            "pilates",
            "beam",
            "production",
            region,
        )
    )

    # If not found, try with "sources/PILATES" in the path (for symlinked or alternate layouts)
    if not os.path.exists(beam_production_path):
        alt_root = os.path.join(pilates_root, "sources", "PILATES")
        alt_beam_production_path = os.path.abspath(
            os.path.join(
                alt_root,
                "pilates",
                "beam",
                "production",
                region,
            )
        )
        if os.path.exists(alt_beam_production_path):
            logger.info(
                f"Primary BEAM production path not found, using alternate: {alt_beam_production_path}"
            )
            beam_production_path = alt_beam_production_path
        else:
            logger.error(
                f"BEAM production input directory does not exist at either {beam_production_path} or {alt_beam_production_path}"
            )
            return
    dest = os.path.join(os.path.abspath(output_dir), region)
    logger.info(
        "Copying BEAM production inputs from {0} to {1}".format(
            beam_production_path, dest
        )
    )

    if not os.path.exists(beam_production_path):
        logger.error(
            f"BEAM production input directory does not exist: {beam_production_path}"
        )
        return

    shutil.copytree(
        beam_production_path,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".git*"),
    )

    git_hash = provenance_tracker.get_git_hash(beam_production_path)
    input_records.append(
        provenance_tracker.record_repo_input(
            "beam",
            repo_path=beam_production_path,
            short_name="beam_prod",
            description="Beam Production Data Repo",
            git_hash=git_hash,
        )
    )
    output_records.append(
        provenance_tracker.record_repo_input(
            "beam",
            repo_path=dest,
            short_name="beam_prod",
            description="Beam Production Data Repo",
        )
    )

    # Optionally copy 'common' configs if present
    common_config_path = os.path.join(os.path.dirname(beam_production_path), "common")
    dest = os.path.join(os.path.abspath(output_dir), "common")
    if os.path.exists(common_config_path):
        shutil.copytree(
            common_config_path,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".git*"),
        )
        input_records.append(
            provenance_tracker.record_repo_input(
                "beam",
                repo_path=common_config_path,
                short_name="beam_common",
                description="Beam Common Data Repo",
                git_hash=git_hash,
            )
        )
        output_records.append(
            provenance_tracker.record_repo_input(
                "beam",
                repo_path=dest,
                short_name="beam_common",
                description="Beam Common Data Repo",
            )
        )

    if hasattr(settings.beam, "skims_shapefile"):
        logger.info(
            f"[BEAM Preprocessor] Updating beam config to use zone id of {settings.beam.skim_zone_geoid_col}"
        )
        update_beam_config(
            settings,
            os.path.split(os.path.split(os.path.split(output_dir)[0])[0])[
                0
            ],  # Sorry...
            "skim_zone_geoid_col",
            valueOverride=settings.beam.skim_zone_geoid_col,
        )
    return RecordStore(recordList=input_records), RecordStore(recordList=output_records)


def update_beam_config(settings, working_dir, param, valueOverride=None):
    """
    Update a BEAM config file parameter with a new value.
    """
    if param in beam_param_map:
        config_header = beam_param_map[param]
        if valueOverride is None:
            pydantic_path = BEAM_PYDANTIC_PATH_MAP.get(param)
            if not pydantic_path:
                logger.warning(
                    f"Parameter '{param}' has no defined Pydantic path. Cannot update beam config."
                )
                return
            config_value = get_setting(settings, pydantic_path)
        else:
            config_value = valueOverride

        if config_value is None:
            logger.debug(
                f"Skipping beam config update for '{param}' because value is None."
            )
            return

        beam_config_path = os.path.join(
            working_dir,
            settings.beam.local_mutable_data_folder,
            settings.run.region,
            settings.beam.config,
        )
        if not os.path.exists(beam_config_path):
            logger.warning(
                f"[BEAM Preprocessor] BEAM config file does not exist: {beam_config_path}"
            )
            return
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
        logger.info(
            f"[BEAM Preprocessor] Updated config {config_header} to {config_value} in {beam_config_path}"
        )
    else:
        logger.warning(
            f"[BEAM Preprocessor] Tried to modify parameter {param} but couldn't find it in settings.yaml"
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


def copy_vehicles_from_atlas(
    settings,
    workspace: "Workspace",
    state: WorkflowState,
    provenance_tracker: "FileProvenanceTracker",
    model_run_hash: str,
):
    beam_scenario_folder = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        settings.beam.scenario_folder,
    )
    beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")
    if state.run_info_path and os.path.exists(state.run_info_path):
        logger.info(
            f"[BeamPreprocessor] Restarted run detected. Using previous run's output path from {state.run_info_path}"
        )
        previous_run_dir = os.path.dirname(state.run_info_path)
        atlas_output_data_dir = os.path.join(previous_run_dir, "atlas", "atlas_output")
    else:
        atlas_output_data_dir = workspace.get_atlas_output_dir()

    # FIX: Look for vehicles2 (with vehicleTypeId), not vehicles
    atlas_vehicle_file_loc = os.path.join(
        atlas_output_data_dir, "vehicles2_{0}.csv".format(state.forecast_year)
    )
    if not os.path.exists(atlas_vehicle_file_loc):
        atlas_vehicle_file_loc = os.path.join(
            atlas_output_data_dir, "vehicles2_{0}.csv".format(state.forecast_year - 1)
        )

    logger.info(
        "Copying atlas vehicles2 file from {0} to {1}".format(
            atlas_vehicle_file_loc, beam_vehicles_path
        )
    )

    input_record = provenance_tracker.record_input_file(
        "beam_preprocessor",
        atlas_vehicle_file_loc,
        short_name="atlas_vehicles2_output",  # Match the short_name from postprocessor
        model_run_id=model_run_hash,
    )

    # FIX: Read uncompressed CSV and write as gzipped
    df = pd.read_csv(atlas_vehicle_file_loc)
    df.to_csv(beam_vehicles_path, compression="gzip", index=False)

    provenance_tracker.record_output_file_with_inputs(
        "beam_preprocessor",
        beam_vehicles_path,
        input_records=[input_record],
        description="BEAM vehicles input copied from ATLAS vehicles2 output",
        short_name="beam_vehicles_in",
        model_run_id=model_run_hash,
    )


def find_activitysim_output_files(
    provenance_tracker: "FileProvenanceTracker",
    workspace: "Workspace",
    required_files: List[str],
) -> dict:
    """
    Find ActivitySim output files by looking through the provenance tracker's file records.
    This handles the case where files have been moved by the postprocessor.

    Args:
        provenance_tracker: The provenance tracker containing file records
        workspace: The workspace object
        required_files: List of required file short names (e.g., ['households', 'persons', 'beam_plans'])

    Returns:
        dict: Mapping of short_name to absolute file path
    """
    found_files = {}

    # Look through all file records to find ActivitySim outputs
    for file_hash, file_record in provenance_tracker.run_info.file_records.items():
        if (
            hasattr(file_record, "short_name")
            and file_record.short_name in required_files
        ):
            if (
                "activitysim" in file_record.models
                or "activitysim_postprocessor" in file_record.models
            ):
                # Convert relative path to absolute path
                if file_record.file_path.startswith("/"):
                    abs_path = file_record.file_path
                else:
                    abs_path = os.path.join(
                        workspace.full_path,
                        file_record.file_path,
                    )

                found_files[file_record.short_name] = abs_path
                logger.info(
                    f"Found ActivitySim output file {file_record.short_name}: {abs_path}"
                )

    # Log missing files
    missing_files = set(required_files) - set(found_files.keys())
    if missing_files:
        logger.warning(f"Could not find ActivitySim output files: {missing_files}")

        # Try to find them in the expected locations as fallback
        asim_output_dir = workspace.get_asim_output_dir()
        for missing_file in missing_files:
            # Map beam_plans back to plans for file system lookup
            file_name = "plans" if missing_file == "beam_plans" else missing_file

            # Try different possible locations
            possible_paths = [
                os.path.join(
                    asim_output_dir, "final_pipeline", file_name, "final.parquet"
                ),
                os.path.join(
                    asim_output_dir,
                    f"year-{workspace.state.current_year}-iteration-{workspace.state.current_inner_iter}",
                    f"{missing_file}.parquet",
                ),
                os.path.join(asim_output_dir, f"{missing_file}.parquet"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    found_files[missing_file] = path
                    logger.info(f"Found {missing_file} at fallback location: {path}")
                    break

    return found_files


def copy_plans_from_asim(
    input_records: RecordStore,
    settings,
    workspace: "Workspace",
    state: WorkflowState,
    provenance_tracker: "FileProvenanceTracker",
    replanning_iteration_number=0,
    model_run_hash: str = None,
) -> RecordStore:
    beam_scenario_folder = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        get_setting(settings, "run.region"),
        get_setting(settings, "beam.scenario_folder"),
    )

    def copy_with_compression_asim_file_to_beam(
        asim_file_path,
        beam_file_name,
        file_format,
        input_record: Optional[Record] = None,
    ) -> Optional[Record]:
        """
        Copy and compress a file from ActivitySim output to BEAM input, with provenance logging.
        """
        beam_file_path = locate_beam_file(
            beam_scenario_folder, beam_file_name, file_format
        )
        logger.info(
            "Copying asim file %s to beam input scenario file %s",
            asim_file_path,
            beam_file_path,
        )

        if isinstance(input_record, FileRecord):
            source_run_id = input_record.producing_run_id
        else:
            source_run_id = None

        # The input_record is already passed in, so we use it directly.
        provenance_tracker.record_input_record(input_record)

        if os.path.exists(asim_file_path):
            if file_format == "csv":
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
            elif file_format == "parquet":
                df = (
                    pd.read_parquet(asim_file_path)
                    .rename(columns={"VEHICL": "cars"})
                    .rename(columns={"auto_ownership": "cars"})
                    .rename(columns={"tripId": "trip_id"})
                )
                if "household_id" in df.columns:
                    df = df.astype({"household_id": pd.Int64Dtype()})
                df.loc[:, ~df.columns.duplicated()].to_parquet(
                    beam_file_path, index=True
                )
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return None

            # Record the copied file as an output of the preprocessor
            record = provenance_tracker.record_output_file_with_inputs(
                "beam_preprocessor",
                beam_file_path,
                input_records=[input_record],
                description=f"Copied from ActivitySim output: {beam_file_name}",
                short_name=beam_file_name + "_beam_in",
                model_run_id=model_run_hash,
            )
            return record
        else:
            logger.error(f"ActivitySim output file does not exist: {asim_file_path}")
            # Still record as missing output for provenance
            provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_file_path,
                description=f"Missing BEAM input file: {beam_file_name}",
                model_run_id=model_run_hash,
                skip_missing=False,
            )
            return None

    def merge_only_updated_households(asim_file_paths: dict) -> List[Record]:
        file_format = get_setting(settings, "activitysim.file_format", "parquet")

        asim_plans_path, asim_plans_record = asim_file_paths.get("beam_plans")
        asim_households_path, asim_households_record = asim_file_paths.get("households")
        asim_persons_path, asim_persons_record = asim_file_paths.get("persons")
        beam_plans_path, beam_plans_record = asim_file_paths.get("beam_plans_out")

        # beam_plans_path = locate_beam_file(beam_scenario_folder, "plans", file_format)
        beam_households_path = locate_beam_file(
            beam_scenario_folder, "households", file_format
        )
        beam_persons_path = locate_beam_file(
            beam_scenario_folder, "persons", file_format
        )

        # Record ActivitySim files as inputs to BEAM preprocessor
        for asim_path, name in [
            (asim_plans_path, "plans"),
            (asim_households_path, "households"),
            (asim_persons_path, "persons"),
        ]:
            if asim_path:
                provenance_tracker.record_input_file(
                    "beam_preprocessor",
                    asim_path,
                    description=f"ActivitySim output for BEAM merge: {name}",
                    model_run_id=model_run_hash,
                )

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
                original_households = (
                    pd.read_parquet(beam_households_path)
                    .rename(columns={"VEHICL": "cars"})
                    .rename(columns={"auto_ownership": "cars"})
                )
                updated_households = (
                    pd.read_parquet(asim_households_path)
                    .rename(columns={"VEHICL": "cars"})
                    .rename(columns={"auto_ownership": "cars"})
                )
                original_persons = pd.read_parquet(beam_persons_path)
                updated_persons = pd.read_parquet(asim_persons_path)
                if beam_plans_path.endswith(".parquet"):
                    original_plans = pd.read_parquet(beam_plans_path).rename(
                        columns={"tripId": "trip_id", "personId": "person_id"}
                    )
                else:
                    try:
                        original_plans = pd.read_csv(
                            beam_plans_path
                        ).rename(  # WHY IS THIS FAILING!!!!!
                            columns={"tripId": "trip_id", "personId": "person_id"}
                        )
                    except gzip.BadGzipFile:
                        logger.warning(
                            "Bad GZip file... Trying to read it as parquet instead?????"
                        )
                        try:
                            original_plans = pd.read_parquet(beam_plans_path).rename(
                                columns={"tripId": "trip_id", "personId": "person_id"}
                            )
                            logger.info("That worked!!!")
                        except Exception as e:
                            logger.warning(
                                f"That didn't work {e}. Just reading in asim plans"
                            )
                            original_plans = pd.read_parquet(asim_plans_path)
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
                    "age": pd.Int64Dtype(),
                    "sex": pd.Int64Dtype(),
                }
            )

            households_final = pd.concat(
                [
                    updated_households,
                    original_households.loc[~original_households.index.isin(hh_u), :],
                ]
            )
            households_final = households_final.astype({"cars": pd.Int64Dtype()})

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
                persons_final.index.unique().astype(float),
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
                persons_final.to_parquet(beam_persons_path, index=True)
                households_final.to_parquet(beam_households_path, index=True)
                plans_final.to_parquet(beam_plans_path, index=True)
            # Record provenance for all three files
            _, asim_persons_record = asim_file_paths.get("persons", (None, None))
            persons_record = provenance_tracker.record_output_file_with_inputs(
                "beam_preprocessor",
                beam_persons_path,
                input_records=[asim_persons_record],
                description="Merged persons for BEAM input",
                model_run_id=model_run_hash,
                state=state,
                short_name=f"persons_beam_in_{state.current_year}_{state.current_inner_iter}",
            )
            _, asim_households_record = asim_file_paths.get("households", (None, None))
            households_record = provenance_tracker.record_output_file_with_inputs(
                "beam_preprocessor",
                beam_households_path,
                input_records=[asim_households_record],
                description="Merged households for BEAM input",
                model_run_id=model_run_hash,
                context=state,
                short_name=f"households_beam_in_{state.current_year}_{state.current_inner_iter}",
            )
            _, asim_plans_record = asim_file_paths.get("beam_plans", (None, None))

            merged_plans_inputs = [asim_plans_record]
            if beam_plans_record:
                merged_plans_inputs.append(beam_plans_record)

            plans_record = provenance_tracker.record_output_file_with_inputs(
                "beam_preprocessor",
                beam_plans_path,
                input_records=merged_plans_inputs,
                description="Merged plans for BEAM input",
                model_run_id=model_run_hash,
                context=state,
                short_name=f"plans_beam_in_{state.current_year}_{state.current_inner_iter}",
            )
            record_list = [plans_record, households_record, persons_record]
        else:
            logger.info("No plans existed already so copying them directly.")
            if asim_plans_path and os.path.exists(asim_plans_path):
                pd.read_parquet(asim_plans_path).to_parquet(
                    beam_plans_path
                )  # Why OSError: Cannot save file into a non-existent directory: '/Users/zaneedell/git/PILATES/tmp/pilates-run-20250714-115638/beam/beam_output/sfbay/year-2011-iteration-0/ITERS/it.2'

                # Record provenance for the plans file at least
                _, asim_plans_record = asim_file_paths.get("beam_plans", (None, None))
                plans_record = provenance_tracker.record_output_file_with_inputs(
                    "beam_preprocessor",
                    beam_plans_path,
                    input_records=[asim_plans_record],
                    description="Copied plans for BEAM input (no merge)",
                    model_run_id=model_run_hash,
                    context=state,
                    short_name=f"plans_beam_in_{state.current_year}_{state.current_inner_iter}",
                )
                record_list = [plans_record]
            else:
                logger.error("No ActivitySim plans file found to copy")
                record_list = []
        return record_list

    # Main logic for copy_plans_from_asim
    if True:  # Replaces legacy `copy_plans_from_asim_outputs` setting
        logger.info(
            "You have chosen to use final ASIM plans. Will attempt to read files from provenance tracker."
        )
        file_format = settings.activitysim.file_format

        # Find ActivitySim output files using provenance tracker
        required_files = [
            "beam_plans",
            "households",
            "persons",
            "beam_plans_out",
        ]
        asim_output_records = (
            provenance_tracker.run_info.get_latest_model_run_output_records(
                "activitysim_postprocessor"
            )
        )

        if len(asim_output_records) == 0:
            logger.error("No ActivitySim output files found in provenance tracker")
            return RecordStore()

        if state.run_info_path and os.path.exists(state.run_info_path):
            logger.info(
                f"[BeamPreprocessor] Restarted run detected. Using previous run's output path from {state.run_info_path}"
            )
            base_path = os.path.dirname(state.run_info_path)
        else:
            base_path = workspace.full_path

        asim_file_paths = {}
        for record in input_records.all_records():
            if (record.short_name.rsplit("_", 2)[0] in required_files) or (
                record.short_name in required_files
            ):
                splt = record.short_name.rsplit("_", 2)
                if len(splt) > 1:
                    if str.isdigit(splt[1]):
                        shortened_name = splt[0]
                    else:
                        shortened_name = record.short_name
                else:
                    shortened_name = record.short_name
                asim_file_paths[shortened_name] = (
                    os.path.join(base_path, record.file_path),
                    record,
                )
                logger.info(
                    f"Found ActivitySim output file {record.short_name}: {record.file_path}"
                )
            else:
                logger.info(
                    f"Skipping non-required ActivitySim output file: {record.short_name}"
                )

        if replanning_iteration_number <= 0:
            record_list = []
            # Map ActivitySim output names to BEAM input names
            asim_to_beam_mapping = [
                (
                    "beam_plans",
                    "plans",
                ),  # ActivitySim outputs beam_plans, BEAM needs plans
                ("households", "households"),
                ("persons", "persons"),
            ]

            for asim_name, beam_name in asim_to_beam_mapping:
                asim_file_path, asim_file_record = asim_file_paths.get(
                    asim_name, (None, None)
                )
                if asim_file_path:
                    record = copy_with_compression_asim_file_to_beam(
                        asim_file_path, beam_name, file_format, asim_file_record
                    )
                    if record:
                        record_list.append(record)
                else:
                    logger.warning(f"ActivitySim output file not found: {asim_name}")
        else:
            record_list = merge_only_updated_households(asim_file_paths)

        record_store = RecordStore(recordList=[r for r in record_list if r is not None])
    else:
        logger.info("Using the plans that were already in the beam scenario folder")
        # Locate and create records for existing plans, households, persons
        file_format = get_setting(settings, "activitysim.file_format", "parquet")
        record_list = []
        for _, beam_name in [
            ("beam_plans", "plans"),
            ("households", "households"),
            ("persons", "persons"),
        ]:
            beam_file_path = locate_beam_file(
                beam_scenario_folder, beam_name, file_format
            )
            record = provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_file_path,
                description=f"Existing BEAM input file: {beam_name}",
                short_name=beam_name,
                model_run_id=model_run_hash,
            )
            if record:
                record_list.append(record)
        record_store = RecordStore(recordList=[r for r in record_list if r is not None])

    return record_store


class BeamPreprocessor(GenericPreprocessor):
    """
    Preprocessor for BEAM model.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)
        self.required_input_data: List[str] = [
            "persons",
            "households",
            "beam_plans",
            "linkstats",
            "beam_plans_out",
        ]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
    ) -> Tuple[RecordStore, RecordStore]:
        # Delegate to the module-level function
        from pilates.beam import preprocessor as beam_pre

        return beam_pre.copy_data_to_mutable_location(
            settings, output_dir, self.provenance_tracker
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Prepares all data needed to run BEAM.
        """
        settings = self.state.full_settings
        iteration_number = self.state.iteration
        previous_beam_records = []

        # Start by retrieving what Initialization stored
        input_records = workspace.output_data.get("beam", RecordStore())
        output_records = RecordStore()

        asim_post_records = previous_records.all_records()
        for record in asim_post_records:
            short_name = record.short_name.rsplit("_", 2)[0]
            if short_name in self.required_input_data:
                input_records.add_record(record)

        # If this is a replanning iteration, we need to get the outputs from the previous BEAM run
        if self.state.current_inner_iter > 0:
            previous_beam_records = (
                self.provenance_tracker.run_info.get_model_run_output_records(
                    "beam",
                    year=self.state.current_year,
                    iteration=self.state.current_inner_iter - 1,
                )
            )
            for record in previous_beam_records:
                short_name = record.short_name.rsplit("_", 2)[0]
                if (short_name in self.required_input_data) and (
                    "_sub" not in record.short_name
                ):
                    input_records.add_record(record)

        model_run_hash = self.provenance_tracker.start_model_run(
            "beam_preprocessor",
            year=self.state.current_year,
            iteration=self.state.current_inner_iter,
            description="Preprocessing for BEAM",
            inputs=input_records,
        )

        # Update BEAM config
        if settings.beam.discard_plans_every_year:
            update_beam_config(
                settings, workspace.full_path, "max_plans_memory", valueOverride=0
            )
        else:
            update_beam_config(settings, workspace.full_path, "max_plans_memory")

        # Prepare the shapefile to ensure consistent zone ordering
        prepare_beam_zone_shapefile(
            settings, workspace, self.provenance_tracker, model_run_hash
        )

        # Copy vehicle data from Atlas if enabled
        # Only copy on first iteration since vehicles are constant across iterations within a year
        if (
            settings.vehicle_ownership_model_enabled
            and self.state.current_inner_iter == 0
        ):
            copy_vehicles_from_atlas(
                settings, workspace, self.state, self.provenance_tracker, model_run_hash
            )

        # Copy plans from ActivitySim
        store = copy_plans_from_asim(
            input_records,
            settings,
            workspace,
            self.state,
            self.provenance_tracker,
            iteration_number,
            model_run_hash,
        )

        beam_prod_repo_record = next(
            (
                repo
                for repo in self.provenance_tracker.run_info.repo_records.values()
                if repo.short_name == "beam_prod"
            ),
            None,
        )

        # Add the BEAM scenario folder to the record store
        if beam_prod_repo_record:
            store.add_record(beam_prod_repo_record)
        store += output_records

        self.provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=store.all_records()
        )

        linkstats_record = next(
            (
                record
                for record in previous_beam_records
                if record.short_name.startswith("linkstats")
                and ("_sub" not in record.short_name)
            ),
            None,
        )
        if not linkstats_record:
            linkstats_path = os.path.join(
                workspace.get_beam_mutable_data_dir(),
                settings.run.region,
                settings.beam.router_directory,
                "init.linkstats.csv.gz",
            )
            if os.path.exists(linkstats_path):
                linkstats_record = self.provenance_tracker.record_output_file(
                    "beam_preprocessor",
                    linkstats_path,
                    self.state.year,
                    description="Initialized linkstats file",
                    short_name="linkstats",
                    state=self.state,
                )
            else:
                logger.warning(
                    "[BEAM Preprocessor] Could not find initlinkstats file at %s",
                    linkstats_path,
                )

        if linkstats_record:
            logger.info(
                "[BEAM Preprocessor] Linkstats file at %s added to BEAM input store",
                linkstats_record.file_path,
            )
            store.add_record(linkstats_record)

        logger.info("[BEAM Preprocessor] BEAM preprocessing complete.")
        return store
