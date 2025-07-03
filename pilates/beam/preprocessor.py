import os
import logging
import shutil
from typing import Optional, List

import pandas as pd
import numpy as np

from pilates.utils.io import locate_asim_file, locate_beam_file
from pilates.utils.provenance import find_project_root
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, Record
from workflow_state import WorkflowState

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
    region = settings["region"]
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

    shutil.copytree(beam_production_path, dest, dirs_exist_ok=True)

    # Log the files that were copied for verification
    if os.path.exists(dest):
        copied_files = []
        for root, dirs, files in os.walk(dest):
            for file in files:
                copied_files.append(os.path.relpath(os.path.join(root, file), dest))
        logger.info(
            f"[BEAM Preprocessor] BEAM config copy complete. Files in {dest}: {copied_files}"
        )
    else:
        logger.warning(
            f"[BEAM Preprocessor] Destination directory {dest} does not exist after copy!"
        )

    # Optionally copy 'common' configs if present
    common_config_path = os.path.join(os.path.dirname(beam_production_path), "common")
    dest = os.path.join(os.path.abspath(output_dir), "common")
    if os.path.exists(common_config_path):
        shutil.copytree(common_config_path, dest, dirs_exist_ok=True)

    if "beam_skims_shapefile" in settings:
        logger.info(
            f"[BEAM Preprocessor] Updating beam config to use zone id of {settings['skim_zone_geoid_col']}"
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
    """
    Update a BEAM config file parameter with a new value.
    """
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
        settings["region"],
        settings["beam_scenario_folder"],
    )
    beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")
    atlas_output_data_dir = workspace.get_atlas_output_dir()
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
    provenance_tracker.record_input_file(
        "beam_preprocessor", atlas_vehicle_file_loc, model_run_id=model_run_hash
    )
    shutil.copy(atlas_vehicle_file_loc, beam_vehicles_path)
    provenance_tracker.record_output_file(
        "beam_preprocessor",
        beam_vehicles_path,
        model_run_id=model_run_hash,
        source_file_paths=[atlas_vehicle_file_loc],
    )


def find_activitysim_output_files(
    provenance_tracker: "FileProvenanceTracker",
    workspace: "Workspace",
    required_files: List[str]
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
        if hasattr(file_record, 'short_name') and file_record.short_name in required_files:
            if 'activitysim' in file_record.models or 'activitysim_postprocessor' in file_record.models:
                # Convert relative path to absolute path
                if file_record.file_path.startswith('/'):
                    abs_path = file_record.file_path
                else:
                    abs_path = os.path.join(workspace.output_path or workspace.full_path, file_record.file_path)
                
                found_files[file_record.short_name] = abs_path
                logger.info(f"Found ActivitySim output file {file_record.short_name}: {abs_path}")
    
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
                os.path.join(asim_output_dir, "final_pipeline", file_name, "final.parquet"),
                os.path.join(asim_output_dir, f"year-{workspace.state.current_year}-iteration-{workspace.state.current_inner_iter}", f"{missing_file}.parquet"),
                os.path.join(asim_output_dir, f"{missing_file}.parquet"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    found_files[missing_file] = path
                    logger.info(f"Found {missing_file} at fallback location: {path}")
                    break
    
    return found_files


def copy_plans_from_asim(
    settings,
    workspace: "Workspace",
    state: WorkflowState,
    provenance_tracker: "FileProvenanceTracker",
    replanning_iteration_number=0,
    model_run_hash: str = None,
) -> RecordStore:
    beam_scenario_folder = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings["region"],
        settings["beam_scenario_folder"],
    )

    def copy_with_compression_asim_file_to_beam(
        asim_file_path, beam_file_name, file_format
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

        # Always record the ActivitySim file as an input to the BEAM preprocessor run
        provenance_tracker.record_input_file(
            "beam_preprocessor",
            asim_file_path,
            description=f"ActivitySim output for BEAM: {beam_file_name}",
            model_run_id=model_run_hash,
        )

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
                df.loc[:, ~df.columns.duplicated()].to_parquet(beam_file_path)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return None

            # Record the copied file as an output of the preprocessor
            record = provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_file_path,
                description=f"Copied from ActivitySim output: {beam_file_name}",
                short_name=beam_file_name,
                source_file_paths=[asim_file_path],
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
        file_format = settings.get("file_format", "parquet")
        
        asim_plans_path = asim_file_paths.get("beam_plans")
        asim_households_path = asim_file_paths.get("households")
        asim_persons_path = asim_file_paths.get("persons")
        
        beam_plans_path = locate_beam_file(beam_scenario_folder, "plans", file_format)
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
            (asim_persons_path, "persons")
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
                persons_final.to_parquet(beam_persons_path, index=False)
                households_final.to_parquet(beam_households_path, index=False)
                plans_final.to_parquet(beam_plans_path, index=False)
            # Record provenance for all three files
            persons_record = provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_persons_path,
                description="Merged persons for BEAM input",
                model_run_id=model_run_hash,
                state=state,
                source_file_paths=[asim_persons_path] if asim_persons_path else [],
            )
            households_record = provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_households_path,
                description="Merged households for BEAM input",
                model_run_id=model_run_hash,
                state=state,
                source_file_paths=[asim_households_path] if asim_households_path else [],
            )
            plans_record = provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_plans_path,
                description="Merged plans for BEAM input",
                model_run_id=model_run_hash,
                state=state,
                source_file_paths=[asim_plans_path] if asim_plans_path else [],
            )
            record_list = [plans_record, households_record, persons_record]
        else:
            logger.info(
                "No plans existed already so copying them directly."
            )
            if asim_plans_path and os.path.exists(asim_plans_path):
                pd.read_parquet(asim_plans_path).to_parquet(beam_plans_path)
                # Record provenance for the plans file at least
                plans_record = provenance_tracker.record_output_file(
                    "beam_preprocessor",
                    beam_plans_path,
                    description="Copied plans for BEAM input (no merge)",
                    model_run_id=model_run_hash,
                    state=state,
                    source_file_paths=[asim_plans_path],
                )
                record_list = [plans_record]
            else:
                logger.error("No ActivitySim plans file found to copy")
                record_list = []
        return record_list

    # Main logic for copy_plans_from_asim
    if settings.get("copy_plans_from_asim_outputs", True):
        logger.info("You have chosen to use final ASIM plans. Will attempt to read files from provenance tracker.")
        file_format = settings.get("file_format", "parquet")
        
        # Find ActivitySim output files using provenance tracker
        required_files = ["beam_plans", "households", "persons"]
        asim_file_paths = find_activitysim_output_files(
            provenance_tracker, workspace, required_files
        )
        
        if not asim_file_paths:
            logger.error("No ActivitySim output files found in provenance tracker")
            return RecordStore()
        
        if replanning_iteration_number <= 0:
            record_list = []
            # Map ActivitySim output names to BEAM input names
            asim_to_beam_mapping = [
                ("beam_plans", "plans"),  # ActivitySim outputs beam_plans, BEAM needs plans
                ("households", "households"),
                ("persons", "persons"),
            ]
            
            for asim_name, beam_name in asim_to_beam_mapping:
                asim_file_path = asim_file_paths.get(asim_name)
                if asim_file_path:
                    record = copy_with_compression_asim_file_to_beam(
                        asim_file_path, beam_name, file_format
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
        file_format = settings.get("file_format", "parquet")
        record_list = []
        for _, beam_name in [("beam_plans", "plans"), ("households", "households"), ("persons", "persons")]:
            beam_file_path = locate_beam_file(beam_scenario_folder, beam_name, file_format)
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

    def preprocess(
        self,
        state: WorkflowState,
        workspace: "Workspace",
        provenance_tracker: "FileProvenanceTracker",
        model_run_hash: str,
    ) -> RecordStore:
        """
        Prepares all data needed to run BEAM.
        - Updates BEAM config with sample size, replanning fraction, etc.
        - Copies plans from ActivitySim outputs.
        - Copies vehicle fleet from Atlas outputs.

        Args:
            state (WorkflowState): The workflow state or context object.
            workspace (Workspace): The workspace containing input data.
            provenance_tracker (FileProvenanceTracker): Tracker for file provenance.
            model_run_hash (str): The unique hash for this preprocessor run.

        Returns:
            RecordStore: Preprocessed input data for the model.
        """
        settings = state.full_settings
        iteration_number = state.iteration

        # Update BEAM config
        if settings["discard_plans_every_year"]:
            update_beam_config(settings, workspace.full_path, "max_plans_memory", 0)
        else:
            update_beam_config(settings, workspace.full_path, "max_plans_memory")

        # Copy vehicle data from Atlas if enabled
        if settings.get("vehicle_ownership_model_enabled"):
            copy_vehicles_from_atlas(
                settings, workspace, state, provenance_tracker, model_run_hash
            )

        # Copy plans from ActivitySim
        store = copy_plans_from_asim(
            settings,
            workspace,
            state,
            provenance_tracker,
            iteration_number,
            model_run_hash,
        )

        logger.info("[BEAM Preprocessor] BEAM preprocessing complete.")
        return store
