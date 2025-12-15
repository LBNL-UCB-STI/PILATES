from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, TYPE_CHECKING
import re

from pilates.utils.consist_adapter import ConsistProvenanceTracker

if TYPE_CHECKING:
    from pilates.workspace import Workspace


import pandas as pd

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, Record, FileRecord
from pilates.generic.model import provenance_logging
from pilates.utils.io import locate_beam_file
from pilates.utils.provenance import find_project_root, FileProvenanceTracker
from pilates.utils.settings_helper import get as get_setting
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

# Mappings for BEAM configuration parameters
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


def _prepare_beam_zone_shapefile(
    workspace: "Workspace", settings: PilatesConfig
) -> Optional[str]:
    """
    Creates a sorted zone shapefile for BEAM from the canonical zone definitions
    and updates the BEAM config to use it.
    """
    logger.info("--- Preparing BEAM Zone Shapefile ---")
    try:
        from pilates.utils.zone_utils import load_canonical_zones

        canonical_zones_gdf = load_canonical_zones(settings, workspace)

        region = settings.run.region
        beam_mutable_folder = workspace.get_beam_mutable_data_dir()
        output_shapefile_name = "canonical_zones_sorted.geojson"
        output_shapefile_path = os.path.join(
            beam_mutable_folder, region, "shape", output_shapefile_name
        )
        os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)

        logger.info(
            f"Saving sorted canonical zones to '{output_shapefile_path}' for BEAM."
        )

        # Reset index so the ID becomes a column
        canonical_zones_gdf = canonical_zones_gdf.reset_index()

        if settings.beam is None:
            raise ValueError("Beam settings are not configured")
        sort_col = settings.beam.skim_zone_geoid_col
        if sort_col in canonical_zones_gdf.columns:
            # If the specific beam config column exists, verify sort order
            canonical_zones_gdf = canonical_zones_gdf.sort_values(by=sort_col)
        else:
            # If column is missing, it was likely renamed during load_canonical_zones.
            # Since load_canonical_zones guarantees a sorted index, we accept that order.
            logger.info(
                f"Beam sort column '{sort_col}' not found (likely renamed to '{canonical_zones_gdf.columns[0]}'). "
                "Using existing sort order from load_canonical_zones."
            )

        canonical_zones_gdf.to_file(output_shapefile_path, driver="GeoJSON")

        logger.info("--- BEAM Zone Shapefile Preparation Complete ---")
        return output_shapefile_path
    except Exception as e:
        logger.error(
            f"An error occurred during BEAM shapefile preparation: {e}", exc_info=True
        )
        raise


class BeamDataHelper:
    """
    Centralizes logic for reading, cleaning, and standardizing BEAM input data.
    """

    # Column Renaming Maps
    RENAMES = {
        "plans": {
            "tripId": "trip_id",
            "tripid": "trip_id",
            "personId": "person_id",
            "personid": "person_id",
            "vehicleId": "vehicle_id",
            "vehicleid": "vehicle_id",
            "legVehicleIds": "leg_vehicle_ids",
            "legvehicleids": "leg_vehicle_ids",
        },
        "households": {"VEHICL": "cars", "auto_ownership": "cars"},
        "persons": {},  # No common renames, but kept for consistency
    }

    # Dtypes for CSV reading to prevent Int64 -> Float conversion
    DTYPES = {
        "plans": {
            "trip_id": pd.Int64Dtype(),
            "person_id": pd.Int64Dtype(),
            "planindex": pd.Int64Dtype(),
        },
        "households": {
            "household_id": pd.Int64Dtype(),
            "cars": pd.Int64Dtype(),
            "auto_ownership": pd.Int64Dtype(),
            "VEHICL": pd.Int64Dtype(),
        },
        "persons": {
            "household_id": pd.Int64Dtype(),
            "person_id": pd.Int64Dtype(),
            "age": pd.Int64Dtype(),
            "sex": pd.Int64Dtype(),
        },
    }

    # Default columns to ensure exist in Plans
    PLAN_DEFAULTS = {
        "planindex": 0,
        "planselected": False,  # Default to False; logic can override to True
        "planscore": 0.0,
        "legtraveltime": 0.0,
        "legroutetype": "",
        "legroutestartlink": 0,
        "legrouteendlink": 0,
        "legroutetraveltime": 0.0,
        "legroutedistance": 0.0,
        "legroutelinks": "",
    }

    @classmethod
    def read_and_clean(
        cls, file_path: str, table_type: str, file_format: str
    ) -> pd.DataFrame:
        """
        Reads a file (CSV/Parquet), applies dtypes, renames columns, and ensures defaults.

        Args:
            file_path: Path to the file.
            table_type: One of 'plans', 'households', 'persons'.
            file_format: 'csv' or 'parquet'.
        """
        if not os.path.exists(file_path):
            # Let the caller handle missing files, or raise specific error
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Read File
        if file_format == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format == "csv":
            # Get specific dtypes for this table type
            dtype_map = cls.DTYPES.get(table_type, {})
            df = pd.read_csv(file_path, dtype=dtype_map)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # 2. Rename Columns
        rename_map = cls.RENAMES.get(table_type, {})
        df = df.rename(columns=rename_map)

        # 3. Deduplicate Columns (Fixes issue seen in households with multiple 'cars' inputs)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # 4. Apply Defaults (specifically for Plans)
        if table_type == "plans":
            for col, default_val in cls.PLAN_DEFAULTS.items():
                if col not in df.columns:
                    df[col] = default_val

        # 5. Set standard Index if applicable
        index_map = {"households": "household_id", "persons": "person_id"}
        index_col = index_map.get(table_type)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True, drop=True)

        return df


class BeamPreprocessor(GenericPreprocessor):
    """
    Preprocessor for the BEAM model. Handles data preparation, configuration updates,
    and copying of inputs from other model components like ActivitySim and ATLAS.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: ConsistProvenanceTracker,
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
        self.settings = self.state.full_settings

    @provenance_logging
    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Prepares all data needed to run BEAM for the current iteration.
        """
        input_records = workspace.output_data.get("beam", RecordStore())
        output_records = RecordStore()

        # Collect necessary records from the previous (ActivitySim) step
        asim_post_records = previous_records.all_records()
        for record in asim_post_records:
            if record.short_name:
                short_name = record.short_name.rsplit("_", 2)[0].replace(
                    "_asim_out", ""
                )  # Updated to handle format e.g. beam_plans_asim_out_2018_0
                if short_name in self.required_input_data:
                    input_records.add_record(record)
                    logger.info(f"Added {short_name} to beam inputs")
                else:
                    logger.info(f"Skipping {record.short_name} produced by activitysim")

        # For replanning iterations, get outputs from the previous BEAM run
        previous_beam_records = []
        if self.state.current_inner_iter > 0:
            previous_beam_records = (
                self.provenance_tracker.run_info.get_latest_model_run_output_records(
                    "beam"
                )
            )
            for record in previous_beam_records:
                short_name = record.short_name.rsplit("_", 2)[0]
                if (short_name in self.required_input_data) and (
                    "_sub" not in record.short_name
                ):
                    input_records.add_record(record)

        model_run_hash = self.provenance_tracker.current_model_run_id

        # Update BEAM config based on settings
        self._update_beam_config(
            "max_plans_memory",
            value_override=0 if self.settings.beam.discard_plans_every_year else None,
            base_path=workspace.full_path,
        )
        # Prepare the zone shapefile to ensure consistent zone ordering
        self.prepare_beam_zone_shapefile(workspace, model_run_hash)

        # Copy vehicle data from Atlas (only on first iteration)
        if (
            self.settings.vehicle_ownership_model_enabled
            and self.state.current_inner_iter == 0
        ):
            self._copy_vehicles_from_atlas(workspace, model_run_hash)

        # Copy and merge plans from ActivitySim
        store = self._copy_plans_from_asim(input_records, workspace, model_run_hash)

        # Add BEAM production data repo to records
        beam_prod_repo_record = next(
            (
                repo
                for repo in self.provenance_tracker.run_info.repo_records.values()
                if repo.short_name == "beam_prod"
            ),
            None,
        )
        if beam_prod_repo_record:
            store.add_record(beam_prod_repo_record)
        store += output_records

        # Ensure linkstats file is present and recorded
        self._handle_linkstats(workspace, previous_beam_records, store)

        logger.info("[BEAM Preprocessor] BEAM preprocessing complete.")
        return store

    def copy_data_to_mutable_location(
        self, settings: PilatesConfig, output_dir: str
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy BEAM input files for the current region from the production directory
        to the run's mutable input directory.
        """
        input_records, output_records = [], []
        region = settings.run.region
        beam_production_path = self._find_beam_production_path(settings, region)

        if not beam_production_path:
            logger.error(
                f"Could not find BEAM production directory for region {region}."
            )
            return RecordStore(), RecordStore()

        dest_region = os.path.join(os.path.abspath(output_dir), region)
        logger.info(
            f"Copying BEAM production inputs from {beam_production_path} to {dest_region}"
        )

        shutil.copytree(
            beam_production_path,
            dest_region,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".git*"),
        )
        git_hash = self.provenance_tracker.get_git_hash(beam_production_path)
        input_records.append(
            self.provenance_tracker.record_repo_input(
                "beam",
                repo_path=beam_production_path,
                short_name="beam_prod",
                description="Beam Production Data Repo",
                git_hash=git_hash,
            )
        )

        output_records.append(
            self.provenance_tracker.record_repo_output(
                "beam",
                repo_path=dest_region,
                short_name="beam_prod",
                description="Beam Production Data Repo",
            )
        )

        # Optionally copy 'common' configs if present
        common_config_path = os.path.join(
            os.path.dirname(beam_production_path), "common"
        )
        if os.path.exists(common_config_path):
            dest_common = os.path.join(os.path.abspath(output_dir), "common")
            shutil.copytree(
                common_config_path,
                dest_common,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", ".git*"),
            )
            input_records.append(
                self.provenance_tracker.record_repo_input(
                    "beam",
                    repo_path=common_config_path,
                    short_name="beam_common",
                    description="Beam Common Data Repo",
                    git_hash=git_hash,
                )
            )
            output_records.append(
                self.provenance_tracker.record_repo_output(
                    "beam",
                    repo_path=dest_common,
                    short_name="beam_common",
                    description="Beam Common Data Repo",
                )
            )

        if hasattr(settings.beam, "skims_shapefile"):
            logger.info(
                f"[BEAM Preprocessor] Updating beam config to use zone id of {settings.beam.skim_zone_geoid_col}"
            )

            # FIX: Calculate base path from output_dir if state.workspace is not guaranteed
            # Original code used triple split logic; here we approximate assuming output_dir is in the workspace
            # or fallback to self.state.workspace if available.
            # A safe fallback for initialization time:
            base_path = None
            if hasattr(self.state, "workspace"):
                base_path = self.state.workspace.full_path
            else:
                # Replicate logic: output_dir is usually "{root}/beam/data"
                # os.path.dirname(os.path.dirname(output_dir)) approx "{root}"
                base_path = os.path.dirname(
                    os.path.dirname(os.path.abspath(output_dir))
                )

            self._update_beam_config(
                "skim_zone_geoid_col",
                value_override=settings.beam.skim_zone_geoid_col,
                base_path=base_path,
            )

        return RecordStore(recordList=input_records), RecordStore(
            recordList=output_records
        )

    def prepare_beam_zone_shapefile(
        self, workspace: "Workspace", model_run_hash: str
    ) -> Optional[str]:
        """
        Creates a sorted zone shapefile for BEAM from the canonical zone definitions
        and updates the BEAM config to use it.
        """

        output_shapefile_path = _prepare_beam_zone_shapefile(workspace, self.settings)
        output_shapefile_name = os.path.basename(output_shapefile_path)

        self.provenance_tracker.record_output_file(
            "beam_preprocessor",
            output_shapefile_path,
            short_name="beam_zone_shapefile_sorted",
            description="BEAM zone shapefile created from sorted canonical zones.",
            model_run_id=model_run_hash,
        )

        logger.info("Updating BEAM configuration to use the new sorted shapefile.")
        relative_shapefile_path = os.path.join("shape", output_shapefile_name)

        # FIX: Pass workspace.full_path explicitly
        self._update_beam_config(
            "beam.agentsim.taz.filePath",
            value_override="${beam.inputDirectory}" + f"/{relative_shapefile_path}",
            base_path=workspace.full_path,
        )
        self._update_beam_config(
            "beam.agentsim.taz.tazIdFieldName",
            value_override=self.settings.shared.geography.zones.activitysim_index_col,
            base_path=workspace.full_path,
        )
        return output_shapefile_path

    def _copy_vehicles_from_atlas(self, workspace: "Workspace", model_run_hash: str):
        """Copies the vehicles file from the ATLAS output to the BEAM input scenario."""
        beam_scenario_folder = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            self.settings.run.region,
            self.settings.beam.scenario_folder,
        )
        beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")

        if self.state.run_info_path and os.path.exists(self.state.run_info_path):
            logger.info(
                f"[BeamPreprocessor] Restarted run detected. Using previous run's output path from {self.state.run_info_path}"
            )
            previous_run_dir = os.path.dirname(self.state.run_info_path)
            atlas_output_data_dir = os.path.join(
                previous_run_dir, "atlas", "atlas_output"
            )
        else:
            atlas_output_data_dir = workspace.get_atlas_output_dir()

        # Look for vehicles2_{year}.csv, falling back to year-1
        atlas_vehicle_file_loc = os.path.join(
            atlas_output_data_dir, f"vehicles2_{self.state.forecast_year}.csv"
        )
        if not os.path.exists(atlas_vehicle_file_loc):
            atlas_vehicle_file_loc = os.path.join(
                atlas_output_data_dir, f"vehicles2_{self.state.forecast_year - 1}.csv"
            )

        logger.info(
            f"Copying atlas vehicles2 file from {atlas_vehicle_file_loc} to {beam_vehicles_path}"
        )

        input_record = self.provenance_tracker.record_input_file(
            "beam_preprocessor",
            atlas_vehicle_file_loc,
            short_name="atlas_vehicles2_output",
            model_run_id=model_run_hash,
        )

        df = pd.read_csv(atlas_vehicle_file_loc)
        df.to_csv(beam_vehicles_path, compression="gzip", index=False)

        self.provenance_tracker.record_output_file_with_inputs(
            "beam_preprocessor",
            beam_vehicles_path,
            input_records=[input_record],
            description="BEAM vehicles input copied from ATLAS vehicles2 output",
            short_name="beam_vehicles_in",
            model_run_id=model_run_hash,
        )

    def _copy_plans_from_asim(
        self,
        input_records: RecordStore,
        workspace: "Workspace",
        model_run_hash: str,
    ) -> RecordStore:
        """Copies plans, households, and persons files from ActivitySim output to BEAM input."""
        logger.info("Attempting to copy final ASIM plans from provenance tracker.")
        file_format = self.settings.activitysim.file_format

        base_path = (
            os.path.dirname(self.state.run_info_path)
            if self.state.run_info_path and os.path.exists(self.state.run_info_path)
            else workspace.full_path
        )

        asim_file_paths = {}
        for record in input_records.all_records():
            splt = record.short_name.rsplit("_", 2)
            shortened_name = (
                splt[0] if len(splt) > 1 and str.isdigit(splt[1]) else record.short_name
            )
            if shortened_name in self.required_input_data:
                asim_file_paths[shortened_name] = (
                    os.path.join(base_path, record.file_path),
                    record,
                )
                logger.info(
                    f"Found ActivitySim output file {record.short_name}: {record.file_path}"
                )

        # Fallback: If required files are not found via provenance, try to locate them directly on the filesystem.
        # This provides robustness against provenance database issues or missing entries.
        required_asim_base_names = [
            "households",
            "persons",
            "beam_plans",  # ActivitySim outputs beam_plans
        ]
        # Construct the full path to the year-iteration specific output directory
        asim_output_iter_dir = os.path.join(
            workspace.get_asim_output_dir(),
            f"year-{self.state.current_year}-iteration-{self.state.current_inner_iter}",
        )

        for base_name in required_asim_base_names:
            if base_name not in asim_file_paths:
                # Construct the full filename including the format extension
                expected_file_name = f"{base_name}.{file_format}"
                expected_full_path = os.path.join(
                    asim_output_iter_dir, expected_file_name
                )

                if os.path.exists(expected_full_path):
                    logger.warning(
                        f"ActivitySim output file '{base_name}' (expected: {expected_file_name}) "
                        f"not found in provenance records. Falling back to filesystem at: {expected_full_path}"
                    )
                    # Create a dummy FileRecord for consistency, linking it to the current run
                    # A more complete FileRecord could be created by parsing file properties if needed,
                    # but for basic path retrieval, this is sufficient.
                    dummy_record = FileRecord(
                        file_path=os.path.relpath(
                            expected_full_path, base_path
                        ),  # Relative path for record
                        short_name=base_name,
                        description=f"ActivitySim output file found via filesystem fallback ({expected_file_name})",
                        producing_run_id=self.provenance_tracker.run_info.run_id,
                        unique_id=f"fallback-{base_name}-{self.provenance_tracker.run_info.run_id}",  # Unique ID for dummy record
                        exists=True,
                        year=self.state.current_year,
                        created_at=str(datetime.now()),
                        uri=(
                            self.provenance_tracker.to_uri(expected_full_path)
                            if self.provenance_tracker and hasattr(self.provenance_tracker, "to_uri")
                            else None
                        ),
                    )
                    asim_file_paths[base_name] = (expected_full_path, dummy_record)
                else:
                    logger.warning(
                        f"Required ActivitySim output file '{base_name}' (expected: {expected_file_name}) "
                        f"not found in provenance records AND not found on filesystem at: {expected_full_path}"
                    )

        if self.state.current_inner_iter <= 0:
            # First iteration: just copy files over
            record_list = self._copy_initial_asim_files(
                asim_file_paths, file_format, model_run_hash, workspace
            )
        else:
            # Subsequent iterations: merge replanned households/persons/plans
            record_list = self._merge_replanned_asim_files(
                asim_file_paths, file_format, model_run_hash, workspace
            )

        return RecordStore(recordList=[r for r in record_list if r is not None])

    def _copy_initial_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        model_run_hash: str,
        workspace: "Workspace",
    ) -> List[Record]:
        """Directly copies ActivitySim outputs for the first BEAM iteration."""
        record_list = []
        asim_to_beam_mapping = [
            (
                "beam_plans",
                "plans",
            ),  # ActivitySim outputs 'beam_plans', BEAM needs 'plans'
            ("households", "households"),
            ("persons", "persons"),
        ]
        beam_scenario_folder = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            self.settings.run.region,
            self.settings.beam.scenario_folder,
        )

        for asim_name, beam_name in asim_to_beam_mapping:
            asim_file_path, asim_file_record = asim_file_paths.get(
                asim_name, (None, None)
            )
            if asim_file_path:
                record = self._copy_with_compression_asim_file_to_beam(
                    asim_file_path,
                    beam_name,
                    file_format,
                    beam_scenario_folder,
                    input_record=asim_file_record,
                    model_run_hash=model_run_hash,
                )
                if record:
                    record_list.append(record)
            else:
                logger.warning(f"ActivitySim output file not found: {asim_name}")
        return record_list

    def _merge_replanned_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        model_run_hash: str,
        workspace: "Workspace",
    ) -> List[Record]:
        """Merges new ActivitySim outputs with existing BEAM inputs for replanning iterations."""
        logger.info("Merging asim outputs with existing beam input scenario files.")
        beam_scenario_folder = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            self.settings.run.region,
            self.settings.beam.scenario_folder,
        )

        asim_plans_path, asim_plans_rec = asim_file_paths.get(
            "beam_plans", (None, None)
        )
        asim_persons_path, asim_persons_rec = asim_file_paths.get(
            "persons", (None, None)
        )
        asim_households_path, asim_households_rec = asim_file_paths.get(
            "households", (None, None)
        )

        def get_data(
            path: str, table_type: str, file_format: str, file_source: str
        ) -> pd.DataFrame:
            if path is None:
                raise FileNotFoundError(
                    f"{file_source} file for table '{table_type}' not found."
                )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{file_source} file for table '{table_type}' not found at {path}."
                )
            return BeamDataHelper.read_and_clean(path, table_type, file_format)

        beam_plans_path = locate_beam_file(beam_scenario_folder, "plans", file_format)
        beam_persons_path = locate_beam_file(
            beam_scenario_folder, "persons", file_format
        )
        beam_households_path = locate_beam_file(
            beam_scenario_folder, "households", file_format
        )

        # Existing BEAM files (previous iteration)
        original_hh = get_data(beam_households_path, "households", file_format, "BEAM")
        original_per = get_data(beam_persons_path, "persons", file_format, "BEAM")
        original_plans = get_data(beam_plans_path, "plans", file_format, "BEAM")

        # New ActivitySim files (current iteration)
        updated_hh = get_data(
            asim_households_path, "households", file_format, "ActivitySim"
        )
        updated_per = get_data(asim_persons_path, "persons", file_format, "ActivitySim")
        updated_plans = get_data(asim_plans_path, "plans", file_format, "ActivitySim")

        # Ensure new plans are marked as selected (override default False)
        updated_plans["planselected"] = True

        # Log overlap
        logger.info(
            f"Replanned {len(updated_per)} persons and {len(updated_hh)} households."
        )

        # Merge logic
        per_idx = updated_per.index
        hh_idx = updated_hh.index

        persons_final = pd.concat(
            [updated_per, original_per.loc[~original_per.index.isin(per_idx)]]
        )
        households_final = pd.concat(
            [updated_hh, original_hh.loc[~original_hh.index.isin(hh_idx)]]
        )
        plans_final = pd.concat(
            [updated_plans, original_plans.loc[~original_plans.person_id.isin(per_idx)]]
        )

        # Save and Record
        record_list = []

        # Map dataframes to their output names and input records
        outputs = [
            (persons_final, "persons", asim_persons_rec),
            (households_final, "households", asim_households_rec),
            (plans_final, "plans", asim_plans_rec),
        ]

        # Retrieve beam_plans_record for provenance mixing
        beam_plans_rec = self.provenance_tracker.run_info.get_most_recent_record(
            short_name="plans_beam_in"
        )

        for df, name, asim_rec in outputs:
            path = locate_beam_file(beam_scenario_folder, name, file_format)

            if file_format == "parquet":
                df.to_parquet(path, index=True)
            else:
                df.to_csv(path, index=(name != "plans"), compression="gzip")

            # Re-enforce CSV behavior from original
            if file_format != "parquet":
                df.to_csv(path, index=False, compression="gzip")

            # Handle provenance inputs
            inputs = [asim_rec] if asim_rec else []
            if name == "plans" and beam_plans_rec:
                inputs.append(beam_plans_rec)

            rec = self.provenance_tracker.record_output_file_with_inputs(
                "beam_preprocessor",
                path,
                input_records=inputs,
                description=f"Merged {name} for BEAM input",
                model_run_id=model_run_hash,
                context=self.state,
                short_name=f"{name}_beam_in_{self.state.current_year}_{self.state.current_inner_iter}",
            )
            record_list.append(rec)

        return record_list

    def _copy_with_compression_asim_file_to_beam(
        self,
        asim_file_path: str,
        beam_file_name: str,
        file_format: str,
        beam_scenario_folder: str,
        input_record: Optional[Record] = None,
        model_run_hash: str = None,
    ) -> Optional[Record]:
        """Copies and compresses a single file from ActivitySim to BEAM, with provenance."""
        beam_file_path = locate_beam_file(
            beam_scenario_folder, beam_file_name, file_format
        )
        logger.info(
            f"Copying asim file {asim_file_path} to beam input scenario file {beam_file_path}"
        )

        if not os.path.exists(asim_file_path):
            logger.error(f"ActivitySim output file does not exist: {asim_file_path}")
            return self.provenance_tracker.record_output_file(
                "beam_preprocessor",
                beam_file_path,
                description=f"Missing BEAM input file: {beam_file_name}",
                model_run_id=model_run_hash,
                skip_missing=False,
            )

        self.provenance_tracker.record_input_record(input_record)

        table_type = "plans" if "plans" in beam_file_name else beam_file_name

        df = BeamDataHelper.read_and_clean(asim_file_path, table_type, file_format)

        if file_format == "parquet":
            df.to_parquet(beam_file_path, index=True)
        else:
            df.to_csv(beam_file_path, compression="gzip", index=False)

        return self.provenance_tracker.record_output_file_with_inputs(
            "beam_preprocessor",
            beam_file_path,
            input_records=[input_record],
            description=f"Copied from ActivitySim output: {beam_file_name}",
            short_name=beam_file_name + "_beam_in",
            model_run_id=model_run_hash,
        )

    def _handle_linkstats(
        self, workspace: "Workspace", previous_beam_records: List, store: RecordStore
    ):
        """
        Ensure a single, explicit warm-start linkstats record is present for this BEAM run.

        Semantics:
        - For the first BEAM run in the PILATES inner-iteration loop, use the initial
          warm-start linkstats file copied into the BEAM mutable input tree
          (typically `.../<router_directory>/init.linkstats.csv.gz`).
        - For subsequent BEAM runs (inner-iterations > 0), use the linkstats produced by
          the most recent BEAM run's *last* BEAM internal iteration (the one logged without
          a `_sub...` suffix).

        Implementation note:
        - We "tag" the selected file via a stable `FileRecord.short_name` of
          `linkstats_warmstart`, so each BEAM step has a consistent lineage key.
        """
        from pilates.generic.records import FileRecord

        def _abs_from_record(rec: FileRecord) -> Optional[str]:
            if not getattr(rec, "file_path", None):
                return None
            if os.path.isabs(rec.file_path):
                return rec.file_path
            return os.path.abspath(os.path.join(str(workspace.full_path), rec.file_path))

        # Prefer the last-sub-iteration BEAM output linkstats (no `_sub`), which is
        # named like `linkstats_<year>_<inner_iter>` by BeamRunner.gather_outputs().
        beam_output_linkstats = None
        pattern = re.compile(r"^linkstats_\d+_\d+$")
        for rec in previous_beam_records or []:
            sn = getattr(rec, "short_name", "") or ""
            if "_sub" in sn:
                continue
            if pattern.match(sn):
                beam_output_linkstats = rec
                break

        # Back-compat fallback: if a previous run only logged an unversioned `linkstats`
        # record, use it, but make it very obvious that this is not ideal.
        if beam_output_linkstats is None:
            for rec in previous_beam_records or []:
                sn = getattr(rec, "short_name", "") or ""
                if sn == "linkstats":
                    beam_output_linkstats = rec
                    logger.warning(
                        "[NOT IDEAL] Using an unversioned `linkstats` record as warm-start input. "
                        "Prefer BEAM outputs logged as `linkstats_<year>_<inner_iter>` so lineage is unambiguous."
                    )
                    break

        # Determine which physical file path to use.
        warmstart_abs_path = None
        warmstart_source = None
        if beam_output_linkstats is not None:
            warmstart_abs_path = _abs_from_record(beam_output_linkstats)
            warmstart_source = "previous_beam_output"

        if warmstart_abs_path is None:
            warmstart_abs_path = os.path.join(
                str(workspace.get_beam_mutable_data_dir()),
                self.settings.run.region,
                self.settings.beam.router_directory,
                "init.linkstats.csv.gz",
            )
            warmstart_source = "initial_inputs"

        if not warmstart_abs_path or not os.path.exists(warmstart_abs_path):
            logger.warning(
                "[BEAM Preprocessor] Warm-start linkstats file not found (source=%s): %s",
                warmstart_source,
                warmstart_abs_path,
            )
            return

        warmstart_rel_path = os.path.relpath(warmstart_abs_path, str(workspace.full_path))
        warmstart_uri = None
        if hasattr(self.provenance_tracker, "to_uri"):
            try:
                warmstart_uri = self.provenance_tracker.to_uri(warmstart_abs_path)
            except Exception:
                warmstart_uri = None

        warmstart_record = FileRecord(
            file_path=warmstart_rel_path,
            short_name="linkstats_warmstart",
            description=f"BEAM warm-start linkstats (source={warmstart_source})",
            models=["beam"],
            year=getattr(self.state, "forecast_year", None),
            iteration=getattr(self.state, "current_inner_iter", None),
            uri=warmstart_uri,
        )

        logger.info(
            "[BEAM Preprocessor] Using warm-start linkstats (source=%s): %s",
            warmstart_source,
            warmstart_rel_path,
        )
        store.add_record(warmstart_record)

    @staticmethod
    def _find_beam_production_path(settings: PilatesConfig, region: str) -> Optional[str]:
        """
        Finds the path to the BEAM production data directory.

        Semantics:
        - `settings.beam.local_input_folder` is interpreted as an inputs-root-relative
          path (relative to the directory containing `run.py`) unless absolute.
        - The returned path should point at the region subdirectory inside that tree.
        """
        pilates_root = find_project_root(start_path=os.path.dirname(__file__))
        if not pilates_root:
            pilates_root = os.path.realpath(os.getcwd())
            logger.warning(
                "[NOT IDEAL] Could not locate PILATES project root via markers; falling back to cwd='%s'. "
                "This can break inputs:// URI virtualization and should be fixed by running from the repo root "
                "or ensuring the repo contains expected markers.",
                pilates_root,
            )

        configured_root = getattr(settings.beam, "local_input_folder", None) if settings.beam else None
        if not configured_root:
            logger.error("BEAM local_input_folder is not configured.")
            return None

        if os.path.isabs(configured_root):
            root_candidate = configured_root
        else:
            root_candidate = os.path.join(pilates_root, configured_root)

        primary_path = os.path.abspath(os.path.join(root_candidate, region))
        if os.path.exists(primary_path):
            return primary_path

        # Legacy fallback observed in some HPC layouts.
        alt_root = os.path.join(pilates_root, "sources", "PILATES")
        alt_path = os.path.abspath(os.path.join(alt_root, configured_root, region))
        if os.path.exists(alt_path):
            logger.warning(
                "[NOT IDEAL] Primary BEAM production path not found; using legacy alternate layout: %s. "
                "Prefer fixing mounts/paths so inputs resolve directly under the configured inputs root.",
                alt_path,
            )
            return alt_path

        logger.error(
            "BEAM production input directory does not exist at either %s or %s",
            primary_path,
            alt_path,
        )
        return None

    def _update_beam_config(
        self, param: str, value_override=None, base_path: str = None
    ):
        """
        Update a BEAM config file parameter with a new value.

        Args:
            param: The config parameter key.
            value_override: The value to set.
            base_path: The root directory of the workspace.
        """
        config_header = beam_param_map.get(param)
        if not config_header:
            logger.warning(
                f"[BEAM Preprocessor] Tried to modify parameter {param} but couldn't find it in settings.yaml"
            )
            return

        if value_override is not None:
            config_value = value_override
        else:
            pydantic_path = BEAM_PYDANTIC_PATH_MAP.get(param)
            if not pydantic_path:
                logger.warning(
                    f"Parameter '{param}' has no defined Pydantic path. Cannot update beam config."
                )
                return
            config_value = get_setting(self.settings, pydantic_path)

        if config_value is None:
            logger.debug(
                f"Skipping beam config update for '{param}' because value is None."
            )
            return

        # FIX: Determine root path robustly
        root = base_path
        if not root and hasattr(self.state, "workspace"):
            root = self.state.workspace.full_path

        if not root:
            logger.error(
                f"Cannot update BEAM config for {param}: Workspace root path could not be determined."
            )
            return

        beam_config_path = os.path.join(
            root,
            self.settings.beam.local_mutable_data_folder,
            self.settings.run.region,
            self.settings.beam.config,
        )

        if not os.path.exists(beam_config_path):
            logger.warning(
                f"[BEAM Preprocessor] BEAM config file does not exist: {beam_config_path}"
            )
            return

        with open(beam_config_path, "r") as file:
            lines = file.readlines()

        modified = False
        with open(beam_config_path, "w") as file:
            for line in lines:
                if line.strip().startswith(config_header):
                    if not modified:  # Write only the first occurrence
                        file.write(f"{config_header} = {config_value}\n")
                        modified = True
                else:
                    file.write(line)
            if not modified:
                file.write(f"\n{config_header} = {config_value}\n")

        logger.info(
            f"[BEAM Preprocessor] Updated config {config_header} to {config_value} in {beam_config_path}"
        )
