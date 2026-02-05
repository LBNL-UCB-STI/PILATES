from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, TYPE_CHECKING, Dict, Any
import re

if TYPE_CHECKING:
    from pilates.workspace import Workspace


import pandas as pd

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.io import locate_beam_file
from pilates.utils.path_utils import find_project_root
from pilates.utils.settings_helper import get as get_setting
from pilates.workflows.artifact_keys import (
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_INPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)
from pilates.activitysim.outputs import has_asim_run_marker
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

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this preprocessor expects from the workflow.
        """
        asim_output_dir = None
        if getattr(settings, "activity_demand_enabled", False):
            asim_output_dir = workspace.get_asim_output_dir()

        atlas_output_dir = None
        if getattr(settings, "vehicle_ownership_model_enabled", False):
            atlas_output_dir = workspace.get_atlas_output_dir()
        return {
            BEAM_MUTABLE_DATA_DIR: workspace.get_beam_mutable_data_dir(),
            ASIM_OUTPUT_DIR: asim_output_dir,
            ATLAS_OUTPUT_DIR: atlas_output_dir,
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this preprocessor produces.

        Notes
        -----
        Output keys
            - ``beam_mutable_data_dir``: Mutable BEAM data directory populated
              with inputs for the runner.
        Related docs
            - See `pilates/beam/inputs.py` for the corresponding input
              descriptions used by BEAM and downstream models.
        """
        return {"beam_mutable_data_dir": workspace.get_beam_mutable_data_dir()}

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)
        self.required_input_data: List[str] = [
            "persons",
            "households",
            "beam_plans",
            "linkstats",
            "beam_plans_out",
        ]
        self.settings = self.state.full_settings

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
            raw_name = getattr(record, "short_name", "") or ""
            if not raw_name:
                continue
            if raw_name in self.required_input_data:
                short_name = raw_name
            elif "_asim_out" in raw_name:
                short_name = raw_name.split("_asim_out")[0]
            else:
                short_name = raw_name

            if short_name in self.required_input_data:
                input_records.add_record(record)
                logger.info(f"Added {short_name} to beam inputs")
            else:
                logger.info(f"Skipping {raw_name} produced by activitysim")

        # For replanning iterations, inputs from the previous BEAM run should be
        # passed in explicitly by the caller.
        previous_beam_records: List[FileRecord] = []
        for record in previous_records.all_records():
            short_name = getattr(record, "short_name", "") or ""
            if short_name.startswith("linkstats"):
                previous_beam_records.append(record)

        # Update BEAM config based on settings
        self._update_beam_config(
            "max_plans_memory",
            value_override=0 if self.settings.beam.discard_plans_every_year else None,
            base_path=workspace.full_path,
        )
        # Prepare the zone shapefile to ensure consistent zone ordering
        self.prepare_beam_zone_shapefile(workspace)

        store = RecordStore()

        # Copy vehicle data from Atlas (only on first iteration)
        if (
            self.settings.vehicle_ownership_model_enabled
            and self.state.current_inner_iter == 0
        ):
            atlas_input_record, beam_output_record = self._copy_vehicles_from_atlas(
                workspace
            )
            if atlas_input_record is not None:
                store.add_record(atlas_input_record)
            if beam_output_record is not None:
                store.add_record(beam_output_record)

        # Copy and merge plans from ActivitySim
        store += self._copy_plans_from_asim(input_records, workspace)

        # Add FileRecord outputs here for any additional BEAM inputs you want
        # tracked as explicit coupler keys (e.g., network files from the
        # initialized BEAM input directory).
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
        # Note: BEAM input directories are captured via the BEAM config adapter;
        # we intentionally avoid tracking them as separate artifacts.

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
            # Note: BEAM common directory is not tracked as a separate artifact.

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

    def prepare_beam_zone_shapefile(self, workspace: "Workspace") -> Optional[str]:
        """
        Creates a sorted zone shapefile for BEAM from the canonical zone definitions
        and updates the BEAM config to use it.
        """

        output_shapefile_path = _prepare_beam_zone_shapefile(workspace, self.settings)
        output_shapefile_name = os.path.basename(output_shapefile_path)

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

    def _copy_vehicles_from_atlas(
        self, workspace: "Workspace"
    ) -> Tuple[Optional[FileRecord], Optional[FileRecord]]:
        """
        Copies the vehicles file from the ATLAS output to the BEAM input scenario.

        Returns
        -------
        tuple of FileRecord or None
            (input_record, output_record) for lineage tracking.
        """
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

        if not os.path.exists(atlas_vehicle_file_loc):
            logger.warning(
                "ATLAS vehicles2 file not found for BEAM input: %s",
                atlas_vehicle_file_loc,
            )
            return None, None

        logger.info(
            f"Copying atlas vehicles2 file from {atlas_vehicle_file_loc} to {beam_vehicles_path}"
        )

        df = pd.read_csv(atlas_vehicle_file_loc)
        df.to_csv(beam_vehicles_path, compression="gzip", index=False)
        input_record = FileRecord(
            file_path=atlas_vehicle_file_loc,
            description="ATLAS vehicles2 input for BEAM",
            short_name=ATLAS_VEHICLES2_INPUT,
            year=getattr(self.state, "forecast_year", None),
            iteration=getattr(self.state, "current_inner_iter", None),
        )
        output_record = FileRecord(
            file_path=beam_vehicles_path,
            description="BEAM vehicles input derived from ATLAS vehicles2",
            short_name="vehicles_beam_in",
            year=getattr(self.state, "forecast_year", None),
            iteration=getattr(self.state, "current_inner_iter", None),
        )
        return input_record, output_record

    def _copy_plans_from_asim(
        self,
        input_records: RecordStore,
        workspace: "Workspace",
    ) -> RecordStore:
        """Copies plans, households, and persons files from ActivitySim output to BEAM input."""
        logger.info("Attempting to copy final ASIM plans from input records.")
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
            if shortened_name.endswith("_asim_out"):
                shortened_name = shortened_name.split("_asim_out")[0]
            if shortened_name in self.required_input_data:
                asim_file_paths[shortened_name] = (
                    os.path.join(base_path, record.file_path),
                    record,
                )
                logger.info(
                    f"Found ActivitySim output file {record.short_name}: {record.file_path}"
                )

        # Fallback: If required files are not found in the input records, try to
        # locate them directly on the filesystem.
        required_asim_base_names = [
            "households",
            "persons",
            "beam_plans",  # ActivitySim outputs beam_plans
        ]
        asim_output_dir = workspace.get_asim_output_dir()
        if base_path and os.path.isabs(base_path):
            rel_asim_output_dir = os.path.relpath(asim_output_dir, workspace.full_path)
            asim_output_dir = os.path.join(base_path, rel_asim_output_dir)

        allow_final_pipeline = has_asim_run_marker(
            asim_output_dir, self.state.current_year, self.state.current_inner_iter
        )
        if not allow_final_pipeline:
            logger.warning(
                "ASim success marker not found for year %s iteration %s; "
                "skipping final_pipeline fallback for BEAM inputs.",
                self.state.current_year,
                self.state.current_inner_iter,
            )

        # Construct the full path to the year-iteration specific output directory
        asim_output_iter_dir = os.path.join(
            asim_output_dir,
            f"year-{self.state.current_year}-iteration-{self.state.current_inner_iter}",
        )

        for base_name in required_asim_base_names:
            if base_name not in asim_file_paths:
                expected_file_name = f"{base_name}.{file_format}"
                candidate_paths = [
                    os.path.join(asim_output_iter_dir, expected_file_name),
                    os.path.join(
                        asim_output_iter_dir,
                        base_name,
                        f"final.{file_format}",
                    ),
                ]
                if allow_final_pipeline:
                    candidate_paths.append(
                        os.path.join(
                            asim_output_dir,
                            "final_pipeline",
                            base_name,
                            f"final.{file_format}",
                        )
                    )
                found_path = next(
                    (path for path in candidate_paths if os.path.exists(path)), None
                )

                if found_path:
                    logger.warning(
                        "ActivitySim output file '%s' (expected: %s) not found in input records. "
                        "Falling back to filesystem at: %s",
                        base_name,
                        expected_file_name,
                        found_path,
                    )
                    dummy_path = (
                        os.path.relpath(found_path, base_path)
                        if base_path and os.path.isabs(base_path)
                        else found_path
                    )
                    dummy_record = FileRecord(
                        file_path=dummy_path,
                        short_name=base_name,
                        description=(
                            "ActivitySim output file found via filesystem fallback "
                            f"({os.path.basename(found_path)})"
                        ),
                        year=self.state.current_year,
                    )
                    asim_file_paths[base_name] = (found_path, dummy_record)
                else:
                    logger.warning(
                        "Required ActivitySim output file '%s' (expected: %s) not found in input "
                        "records AND not found on filesystem at any of: %s",
                        base_name,
                        expected_file_name,
                        ", ".join(candidate_paths),
                    )

        if self.state.current_inner_iter <= 0:
            # First iteration: just copy files over
            record_list = self._copy_initial_asim_files(
                asim_file_paths, file_format, workspace
            )
        else:
            # Subsequent iterations: merge replanned households/persons/plans
            record_list = self._merge_replanned_asim_files(
                asim_file_paths, file_format, workspace
            )

        return RecordStore(recordList=[r for r in record_list if r is not None])

    def _copy_initial_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        workspace: "Workspace",
    ) -> List[FileRecord]:
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
                records = self._copy_with_compression_asim_file_to_beam(
                    asim_file_path,
                    beam_name,
                    file_format,
                    beam_scenario_folder,
                    input_record=asim_file_record,
                )
                if records:
                    record_list.extend(records)
            else:
                logger.warning(f"ActivitySim output file not found: {asim_name}")
        return record_list

    def _merge_replanned_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        workspace: "Workspace",
    ) -> List[FileRecord]:
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

        for df, name, asim_rec in outputs:
            path = locate_beam_file(beam_scenario_folder, name, file_format)

            if file_format == "parquet":
                df.to_parquet(path, index=True)
            else:
                df.to_csv(path, index=(name != "plans"), compression="gzip")

            # Re-enforce CSV behavior from original
            if file_format != "parquet":
                df.to_csv(path, index=False, compression="gzip")

            record_list.extend(
                self._format_specific_output_records(
                    name, path, file_format, "Merged BEAM input file"
                )
            )

        return record_list

    def _copy_with_compression_asim_file_to_beam(
        self,
        asim_file_path: str,
        beam_file_name: str,
        file_format: str,
        beam_scenario_folder: str,
        input_record: Optional[FileRecord] = None,
    ) -> Optional[List[FileRecord]]:
        """Copies and compresses a single file from ActivitySim to BEAM."""
        beam_file_path = locate_beam_file(
            beam_scenario_folder, beam_file_name, file_format
        )
        logger.info(
            f"Copying asim file {asim_file_path} to beam input scenario file {beam_file_path}"
        )

        if not os.path.exists(asim_file_path):
            logger.error(f"ActivitySim output file does not exist: {asim_file_path}")
            return [
                FileRecord(
                    file_path=beam_file_path,
                    description=f"Missing BEAM input file: {beam_file_name}",
                    short_name=f"{beam_file_name}_beam_in_missing",
                    year=self.state.current_year,
                    iteration=self.state.current_inner_iter,
                )
            ]

        table_type = "plans" if "plans" in beam_file_name else beam_file_name

        df = BeamDataHelper.read_and_clean(asim_file_path, table_type, file_format)

        if file_format == "parquet":
            df.to_parquet(beam_file_path, index=True)
        else:
            df.to_csv(beam_file_path, compression="gzip", index=False)

        records = self._format_specific_output_records(
            beam_file_name,
            beam_file_path,
            file_format,
            f"Copied from ActivitySim output: {beam_file_name}",
        )
        return records

    def _format_specific_output_records(
        self,
        file_stem: str,
        file_path: str,
        file_format: str,
        description_prefix: str,
    ) -> List[FileRecord]:
        """
        Emit explicit output records for BEAM input files by format.

        Parameters
        ----------
        file_stem : str
            Base filename (e.g., "persons").
        file_path : str
            Full path to the output file.
        file_format : str
            File format ("csv" or "parquet").
        description_prefix : str
            Description prefix for the record.
        """
        short_name_map = {
            "plans": BEAM_PLANS_IN,
            "households": BEAM_HOUSEHOLDS_IN,
            "persons": BEAM_PERSONS_IN,
        }
        short_name = short_name_map.get(file_stem, f"{file_stem}_beam_in")
        return [
            FileRecord(
                file_path=file_path,
                description=description_prefix,
                short_name=short_name,
                year=getattr(self.state, "current_year", None),
                iteration=getattr(self.state, "current_inner_iter", None),
            )
        ]

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
            return os.path.abspath(
                os.path.join(str(workspace.full_path), rec.file_path)
            )

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
            base_dir = os.path.join(
                str(workspace.get_beam_mutable_data_dir()),
                self.settings.run.region,
                self.settings.beam.router_directory,
            )
            parquet_candidate = os.path.join(base_dir, "init.linkstats.parquet")
            csv_candidate = os.path.join(base_dir, "init.linkstats.csv.gz")
            if os.path.exists(parquet_candidate):
                warmstart_abs_path = parquet_candidate
            else:
                warmstart_abs_path = csv_candidate
            warmstart_source = "initial_inputs"

        if not warmstart_abs_path or not os.path.exists(warmstart_abs_path):
            logger.warning(
                "[BEAM Preprocessor] Warm-start linkstats file not found (source=%s): %s",
                warmstart_source,
                warmstart_abs_path,
            )
            return

        warmstart_rel_path = os.path.relpath(
            warmstart_abs_path, str(workspace.full_path)
        )
        warmstart_record = FileRecord(
            file_path=warmstart_rel_path,
            short_name="linkstats_warmstart",
            description=f"BEAM warm-start linkstats (source={warmstart_source})",
            year=getattr(self.state, "forecast_year", None),
            iteration=getattr(self.state, "current_inner_iter", None),
        )

        logger.info(
            "[BEAM Preprocessor] Using warm-start linkstats (source=%s): %s",
            warmstart_source,
            warmstart_rel_path,
        )
        store.add_record(warmstart_record)

    @staticmethod
    def _find_beam_production_path(
        settings: PilatesConfig, region: str
    ) -> Optional[str]:
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

        configured_root = (
            getattr(settings.beam, "local_input_folder", None)
            if settings.beam
            else None
        )
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
