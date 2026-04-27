from __future__ import annotations

import logging
import json
import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING, Dict, Any, Mapping

if TYPE_CHECKING:
    from pilates.workspace import Workspace

from pilates.config import PilatesConfig
from pilates.beam import beam_exchange
from pilates.beam.config_hocon import (
    BeamConfigHoconError,
    beam_config_debug_snapshot,
    beam_config_env_overrides,
    beam_primary_config_path,
    update_staged_beam_config_value,
)
from pilates.beam import beam_input_staging
from pilates.beam.outputs import BeamPreprocessOutputs
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.consist_runtime import artifact_fingerprint
from pilates.utils.io import is_activity_demand_enabled
from pilates.utils.path_utils import find_project_root
from pilates.workflows.artifact_keys import (
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

def _record_store_from_artifact_mappings(
    *,
    activity_demand_outputs: Optional[Mapping[str, Any]],
    previous_beam_outputs: Optional[Mapping[str, Any]],
    beam_preprocess_inputs: Optional[Mapping[str, Any]],
) -> RecordStore:
    """
    Build the BEAM-internal input store from plain artifact mappings.
    """

    def _append_mapping(
        store: RecordStore,
        artifact_mapping: Optional[Mapping[str, Any]],
        *,
        description_prefix: str,
        key_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        if not artifact_mapping:
            return
        for key, value in artifact_mapping.items():
            path = artifact_to_path(value, None)
            if path is None and isinstance(value, (str, os.PathLike)):
                path = os.fspath(value)
            if not path:
                continue
            record_key = key_aliases.get(key, key) if key_aliases else key
            store.add_record(
                FileRecord(
                    file_path=str(path),
                    short_name=record_key,
                    description=f"{description_prefix}: {record_key}",
                    content_hash=artifact_fingerprint(value),
                )
            )

    combined = RecordStore()
    _append_mapping(
        combined,
        activity_demand_outputs,
        description_prefix="BEAM preprocess activity-demand input",
    )
    _append_mapping(
        combined,
        previous_beam_outputs,
        description_prefix="BEAM preprocess warm-start input",
    )
    _append_mapping(
        combined,
        beam_preprocess_inputs,
        description_prefix="BEAM preprocess provided input",
        key_aliases={
            BEAM_PLANS_IN: "beam_plans",
            BEAM_HOUSEHOLDS_IN: "households",
            BEAM_PERSONS_IN: "persons",
            LINKSTATS_WARMSTART: "linkstats",
        },
    )
    return combined

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

        # Reset index so the canonical ActivitySim zone ID becomes a column.
        canonical_zones_gdf = canonical_zones_gdf.reset_index()

        if settings.beam is None:
            raise ValueError("Beam settings are not configured")
        sort_col = settings.beam.skim_zone_geoid_col
        canonical_id_col = settings.shared.geography.zones.activitysim_index_col
        if sort_col in canonical_zones_gdf.columns:
            canonical_order = canonical_zones_gdf[canonical_id_col].astype(str).tolist()
            sort_order = (
                canonical_zones_gdf.sort_values(by=sort_col, kind="stable")[
                    canonical_id_col
                ]
                .astype(str)
                .tolist()
            )
            if sort_order == canonical_order:
                logger.info(
                    "Beam sort column '%s' matches canonical zone order; preserving export.",
                    sort_col,
                )
            else:
                logger.warning(
                    "Beam sort column '%s' would reorder canonical zones. "
                    "Preserving canonical order based on '%s' instead.",
                    sort_col,
                    canonical_id_col,
                )
        else:
            logger.warning(
                "Beam sort column '%s' not found in canonical zones export. "
                "Preserving canonical order based on '%s'. Available columns: %s",
                sort_col,
                canonical_id_col,
                canonical_zones_gdf.columns.tolist(),
            )

        canonical_zones_gdf.to_file(output_shapefile_path, driver="GeoJSON")

        logger.info("--- BEAM Zone Shapefile Preparation Complete ---")
        return output_shapefile_path
    except Exception as e:
        logger.error(
            f"An error occurred during BEAM shapefile preparation: {e}", exc_info=True
        )
        raise


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
        beam_scenario_dir = beam_exchange.resolve_beam_exchange_scenario_folder(
            settings,
            workspace,
        )
        preferred_format = (
            getattr(getattr(settings, "activitysim", None), "file_format", None)
            or "parquet"
        )
        beam_input_paths = {
            BEAM_PLANS_IN: beam_exchange.locate_existing_beam_exchange_input(
                beam_scenario_dir, "plans", preferred_format
            )[0],
            BEAM_HOUSEHOLDS_IN: beam_exchange.locate_existing_beam_exchange_input(
                beam_scenario_dir, "households", preferred_format
            )[0],
            BEAM_PERSONS_IN: beam_exchange.locate_existing_beam_exchange_input(
                beam_scenario_dir, "persons", preferred_format
            )[0],
            LINKSTATS_WARMSTART: beam_exchange.locate_existing_beam_exchange_input(
                beam_scenario_dir, "linkstats", preferred_format
            )[0],
        }
        asim_output_dir = None
        if settings.runtime.flags.activity_demand_enabled:
            asim_output_dir = workspace.get_asim_output_dir()

        atlas_output_dir = None
        if settings.runtime.flags.vehicle_ownership_model_enabled:
            atlas_output_dir = workspace.get_atlas_output_dir()
        beam_config = getattr(getattr(settings, "beam", None), "config", None)
        beam_config_path = (
            os.path.join(workspace.get_beam_mutable_data_dir(), settings.run.region, beam_config)
            if beam_config
            else None
        )
        atlas_vehicle_input = None
        if atlas_output_dir is not None:
            forecast_year = getattr(state, "forecast_year", None)
            if forecast_year is not None:
                for candidate in (
                    os.path.join(atlas_output_dir, f"vehicles2_{forecast_year}.csv"),
                    os.path.join(atlas_output_dir, f"vehicles2_{forecast_year - 1}.csv"),
                ):
                    if os.path.exists(candidate):
                        atlas_vehicle_input = candidate
                        break
        return {
            BEAM_MUTABLE_DATA_DIR: workspace.get_beam_mutable_data_dir(),
            BEAM_CONFIG_FILE: beam_config_path,
            ASIM_OUTPUT_DIR: asim_output_dir,
            ATLAS_OUTPUT_DIR: atlas_output_dir,
            ATLAS_VEHICLES2_OUTPUT: atlas_vehicle_input,
            **beam_input_paths,
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
    ):
        super().__init__(model_name, state)
        self.required_input_data: List[str] = [
            "persons",
            "households",
            "beam_plans",
            "linkstats",
            "beam_plans_out",
        ]
        self.settings = self.state.full_settings

    def _default_beam_exchange_scenario_folder(self, workspace: "Workspace") -> str:
        return beam_exchange.default_beam_exchange_scenario_folder(
            self.settings,
            workspace,
        )

    def _config_beam_exchange_scenario_folder(
        self, workspace: "Workspace"
    ) -> Optional[str]:
        return beam_exchange.config_beam_exchange_scenario_folder(
            self.settings,
            workspace,
        )

    def _resolve_beam_exchange_scenario_folder(self, workspace: "Workspace") -> str:
        return beam_exchange.resolve_beam_exchange_scenario_folder(
            self.settings,
            workspace,
        )

    def _beam_exchange_scenario_folder_candidates(
        self, workspace: "Workspace"
    ) -> List[str]:
        return beam_exchange.beam_exchange_scenario_folder_candidates(
            self.settings,
            workspace,
        )

    @staticmethod
    def _beam_exchange_format_candidates(preferred_format: Optional[str]) -> List[str]:
        return beam_exchange.beam_exchange_format_candidates(preferred_format)

    @classmethod
    def _locate_existing_beam_exchange_input(
        cls,
        beam_scenario_folder: str,
        stem: str,
        preferred_format: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        return beam_exchange.locate_existing_beam_exchange_input(
            beam_scenario_folder,
            stem,
            preferred_format,
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Prepares all data needed to run BEAM for the current iteration.
        """
        input_records = RecordStore()
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
        if self.settings.shared.geography.zones is not None:
            self.prepare_beam_zone_shapefile(workspace)
        else:
            logger.info("No zones configured; skipping zone shapefile preparation.")

        store = RecordStore()

        # Copy vehicle data from Atlas (only on first iteration)
        if (
            self.settings.vehicle_ownership_model_enabled
            and self.state.current_inner_iter == 0
        ):
            restored_vehicle_source = next(
                (
                    record.file_path
                    for record in previous_records.all_records()
                    if getattr(record, "short_name", None)
                    in (ATLAS_VEHICLES2_OUTPUT, "vehicles_beam_in")
                ),
                None,
            )
            beam_output_record = self._copy_vehicles_from_atlas(
                workspace,
                source_path=restored_vehicle_source,
            )
            if beam_output_record is not None:
                store.add_record(beam_output_record)

        canonical_input_keys = {
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        }

        # In beam-only mode, the copied BEAM input repo already contains default
        # plans/households/persons files. In ActivitySim mode, those canonical
        # files must come from ActivitySim staging rather than silently falling
        # back to stale defaults.
        if self._activity_demand_enabled():
            store += self._copy_plans_from_asim(input_records, workspace)
            staged_keys = {
                record.short_name
                for record in store.all_records()
                if getattr(record, "short_name", "") in canonical_input_keys
            }
            missing_keys = canonical_input_keys - staged_keys
            if missing_keys:
                raise RuntimeError(
                    "beam_preprocess expected ActivitySim to stage the canonical "
                    f"BEAM input trio, but missing outputs were {sorted(missing_keys)}"
                )
            if self.state.current_inner_iter == 0:
                self._validate_population_consistency(workspace)
        else:
            store += self._register_existing_beam_exchange_inputs(workspace)

        # Add FileRecord outputs here for any additional BEAM inputs you want
        # tracked as explicit coupler keys (e.g., network files from the
        # initialized BEAM input directory).
        store += output_records

        # Ensure linkstats file is present and recorded
        self._handle_linkstats(workspace, previous_beam_records, store)

        try:
            debug_snapshot = beam_config_debug_snapshot(
                self.settings,
                workspace=workspace,
            )
            logger.warning(
                "[BEAM DEBUG][preprocess] staged config snapshot: %s",
                json.dumps(debug_snapshot, sort_keys=True, default=str),
            )
        except Exception:
            logger.exception(
                "[BEAM DEBUG][preprocess] failed to capture staged config snapshot"
            )

        logger.info("[BEAM Preprocessor] BEAM preprocessing complete.")
        return store

    def _activity_demand_enabled(self) -> bool:
        """Return whether ActivitySim is enabled for the current run."""
        return is_activity_demand_enabled(self.settings)

    def _register_existing_beam_exchange_inputs(
        self,
        workspace: "Workspace",
    ) -> RecordStore:
        return beam_exchange.register_existing_beam_exchange_inputs(
            settings=self.settings,
            state=self.state,
            workspace=workspace,
        )

    def existing_beam_exchange_inputs(self, workspace: "Workspace") -> RecordStore:
        """
        Return the canonical BEAM exchange inputs already staged in the mutable repo.
        """
        return self._register_existing_beam_exchange_inputs(workspace)

    def preprocess(
        self,
        workspace: "Workspace",
        *,
        activity_demand_outputs: Optional[Mapping[str, Any]] = None,
        previous_beam_outputs: Optional[Mapping[str, Any]] = None,
        beam_preprocess_inputs: Optional[Mapping[str, Any]] = None,
    ) -> BeamPreprocessOutputs:
        """
        Build BEAM inputs from plain artifact mappings and return typed outputs.
        """
        self.state.set_sub_stage_progress("preprocessor")
        input_store = _record_store_from_artifact_mappings(
            activity_demand_outputs=activity_demand_outputs,
            previous_beam_outputs=previous_beam_outputs,
            beam_preprocess_inputs=beam_preprocess_inputs,
        )
        record_store = self._preprocess(workspace, input_store)
        prepared_inputs: Dict[str, Path] = {}
        for key, value in record_store.to_mapping().items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            prepared_inputs[key] = Path(path)
        return BeamPreprocessOutputs(
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared_inputs,
        )

    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir: str,
        workspace: Optional["Workspace"] = None,
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
            logger.debug(
                "[BEAM Preprocessor] Deferring zone-id config updates until the "
                "sorted zone shapefile is prepared."
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
        self,
        workspace: "Workspace",
        *,
        source_path: Optional[str] = None,
    ) -> Optional[FileRecord]:
        return beam_input_staging.copy_vehicles_from_atlas(
            workspace=workspace,
            state=self.state,
            resolve_beam_exchange_scenario_folder_fn=self._resolve_beam_exchange_scenario_folder,
            source_path=source_path,
            preferred_format=(
                getattr(getattr(self.settings, "activitysim", None), "file_format", None)
                or "csv"
            ),
        )

    def _copy_plans_from_asim(
        self,
        input_records: RecordStore,
        workspace: "Workspace",
    ) -> RecordStore:
        return beam_input_staging.copy_plans_from_asim(
            input_records=input_records,
            workspace=workspace,
            state=self.state,
            settings=self.settings,
            required_input_data=self.required_input_data,
            copy_initial_asim_files_fn=self._copy_initial_asim_files,
            merge_replanned_asim_files_fn=self._merge_replanned_asim_files,
        )

    def _copy_initial_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        workspace: "Workspace",
    ) -> List[FileRecord]:
        return beam_input_staging.copy_initial_asim_files(
            asim_file_paths=asim_file_paths,
            file_format=file_format,
            workspace=workspace,
            resolve_beam_exchange_scenario_folder_fn=self._resolve_beam_exchange_scenario_folder,
            copy_with_compression_asim_file_to_beam_fn=self._copy_with_compression_asim_file_to_beam,
        )

    def _merge_replanned_asim_files(
        self,
        asim_file_paths: dict,
        file_format: str,
        workspace: "Workspace",
    ) -> List[FileRecord]:
        return beam_input_staging.merge_replanned_asim_files(
            asim_file_paths=asim_file_paths,
            file_format=file_format,
            workspace=workspace,
            resolve_beam_exchange_scenario_folder_fn=self._resolve_beam_exchange_scenario_folder,
            format_specific_output_records_fn=self._format_specific_output_records,
        )

    def _copy_with_compression_asim_file_to_beam(
        self,
        asim_file_path: str,
        beam_file_name: str,
        file_format: str,
        beam_scenario_folder: str,
        input_record: Optional[FileRecord] = None,
    ) -> Optional[List[FileRecord]]:
        _ = input_record
        return beam_input_staging.copy_with_compression_asim_file_to_beam(
            asim_file_path=asim_file_path,
            beam_file_name=beam_file_name,
            file_format=file_format,
            beam_scenario_folder=beam_scenario_folder,
            state=self.state,
        )

    def _format_specific_output_records(
        self,
        file_stem: str,
        file_path: str,
        file_format: str,
        description_prefix: str,
    ) -> List[FileRecord]:
        _ = file_format
        return beam_input_staging.format_specific_output_records(
            file_stem=file_stem,
            file_path=file_path,
            description_prefix=description_prefix,
            state=self.state,
        )

    def _handle_linkstats(
        self, workspace: "Workspace", previous_beam_records: List, store: RecordStore
    ):
        beam_input_staging.handle_linkstats(
            workspace=workspace,
            previous_beam_records=previous_beam_records,
            store=store,
            state=self.state,
            settings=self.settings,
        )

    def _validate_population_consistency(self, workspace: "Workspace") -> None:
        beam_input_staging.validate_population_consistency(
            workspace=workspace,
            settings=self.settings,
            resolve_beam_exchange_scenario_folder_fn=self._resolve_beam_exchange_scenario_folder,
        )

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
            beam_settings = self.settings.beam
            if beam_settings is None:
                logger.warning(
                    "BEAM config is missing; cannot resolve parameter '%s'.",
                    param,
                )
                return
            if pydantic_path == "beam.sample":
                config_value = beam_settings.sample
            elif pydantic_path == "beam.replanning_portion":
                config_value = beam_settings.replanning_portion
            elif pydantic_path == "beam.max_plans_memory":
                config_value = beam_settings.max_plans_memory
            elif pydantic_path == "beam.skim_zone_geoid_col":
                config_value = beam_settings.skim_zone_geoid_col
            else:
                logger.warning(
                    "Unsupported BEAM Pydantic path '%s' for parameter '%s'.",
                    pydantic_path,
                    param,
                )
                return

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

        beam_config_path = beam_primary_config_path(
            self.settings,
            workspace_path=root,
        )

        if not beam_config_path.exists():
            logger.warning(
                f"[BEAM Preprocessor] BEAM config file does not exist: {beam_config_path}"
            )
            return

        try:
            changed = update_staged_beam_config_value(
                beam_config_path,
                key=config_header,
                value=config_value,
                env_overrides=beam_config_env_overrides(
                    self.settings,
                    workspace_path=root,
                ),
            )
        except BeamConfigHoconError:
            logger.error(
                "[BEAM Preprocessor] Failed to update staged BEAM config key %s in %s",
                config_header,
                beam_config_path,
                exc_info=True,
            )
            raise

        if not changed:
            logger.info(
                "[BEAM Preprocessor] Config already up to date for %s in %s",
                config_header,
                beam_config_path,
            )
            return

        logger.info(
            f"[BEAM Preprocessor] Updated config {config_header} to {config_value} in {beam_config_path}"
        )
