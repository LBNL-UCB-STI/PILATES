#!/usr/bin/env python
"""
Migrate legacy PILATES configuration files to new hierarchical structure.

Usage:
    python scripts/migrate_config.py old_config.yaml new_config.yaml [--validate]

This script converts flat configuration files to the new nested structure
designed for better configuration hashing and provenance tracking.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

# Add pilates to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pilates.config.models import PilatesConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ZONE_DEFAULTS = {
    "seattle": {
        "source_file": "pilates/activitysim/data/seattle/block_groups_seattle_4326.geojson",
        "fallback_source_files": [
            "pilates/beam/production/seattle/shape/block-groups-32048.shp"
        ],
        "canonical_id_col": "OBJECTID",
        "activitysim_index_col": "TAZ",
    },
    "sfbay": {
        "source_file": "pilates/activitysim/data/sfbay/taz_sfbay.geojson",
        "fallback_source_files": [],
        "canonical_id_col": "taz1454",
        "activitysim_index_col": "TAZ",
    },
}


class ConfigMigrator:
    """
    Migrates legacy PILATES configurations to new hierarchical structure.
    """

    def __init__(self, legacy_config: Dict[str, Any]):
        """
        Initialize migrator with legacy config.

        Args:
            legacy_config: Legacy configuration dictionary
        """
        self.legacy = legacy_config
        self.warnings: list[str] = []

    def migrate(self) -> Dict[str, Any]:
        """
        Perform migration from legacy to new format.

        Returns:
            New configuration dictionary
        """
        logger.info("Starting configuration migration...")

        new_config = {
            "run": self._migrate_run_config(),
            "shared": self._migrate_shared_config(),
            "infrastructure": self._migrate_infrastructure_config(),
            "postprocessing": self._migrate_postprocessing_config(),
        }

        # Add model-specific configs (only if models are enabled)
        new_config["urbansim"] = self._migrate_urbansim_config()

        if self.legacy.get("vehicle_ownership_model"):
            new_config["atlas"] = self._migrate_atlas_config()

        if self.legacy.get("activity_demand_model"):
            new_config["activitysim"] = self._migrate_activitysim_config()

        if self.legacy.get("travel_model"):
            new_config["beam"] = self._migrate_beam_config()

        logger.info(f"Migration complete with {len(self.warnings)} warnings")
        return new_config

    def _migrate_run_config(self) -> Dict[str, Any]:
        """Migrate run-level configuration."""
        logger.info("Migrating run configuration...")

        return {
            "region": self.legacy["region"],
            "scenario": self.legacy.get("scenario", "base"),
            "start_year": self.legacy["start_year"],
            "end_year": self.legacy["end_year"],
            "land_use_freq": self.legacy.get("land_use_freq", 1),
            "travel_model_freq": self.legacy.get("travel_model_freq", 1),
            "vehicle_ownership_freq": self.legacy.get("vehicle_ownership_freq", 1),
            "supply_demand_iters": self.legacy.get("supply_demand_iters", 6),
            "output_directory": self.legacy.get("output_directory", "./output"),
            "output_run_name": self.legacy.get("output_run_name", "pilates_run"),
            "consist_db_local_run": self.legacy.get("consist_db_local_run", True),
            "consist_db_filename": self.legacy.get(
                "consist_db_filename", "provenance.duckdb"
            ),
            "models": {
                "land_use": self.legacy.get("land_use_model") or None,
                "travel": self.legacy.get("travel_model") or None,
                "activity_demand": self.legacy.get("activity_demand_model") or None,
                "vehicle_ownership": self.legacy.get("vehicle_ownership_model") or None,
            },
        }

    def _migrate_shared_config(self) -> Dict[str, Any]:
        """Migrate shared configuration."""
        logger.info("Migrating shared configuration...")

        # Geography
        region = self.legacy["region"]
        fips_data = self.legacy.get("FIPS", {}).get(region, {})

        geography = {
            "FIPS": fips_data,
            "local_crs": self.legacy.get("local_crs", {}).get(region, "EPSG:4326"),
        }
        zone_defaults = ZONE_DEFAULTS.get(region)
        skims_zone_type = self.legacy.get("skims_zone_type")
        if zone_defaults and skims_zone_type:
            geography["zones"] = {
                "zone_type": skims_zone_type,
                "source_file": zone_defaults["source_file"],
                "fallback_source_files": list(
                    zone_defaults.get("fallback_source_files", [])
                ),
                "canonical_id_col": zone_defaults["canonical_id_col"],
                "activitysim_index_col": zone_defaults["activitysim_index_col"],
            }

        # Skims
        skims = {
            "zone_type": self.legacy.get("skims_zone_type", "taz"),
            "fname": self.legacy.get("skims_fname", "skims.omx"),
            "origin_fname": self.legacy.get("origin_skims_fname"),
            "geoms_fname": self.legacy.get("beam_geoms_fname", "geoms.csv"),
            "geoms_index_col": self.legacy.get("geoms_index_col", "GEOID"),
            "hwy_paths": self.legacy.get("hwy_paths", []),
            "periods": self.legacy.get("periods", []),
            "transit_paths": self.legacy.get("transit_paths"),
        }

        # Database
        database_config = self.legacy.get("database", {})
        default_database_path = f"pilates/database/{region}_pilates_data.duckdb"
        database = {
            "enabled": database_config.get("enabled", True),
            "type": database_config.get("type", "duckdb"),
            "path": database_config.get("path", default_database_path),
        }

        return {
            "geography": geography,
            "skims": skims,
            "database": database,
        }

    def _migrate_infrastructure_config(self) -> Dict[str, Any]:
        """Migrate infrastructure configuration."""
        logger.info("Migrating infrastructure configuration...")

        # Filter out None values from image dicts
        singularity_images = {
            k: v
            for k, v in self.legacy.get("singularity_images", {}).items()
            if v is not None
        }
        docker_images = {
            k: v
            for k, v in self.legacy.get("docker_images", {}).items()
            if v is not None
        }

        return {
            "container_manager": self.legacy.get("container_manager", "docker"),
            "singularity_images": singularity_images,
            "docker_images": docker_images,
            "docker_config": {
                "stdout": self.legacy.get("docker_stdout", False),
                "pull_latest": self.legacy.get("pull_latest", False),
            },
        }

    def _migrate_urbansim_config(self) -> Dict[str, Any]:
        """Migrate UrbanSim configuration."""
        logger.info("Migrating UrbanSim configuration...")

        region = self.legacy["region"]
        region_to_id = self.legacy.get("region_to_region_id", {})

        # Ensure all region IDs are strings (for proper YAML quoting)
        region_id = region_to_id.get(region)
        if region_id and not isinstance(region_id, str):
            region_id = str(region_id)

        # Ensure all region_to_region_id values are strings
        region_mappings = {}
        for k, v in self.legacy.get("region_to_region_id", {}).items():
            region_mappings[k] = str(v) if v is not None else v

        return {
            "region_id": region_id,
            "local_data_input_folder": self.legacy.get(
                "usim_local_data_input_folder", "pilates/urbansim/data/"
            ),
            "local_mutable_data_folder": self.legacy.get(
                "usim_local_mutable_data_folder", "urbansim/data/"
            ),
            "client_base_folder": self.legacy.get(
                "usim_client_base_folder", "/base/demos_urbansim"
            ),
            "client_data_folder": self.legacy.get(
                "usim_client_data_folder", "/base/demos_urbansim/data/"
            ),
            "input_file_template": self.legacy.get(
                "usim_formattable_input_file_name",
                "custom_mpo_{region_id}_model_data.h5",
            ),
            "input_file_template_year": self.legacy.get(
                "usim_formattable_input_file_name_year",
                "custom_mpo_{region_id}_model_data_{start_year}.h5",
            ),
            "output_file_template": self.legacy.get(
                "usim_formattable_output_file_name", "model_data_{year}.h5"
            ),
            "command_template": self.legacy.get(
                "usim_formattable_command", "-r {0} -i {1} -y {2} -f {3} -t {4}"
            ),
            "region_mappings": {"region_to_region_id": region_mappings},
        }

    def _migrate_atlas_config(self) -> Dict[str, Any]:
        """Migrate ATLAS configuration."""
        logger.info("Migrating ATLAS configuration...")

        return {
            "host_input_folder": self.legacy.get(
                "atlas_host_input_folder", "pilates/atlas/atlas_input"
            ),
            "warmstart_input_folder": self.legacy.get(
                "atlas_warmstart_input_folder", "pilates/atlas/atlas_input"
            ),
            "host_mutable_input_folder": self.legacy.get(
                "atlas_host_mutable_input_folder", "atlas/atlas_input"
            ),
            "host_output_folder": self.legacy.get(
                "atlas_host_output_folder", "atlas/atlas_output"
            ),
            "container_input_folder": self.legacy.get(
                "atlas_container_input_folder", "/atlas_input"
            ),
            "container_output_folder": self.legacy.get(
                "atlas_container_output_folder", "/atlas_output"
            ),
            "basedir": self.legacy.get("atlas_basedir", "/"),
            "codedir": self.legacy.get("atlas_codedir", "/"),
            "sample_size": self.legacy.get("atlas_sample_size", 0),
            "num_processes": self.legacy.get("atlas_num_processes", 40),
            "beamac": self.legacy.get("atlas_beamac", 0),
            "mod": self.legacy.get("atlas_mod", 2),
            "scenario": self.legacy.get("atlas_scenario", "baseline"),
            "adscen": self.legacy.get("atlas_adscen", "baseline"),
            "rebfactor": self.legacy.get("atlas_rebfactor", 1),
            "taxfactor": self.legacy.get("atlas_taxfactor", 1),
            "discIncent": self.legacy.get("atlas_discIncent", 0),
            "command_template": self.legacy.get("atlas_formattable_command", ""),
        }

    def _migrate_activitysim_config(self) -> Dict[str, Any]:
        """Migrate ActivitySim configuration."""
        logger.info("Migrating ActivitySim configuration...")

        region = self.legacy["region"]
        region_to_subdir = self.legacy.get("region_to_asim_subdir", {})
        region_to_bucket = self.legacy.get("region_to_asim_bucket", {})

        # Database config
        asim_db_config = self.legacy.get("activitysim_database", {})

        return {
            "household_sample_size": self.legacy.get("household_sample_size", 0),
            "chunk_size": self.legacy.get("chunk_size", 12_000_000_000),
            "num_processes": self.legacy.get("num_processes", 25),
            "file_format": self.legacy.get("file_format", "parquet"),
            "local_input_folder": self.legacy.get(
                "asim_local_input_folder", "pilates/activitysim/data/"
            ),
            "local_mutable_data_folder": self.legacy.get(
                "asim_local_mutable_data_folder", "activitysim/data/"
            ),
            "local_output_folder": self.legacy.get(
                "asim_local_output_folder", "activitysim/output/"
            ),
            "local_configs_folder": self.legacy.get(
                "asim_local_configs_folder", "pilates/activitysim/configs/"
            ),
            "local_mutable_configs_folder": self.legacy.get(
                "asim_local_mutable_configs_folder", "activitysim/configs/"
            ),
            "validation_folder": self.legacy.get(
                "asim_validation_folder", "pilates/activitysim/validation"
            ),
            "subdir": self.legacy.get("asim_subdir", "configs"),
            "main_configs_dir": self.legacy.get(
                "asim_main_configs_dir", "configs_extended"
            ),
            "region_mappings": {
                "region_to_subdir": region_to_subdir,
                "region_to_bucket": region_to_bucket,
            },
            "from_urbansim_col_maps": self.legacy.get("asim_from_usim_col_maps", {}),
            "to_urbansim_col_maps": self.legacy.get("asim_to_usim_col_maps", {}),
            "output_tables": self.legacy.get("asim_output_tables", {}),
            "command_template": self.legacy.get("asim_formattable_command", ""),
            "database": {
                "enabled": asim_db_config.get("enabled", False),
                "use_processed_data": asim_db_config.get("use_processed_data", True),
                "year": asim_db_config.get("year"),
            },
            "warm_start_activities": self.legacy.get("warm_start_activities", False),
            "replan_iters": self.legacy.get("replan_iters", 0),
            "replan_hh_samp_size": self.legacy.get("replan_hh_samp_size", 0),
            "replan_after": self.legacy.get(
                "replan_after", "non_mandatory_tour_scheduling"
            ),
            "final_plans_folder": self.legacy.get(
                "final_asim_plans_folder", "pilates/activitysim/output/final_plans"
            ),
        }

    def _migrate_beam_config(self) -> Dict[str, Any]:
        """Migrate BEAM configuration."""
        logger.info("Migrating BEAM configuration...")

        beam_config = {
            "config": self.legacy.get("beam_config", "beam.conf"),
            "sample": self.legacy.get("beam_sample", 1.0),
            "replanning_portion": self.legacy.get("beam_replanning_portion", 0.4),
            "memory": self.legacy.get("beam_memory", "180g"),
            "local_input_folder": self.legacy.get(
                "beam_local_input_folder", "pilates/beam/production/"
            ),
            "local_mutable_data_folder": self.legacy.get(
                "beam_local_mutable_data_folder", "beam/input/"
            ),
            "local_output_folder": self.legacy.get(
                "beam_local_output_folder", "beam/beam_output/"
            ),
            "scenario_folder": self.legacy.get("beam_scenario_folder", "urbansim/"),
            "router_directory": self.legacy.get("beam_router_directory", "r5/network"),
            "skims_shapefile": self.legacy.get(
                "beam_skims_shapefile", "shape/zones.shp"
            ),
            "skim_zone_source_id_col": self.legacy.get(
                "skim_zone_source_id_col", "sk_zone"
            ),
            "skim_zone_geoid_col": self.legacy.get("skim_zone_geoid_col", "geoid10"),
            "discard_plans_every_year": self.legacy.get(
                "discard_plans_every_year", False
            ),
            "max_plans_memory": self.legacy.get("max_plans_memory", 5),
            "simulated_hwy_paths": self.legacy.get("beam_simulated_hwy_paths", []),
            "asim_hwy_measure_map": self.legacy.get("beam_asim_hwy_measure_map", {}),
            "asim_transit_measure_map": self.legacy.get(
                "beam_asim_transit_measure_map", {}
            ),
            "asim_ridehail_measure_map": self.legacy.get(
                "beam_asim_ridehail_measure_map", {}
            ),
            "ridehail_path_map": self.legacy.get("ridehail_path_map", {}),
        }
        if self.legacy.get("beam_full_skim_run_schedule") is not None:
            logger.info("Migrating BEAM full_skim configuration...")

            modes_to_build: Dict[str, bool] = {}
            if self.legacy.get("beam_full_skim_modes_drive") is not None:
                modes_to_build["drive"] = self.legacy.get(
                    "beam_full_skim_modes_drive"
                )
            if self.legacy.get("beam_full_skim_modes_walk") is not None:
                modes_to_build["walk"] = self.legacy.get("beam_full_skim_modes_walk")
            if self.legacy.get("beam_full_skim_modes_transit") is not None:
                modes_to_build["transit"] = self.legacy.get(
                    "beam_full_skim_modes_transit"
                )
            if not modes_to_build:
                modes_to_build = {"drive": True, "walk": False, "transit": False}

            skim_key_prefix = "beam_full_skim"
            full_skim_config: Dict[str, Any] = {
                "run_schedule": self.legacy.get(
                    f"{skim_key_prefix}_run_schedule", "standalone"
                ),
                "router_type": self.legacy.get(f"{skim_key_prefix}_router_type", "r5+gh"),
                "skims_geo_type": self.legacy.get(
                    f"{skim_key_prefix}_skims_geo_type", "taz"
                ),
                "skims_kind": self.legacy.get(f"{skim_key_prefix}_skims_kind", "od"),
                "peak_hours": self.legacy.get(f"{skim_key_prefix}_peak_hours", [8.5]),
                "modes_to_build": modes_to_build,
            }

            parallelism_value = None
            if "beam_full_skim_parallelism_thread_ratio" in self.legacy:
                parallelism_value = self.legacy.get(
                    "beam_full_skim_parallelism_thread_ratio"
                )
            elif "beam_full_skim_parallelism_thread_pct" in self.legacy:
                parallelism_value = self.legacy.get(
                    "beam_full_skim_parallelism_thread_pct"
                )
            elif "beam_full_skim_parallelism" in self.legacy:
                parallelism_value = self.legacy.get("beam_full_skim_parallelism")

            if parallelism_value is not None:
                if parallelism_value <= 1.0:
                    full_skim_config["parallelism_thread_ratio"] = parallelism_value
                else:
                    full_skim_config["parallelism_thread_ratio"] = (
                        parallelism_value / 100.0
                    )

            beam_config["full_skim"] = full_skim_config

        return beam_config

        # Migrate full_skim configuration if present
        if self.legacy.get("beam_full_skim_run_schedule") is not None:
            logger.info("Migrating BEAM full_skim configuration...")

            # Build modes_to_build dict from individual mode flags
            modes_to_build = {}
            if self.legacy.get("beam_full_skim_modes_drive") is not None:
                modes_to_build["drive"] = self.legacy.get("beam_full_skim_modes_drive")
            if self.legacy.get("beam_full_skim_modes_walk") is not None:
                modes_to_build["walk"] = self.legacy.get("beam_full_skim_modes_walk")
            if self.legacy.get("beam_full_skim_modes_transit") is not None:
                modes_to_build["transit"] = self.legacy.get("beam_full_skim_modes_transit")

            # Default to drive-only if no modes specified
            if not modes_to_build:
                modes_to_build = {"drive": True, "walk": False, "transit": False}

            skim_key_prefix = "beam_full_skim"
            full_skim_config = {
                "run_schedule": self.legacy.get(
                    f"{skim_key_prefix}_run_schedule", "standalone"
                ),
                "router_type": self.legacy.get(f"{skim_key_prefix}_router_type", "r5+gh"),
                "skims_geo_type": self.legacy.get(f"{skim_key_prefix}_skims_geo_type", "taz"),
                "skims_kind": self.legacy.get(f"{skim_key_prefix}_skims_kind", "od"),
                "peak_hours": self.legacy.get(f"{skim_key_prefix}_peak_hours", [8.5]),
                "modes_to_build": modes_to_build,
            }

            # Add parallelism_thread_ratio only if explicitly set (otherwise auto-calculate at 0.8)
            parallelism_value = None
            if "beam_full_skim_parallelism_thread_ratio" in self.legacy:
                parallelism_value = self.legacy.get("beam_full_skim_parallelism_thread_ratio")
            elif "beam_full_skim_parallelism_thread_pct" in self.legacy:
                parallelism_value = self.legacy.get("beam_full_skim_parallelism_thread_pct")
            elif "beam_full_skim_parallelism" in self.legacy:
                parallelism_value = self.legacy.get("beam_full_skim_parallelism")

            if parallelism_value is not None:
                # Interpret <=1.0 as ratio, otherwise treat as percent
                if parallelism_value <= 1.0:
                    full_skim_config["parallelism_thread_ratio"] = parallelism_value
                else:
                    full_skim_config["parallelism_thread_ratio"] = parallelism_value / 100.0

            beam_config["full_skim"] = full_skim_config

        return beam_config

    def _migrate_postprocessing_config(self) -> Dict[str, Any]:
        """Migrate postprocessing configuration."""
        logger.info("Migrating postprocessing configuration...")

        return {
            "output_folder": self.legacy.get(
                "postprocessing_output_folder", "pilates/postprocessing/output"
            ),
            "mep_output_folder": self.legacy.get(
                "mep_local_output_folder", "pilates/postprocessing/MEP/"
            ),
            "scenario_definitions": self.legacy.get("scenario_definitions", {}),
            "validation_metrics": self.legacy.get("validation_metrics", {}),
        }

    def get_warnings(self) -> list[str]:
        """Get list of migration warnings."""
        return self.warnings


def migrate_config_file(
    input_path: str, output_path: str, validate: bool = True
) -> bool:
    """
    Migrate a configuration file from legacy to new format.

    Args:
        input_path: Path to legacy config file
        output_path: Path to write new config file
        validate: Whether to validate the new config with Pydantic

    Returns:
        True if migration successful, False otherwise
    """
    try:
        # Load legacy config
        logger.info(f"Loading legacy config from: {input_path}")
        with open(input_path, "r") as f:
            legacy_config = yaml.safe_load(f)

        # Migrate
        migrator = ConfigMigrator(legacy_config)
        new_config = migrator.migrate()

        # Validate if requested
        if validate:
            logger.info("Validating new configuration...")
            try:
                validated = PilatesConfig(**new_config)
                logger.info("✓ Configuration validation passed!")
            except Exception as e:
                logger.error(f"✗ Configuration validation failed: {e}")
                logger.info(
                    "Writing config anyway (use --no-validate to skip validation)"
                )
                # Continue to write even if validation fails

        # Write new config with proper string quoting and comments
        logger.info(f"Writing new config to: {output_path}")

        # Custom YAML representer to ensure strings with { } or numbers are quoted
        def str_representer(dumper, data):
            # Always quote if contains curly braces or looks like a number
            if "{" in data or "}" in data or data.isdigit():
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer)

        # Write with comments
        with open(output_path, "w") as f:
            # Header
            f.write("# PILATES Configuration File (Hierarchical Format)\n")
            f.write("# Generated by scripts/migrate_config.py\n")
            f.write("# See docs/config_restructure_proposal.md for details\n\n")

            # Run configuration
            f.write(
                "# =============================================================================\n"
            )
            f.write("# RUN CONFIGURATION\n")
            f.write("# Core run parameters and model selection\n")
            f.write(
                "# =============================================================================\n"
            )
            yaml.dump(
                {"run": new_config["run"]},
                f,
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )
            f.write("\n")

            # Shared configuration
            f.write(
                "# =============================================================================\n"
            )
            f.write("# SHARED CONFIGURATION\n")
            f.write("# Geography, skims, and database settings shared across models\n")
            f.write(
                "# =============================================================================\n"
            )
            yaml.dump(
                {"shared": new_config["shared"]},
                f,
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )
            f.write("\n")

            # Infrastructure
            f.write(
                "# =============================================================================\n"
            )
            f.write("# INFRASTRUCTURE CONFIGURATION\n")
            f.write(
                "# Container management (Docker/Singularity) and execution environment\n"
            )
            f.write(
                "# =============================================================================\n"
            )
            yaml.dump(
                {"infrastructure": new_config["infrastructure"]},
                f,
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )
            f.write("\n")

            # Model-specific configs
            if "urbansim" in new_config:
                f.write(
                    "# =============================================================================\n"
                )
                f.write("# URBANSIM CONFIGURATION\n")
                f.write("# Land use model settings\n")
                f.write(
                    "# =============================================================================\n"
                )
                yaml.dump(
                    {"urbansim": new_config["urbansim"]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    width=120,
                )
                f.write("\n")

            if "atlas" in new_config:
                f.write(
                    "# =============================================================================\n"
                )
                f.write("# ATLAS CONFIGURATION\n")
                f.write("# Vehicle fleet microsimulation settings\n")
                f.write(
                    "# =============================================================================\n"
                )
                yaml.dump(
                    {"atlas": new_config["atlas"]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    width=120,
                )
                f.write("\n")

            if "activitysim" in new_config:
                f.write(
                    "# =============================================================================\n"
                )
                f.write("# ACTIVITYSIM CONFIGURATION\n")
                f.write("# Activity-based travel demand model settings\n")
                f.write(
                    "# =============================================================================\n"
                )
                yaml.dump(
                    {"activitysim": new_config["activitysim"]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    width=120,
                )
                f.write("\n")

            if "beam" in new_config:
                f.write(
                    "# =============================================================================\n"
                )
                f.write("# BEAM CONFIGURATION\n")
                f.write("# Transportation network simulation settings\n")
                f.write(
                    "# =============================================================================\n"
                )
                yaml.dump(
                    {"beam": new_config["beam"]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    width=120,
                )
                f.write("\n")

            # Postprocessing
            f.write(
                "# =============================================================================\n"
            )
            f.write("# POSTPROCESSING CONFIGURATION\n")
            f.write("# Output processing and validation settings\n")
            f.write(
                "# =============================================================================\n"
            )
            yaml.dump(
                {"postprocessing": new_config["postprocessing"]},
                f,
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )

        # Report warnings
        warnings = migrator.get_warnings()
        if warnings:
            logger.warning(f"Migration completed with {len(warnings)} warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("✓ Migration completed successfully with no warnings!")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate PILATES configuration to new hierarchical structure"
    )
    parser.add_argument("input", help="Path to legacy configuration file")
    parser.add_argument("output", help="Path to write new configuration file")
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate new config with Pydantic (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip Pydantic validation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = migrate_config_file(args.input, args.output, args.validate)

    sys.exit(0 if success else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
