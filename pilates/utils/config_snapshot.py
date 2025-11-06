"""
Configuration snapshot management for PILATES.

This module provides utilities for capturing complete configuration state
including git hashes, config file contents, and PILATES settings for
reproducibility and database upload purposes.

Supports hierarchical configuration hashing for intelligent caching and
output reuse across runs.
"""

import hashlib
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import glob

from pilates.config import PilatesConfig
from pilates.generic.config_hashing import ConfigHasher
from pilates.config.schema import get_field_annotations, get_dependency_graph
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


class ConfigSnapshotManager:
    """
    Manages creation and storage of configuration snapshots for PILATES runs.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.project_root = self._find_project_root()

    def _find_project_root(self) -> str:
        """Find the PILATES project root directory."""
        current = Path(self.workspace_path).resolve()
        while current != current.parent:
            if (current / "pilates").exists() and (current / "run.py").exists():
                return str(current)
            current = current.parent
        # Fallback to workspace path if project root not found
        return self.workspace_path

    def get_git_hash(self, repo_path: str) -> Optional[str]:
        """Get git commit hash for a repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"Could not get git hash for {repo_path}")
            return None

    def _get_config_file_content(self, file_path: str) -> Optional[str]:
        """Read and return the content of a configuration file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not read config file {file_path}: {e}")
        return None

    def _collect_beam_configs(self, settings: PilatesConfig) -> Dict[str, str]:
        """Collect BEAM configuration file contents."""
        configs = {}

        # Main BEAM config file
        if "beam_config" in settings:
            region = settings.run.region
            beam_config_path = os.path.join(
                self.project_root,
                "pilates",
                "beam",
                "production",
                region,
                settings.beam.config,
            )
            content = self._get_config_file_content(beam_config_path)
            if content:
                configs["beam_main_config"] = content
                configs["beam_config_path"] = beam_config_path

        # Common BEAM configs
        common_config_dir = os.path.join(
            self.project_root, "pilates", "beam", "production", "common"
        )
        if os.path.exists(common_config_dir):
            for config_file in ["akka.conf", "matsim.conf", "metrics.conf"]:
                config_path = os.path.join(common_config_dir, config_file)
                content = self._get_config_file_content(config_path)
                if content:
                    configs[f"beam_common_{config_file}"] = content

        return configs

    def _collect_activitysim_configs(self, settings: PilatesConfig) -> Dict[str, str]:
        """Collect ActivitySim configuration file contents."""
        configs = {}

        # ActivitySim configs directory
        region = settings.run.region
        region_to_subdir = settings.activitysim.region_mappings
        asim_subdir = region_to_subdir.get(region, region)

        asim_config_dir = os.path.join(
            self.project_root, "pilates", "activitysim", "configs", asim_subdir
        )

        if os.path.exists(asim_config_dir):
            # Collect all YAML files in the config directory
            yaml_files = glob.glob(
                os.path.join(asim_config_dir, "**", "*.yaml"), recursive=True
            )
            yaml_files.extend(
                glob.glob(os.path.join(asim_config_dir, "**", "*.yml"), recursive=True)
            )

            for yaml_file in sorted(yaml_files):
                relative_path = os.path.relpath(yaml_file, asim_config_dir)
                content = self._get_config_file_content(yaml_file)
                if content:
                    # Use relative path as key to maintain structure
                    configs[
                        f"asim_{relative_path.replace('/', '_').replace('.', '_')}"
                    ] = content

        return configs

    def _collect_urbansim_configs(self, settings: PilatesConfig) -> Dict[str, str]:
        """Collect UrbanSim configuration file contents."""
        configs = {}

        # UrbanSim doesn't have explicit config files in the current structure,
        # but we can capture any relevant files if they exist
        usim_data_dir = os.path.join(self.project_root, "pilates", "urbansim", "data")
        if os.path.exists(usim_data_dir):
            # Look for any config-like files
            for ext in ["*.yaml", "*.yml", "*.json", "*.conf"]:
                config_files = glob.glob(os.path.join(usim_data_dir, ext))
                for config_file in config_files:
                    filename = os.path.basename(config_file)
                    content = self._get_config_file_content(config_file)
                    if content:
                        configs[f"usim_{filename}"] = content

        return configs

    def _collect_atlas_configs(self, settings: PilatesConfig) -> Dict[str, str]:
        """Collect ATLAS configuration file contents."""
        configs = {}

        # ATLAS input directory might contain config files
        atlas_input_dir = os.path.join(
            self.project_root, "pilates", "atlas", "atlas_input"
        )
        if os.path.exists(atlas_input_dir):
            # Look for config-like files
            for ext in ["*.yaml", "*.yml", "*.json", "*.conf", "*.R", "*.csv"]:
                config_files = glob.glob(os.path.join(atlas_input_dir, ext))
                for config_file in config_files:
                    filename = os.path.basename(config_file)
                    # Only include small files (likely configs, not data)
                    if os.path.getsize(config_file) < 1024 * 1024:  # 1MB limit
                        content = self._get_config_file_content(config_file)
                        if content:
                            configs[f"atlas_{filename}"] = content

        return configs

    def extract_relevant_pilates_settings(
        self, settings: PilatesConfig
    ) -> Dict[str, Any]:
        """
        Extract PILATES settings that affect model behavior.

        This method filters out runtime-specific settings and keeps only
        those that affect reproducibility.
        """
        # Settings that affect model behavior and should be tracked
        relevant_keys = [
            # Core simulation parameters
            "region",
            "start_year",
            "end_year",
            "land_use_freq",
            "travel_model_freq",
            "vehicle_ownership_freq",
            # Model selection
            "land_use_model",
            "travel_model",
            "activity_demand_model",
            "vehicle_ownership_model",
            # Model-specific configs
            "beam_config",
            "beam_sample",
            "beam_replanning_portion",
            "household_sample_size",
            "chunk_size",
            # Skim and routing settings
            "skims_zone_type",
            "skims_fname",
            "beam_router_directory",
            "beam_geoms_fname",
            "geoms_index_col",
            # Region mappings
            "region_to_region_id",
            "region_to_asim_subdir",
            "region_to_asim_bucket",
            # ATLAS settings
            "atlas_num_processes",
            "atlas_sample_size",
            "atlas_scenario",
            "atlas_mod",
            "atlas_beamac",
            # ActivitySim settings
            "periods",
            "transit_paths",
            "hwy_paths",
            # Geographic settings
            "FIPS",
            "local_crs",
        ]

        relevant_settings = {}
        for key in relevant_keys:
            if get_setting(settings, key) is not None:
                relevant_settings[key] = get_setting(settings, key)

        return relevant_settings

    def create_config_snapshot(self, settings: PilatesConfig) -> Dict[str, Any]:
        """
        Create a complete configuration snapshot.

        Returns a dictionary containing all configuration information
        needed for reproducibility and database upload.
        """
        snapshot_id = str(uuid.uuid4())

        # Collect git hashes
        git_hashes = {
            "pilates_main": self.get_git_hash(self.project_root),
            "beam_configs": self.get_git_hash(
                os.path.join(self.project_root, "pilates", "beam")
            ),
            "asim_configs": self.get_git_hash(
                os.path.join(self.project_root, "pilates", "activitysim")
            ),
            "usim_configs": self.get_git_hash(
                os.path.join(self.project_root, "pilates", "urbansim")
            ),
            "atlas_configs": self.get_git_hash(
                os.path.join(self.project_root, "pilates", "atlas")
            ),
        }

        # Collect all config file contents
        all_config_files = {}
        all_config_files.update(self._collect_beam_configs(settings))
        all_config_files.update(self._collect_activitysim_configs(settings))
        all_config_files.update(self._collect_urbansim_configs(settings))
        all_config_files.update(self._collect_atlas_configs(settings))

        # Extract relevant PILATES settings
        relevant_settings = self.extract_relevant_pilates_settings(settings)

        # Create overall config content hash
        all_content = {
            "git_hashes": git_hashes,
            "config_files": all_config_files,
            "pilates_settings": relevant_settings,
        }
        config_content_hash = hashlib.sha256(
            json.dumps(all_content, sort_keys=True).encode()
        ).hexdigest()

        snapshot = {
            "snapshot_id": snapshot_id,
            "created_timestamp": datetime.utcnow().isoformat(),
            "config_content_hash": config_content_hash,
            "git_hashes": git_hashes,
            "config_files": all_config_files,
            "pilates_settings": relevant_settings,
            # Specific config references for easy access
            "beam_config": get_setting(settings, "beam.config"),
            "asim_subdir": get_setting(settings, "activitysim.region_mappings", {}).get(get_setting(settings, "run.region"), {}).get("asim_subdir"),
            "region": get_setting(settings, "run.region"),
        }

        logger.info(
            f"Created config snapshot {snapshot_id} with hash {config_content_hash[:8]}"
        )
        return snapshot

    def create_config_hash_for_model(
        self, model_name: str, config_snapshot: Dict[str, Any]
    ) -> str:
        """
        Create a config hash specific to a model.

        This extracts only the configuration that affects the specific model
        and creates a hash for existence queries.
        """
        relevant_config = {}

        if model_name.lower() == "beam":
            relevant_config = {
                "git_hash": config_snapshot["git_hashes"].get("beam_configs"),
                "pilates_git": config_snapshot["git_hashes"].get("pilates_main"),
                "beam_config": config_snapshot.get("beam_config"),
                "beam_sample": config_snapshot["pilates_settings"].get("beam_sample"),
                "beam_replanning": config_snapshot["pilates_settings"].get(
                    "beam_replanning_portion"
                ),
                "config_files": {
                    k: v
                    for k, v in config_snapshot["config_files"].items()
                    if k.startswith("beam_")
                },
            }
        elif model_name.lower() == "activitysim":
            relevant_config = {
                "git_hash": config_snapshot["git_hashes"].get("asim_configs"),
                "pilates_git": config_snapshot["git_hashes"].get("pilates_main"),
                "asim_subdir": config_snapshot.get("asim_subdir"),
                "household_sample": config_snapshot["pilates_settings"].get(
                    "household_sample_size"
                ),
                "chunk_size": config_snapshot["pilates_settings"].get("chunk_size"),
                "config_files": {
                    k: v
                    for k, v in config_snapshot["config_files"].items()
                    if k.startswith("asim_")
                },
            }
        elif model_name.lower() == "urbansim":
            relevant_config = {
                "git_hash": config_snapshot["git_hashes"].get("usim_configs"),
                "pilates_git": config_snapshot["git_hashes"].get("pilates_main"),
                "region": config_snapshot.get("region"),
                "config_files": {
                    k: v
                    for k, v in config_snapshot["config_files"].items()
                    if k.startswith("usim_")
                },
            }
        elif model_name.lower() == "atlas":
            relevant_config = {
                "git_hash": config_snapshot["git_hashes"].get("atlas_configs"),
                "pilates_git": config_snapshot["git_hashes"].get("pilates_main"),
                "atlas_settings": {
                    k: v
                    for k, v in config_snapshot["pilates_settings"].items()
                    if k.startswith("atlas_")
                },
                "config_files": {
                    k: v
                    for k, v in config_snapshot["config_files"].items()
                    if k.startswith("atlas_")
                },
            }

        # Create hash of relevant config
        config_json = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def create_hierarchical_config_hashes(
        self,
        config_snapshot: Dict[str, Any],
        enabled_models: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create hierarchical config hashes for intelligent caching (Phase 1).

        This method computes separate hashes for:
        - base: Run-level config affecting all models
        - Each enabled model: Model-specific config + upstream dependencies

        Args:
            config_snapshot: Complete config snapshot from create_config_snapshot()
            enabled_models: List of enabled model names (e.g., ['urbansim', 'activitysim', 'beam'])

        Returns:
            Dict mapping model name → {'hash': str, 'config_data': dict}

        Example:
            >>> enabled_models = ['activitysim', 'beam']
            >>> hashes = manager.create_hierarchical_config_hashes(snapshot, enabled_models)
            >>> print(hashes['activitysim']['hash'])
            'a1b2c3d4...'
        """
        # Build config dict suitable for ConfigHasher
        # (using pilates_settings from snapshot)
        config_for_hashing = config_snapshot.get('pilates_settings', {})

        # Get field annotations and dependency graph
        field_annotations = get_field_annotations()
        dependency_graph = get_dependency_graph()

        # Create hasher
        hasher = ConfigHasher(
            config=config_for_hashing,
            field_annotations=field_annotations,
            dependency_graph=dependency_graph
        )

        # Get hierarchical hashes
        hash_results = hasher.get_hierarchical_hashes(enabled_models)

        # Package results with config data for database storage
        result = {}

        for model_name, hash_value in hash_results.items():
            # Extract the config data that was hashed
            if model_name == 'base':
                # Base config: global fields only
                config_data = hasher._extract_fields_by_scope(
                    hasher.field_annotations.get('run', {}).get('hash_scope', 'global')
                )
            else:
                # Model config: extract model-specific section
                config_data = config_for_hashing.get(model_name, {})

            result[model_name] = {
                'hash': hash_value,
                'config_data': config_data,
                'config_type': model_name,
                'model_name': model_name
            }

        logger.info(
            f"Created hierarchical config hashes for {len(result)} layers: "
            f"{', '.join(result.keys())}"
        )

        return result
