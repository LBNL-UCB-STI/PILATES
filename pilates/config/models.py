"""
Pydantic models for PILATES configuration.

This module defines the structure and validation for PILATES configuration files.
The models use the new hierarchical structure designed for better config hashing
and provenance tracking.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# RUN-LEVEL CONFIGURATION
# =============================================================================


class ModelSelection(BaseModel):
    """Which models are enabled for this run."""

    land_use: Optional[str] = Field(
        None, description="Land use model (urbansim, or null to disable)"
    )
    travel: Optional[str] = Field(
        None, description="Travel model (beam, polaris, or null to disable)"
    )
    activity_demand: Optional[str] = Field(
        None, description="Activity demand model (activitysim, or null to disable)"
    )
    vehicle_ownership: Optional[str] = Field(
        None, description="Vehicle ownership model (atlas, or null to disable)"
    )


class RunConfig(BaseModel):
    """Top-level run configuration affecting workflow orchestration."""

    # Core simulation parameters (GLOBAL scope)
    region: str = Field(
        ..., description="Geographic region (seattle, sfbay, austin, etc.)"
    )
    scenario: str = Field(..., description="Scenario name")
    start_year: int = Field(..., description="First simulation year", ge=1900, le=2100)
    end_year: int = Field(..., description="Final simulation year", ge=1900, le=2100)
    use_stubs: bool = Field(
        False,
        description="Whether to substitute stub models for containers -- used for testing",
    )

    # Model execution frequencies (GLOBAL scope)
    land_use_freq: int = Field(1, description="How often land use model runs", ge=0)
    travel_model_freq: int = Field(1, description="How often travel model runs", ge=0)
    vehicle_ownership_freq: int = Field(
        1, description="How often vehicle model runs", ge=0
    )
    supply_demand_iters: int = Field(
        1, description="Supply-demand loop iterations", ge=1
    )

    # Output configuration (METADATA scope)
    output_directory: str = Field(..., description="Where to write outputs")
    output_run_name: str = Field(..., description="Human-readable run name")
    local_workspace_root: Optional[str] = Field(
        None,
        description=(
            "Optional node-local workspace root (defaults to output_directory)"
        ),
    )
    enable_archive_copy: bool = Field(
        False, description="Copy logged outputs to archive root as they are produced"
    )
    bootstrap_cache_enabled: bool = Field(
        True,
        description=(
            "Enable cache probing for the pre-scenario bootstrap initialization phase"
        ),
    )
    restart_rehydrate_mode: Literal["native", "off"] = Field(
        "native",
        description=(
            "Restart hydration mode: native reconstructs completed runs via "
            "Consist materialization, off disables restart hydration"
        ),
    )
    restart_strict: bool = Field(
        False,
        description=(
            "Fail startup if required restart artifacts are still missing after "
            "startup hydration/preflight checks"
        ),
    )
    consist_db_local_run: bool = Field(
        True,
        description=(
            "Store Consist provenance DB in the node-local run directory and mirror it "
            "to the archive run directory at shutdown"
        ),
    )
    consist_db_filename: str = Field(
        "provenance.duckdb",
        description=(
            "Filename to use for the run-local Consist DB when consist_db_local_run "
            "is enabled"
        ),
    )
    consist_db_snapshot_enabled: bool = Field(
        True,
        description="Enable periodic checkpoint snapshots of the run-local Consist DB",
    )
    consist_db_snapshot_interval_seconds: int = Field(
        600,
        description=(
            "Minimum seconds between interval-based Consist DB snapshots at safe points"
        ),
        ge=0,
    )
    consist_db_snapshot_on_outer_iteration: bool = Field(
        True,
        description=(
            "Create a Consist DB snapshot at each supply-demand outer iteration boundary"
        ),
    )
    consist_db_snapshot_keep_last: int = Field(
        3,
        description="Number of historical Consist DB snapshots to retain per run",
        ge=1,
    )
    consist_db_restore_on_start: bool = Field(
        True,
        description=(
            "Restore run-local Consist DB from latest archived snapshot when local DB is "
            "missing at startup"
        ),
    )
    consist_db_restore_strict: bool = Field(
        False,
        description=(
            "Fail startup if Consist DB restore from archive snapshot fails when restore "
            "is enabled"
        ),
    )
    consist_db_seed_from_shared_on_start: bool = Field(
        False,
        description=(
            "When run-local DB mode is enabled and local DB is missing, seed it from "
            "shared.database.path if no run snapshot restore is available"
        ),
    )
    consist_db_seed_strict: bool = Field(
        False,
        description=(
            "Fail startup if seed-from-shared is enabled but seeding the local Consist "
            "DB from shared.database.path fails"
        ),
    )
    consist_code_identity: Optional[
        Literal["repo_git", "callable_module", "callable_source"]
    ] = Field(
        None,
        description=(
            "Optional Consist cache code-identity mode override for step runs. "
            "When omitted, Consist defaults to repo_git."
        ),
    )
    consist_hashing_strategy: Literal["fast", "full"] = Field(
        "fast",
        description=(
            "Consist artifact hashing mode for cache identity: "
            "'fast' uses file metadata (mtime/size), 'full' hashes file content."
        ),
    )

    # Model selection (GLOBAL scope)
    models: ModelSelection = Field(..., description="Which models are enabled")

    @field_validator("output_directory")
    @classmethod
    def expand_env_vars(cls, v):
        """Expand environment variables in output_directory."""
        return os.path.expandvars(v)

    @field_validator("local_workspace_root")
    @classmethod
    def expand_local_workspace_root(cls, v):
        """Expand environment variables in local_workspace_root."""
        if v is None:
            return v
        return os.path.expandvars(v)

    @field_validator("consist_db_filename")
    @classmethod
    def validate_consist_db_filename(cls, v):
        """Require a basename-only filename for local run DB placement."""
        if not v:
            raise ValueError("consist_db_filename must not be empty")
        if os.path.basename(v) != v:
            raise ValueError("consist_db_filename must be a filename, not a path")
        return v

    @field_validator("end_year")
    @classmethod
    def validate_end_year(cls, v, info):
        """Ensure end_year >= start_year."""
        if "start_year" in info.data and v < info.data["start_year"]:
            raise ValueError(
                f"end_year ({v}) must be >= start_year ({info.data['start_year']})"
            )
        return v

    def to_consist_facet(self) -> Dict[str, Any]:
        return self.model_dump()


# =============================================================================
# SHARED CONFIGURATION
# =============================================================================


class GeographyConfig(BaseModel):
    """Geographic configuration used by multiple models."""

    FIPS: Dict[str, Any] = Field(..., description="FIPS codes for study area")
    local_crs: str = Field(
        ..., description="Local coordinate reference system (e.g., EPSG:32048)"
    )
    zones: Optional["ZonesConfig"] = Field(
        None, description="Canonical zone definitions (optional for tests)"
    )
    alternative_zones: Optional["ZoneSourceConfig"] = Field(
        None,
        description="Optional alternative canonical zone source used if primary zones source is unavailable.",
    )


class ZoneSourceConfig(BaseModel):
    """Alternate canonical zone source metadata."""

    zone_type: str = Field(
        ...,
        description="The geographic resolution of zones (e.g., 'taz', 'block_group')",
    )
    source_file: str = Field(
        ..., description="Path to a zone geometry source file."
    )
    canonical_id_col: str = Field(
        ..., description="Column in source_file with the canonical zone ID."
    )
    activitysim_index_col: str = Field(
        "TAZ", description="Column name for ActivitySim's 0-based internal index."
    )
    source_crs: Optional[str] = Field(
        None,
        description=(
            "Optional CRS override or provenance hint for the source file "
            "(e.g., 'EPSG:26910')."
        ),
    )


class ZonesConfig(BaseModel):
    """Canonical zone definition configuration."""

    zone_type: str = Field(
        ...,
        description="The geographic resolution of zones (e.g., 'taz', 'block_group')",
    )
    source_file: Optional[str] = Field(
        None,
        description="User-provided path to the canonical zone geometry source file.",
    )
    canonical_id_col: str = Field(
        ..., description="Column in source_file with the canonical zone ID."
    )
    activitysim_index_col: str = Field(
        "TAZ", description="Column name for ActivitySim's 0-based internal index."
    )
    source_crs: Optional[str] = Field(
        None,
        description=(
            "Optional CRS override or provenance hint for the primary source file "
            "(e.g., 'EPSG:4326')."
        ),
    )


class SkimsConfig(BaseModel):
    """Skim configuration affecting ActivitySim and BEAM."""

    fname: str = Field(..., description="Skim file name")
    origin_fname: Optional[str] = Field(None, description="Origin skim file name")
    hwy_paths: List[str] = Field(default_factory=list, description="Highway path types")
    periods: List[str] = Field(default_factory=list, description="Time periods")
    transit_paths: Optional[Dict[str, List[str]]] = Field(
        None, description="Transit path definitions"
    )


class DatabaseConfig(BaseModel):
    """Database configuration for provenance and data storage."""

    enabled: bool = Field(False, description="Enable database storage")
    type: str = Field("duckdb", description="Database type")
    path: str = Field(..., description="Database file path")
    shapshot_path: Optional[str] = Field(
        None, description="Matrix snapshot file path (legacy misspelling)."
    )
    snapshot_path: Optional[str] = Field(
        None,
        description="Alias for shapshot_path (preferred spelling).",
    )
    use_consist: bool = Field(
        True,
        description=(
            "Deprecated toggle retained for config compatibility. "
            "Consist is mandatory and this value is ignored."
        ),
    )

    @model_validator(mode="after")
    def _coalesce_snapshot_path(self):
        if self.snapshot_path and not self.shapshot_path:
            self.shapshot_path = self.snapshot_path
        return self

    @field_validator("path")
    @classmethod
    def expand_env_vars(cls, v):
        """Expand environment variables in database path."""
        return os.path.expandvars(v)


class SharedConfig(BaseModel):
    """Configuration shared across multiple models."""

    geography: GeographyConfig
    skims: SkimsConfig
    database: DatabaseConfig


# =============================================================================
# INFRASTRUCTURE CONFIGURATION
# =============================================================================


class DockerConfig(BaseModel):
    """Docker-specific settings."""

    stdout: bool = Field(False, description="Show container stdout")
    pull_latest: bool = Field(False, description="Pull latest images before run")


class InfrastructureConfig(BaseModel):
    """Container management and execution environment."""

    container_manager: Literal["docker", "singularity"] = Field(
        "docker", description="Container manager: docker or singularity"
    )
    singularity_images: Dict[str, str] = Field(
        default_factory=dict, description="Singularity image URIs"
    )
    docker_images: Dict[str, str] = Field(
        default_factory=dict, description="Docker image tags"
    )
    docker_config: DockerConfig = Field(
        default_factory=DockerConfig, description="Docker settings"
    )


# =============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================


class UrbanSimConfig(BaseModel):
    """UrbanSim land use model configuration."""

    region_id: Optional[str] = Field(
        None, description="Region ID (computed from region if not provided)"
    )
    local_data_input_folder: str = Field(..., description="Local input data folder")
    local_mutable_data_folder: str = Field(..., description="Mutable data folder")
    client_base_folder: str = Field(..., description="Container base folder")
    client_data_folder: str = Field(..., description="Container data folder")
    input_file_template: str = Field(..., description="Input file naming template")
    input_file_template_year: str = Field(
        ..., description="Year-specific input file template"
    )
    output_file_template: str = Field(..., description="Output file naming template")
    command_template: str = Field(..., description="Command template")
    region_mappings: Dict[str, Any] = Field(
        default_factory=dict, description="Region-specific mappings"
    )


class AtlasConfig(BaseModel):
    """ATLAS vehicle fleet model configuration."""

    host_input_folder: str
    warmstart_input_folder: str
    host_mutable_input_folder: str
    host_output_folder: str
    container_input_folder: str
    container_output_folder: str
    basedir: str
    codedir: str
    max_retries: int = Field(
        3, description="Number of times ATLAS can re-run due to flakiness"
    )
    sample_size: int = Field(0, description="Sample size (0 = full population)")
    num_processes: int = Field(40, description="Number of parallel processes")
    beamac: int = Field(0, description="BEAM accessibility mode")
    mod: int = Field(2, description="Model mode: 1=static, 2=dynamic")
    scenario: str = Field("baseline", description="Scenario name")
    adscen: str = Field("baseline", description="Adoption scenario")
    rebfactor: int = Field(1, description="Rebate incentive: 0=NO, 1=YES")
    taxfactor: int = Field(1, description="Tax credit incentive: 0=NO, 1=YES")
    discIncent: int = Field(0, description="Discount incentive: 0=NO, 1=YES")
    command_template: str = Field(..., description="Command template")


class ActivitySimDatabaseConfig(BaseModel):
    """ActivitySim database input mode configuration."""

    enabled: bool = Field(False, description="Enable database input mode")
    use_processed_data: bool = Field(
        True, description="Use pre-processed data (fast path)"
    )
    year: Optional[int] = Field(None, description="Year for data retrieval")


class ActivitySimConfig(BaseModel):
    """ActivitySim activity-based demand model configuration."""

    household_sample_size: int = Field(
        0, description="Household sample size (0 = full population)"
    )
    chunk_size: int = Field(12_000_000_000, description="Memory chunk size")
    num_processes: int = Field(25, description="Number of parallel processes")
    file_format: Literal["parquet", "csv"] = Field(
        "parquet", description="Output file format"
    )
    persist_sharrow_cache: Optional[bool] = Field(
        None,
        description=(
            "Persist ActivitySim sharrow/numba compile cache directory. "
            "When unset, defaults to enabled for parquet format."
        ),
    )
    local_input_folder: str
    local_mutable_data_folder: str
    local_output_folder: str
    local_configs_folder: str
    local_mutable_configs_folder: str
    validation_folder: str
    clipped_geoms_path: Optional[str] = Field(
        None,
        description="Path to BEAM's clipped zone geometries for constraining activity locations.",
    )
    subdir: str = Field("configs", description="Config subdirectory")
    main_configs_dir: str = Field(
        "configs_extended", description="Main configs directory"
    )
    region_mappings: Dict[str, Any] = Field(
        default_factory=dict, description="Region-specific mappings"
    )
    from_urbansim_col_maps: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="UrbanSim → ActivitySim column mappings"
    )
    to_urbansim_col_maps: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="ActivitySim → UrbanSim column mappings"
    )
    output_tables: Dict[str, Any] = Field(
        default_factory=dict, description="Output table configuration"
    )
    command_template: str = Field(..., description="Command template")
    database: ActivitySimDatabaseConfig = Field(
        default_factory=ActivitySimDatabaseConfig,
        description="Database input configuration",
    )
    warm_start_activities: bool = Field(False, description="Warm start activities")
    replan_iters: int = Field(0, description="Replanning iterations")
    replan_hh_samp_size: int = Field(0, description="Replanning household sample size")
    replan_after: str = Field(
        "non_mandatory_tour_scheduling", description="Replan after this step"
    )
    random_seed: Optional[int] = Field(
        None, description="Base random number generator seed"
    )
    final_plans_folder: str = Field(..., description="Final plans output folder")

    def to_consist_facet(self) -> Dict[str, Any]:
        return {
            "household_sample_size": self.household_sample_size,
            "chunk_size": self.chunk_size,
            "num_processes": self.num_processes,
            "file_format": self.file_format,
            "persist_sharrow_cache": self.persist_sharrow_cache,
            "warm_start_activities": self.warm_start_activities,
            "replan_iters": self.replan_iters,
            "replan_hh_samp_size": self.replan_hh_samp_size,
            "replan_after": self.replan_after,
            "random_seed": self.random_seed,
            "database": self.database.model_dump() if self.database else None,
        }


class FullSkimsCreatorConfig(BaseModel):
    """Configuration for BEAM FullSkimsCreatorApp (full-skim mode)."""

    run_schedule: Literal[
        "standalone",
        "after_each_iteration",
        "after_final_iteration",
        "disabled",
    ] = Field(
        "standalone",
        description=(
            "When to run full-skim. "
            "'standalone' replaces normal BEAM runs, "
            "'after_each_iteration' runs after each BEAM iteration, "
            "'after_final_iteration' runs after the final BEAM iteration, "
            "'disabled' never runs."
        ),
    )
    router_type: str = Field("r5+gh", description="Router: r5, r5+gh, or gh")
    skims_geo_type: str = Field("taz", description="Geography: taz or h3")
    skims_kind: str = Field("od", description="Skim type: od")
    peak_hours: List[float] = Field(
        default_factory=lambda: [8.5], description="Peak hours (8.5 = 8:30 AM)"
    )
    modes_to_build: Dict[str, bool] = Field(
        default_factory=lambda: {"drive": True, "walk": False, "transit": False},
        description="Modes to generate skims for",
    )
    parallelism_thread_ratio: Optional[float] = Field(
        None,
        description=(
            "Ratio of CPU cores to use (0.0-1.0). "
            "Default behavior is auto-compute when unset."
        ),
        ge=0.0,
        le=1.0,
    )


class BeamConfig(BaseModel):
    """BEAM transportation network simulation configuration."""

    config: str = Field(..., description="Main BEAM config file")
    sample: float = Field(1.0, description="Population sample fraction", ge=0.0, le=1.0)
    replanning_portion: float = Field(
        0.4, description="Replanning portion", ge=0.0, le=1.0
    )
    memory: str = Field("180g", description="JVM memory allocation")
    local_input_folder: str
    local_mutable_data_folder: str
    local_output_folder: str
    scenario_folder: str
    router_directory: str
    skims_shapefile: str
    skim_zone_source_id_col: str
    skim_zone_geoid_col: str
    discard_plans_every_year: bool = Field(
        False, description="Discard plans every year"
    )
    max_plans_memory: int = Field(5, description="Max plans in memory")
    simulated_hwy_paths: List[str] = Field(
        default_factory=list, description="Simulated highway paths"
    )
    asim_hwy_measure_map: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Highway measure mappings"
    )
    asim_transit_measure_map: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Transit measure mappings"
    )
    asim_ridehail_measure_map: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Ridehail measure mappings"
    )
    ridehail_path_map: Dict[str, str] = Field(
        default_factory=dict, description="Ridehail path mappings"
    )
    skim_previous_weight: float = Field(
        0.9,
        description=(
            "Weight on previous skims when blending trip counts. "
            "New counts are always fully applied."
        ),
        ge=0.0,
        le=1.0,
    )
    full_skim: Optional[FullSkimsCreatorConfig] = Field(
        None,
        description="Optional full-skim mode configuration.",
    )


# =============================================================================
# POSTPROCESSING CONFIGURATION
# =============================================================================


class PostprocessingConfig(BaseModel):
    """Postprocessing and validation configuration."""

    output_folder: str
    mep_output_folder: str
    scenario_definitions: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)

    def to_consist_facet(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "sample": self.sample,
            "replanning_portion": self.replanning_portion,
            "memory": self.memory,
            "discard_plans_every_year": self.discard_plans_every_year,
            "max_plans_memory": self.max_plans_memory,
            "simulated_hwy_paths": list(self.simulated_hwy_paths or []),
        }


# =============================================================================
# TOP-LEVEL PILATES CONFIGURATION
# =============================================================================


class PilatesConfig(BaseModel):
    """
    Complete PILATES configuration.

    This is the top-level model that encompasses all configuration sections.
    """

    model_config = ConfigDict(extra="allow")

    run: RunConfig
    shared: SharedConfig
    infrastructure: InfrastructureConfig
    urbansim: Optional[UrbanSimConfig] = None
    atlas: Optional[AtlasConfig] = None
    activitysim: Optional[ActivitySimConfig] = None
    beam: Optional[BeamConfig] = None
    postprocessing: Optional[PostprocessingConfig] = None

    @model_validator(mode="after")
    def validate_model_configs(self):
        """Ensure enabled models have configurations."""
        if self.run.models.land_use and not self.urbansim:
            raise ValueError("land_use model is enabled but urbansim config is missing")
        if self.run.models.vehicle_ownership and not self.atlas:
            raise ValueError(
                "vehicle_ownership model is enabled but atlas config is missing"
            )
        if self.run.models.activity_demand and not self.activitysim:
            raise ValueError(
                "activity_demand model is enabled but activitysim config is missing"
            )
        if self.run.models.travel and not self.beam:
            raise ValueError("travel model is enabled but beam config is missing")
        return self

    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        enabled = []
        if self.run.models.land_use:
            enabled.append(self.run.models.land_use)
        if self.run.models.vehicle_ownership:
            enabled.append(self.run.models.vehicle_ownership)
        if self.run.models.activity_demand:
            enabled.append(self.run.models.activity_demand)
        if self.run.models.travel:
            enabled.append(self.run.models.travel)
        return enabled

    def get_initialization_signature(self) -> Dict[str, Any]:
        """
        Extracts configuration strictly required for model initialization.

        This dictionary is used by Consist to hash the 'Input/Init' step.
        It defines the 'Logical World' (Space, Time, Topology).
        """
        return {
            # 1. Context: The boundaries of the simulation
            "context": {
                "region": self.run.region,
                "scenario": self.run.scenario,
                "start_year": self.run.start_year,
            },
            # 2. Geography: The spatial resolution and zone system
            # IdentityManager will auto-serialize this Pydantic model
            "geography": self.shared.geography,
            # 3. Initial Conditions: Baseline costs/impedances (filenames)
            # Note: Content hashing of these files happens via input_artifacts=[...],
            # this only hashes the *pointer* to the file.
            "initial_conditions": self.shared.skims,
            # 4. Orchestration: The DAG topology
            # Changes here alter the workflow state machine construction
            "orchestration": {
                "use_stubs": self.run.use_stubs,
                "supply_demand_iters": self.run.supply_demand_iters,
                "frequencies": {
                    "land_use": self.run.land_use_freq,
                    "travel": self.run.travel_model_freq,
                    "vehicle": self.run.vehicle_ownership_freq,
                },
                # Determines which sub-models are active
                "enabled_models": self.run.models,
            },
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_config(config_path: str) -> PilatesConfig:
    """
    Load and validate PILATES configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PilatesConfig object

    Raises:
        ValidationError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Pydantic will validate on instantiation
    return PilatesConfig(**config_dict)


def validate_config(config_dict: Dict[str, Any]) -> PilatesConfig:
    """
    Validate a configuration dictionary.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        Validated PilatesConfig object

    Raises:
        ValidationError: If config is invalid
    """
    return PilatesConfig(**config_dict)
