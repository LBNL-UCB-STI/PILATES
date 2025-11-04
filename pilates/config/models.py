"""
Pydantic models for PILATES configuration.

This module defines the structure and validation for PILATES configuration files.
The models use the new hierarchical structure designed for better config hashing
and provenance tracking.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# RUN-LEVEL CONFIGURATION
# =============================================================================

class ModelSelection(BaseModel):
    """Which models are enabled for this run."""
    land_use: Optional[str] = Field(
        None,
        description="Land use model (urbansim, or null to disable)"
    )
    travel: Optional[str] = Field(
        None,
        description="Travel model (beam, polaris, or null to disable)"
    )
    activity_demand: Optional[str] = Field(
        None,
        description="Activity demand model (activitysim, or null to disable)"
    )
    vehicle_ownership: Optional[str] = Field(
        None,
        description="Vehicle ownership model (atlas, or null to disable)"
    )


class RunConfig(BaseModel):
    """Top-level run configuration affecting workflow orchestration."""

    # Core simulation parameters (GLOBAL scope)
    region: str = Field(..., description="Geographic region (seattle, sfbay, austin, etc.)")
    scenario: str = Field(..., description="Scenario name")
    start_year: int = Field(..., description="First simulation year", ge=1900, le=2100)
    end_year: int = Field(..., description="Final simulation year", ge=1900, le=2100)

    # Model execution frequencies (GLOBAL scope)
    land_use_freq: int = Field(1, description="How often land use model runs", ge=1)
    travel_model_freq: int = Field(1, description="How often travel model runs", ge=1)
    vehicle_ownership_freq: int = Field(1, description="How often vehicle model runs", ge=1)
    supply_demand_iters: int = Field(6, description="Supply-demand loop iterations", ge=1)

    # Output configuration (METADATA scope)
    output_directory: str = Field(..., description="Where to write outputs")
    output_run_name: str = Field(..., description="Human-readable run name")

    # Model selection (GLOBAL scope)
    models: ModelSelection = Field(..., description="Which models are enabled")

    @field_validator('end_year')
    @classmethod
    def validate_end_year(cls, v, info):
        """Ensure end_year >= start_year."""
        if 'start_year' in info.data and v < info.data['start_year']:
            raise ValueError(f"end_year ({v}) must be >= start_year ({info.data['start_year']})")
        return v


# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

class GeographyConfig(BaseModel):
    """Geographic configuration used by multiple models."""

    FIPS: Dict[str, Any] = Field(
        ...,
        description="FIPS codes for study area"
    )
    local_crs: str = Field(
        ...,
        description="Local coordinate reference system (e.g., EPSG:32048)"
    )


class SkimsConfig(BaseModel):
    """Skim configuration affecting ActivitySim and BEAM."""

    zone_type: str = Field(..., description="Zone type: taz, block_group, or block")
    fname: str = Field(..., description="Skim file name")
    origin_fname: Optional[str] = Field(None, description="Origin skim file name")
    geoms_fname: str = Field(..., description="Geometries file name")
    geoms_index_col: str = Field(..., description="Geometry index column")
    hwy_paths: List[str] = Field(default_factory=list, description="Highway path types")
    periods: List[str] = Field(default_factory=list, description="Time periods")
    transit_paths: Optional[Dict[str, List[str]]] = Field(None, description="Transit path definitions")


class DatabaseConfig(BaseModel):
    """Database configuration for provenance and data storage."""

    enabled: bool = Field(False, description="Enable database storage")
    type: str = Field("duckdb", description="Database type")
    path: str = Field(..., description="Database file path")


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

    container_manager: str = Field("docker", description="Container manager: docker or singularity")
    singularity_images: Dict[str, str] = Field(default_factory=dict, description="Singularity image URIs")
    docker_images: Dict[str, str] = Field(default_factory=dict, description="Docker image tags")
    docker_config: DockerConfig = Field(default_factory=DockerConfig, description="Docker settings")


# =============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================

class UrbanSimConfig(BaseModel):
    """UrbanSim land use model configuration."""

    region_id: Optional[str] = Field(None, description="Region ID (computed from region if not provided)")
    local_data_input_folder: str = Field(..., description="Local input data folder")
    local_mutable_data_folder: str = Field(..., description="Mutable data folder")
    client_base_folder: str = Field(..., description="Container base folder")
    client_data_folder: str = Field(..., description="Container data folder")
    input_file_template: str = Field(..., description="Input file naming template")
    input_file_template_year: str = Field(..., description="Year-specific input file template")
    output_file_template: str = Field(..., description="Output file naming template")
    command_template: str = Field(..., description="Command template")
    region_mappings: Dict[str, Any] = Field(default_factory=dict, description="Region-specific mappings")


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
    use_processed_data: bool = Field(True, description="Use pre-processed data (fast path)")
    year: Optional[int] = Field(None, description="Year for data retrieval")


class ActivitySimConfig(BaseModel):
    """ActivitySim activity-based demand model configuration."""

    household_sample_size: int = Field(0, description="Household sample size (0 = full population)")
    chunk_size: int = Field(12_000_000_000, description="Memory chunk size")
    num_processes: int = Field(25, description="Number of parallel processes")
    file_format: str = Field("parquet", description="Output file format")
    local_input_folder: str
    local_mutable_data_folder: str
    local_output_folder: str
    local_configs_folder: str
    local_mutable_configs_folder: str
    validation_folder: str
    subdir: str = Field("configs", description="Config subdirectory")
    main_configs_dir: str = Field("configs_extended", description="Main configs directory")
    region_mappings: Dict[str, Any] = Field(default_factory=dict, description="Region-specific mappings")
    from_urbansim_col_maps: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="UrbanSim → ActivitySim column mappings")
    to_urbansim_col_maps: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="ActivitySim → UrbanSim column mappings")
    output_tables: Dict[str, Any] = Field(default_factory=dict, description="Output table configuration")
    command_template: str = Field(..., description="Command template")
    database: ActivitySimDatabaseConfig = Field(default_factory=ActivitySimDatabaseConfig, description="Database input configuration")
    warm_start_activities: bool = Field(False, description="Warm start activities")
    replan_iters: int = Field(0, description="Replanning iterations")
    replan_hh_samp_size: int = Field(0, description="Replanning household sample size")
    replan_after: str = Field("non_mandatory_tour_scheduling", description="Replan after this step")
    final_plans_folder: str = Field(..., description="Final plans output folder")


class BeamConfig(BaseModel):
    """BEAM transportation network simulation configuration."""

    config: str = Field(..., description="Main BEAM config file")
    sample: float = Field(1.0, description="Population sample fraction", ge=0.0, le=1.0)
    replanning_portion: float = Field(0.4, description="Replanning portion", ge=0.0, le=1.0)
    memory: str = Field("180g", description="JVM memory allocation")
    local_input_folder: str
    local_mutable_data_folder: str
    local_output_folder: str
    scenario_folder: str
    router_directory: str
    skims_shapefile: str
    skim_zone_source_id_col: str
    skim_zone_geoid_col: str
    discard_plans_every_year: bool = Field(False, description="Discard plans every year")
    max_plans_memory: int = Field(5, description="Max plans in memory")
    simulated_hwy_paths: List[str] = Field(default_factory=list, description="Simulated highway paths")
    asim_hwy_measure_map: Dict[str, Optional[str]] = Field(default_factory=dict, description="Highway measure mappings")
    asim_transit_measure_map: Dict[str, Optional[str]] = Field(default_factory=dict, description="Transit measure mappings")
    asim_ridehail_measure_map: Dict[str, Optional[str]] = Field(default_factory=dict, description="Ridehail measure mappings")
    ridehail_path_map: Dict[str, str] = Field(default_factory=dict, description="Ridehail path mappings")


# =============================================================================
# POSTPROCESSING CONFIGURATION
# =============================================================================

class PostprocessingConfig(BaseModel):
    """Postprocessing and validation configuration."""

    output_folder: str
    mep_output_folder: str
    scenario_definitions: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# TOP-LEVEL PILATES CONFIGURATION
# =============================================================================

class PilatesConfig(BaseModel):
    """
    Complete PILATES configuration.

    This is the top-level model that encompasses all configuration sections.
    """

    run: RunConfig
    shared: SharedConfig
    infrastructure: InfrastructureConfig
    urbansim: Optional[UrbanSimConfig] = None
    atlas: Optional[AtlasConfig] = None
    activitysim: Optional[ActivitySimConfig] = None
    beam: Optional[BeamConfig] = None
    postprocessing: Optional[PostprocessingConfig] = None

    @model_validator(mode='after')
    def validate_model_configs(self):
        """Ensure enabled models have configurations."""
        if self.run.models.land_use and not self.urbansim:
            raise ValueError("land_use model is enabled but urbansim config is missing")
        if self.run.models.vehicle_ownership and not self.atlas:
            raise ValueError("vehicle_ownership model is enabled but atlas config is missing")
        if self.run.models.activity_demand and not self.activitysim:
            raise ValueError("activity_demand model is enabled but activitysim config is missing")
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

    with open(config_path, 'r') as f:
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


def config_to_dict(config: PilatesConfig) -> Dict[str, Any]:
    """
    Convert PilatesConfig to dictionary with both nested and flat keys for backward compatibility.

    Args:
        config: PilatesConfig object

    Returns:
        Configuration as dictionary with nested structure PLUS flattened legacy keys
    """
    # Start with nested structure
    result = config.model_dump(exclude_none=True)

    # Add special legacy aliases
    if 'shared' in result and 'skims' in result['shared']:
        skims = result['shared']['skims']
        if 'geoms_fname' in skims:
            result['beam_geoms_fname'] = skims['geoms_fname']
        if 'geoms_index_col' in skims:
            result['geoms_index_col'] = skims['geoms_index_col']

    if 'beam' in result and 'router_directory' in result['beam']:
        result['beam_router_directory'] = result['beam']['router_directory']

    return result
