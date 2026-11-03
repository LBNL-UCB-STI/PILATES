"""
Pydantic models for PILATES configuration.

This module defines the structure and validation for PILATES configuration files.
Consist-specific identity and facet mapping lives in pilates.workflows.consist_metadata.
"""

import argparse
import copy
import logging
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
import yaml

from .activitysim import (
    ActivitySimAdmissionConfig,
    ActivitySimConfig,
    ActivitySimConfigRoot,
    ActivitySimDatabaseConfig,
    required_activitysim_config_roots,
)
from .admission import (
    AdmissionExpectation,
    AdmissionPolicyConfig,
    DeclaredDigestExpectation,
    PriorRunExpectation,
    admission_policy_fingerprint,
)
from .beam import (
    BeamAdmissionConfig,
    BeamArtifactFormatsConfig,
    BeamConfig,
    BeamLinkstatsAdmissionConfig,
    FullSkimsCreatorConfig,
)
from .runtime import PilatesRuntime, RuntimeFlags, RuntimeOptions

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
    max_year_intervals: Optional[int] = Field(
        None,
        description=(
            "Optional cap on planned land-use forecast intervals; useful for "
            "bounded representative runs"
        ),
        ge=1,
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
    recovery_archive_roots: List[str] = Field(
        default_factory=list,
        description=(
            "Optional long-term recovery/archive roots used for post-run promotion "
            "of a completed archive run"
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
        "callable_module",
        description=(
            "Optional Consist cache code-identity mode override for step runs. "
            "Defaults to callable_module; when set to None, Consist falls back to repo_git."
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

    @field_validator("recovery_archive_roots")
    @classmethod
    def expand_recovery_archive_roots(cls, v):
        """Expand environment variables in recovery archive roots."""
        return [os.path.expandvars(str(root)) for root in (v or [])]

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
    source_file: str = Field(..., description="Path to a zone geometry source file.")
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
    snapshot_path: Optional[str] = Field(
        None,
        description="Matrix snapshot file path.",
    )

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


class UrbanSimAdmissionConfig(BaseModel):
    """Optional inbound-admission policy for the UrbanSim bootstrap datastore."""

    initial_datastore: Optional[AdmissionPolicyConfig] = Field(
        None,
        description="Optional expectation for the bootstrap-staged UrbanSim datastore.",
    )


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
    admission: Optional[UrbanSimAdmissionConfig] = Field(
        None,
        description="Optional inbound-admission policy; omitted means disabled.",
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
            "output_folder": self.output_folder,
            "mep_output_folder": self.mep_output_folder,
            "scenario_definitions": self.scenario_definitions,
            "validation_metrics": self.validation_metrics,
        }


class WorkflowConfig(BaseModel):
    """Workflow orchestration configuration."""

    @model_validator(mode="before")
    @classmethod
    def reject_retired_manifest_recovery(cls, value: Any) -> Any:
        """Reject the removed YAML workflow-recovery configuration explicitly."""
        if isinstance(value, dict) and "manifests" in value:
            raise ValueError(
                "workflow.manifests has been removed; native Consist execution "
                "does not support YAML manifest recovery."
            )
        return value


# =============================================================================
# TOP-LEVEL PILATES CONFIGURATION
# =============================================================================


class PilatesConfig(BaseModel):
    """
    Complete PILATES configuration.

    This is the top-level model that encompasses all configuration sections.
    """

    model_config = ConfigDict(extra="allow")
    _runtime: PilatesRuntime = PrivateAttr(default_factory=PilatesRuntime)

    base: Optional[str] = Field(
        None,
        exclude=True,
        description="Optional path to a base YAML configuration file to inherit from.",
    )
    run: RunConfig
    write_skims_to_omx: bool = Field(
        False,
        description=(
            "Write final BEAM Zarr skims to OMX during BEAM postprocess. "
            "UrbanSim and ATLAS still enable OMX export automatically because "
            "their preprocessors consume that handoff."
        ),
    )
    shared: SharedConfig
    infrastructure: InfrastructureConfig
    urbansim: Optional[UrbanSimConfig] = None
    atlas: Optional[AtlasConfig] = None
    activitysim: Optional[ActivitySimConfig] = None
    beam: Optional[BeamConfig] = None
    postprocessing: Optional[PostprocessingConfig] = None
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)

    @model_validator(mode="before")
    @classmethod
    def resolve_inline_base_if_present(cls, value: Any) -> Any:
        """Resolve top-level ``base``/``extends`` when validating a raw YAML dictionary."""
        if not isinstance(value, dict):
            return value
        base_ref = value.get("base")
        if base_ref is None and "extends" in value:
            base_ref = value.get("extends")
        if not isinstance(base_ref, (str, Path)) or not str(base_ref).strip():
            return value

        required_sections_present = all(
            section in value for section in ("run", "shared", "infrastructure")
        )
        try:
            resolved_base_path = _resolve_base_config_path(
                base_ref,
                referencing_path=None,
                config_path="<inline dict>",
                include_scenario_dirs=True,
            )
        except FileNotFoundError:
            if required_sections_present:
                return value
            raise

        base_dict = resolve_config_dict(resolved_base_path)
        child_overrides = {
            k: v for k, v in value.items() if k not in {"base", "extends"}
        }
        return _deep_merge_config_dicts(base_dict, child_overrides)

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

    def _sync_runtime_flags(self) -> PilatesRuntime:
        """Compute default runtime flags from current config state and sync non-overridden fields."""
        extra = self.__pydantic_extra__ or {}
        warm_start_skims = bool(extra.get("warm_start_skims", False))
        static_skims = bool(extra.get("static_skims", False))
        try:
            replan_iters = (
                int(self.activitysim.replan_iters)
                if self.activitysim is not None
                else int(extra.get("replan_iters", 0) or 0)
            )
        except AttributeError:
            replan_iters = int(extra.get("replan_iters", 0) or 0)
        land_use_enabled = bool(self.run.models.land_use) and not warm_start_skims
        vehicle_ownership_model_enabled = bool(self.run.models.vehicle_ownership)
        activity_demand_model = self.run.models.activity_demand
        activity_demand_enabled = bool(activity_demand_model)
        travel_model = extra.get("traffic_assignment") or self.run.models.travel
        traffic_assignment_enabled = bool(travel_model) and not static_skims
        replanning_enabled = (replan_iters > 0) and not (
            activity_demand_enabled and activity_demand_model == "polaris"
        )
        self._runtime.flags.sync_defaults(
            land_use_enabled=land_use_enabled,
            vehicle_ownership_model_enabled=vehicle_ownership_model_enabled,
            activity_demand_enabled=activity_demand_enabled,
            traffic_assignment_enabled=traffic_assignment_enabled,
            replanning_enabled=replanning_enabled,
        )
        return self._runtime

    @property
    def runtime(self) -> PilatesRuntime:
        """Typed runtime metadata derived live from the configuration."""
        return self._sync_runtime_flags()

    @property
    def land_use_enabled(self) -> bool:
        return self.runtime.flags.land_use_enabled

    @land_use_enabled.setter
    def land_use_enabled(self, value: bool) -> None:
        self._runtime.flags.land_use_enabled = value

    @property
    def vehicle_ownership_model_enabled(self) -> bool:
        return self.runtime.flags.vehicle_ownership_model_enabled

    @vehicle_ownership_model_enabled.setter
    def vehicle_ownership_model_enabled(self, value: bool) -> None:
        self._runtime.flags.vehicle_ownership_model_enabled = value

    @property
    def activity_demand_enabled(self) -> bool:
        return self.runtime.flags.activity_demand_enabled

    @activity_demand_enabled.setter
    def activity_demand_enabled(self, value: bool) -> None:
        self._runtime.flags.activity_demand_enabled = value

    @property
    def traffic_assignment_enabled(self) -> bool:
        return self.runtime.flags.traffic_assignment_enabled

    @traffic_assignment_enabled.setter
    def traffic_assignment_enabled(self, value: bool) -> None:
        self._runtime.flags.traffic_assignment_enabled = value

    @property
    def replanning_enabled(self) -> bool:
        return self.runtime.flags.replanning_enabled

    @replanning_enabled.setter
    def replanning_enabled(self, value: bool) -> None:
        self._runtime.flags.replanning_enabled = value

    @property
    def state_file_loc(self) -> Optional[str]:
        return self._runtime.options.state_file_loc

    @state_file_loc.setter
    def state_file_loc(self, value: Optional[str]) -> None:
        self._runtime.options.state_file_loc = value

    @property
    def settings_file(self) -> Optional[str]:
        return self._runtime.options.settings_file

    @settings_file.setter
    def settings_file(self, value: Optional[str]) -> None:
        self._runtime.options.settings_file = value

    @property
    def allow_rewind_resume(self) -> bool:
        return self._runtime.options.allow_rewind_resume

    @allow_rewind_resume.setter
    def allow_rewind_resume(self, value: bool) -> None:
        self._runtime.options.allow_rewind_resume = bool(value)

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
                "max_year_intervals": self.run.max_year_intervals,
                "frequencies": {
                    "land_use": self.run.land_use_freq,
                    "travel": self.run.travel_model_freq,
                    "vehicle": self.run.vehicle_ownership_freq,
                },
                # Determines which sub-models are active
                "enabled_models": self.run.models,
            },
            "urbansim_admission": (
                {
                    "initial_datastore": admission_policy_fingerprint(
                        self.urbansim.admission.initial_datastore
                    )
                }
                if self.urbansim is not None and self.urbansim.admission is not None
                else None
            ),
            **(
                {
                    "activitysim_admission": {
                        "initial_skims": admission_policy_fingerprint(
                            self.activitysim.admission.initial_skims
                        )
                    }
                }
                if self.activitysim is not None
                and self.activitysim.admission is not None
                else {}
            ),
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _deep_merge_config_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively deep-merge ``override`` onto ``base``.

    Semantics:
    - Dictionaries are merged recursively.
    - Lists in ``override`` replace lists in ``base``.
    - Scalars and explicit ``None`` values in ``override`` overwrite ``base``.
    """
    merged: Dict[str, Any] = copy.deepcopy(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge_config_dicts(base_value, override_value)
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def _resolve_base_config_path(
    base_path: Union[str, Path],
    referencing_path: Optional[Path],
    config_path: Union[str, Path],
    include_scenario_dirs: bool = True,
) -> Path:
    """Resolve a base configuration path relative to the child file, cwd, or repo root."""
    base_candidate = Path(str(base_path).strip())
    if base_candidate.is_absolute():
        if base_candidate.exists():
            return base_candidate.resolve()
        raise FileNotFoundError(
            f"Base config not found: {base_path} (referenced by {config_path})"
        )

    repo_root = Path(__file__).resolve().parents[2]
    search_roots: List[Path] = []
    if referencing_path is not None:
        search_roots.append(referencing_path.parent)
    search_roots.extend([Path.cwd(), repo_root])
    if include_scenario_dirs:
        scenarios_dir = repo_root / "scenarios"
        if scenarios_dir.is_dir():
            search_roots.append(scenarios_dir)
            search_roots.extend(
                sorted(p for p in scenarios_dir.iterdir() if p.is_dir())
            )

    for root in search_roots:
        candidate = root / base_candidate
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Base config not found: {base_path} (referenced by {config_path})"
    )


def resolve_config_dict(
    config_path: Union[str, Path],
    visited: Optional[set[Path]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file and recursively resolve ``base:`` inheritance.

    Args:
        config_path: Path to YAML configuration file.
        visited: Canonical paths already visited in the current inheritance chain.

    Returns:
        Fully merged configuration dictionary with ``base``/``extends`` removed.

    Raises:
        FileNotFoundError: If ``config_path`` or any referenced ``base`` file does not exist.
        ValueError: If circular configuration inheritance is detected or YAML is not a mapping.
    """
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    canonical_path = path_obj.resolve()
    active_visited: set[Path] = set() if visited is None else set(visited)
    if canonical_path in active_visited:
        raise ValueError(
            f"Circular configuration inheritance detected: {canonical_path} already in {active_visited}"
        )

    with open(canonical_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        config_dict: Dict[str, Any] = {}
    elif isinstance(loaded, dict):
        config_dict = dict(loaded)
    else:
        raise ValueError(
            f"Configuration file must deserialize to a mapping, got {type(loaded).__name__}: {config_path}"
        )

    has_base_key = "base" in config_dict or "extends" in config_dict
    base_ref = config_dict.get("base")
    if base_ref is None and "extends" in config_dict:
        base_ref = config_dict.get("extends")

    child_overrides = {
        k: v for k, v in config_dict.items() if k not in {"base", "extends"}
    }
    if not has_base_key or base_ref is None or str(base_ref).strip() == "":
        return child_overrides

    resolved_base_path = _resolve_base_config_path(
        base_ref,
        referencing_path=canonical_path,
        config_path=config_path,
    )
    base_dict = resolve_config_dict(
        resolved_base_path,
        visited={*active_visited, canonical_path},
    )
    return _deep_merge_config_dicts(base_dict, child_overrides)


def load_config(config_path: Union[str, Path]) -> PilatesConfig:
    """
    Load and validate PILATES configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PilatesConfig object

    Raises:
        ValidationError: If config is invalid
        FileNotFoundError: If config file or base config doesn't exist
        ValueError: If circular configuration inheritance is detected
    """
    merged_dict = resolve_config_dict(config_path)
    return PilatesConfig(**merged_dict)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint to dump a resolved PILATES configuration YAML to stdout."""
    parser = argparse.ArgumentParser(
        prog="python -m pilates.config.models",
        description="Resolve PILATES scenario configuration inheritance and dump YAML to stdout.",
    )
    parser.add_argument(
        "config_path",
        help="Path to the scenario YAML configuration file.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the resolved configuration against PilatesConfig before dumping.",
    )
    args = parser.parse_args(argv)
    resolved = resolve_config_dict(args.config_path)
    if args.validate:
        PilatesConfig(**resolved)
    sys.stdout.write(yaml.safe_dump(resolved, sort_keys=False))
    return 0


__all__ = [
    "ActivitySimAdmissionConfig",
    "ActivitySimConfig",
    "ActivitySimConfigRoot",
    "ActivitySimDatabaseConfig",
    "AdmissionExpectation",
    "AdmissionPolicyConfig",
    "AtlasConfig",
    "BeamAdmissionConfig",
    "BeamArtifactFormatsConfig",
    "BeamConfig",
    "BeamLinkstatsAdmissionConfig",
    "DatabaseConfig",
    "DeclaredDigestExpectation",
    "DockerConfig",
    "FullSkimsCreatorConfig",
    "GeographyConfig",
    "InfrastructureConfig",
    "ModelSelection",
    "PilatesConfig",
    "PilatesRuntime",
    "PostprocessingConfig",
    "PriorRunExpectation",
    "RunConfig",
    "RuntimeFlags",
    "RuntimeOptions",
    "SharedConfig",
    "SkimsConfig",
    "UrbanSimAdmissionConfig",
    "UrbanSimConfig",
    "WorkflowConfig",
    "ZoneSourceConfig",
    "ZonesConfig",
    "admission_policy_fingerprint",
    "load_config",
    "main",
    "required_activitysim_config_roots",
    "resolve_config_dict",
    "validate_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
