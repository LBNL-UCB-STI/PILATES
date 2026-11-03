"""ActivitySim configuration models and logical config root paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .admission import AdmissionPolicyConfig, admission_policy_fingerprint


@dataclass(frozen=True, slots=True)
class ActivitySimConfigRoot:
    """One normalized path relative to the ActivitySim configuration tree."""

    relative_path: PurePosixPath

    @classmethod
    def parse(cls, value: str) -> "ActivitySimConfigRoot":
        if not isinstance(value, str) or not value:
            raise ValueError(
                "ActivitySim config root must be a non-empty logical relative path"
            )
        windows_path = PureWindowsPath(value)
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or "\\" in value
            or path == PurePosixPath(".")
            or ".." in path.parts
        ):
            raise ValueError(
                "ActivitySim config root must be a logical relative path without "
                f"absolute or parent traversal components: {value!r}"
            )
        return cls(relative_path=path)

    def as_posix(self) -> str:
        return self.relative_path.as_posix()

    def path_under(self, root: Path) -> Path:
        return root.joinpath(*self.relative_path.parts)


def required_activitysim_config_roots(
    main_configs_dir: str,
) -> tuple[ActivitySimConfigRoot, ...]:
    """Return the validated config roots in adapter and launch order."""

    candidates = (
        main_configs_dir,
        "configs",
        "configs_extended",
        "configs_mp",
        "configs_sh_compile",
    )
    ordered: list[ActivitySimConfigRoot] = []
    seen: set[str] = set()
    for candidate in candidates:
        root = ActivitySimConfigRoot.parse(candidate)
        logical_path = root.as_posix()
        if logical_path in seen:
            continue
        seen.add(logical_path)
        ordered.append(root)
    return tuple(ordered)


class ActivitySimDatabaseConfig(BaseModel):
    """ActivitySim database input mode configuration."""

    enabled: bool = Field(False, description="Enable database input mode")
    use_processed_data: bool = Field(
        True, description="Use pre-processed data (fast path)"
    )
    year: Optional[int] = Field(None, description="Year for data retrieval")


class ActivitySimAdmissionConfig(BaseModel):
    """Optional inbound-admission policy for ActivitySim bootstrap skims."""

    initial_skims: Optional[AdmissionPolicyConfig] = Field(
        None,
        description="Optional expectation for the initial ActivitySim OMX skims.",
    )


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

    @field_validator("main_configs_dir")
    @classmethod
    def validate_main_configs_dir(cls, value: str) -> str:
        return ActivitySimConfigRoot.parse(value).as_posix()

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
    replan_iters: int = Field(0, description="Replanning iterations")
    replan_hh_samp_size: int = Field(0, description="Replanning household sample size")
    replan_after: str = Field(
        "non_mandatory_tour_scheduling", description="Replan after this step"
    )
    random_seed: Optional[int] = Field(
        None, description="Base random number generator seed"
    )
    workplace_reassignment_share: float = Field(
        0.0,
        description=(
            "Share of workers with valid workplace assignments to mark for "
            "mandatory workplace re-assignment in ActivitySim preprocess."
        ),
        ge=0.0,
        le=1.0,
    )
    school_reassignment_share: float = Field(
        0.0,
        description=(
            "Share of students with valid school assignments to mark for "
            "mandatory school re-assignment in ActivitySim preprocess."
        ),
        ge=0.0,
        le=1.0,
    )
    final_plans_folder: str = Field(..., description="Final plans output folder")
    admission: Optional[ActivitySimAdmissionConfig] = Field(
        None,
        description="Optional inbound-admission policy; omitted means disabled.",
    )

    def to_consist_facet(self) -> Dict[str, Any]:
        facet: Dict[str, Any] = {
            "household_sample_size": self.household_sample_size,
            "chunk_size": self.chunk_size,
            "num_processes": self.num_processes,
            "file_format": self.file_format,
            "persist_sharrow_cache": self.persist_sharrow_cache,
            "replan_iters": self.replan_iters,
            "replan_hh_samp_size": self.replan_hh_samp_size,
            "replan_after": self.replan_after,
            "random_seed": self.random_seed,
            "workplace_reassignment_share": self.workplace_reassignment_share,
            "school_reassignment_share": self.school_reassignment_share,
            "database": self.database.model_dump() if self.database else None,
        }
        if self.admission is not None:
            facet["admission"] = {
                "initial_skims": admission_policy_fingerprint(
                    self.admission.initial_skims
                )
            }
        return facet
