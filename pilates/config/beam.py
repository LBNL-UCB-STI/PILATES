"""BEAM transportation network simulation configuration models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .admission import AdmissionExpectation, AdmissionPolicyConfig


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
    skims_kind: Literal["activitySimOmx", "activitySimZarr", "activitySim", "od"] = (
        Field(
            "activitySimOmx",
            description=(
                "Skim output format: activitySimOmx, activitySimZarr, "
                "activitySim, or od"
            ),
        )
    )
    peak_hours: List[float] = Field(
        default_factory=lambda: [8.5], description="Peak hours (8.5 = 8:30 AM)"
    )
    modes_to_build: Dict[str, bool] = Field(
        default_factory=lambda: {"drive": True, "walk": True, "transit": True},
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
    bifurcate_tolls: bool = Field(
        True,
        description=(
            "When true, auto trips are routed twice: once toll-free (SOV) "
            "and once toll-permitted (SOVTOLL)."
        ),
    )
    generate_return_trips: bool = Field(
        True,
        description=(
            "When true, generates both outbound Drive-Transit-Walk and "
            "inbound Walk-Transit-Drive itineraries."
        ),
    )
    toll_file_path: Optional[str] = Field(
        None,
        description="Optional path to toll-prices.csv (overrides beam.agentsim.toll.filePath).",
    )
    directory2: Optional[str] = Field(
        None,
        description="Optional path to secondary R5 network directory for dual-router mode.",
    )
    od_skims_path: Optional[str] = Field(
        None,
        description="Optional path to an existing skim .csv.gz to reuse pre-computed OD pairs or warm-start.",
    )
    input_path: Optional[str] = Field(
        None,
        description="Optional path to a custom OD pairs CSV with origin,destination columns.",
    )
    linkstats_path: Optional[str] = Field(
        None,
        description="Optional explicit path to congested linkstats.csv.gz or .parquet.",
    )
    calculation_timeout_hours: int = Field(
        72,
        ge=1,
        description=(
            "Timeout in hours for BackgroundFullSkimsCreator "
            "(beam.urbansim.backgroundODSkimsCreator.calculationTimeoutHours)."
        ),
    )

    def to_consist_facet(self) -> Dict[str, Any]:
        return {
            "run_schedule": self.run_schedule,
            "router_type": self.router_type,
            "skims_geo_type": self.skims_geo_type,
            "skims_kind": self.skims_kind,
            "peak_hours": list(self.peak_hours),
            "modes_to_build": dict(self.modes_to_build),
            "parallelism_thread_ratio": self.parallelism_thread_ratio,
            "bifurcate_tolls": self.bifurcate_tolls,
            "generate_return_trips": self.generate_return_trips,
            "toll_file_path": self.toll_file_path,
            "directory2": self.directory2,
            "od_skims_path": self.od_skims_path,
            "input_path": self.input_path,
            "linkstats_path": self.linkstats_path,
        }


class BeamLinkstatsAdmissionConfig(AdmissionPolicyConfig):
    """Explicit prior-run or declared-digest expectation for one staged BEAM warm-start file."""

    expectation: AdmissionExpectation


class BeamAdmissionConfig(BaseModel):
    """Optional inbound-admission settings for BEAM runtime roles."""

    linkstats: Optional[BeamLinkstatsAdmissionConfig] = Field(
        None,
        description="Optional prior-run or declared-digest admission for staged warm-start linkstats.",
    )


class BeamArtifactFormatsConfig(BaseModel):
    """Preferred interchange formats for artifacts produced by BEAM."""

    activitysim_skims: Literal["zarr", "omx"] = Field(
        "zarr", description="BEAM activity-sim skimmer output format"
    )
    exchange: Literal["parquet", "csv"] = Field(
        "parquet", description="BEAM exchange-scenario population input format"
    )
    events: Literal["parquet", "csv.gz"] = Field(
        "parquet", description="BEAM events output format"
    )
    linkstats: Literal["parquet", "csv.gz"] = Field(
        "parquet", description="BEAM linkstats output format"
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
    warmstart_linkstats_path: Optional[str] = Field(
        None,
        description=(
            "Optional path to the initial BEAM warm-start linkstats file. "
            "If relative, it is resolved against the mutable BEAM region input root."
        ),
    )
    activitysim_skims_file_base_name: str = Field(
        "skimsActivitySimOD",
        min_length=1,
        description=(
            "BEAM activity-sim skimmer fileBaseName. BEAM appends its current "
            "marker and the configured artifact-format extension."
        ),
    )
    artifact_formats: BeamArtifactFormatsConfig = Field(
        default_factory=BeamArtifactFormatsConfig,
        description=(
            "PILATES-owned preferred formats applied to the canonical BEAM launch "
            "configuration."
        ),
    )
    admission: Optional[BeamAdmissionConfig] = Field(
        None,
        description="Optional inbound-admission policy; omitted means disabled.",
    )
    skims_shapefile: str
    skim_zone_source_id_col: str
    skim_zone_geoid_col: str
    discard_plans_every_year: bool = Field(
        False, description="Discard plans every year"
    )
    max_plans_memory: int = Field(5, description="Max plans in memory")
    plan_memory_beta: float = Field(
        1.0,
        description="Logit scale parameter (brainExpBeta) for ExpBetaPlanSelector in PILATES plan merging.",
    )
    new_plan_selection_policy: Literal["force_new", "exp_beta", "probabilistic_new"] = (
        Field(
            "force_new",
            description=(
                "How to select the active plan when ActivitySim adds a new candidate plan: "
                "'force_new' always selects the newly generated plan; "
                "'exp_beta' assigns transient new_plan_initial_score (default 0.0) to unexecuted plans and samples via ExpBetaPlanSelector; "
                "'probabilistic_new' forces the new plan with probability new_plan_adoption_rate and otherwise uses exp_beta."
            ),
        )
    )
    new_plan_initial_score: float = Field(
        0.0,
        description="Transient prior score assigned to unexecuted plans during ExpBetaPlanSelector sampling.",
    )
    new_plan_adoption_rate: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Probability of forcing the newly generated ActivitySim plan when policy is 'probabilistic_new'.",
    )
    plan_selection_random_seed: int = Field(
        42,
        description="Base random seed for deterministic, reproducible plan selection across iterations.",
    )
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
