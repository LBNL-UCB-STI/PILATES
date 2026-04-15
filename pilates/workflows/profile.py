from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Set

from pilates.utils.io import apply_runtime_flags, compute_model_enabled_flags


@dataclass(frozen=True)
class WorkflowProfile:
    land_use_enabled: bool
    vehicle_ownership_model_enabled: bool
    activity_demand_enabled: bool
    traffic_assignment_enabled: bool
    replanning_enabled: bool
    supply_demand_loop_enabled: bool
    activity_demand_direct_from_land_use: bool

    def model_enabled(self, flag_attr: str) -> bool:
        return bool(getattr(self, flag_attr, False))

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


def workflow_profile_from_flags(enabled_flags: Dict[str, bool]) -> WorkflowProfile:
    land_use_enabled = bool(enabled_flags["land_use_enabled"])
    activity_demand_enabled = bool(enabled_flags["activity_demand_enabled"])
    traffic_assignment_enabled = bool(enabled_flags["traffic_assignment_enabled"])
    return WorkflowProfile(
        land_use_enabled=land_use_enabled,
        vehicle_ownership_model_enabled=bool(
            enabled_flags["vehicle_ownership_model_enabled"]
        ),
        activity_demand_enabled=activity_demand_enabled,
        traffic_assignment_enabled=traffic_assignment_enabled,
        replanning_enabled=bool(enabled_flags["replanning_enabled"]),
        supply_demand_loop_enabled=(
            activity_demand_enabled or traffic_assignment_enabled
        ),
        activity_demand_direct_from_land_use=(
            land_use_enabled and not activity_demand_enabled
        ),
    )


def build_workflow_profile(settings: Any) -> WorkflowProfile:
    runtime = getattr(settings, "runtime", None)
    runtime_flags = getattr(runtime, "flags", None)
    flags_initialized = bool(getattr(runtime, "flags_initialized", False))
    if runtime_flags is not None and flags_initialized:
        return workflow_profile_from_flags(
            {
                "land_use_enabled": bool(runtime_flags.land_use_enabled),
                "vehicle_ownership_model_enabled": bool(
                    runtime_flags.vehicle_ownership_model_enabled
                ),
                "activity_demand_enabled": bool(runtime_flags.activity_demand_enabled),
                "traffic_assignment_enabled": bool(
                    runtime_flags.traffic_assignment_enabled
                ),
                "replanning_enabled": bool(runtime_flags.replanning_enabled),
            }
        )

    enabled_flags = compute_model_enabled_flags(settings)
    apply_runtime_flags(settings, enabled_flags)
    return workflow_profile_from_flags(enabled_flags)


def profile_enabled_schema_models(
    profile: WorkflowProfile,
    *,
    include_optional: bool = True,
) -> Set[str]:
    from pilates.workflows.catalog import schema_step_specs

    enabled_models: Set[str] = set()
    for spec in schema_step_specs(include_optional=include_optional):
        flag_attr = spec.enabled_flag_attr
        model_attr = spec.enabled_model_attr
        if flag_attr is None or model_attr is None:
            enabled_models.add(spec.step_name)
            continue
        if profile.model_enabled(flag_attr):
            enabled_models.add(spec.step_name)
    return enabled_models
