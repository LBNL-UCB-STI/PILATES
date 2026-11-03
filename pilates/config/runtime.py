"""Typed runtime flags and options associated with a loaded PilatesConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_FLAG_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "land_use_enabled",
        "vehicle_ownership_model_enabled",
        "activity_demand_enabled",
        "traffic_assignment_enabled",
        "replanning_enabled",
    }
)


@dataclass
class RuntimeFlags:
    """Derived runtime enablement flags attached to a loaded config."""

    land_use_enabled: bool = False
    vehicle_ownership_model_enabled: bool = False
    activity_demand_enabled: bool = False
    traffic_assignment_enabled: bool = False
    replanning_enabled: bool = False
    _overridden_fields: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    def __setattr__(self, name: str, value: object) -> None:
        if "_overridden_fields" in self.__dict__ and name in _FLAG_FIELD_NAMES:
            object.__setattr__(self, name, bool(value))
            self._overridden_fields.add(name)
            return
        object.__setattr__(self, name, value)

    def sync_defaults(
        self,
        *,
        land_use_enabled: bool,
        vehicle_ownership_model_enabled: bool,
        activity_demand_enabled: bool,
        traffic_assignment_enabled: bool,
        replanning_enabled: bool,
    ) -> None:
        """Sync computed flag defaults for any flag not explicitly overridden."""
        computed = {
            "land_use_enabled": bool(land_use_enabled),
            "vehicle_ownership_model_enabled": bool(vehicle_ownership_model_enabled),
            "activity_demand_enabled": bool(activity_demand_enabled),
            "traffic_assignment_enabled": bool(traffic_assignment_enabled),
            "replanning_enabled": bool(replanning_enabled),
        }
        for flag_name, flag_value in computed.items():
            if flag_name not in self._overridden_fields:
                object.__setattr__(self, flag_name, flag_value)


@dataclass
class RuntimeOptions:
    """Ephemeral runtime options attached outside the persisted config schema."""

    state_file_loc: Optional[str] = None
    settings_file: Optional[str] = None
    allow_rewind_resume: bool = False


@dataclass
class PilatesRuntime:
    """Typed runtime state associated with a loaded ``PilatesConfig`` instance."""

    flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    options: RuntimeOptions = field(default_factory=RuntimeOptions)
    flags_initialized: bool = True
