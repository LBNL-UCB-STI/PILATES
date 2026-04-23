from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
)


def _ordered_unique(*groups: Sequence[str]) -> Tuple[str, ...]:
    """Preserve stable key ordering while applying additive overlay rules."""
    return tuple(dict.fromkeys(key for group in groups for key in group))


def _run_mode_name(run_mode: Any) -> str:
    """Normalize enum-like run modes to the string values used in the registry."""
    return str(getattr(run_mode, "value", run_mode))


@dataclass(frozen=True)
class StepRuntimeOverlayRule:
    """Small declarative overlay for runtime-only step policy exceptions.

    The workflow catalog and binding specs remain the primary sources of truth.
    This rule exists only for the narrow cases where the effective runtime
    contract depends on enabled run shape or restart mode in a way that the
    static catalog cannot currently express.

    Keep this intentionally tiny. If a new rule starts to look like a general
    binding or catalog feature, that concern belongs in those modules instead
    of expanding this registry.
    """

    step_name: str
    run_mode: Optional[str] = None
    land_use_enabled: Optional[bool] = None
    add_required_inputs: Tuple[str, ...] = ()
    add_optional_inputs: Tuple[str, ...] = ()
    remove_optional_inputs: frozenset[str] = field(default_factory=frozenset)
    role_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def applies(self, *, step_name: str, profile: Any, run_mode: Any) -> bool:
        """Return whether this overlay matches the current runtime shape."""
        if self.step_name != step_name:
            return False
        if self.run_mode is not None and _run_mode_name(run_mode) != self.run_mode:
            return False
        if (
            self.land_use_enabled is not None
            and bool(getattr(profile, "land_use_enabled", False))
            != self.land_use_enabled
        ):
            return False
        return True


_STEP_RUNTIME_OVERLAY_RULES: Tuple[StepRuntimeOverlayRule, ...] = (
    StepRuntimeOverlayRule(
        step_name="activitysim_preprocess",
        run_mode="restart",
        land_use_enabled=True,
        add_optional_inputs=(USIM_DATASTORE_CURRENT_H5,),
        role_overrides={
            USIM_POPULATION_SOURCE_H5: {
                "workspace_archive_fallback_allowed": False,
                "restart_may_restore_synthetically": False,
                "restart_requires_explicit_before_execution": True,
            },
            USIM_DATASTORE_CURRENT_H5: {
                "workspace_archive_fallback_allowed": False,
                "restart_may_restore_synthetically": False,
                "restart_requires_explicit_before_execution": True,
            },
        },
    ),
    StepRuntimeOverlayRule(
        step_name="activitysim_postprocess",
        land_use_enabled=False,
        remove_optional_inputs=frozenset(
            {
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            }
        ),
    ),
    StepRuntimeOverlayRule(
        step_name="activitysim_postprocess",
        land_use_enabled=True,
        add_required_inputs=(USIM_POPULATION_SOURCE_H5, USIM_DATASTORE_CURRENT_H5),
        add_optional_inputs=(USIM_DATASTORE_BASE_H5,),
        remove_optional_inputs=frozenset(
            {
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
            }
        ),
    ),
)


def resolve_runtime_input_output_overrides(
    *,
    step_name: str,
    profile: Any,
    run_mode: Any,
    required_inputs: Tuple[str, ...],
    optional_inputs: Tuple[str, ...],
) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Apply the matching runtime overlay rules to required/optional inputs.

    The return value is still just a projection over catalog-declared keys. The
    registry may promote, demote, or append a small number of inputs for the
    effective runtime shape, but it must not invent a parallel step contract.
    """
    required = required_inputs
    optional = optional_inputs
    for rule in _STEP_RUNTIME_OVERLAY_RULES:
        if not rule.applies(step_name=step_name, profile=profile, run_mode=run_mode):
            continue
        required = _ordered_unique(required, rule.add_required_inputs)
        optional = tuple(
            key
            for key in _ordered_unique(optional, rule.add_optional_inputs)
            if key not in rule.remove_optional_inputs
        )
    return required, optional


def resolve_runtime_role_policy_overrides(
    *,
    step_name: str,
    profile: Any,
    run_mode: Any,
) -> Dict[str, Dict[str, Any]]:
    """Collect role-policy overlays for the current step/runtime shape.

    These values are merged into binding-derived `InputRolePolicy` instances by
    the surface builder. They are intentionally runtime-policy deltas rather
    than a replacement for binding rules.
    """
    overrides: Dict[str, Dict[str, Any]] = {}
    for rule in _STEP_RUNTIME_OVERLAY_RULES:
        if not rule.applies(step_name=step_name, profile=profile, run_mode=run_mode):
            continue
        for key, policy in rule.role_overrides.items():
            current = overrides.setdefault(key, {})
            current.update(policy)
    return overrides
