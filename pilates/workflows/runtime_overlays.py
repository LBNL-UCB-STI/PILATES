from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

from pilates.activitysim.outputs import configured_asim_output_keys
from pilates.beam.outputs import requires_omx_skim_output
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
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

    def applies(
        self,
        *,
        step_name: str,
        land_use_enabled: bool,
        run_mode: Any,
    ) -> bool:
        """Return whether this overlay matches the current runtime shape."""
        if self.step_name != step_name:
            return False
        if self.run_mode is not None and _run_mode_name(run_mode) != self.run_mode:
            return False
        if (
            self.land_use_enabled is not None
            and land_use_enabled != self.land_use_enabled
        ):
            return False
        return True


_STEP_RUNTIME_OVERLAY_RULES: Tuple[StepRuntimeOverlayRule, ...] = (
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
    land_use_enabled: bool,
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
        if not rule.applies(
            step_name=step_name,
            land_use_enabled=land_use_enabled,
            run_mode=run_mode,
        ):
            continue
        required = _ordered_unique(required, rule.add_required_inputs)
        optional = tuple(
            key
            for key in _ordered_unique(optional, rule.add_optional_inputs)
            if key not in rule.remove_optional_inputs
        )
    return required, optional


def resolve_runtime_output_overrides(
    *,
    step_name: str,
    settings: Any,
    required_outputs: Tuple[str, ...],
    optional_outputs: Tuple[str, ...],
) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Promote configured optional outputs into the effective run contract.

    Static step metadata describes the complete output vocabulary. The runtime
    surface promotes only outputs that this configuration actually requests or
    that a downstream enabled model requires.
    """
    promoted: Tuple[str, ...] = ()
    if step_name in {"activitysim_run", "activitysim_postprocess"}:
        activitysim_settings = getattr(settings, "activitysim", None)
        if activitysim_settings is not None:
            configured = set(configured_asim_output_keys(settings))
            promoted = tuple(key for key in optional_outputs if key in configured)
    elif step_name == "atlas_preprocess":
        atlas_settings = getattr(settings, "atlas", None)
        beamac = (
            getattr(atlas_settings, "beamac", 0) if atlas_settings is not None else 0
        )
        if beamac and beamac > 0:
            promoted = ("atlas_accessibility_csv",)
    elif step_name == "beam_postprocess" and requires_omx_skim_output(settings):
        promoted = (FINAL_SKIMS_OMX,)

    if not promoted:
        return required_outputs, optional_outputs

    promoted_set = set(promoted)
    return (
        _ordered_unique(required_outputs, promoted),
        tuple(key for key in optional_outputs if key not in promoted_set),
    )
