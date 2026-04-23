from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.workflows.artifact_keys import (
    ZARR_SKIMS,
)
from pilates.workflows.catalog import (
    schema_step_specs,
    workflow_step_contracts_by_name,
    workflow_step_spec_for_step_name,
)
from pilates.workflows._profile import (
    WorkflowProfile,
    ensure_runtime_flags_initialized,
    workflow_profile_from_flags,
)
from pilates.workflows.runtime_overlays import (
    resolve_runtime_input_output_overrides,
    resolve_runtime_role_policy_overrides,
)


class RunMode(str, Enum):
    """Execution mode for the current run shape.

    The surface uses this to apply restart-only overlays without introducing
    restart conditionals at every call site.
    """

    FRESH = "fresh"
    RESTART = "restart"


def _enabled_schema_step_names_from_profile(
    profile: WorkflowProfile,
    *,
    include_optional: bool,
) -> FrozenSet[str]:
    enabled_steps = set()
    for spec in schema_step_specs(include_optional=include_optional):
        flag_attr = spec.enabled_flag_attr
        model_attr = spec.enabled_model_attr
        if flag_attr is None or model_attr is None:
            enabled_steps.add(spec.step_name)
            continue
        if profile.model_enabled(flag_attr):
            enabled_steps.add(spec.step_name)
    return frozenset(enabled_steps)


@dataclass(frozen=True)
class RestartFrontierContract:
    """Effective restart frontier for the enabled run shape.

    The frontier answers a narrow question for restart/bootstrap code: if we
    are resuming into a specific stage, which artifacts must already exist so
    replay can safely continue without re-running upstream work?
    """

    frontier_stage: str
    frontier_step: str
    required_keys: Tuple[str, ...]


@dataclass(frozen=True)
class InputRolePolicy:
    """Runtime binding policy for one semantic input role.

    This is a projection of the binding spec plus any run-shape overlays.
    `binding.py` still owns the resolution mechanics; this object captures the
    effective policy that restart guards, stage code, and binding should share.
    """

    explicit_inputs_allowed: bool
    coupler_fallback_allowed: bool
    workspace_archive_fallback_allowed: bool
    restart_may_restore_synthetically: bool
    restart_requires_explicit_before_execution: bool

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class StepRuntimeSurface:
    """Runtime contract view for one workflow step.

    `catalog.py` remains the semantic source of truth. This dataclass is the
    enabled/run-mode projection that callers should consult when they need to
    know what the step effectively requires in this run, not merely what the
    static catalog says in the abstract.
    """

    step_name: str
    stage_name: str
    phase: str
    optional: bool
    enabled: bool
    required_input_keys: Tuple[str, ...]
    optional_input_keys: Tuple[str, ...]
    required_output_keys: Tuple[str, ...]
    optional_output_keys: Tuple[str, ...]
    input_role_policies: Mapping[str, InputRolePolicy]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "stage_name": self.stage_name,
            "phase": self.phase,
            "optional": self.optional,
            "enabled": self.enabled,
            "required_input_keys": list(self.required_input_keys),
            "optional_input_keys": list(self.optional_input_keys),
            "required_output_keys": list(self.required_output_keys),
            "optional_output_keys": list(self.optional_output_keys),
            "input_role_policies": {
                key: policy.to_dict()
                for key, policy in self.input_role_policies.items()
            },
        }


@dataclass(frozen=True)
class EnabledWorkflowSurface:
    """Single runtime authority for enabled-step contract decisions.

    The surface is intentionally small: it answers "what is enabled and what is
    the effective contract for that enabled shape?" without replacing the
    catalog or the binding engine. Runtime callers should prefer this object
    over re-deriving enabled steps, bootstrap ownership, or restart frontier
    rules on their own.
    """

    profile: WorkflowProfile
    run_mode: RunMode
    enabled_step_names: FrozenSet[str]
    enabled_model_names: FrozenSet[str]
    enabled_stage_names: FrozenSet[str]
    step_surfaces: Mapping[str, StepRuntimeSurface]
    bootstrap_owned_artifact_keys: FrozenSet[str]
    restart_prebootstrap_deferred_artifact_keys: FrozenSet[str]
    restart_frontier_contract: Optional[RestartFrontierContract]
    _required_enabled_step_names: FrozenSet[str] = field(
        default_factory=frozenset,
        repr=False,
        compare=False,
    )

    def step_enabled(self, step_name: str, *, include_optional: bool = True) -> bool:
        """Return whether a step is enabled in the current run shape."""
        enabled_names = (
            self.enabled_step_names
            if include_optional
            else self._required_enabled_step_names
        )
        return str(step_name) in enabled_names

    def stage_enabled(self, stage_name: str) -> bool:
        return str(stage_name) in self.enabled_stage_names

    def step_surface(self, step_name: str) -> Optional[StepRuntimeSurface]:
        return self.step_surfaces.get(step_name)

    def enabled_schema_step_names(self, *, include_optional: bool = True) -> FrozenSet[str]:
        """Return the enabled schema-step set used by planning/runtime callers."""
        return (
            self.enabled_step_names
            if include_optional
            else self._required_enabled_step_names
        )

    def is_bootstrap_owned_artifact_key(self, key: str) -> bool:
        return str(key) in self.bootstrap_owned_artifact_keys

    def is_restart_prebootstrap_deferred_artifact_key(self, key: str) -> bool:
        return str(key) in self.restart_prebootstrap_deferred_artifact_keys

    def restart_frontier(self) -> Optional[RestartFrontierContract]:
        """Return the effective restart frontier for this run, if any."""
        return self.restart_frontier_contract

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the surface for debugging and failure diagnostics."""
        return {
            "profile": self.profile.to_dict(),
            "run_mode": self.run_mode.value,
            "enabled_step_names": sorted(self.enabled_step_names),
            "enabled_model_names": sorted(self.enabled_model_names),
            "enabled_stage_names": sorted(self.enabled_stage_names),
            "bootstrap_owned_artifact_keys": sorted(self.bootstrap_owned_artifact_keys),
            "restart_prebootstrap_deferred_artifact_keys": sorted(
                self.restart_prebootstrap_deferred_artifact_keys
            ),
            "restart_frontier_contract": (
                asdict(self.restart_frontier_contract)
                if self.restart_frontier_contract is not None
                else None
            ),
            "step_surfaces": {
                name: surface.to_dict()
                for name, surface in sorted(self.step_surfaces.items())
            },
        }


def _ordered_unique(*groups: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


def _enabled_model_names(settings: Any) -> FrozenSet[str]:
    models = getattr(getattr(settings, "run", None), "models", None)
    if models is None:
        return frozenset()

    values = []
    for attr_name in (
        "land_use",
        "vehicle_ownership",
        "activity_demand",
        "traffic_assignment",
        "travel",
        "postprocessing",
    ):
        value = getattr(models, attr_name, None)
        text = str(value).strip() if value is not None else ""
        if text:
            values.append(text)
    return frozenset(dict.fromkeys(values))


def _enabled_stage_names(profile: WorkflowProfile) -> FrozenSet[str]:
    stage_names = []
    if profile.land_use_enabled:
        stage_names.append("land_use")
    if profile.vehicle_ownership_model_enabled:
        stage_names.append("vehicle_ownership_model")
    if profile.activity_demand_enabled:
        stage_names.append("activity_demand")
    elif profile.land_use_enabled:
        stage_names.append("activity_demand_directly_from_land_use")
    if profile.traffic_assignment_enabled:
        stage_names.append("traffic_assignment")
    if profile.supply_demand_loop_enabled:
        stage_names.append("supply_demand_loop")
    return frozenset(stage_names)


def _bootstrap_owned_artifact_keys(settings: Any) -> FrozenSet[str]:
    keys = []
    models = getattr(getattr(settings, "run", None), "models", None)
    activity_demand = getattr(models, "activity_demand", None) if models is not None else None
    land_use = getattr(models, "land_use", None) if models is not None else None
    vehicle_ownership = (
        getattr(models, "vehicle_ownership", None) if models is not None else None
    )
    traffic_assignment = None
    if models is not None:
        traffic_assignment = getattr(models, "traffic_assignment", None) or getattr(
            models, "travel", None
        )

    if (
        land_use == "urbansim"
        or activity_demand == "activitysim"
        or vehicle_ownership == "atlas"
    ):
        keys.append("usim_datastore_base_h5")

    if activity_demand == "activitysim":
        main_configs_dir = (
            getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
            or "configs"
        )
        keys.extend(
            f"activitysim_config_settings_yaml_{dirname}"
            for dirname in required_asim_config_dirs(main_configs_dir)
        )

    if traffic_assignment == "beam":
        keys.extend(
            (
                "beam_mutable_data_dir",
                "beam_region_input_dir",
                "beam_primary_config_file",
            )
        )

    return frozenset(keys)


def _restart_frontier_contract(
    *,
    settings: Any,
    state: Any,
) -> Optional[RestartFrontierContract]:
    current_major_stage = getattr(state, "current_major_stage", None)
    current_sub_stage = getattr(state, "current_sub_stage", None)
    stage_enum = getattr(state, "Stage", None)
    if stage_enum is None:
        return None
    if current_major_stage != getattr(stage_enum, "supply_demand_loop", None):
        return None
    if current_sub_stage != getattr(stage_enum, "traffic_assignment", None):
        return None

    models = getattr(getattr(settings, "run", None), "models", None)
    if models is None:
        return None
    if getattr(models, "activity_demand", None) != "activitysim":
        return None
    traffic_assignment = getattr(models, "traffic_assignment", None) or getattr(
        models, "travel", None
    )
    if traffic_assignment != "beam":
        return None

    return RestartFrontierContract(
        frontier_stage="traffic_assignment",
        frontier_step="beam_preprocess",
        required_keys=(
            "beam_plans_asim_out",
            "households_asim_out",
            "persons_asim_out",
            ZARR_SKIMS,
        ),
    )


def _make_input_role_policy(
    *,
    rule: Any,
    overrides: Mapping[str, Any],
) -> InputRolePolicy:
    workspace_archive_fallback_allowed = bool(
        overrides.get("workspace_archive_fallback_allowed", rule.allow_fallback)
    )
    restart_may_restore_synthetically = bool(
        overrides.get("restart_may_restore_synthetically", rule.allow_fallback)
    )
    return InputRolePolicy(
        explicit_inputs_allowed=bool(rule.allow_explicit),
        coupler_fallback_allowed=bool(rule.allow_coupler),
        workspace_archive_fallback_allowed=workspace_archive_fallback_allowed,
        restart_may_restore_synthetically=restart_may_restore_synthetically,
        restart_requires_explicit_before_execution=bool(
            overrides.get("restart_requires_explicit_before_execution", False)
        ),
    )


def _build_step_runtime_surface(
    *,
    settings: Any,
    step_name: str,
    profile: WorkflowProfile,
    run_mode: RunMode,
    enabled_step_names: FrozenSet[str],
) -> StepRuntimeSurface:
    """Project one step's static contract into the current runtime shape.

    This combines:
    - catalog-declared required/optional inputs and outputs
    - binding-spec fallback/explicit/coupler policy
    - restart- or run-shape-specific overlays
    """
    from pilates.workflows.binding import binding_spec_for_step_name

    step_spec = workflow_step_spec_for_step_name(step_name)
    if step_spec is None:
        raise KeyError(f"Unknown workflow step '{step_name}'.")
    contracts = workflow_step_contracts_by_name(settings=settings)
    contract = contracts.get(step_name, {})
    binding_spec = binding_spec_for_step_name(step_name, settings=settings)

    required_inputs = tuple(contract.get("input_keys", ()))
    optional_inputs = tuple(contract.get("optional_input_keys", ()))
    required_outputs = tuple(contract.get("output_keys", ()))
    optional_outputs = tuple(contract.get("optional_output_keys", ()))
    # Runtime overlays are intentionally resolved through a tiny registry module
    # so `surface.py` stays a projection engine instead of accumulating
    # step-specific conditionals over time.
    required_inputs, optional_inputs = resolve_runtime_input_output_overrides(
        step_name=step_name,
        profile=profile,
        run_mode=run_mode,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
    )

    policy_overrides = resolve_runtime_role_policy_overrides(
        step_name=step_name,
        profile=profile,
        run_mode=run_mode,
    )
    input_role_policies: Dict[str, InputRolePolicy] = {}
    declared_input_keys = set(_ordered_unique(required_inputs, optional_inputs))
    if binding_spec is not None:
        for rule in binding_spec.artifact_rules:
            if (
                rule.semantic_key not in declared_input_keys
                and rule.semantic_key not in policy_overrides
            ):
                continue
            input_role_policies[rule.semantic_key] = _make_input_role_policy(
                rule=rule,
                overrides=policy_overrides.get(rule.semantic_key, {}),
            )
    for semantic_key, overrides in policy_overrides.items():
        if semantic_key in input_role_policies:
            continue
        input_role_policies[semantic_key] = InputRolePolicy(
            explicit_inputs_allowed=True,
            coupler_fallback_allowed=True,
            workspace_archive_fallback_allowed=bool(
                overrides.get("workspace_archive_fallback_allowed", False)
            ),
            restart_may_restore_synthetically=bool(
                overrides.get("restart_may_restore_synthetically", False)
            ),
            restart_requires_explicit_before_execution=bool(
                overrides.get("restart_requires_explicit_before_execution", False)
            ),
        )
    for key in _ordered_unique(required_inputs, optional_inputs):
        if key not in input_role_policies:
            input_role_policies[key] = InputRolePolicy(
                explicit_inputs_allowed=True,
                coupler_fallback_allowed=True,
                workspace_archive_fallback_allowed=False,
                restart_may_restore_synthetically=False,
                restart_requires_explicit_before_execution=False,
            )

    return StepRuntimeSurface(
        step_name=step_name,
        stage_name=step_spec.stage_name,
        phase=step_spec.phase,
        optional=bool(step_spec.optional),
        enabled=step_name in enabled_step_names,
        required_input_keys=required_inputs,
        optional_input_keys=optional_inputs,
        required_output_keys=required_outputs,
        optional_output_keys=optional_outputs,
        input_role_policies=input_role_policies,
    )


def _validate_surface_invariants(
    *,
    step_surfaces: Mapping[str, StepRuntimeSurface],
    enabled_step_names: FrozenSet[str],
    profile: WorkflowProfile,
) -> None:
    """Reject inconsistent projections before runtime execution starts.

    These checks are structural only. They deliberately do not touch the
    filesystem or validate artifact existence; bootstrap/restart code still
    owns operational checks.
    """
    for step_name in enabled_step_names:
        surface = step_surfaces[step_name]
        step_spec = workflow_step_spec_for_step_name(step_name)
        if step_spec is None:
            raise RuntimeError(f"Enabled workflow surface references unknown step '{step_name}'.")
        for dependency in step_spec.depends_on:
            if dependency in enabled_step_names:
                continue
            dependency_spec = workflow_step_spec_for_step_name(dependency)
            dependency_flag = (
                getattr(dependency_spec, "enabled_flag_attr", None)
                if dependency_spec is not None
                else None
            )
            if dependency_flag and not profile.model_enabled(dependency_flag):
                continue
            raise RuntimeError(
                f"Enabled step '{step_name}' depends on disabled step '{dependency}'."
            )
        known_inputs = set(_ordered_unique(surface.required_input_keys, surface.optional_input_keys))
        unknown = sorted(
            key
            for key, policy in surface.input_role_policies.items()
            if key not in known_inputs
            and not policy.restart_requires_explicit_before_execution
        )
        if unknown:
            raise RuntimeError(
                f"Step surface '{step_name}' has policies for undeclared input keys: {unknown}"
            )


def build_enabled_workflow_surface(
    settings: Any,
    *,
    state: Any = None,
) -> EnabledWorkflowSurface:
    """Build the runtime-enabled workflow surface for the current settings/state.

    Args:
        settings: Parsed PILATES settings.
        state: Optional workflow state. When present, restart-aware projections
            such as run mode and restart frontier are included.

    Returns:
        The enabled workflow surface that runtime callers should share.
    """
    enabled_flags = ensure_runtime_flags_initialized(settings)
    profile = workflow_profile_from_flags(enabled_flags)
    run_mode = (
        RunMode.RESTART
        if bool(getattr(state, "is_restart_run", False) or getattr(state, "run_info_path", None))
        else RunMode.FRESH
    )

    enabled_step_names = _enabled_schema_step_names_from_profile(
        profile,
        include_optional=True,
    )
    required_enabled_step_names = _enabled_schema_step_names_from_profile(
        profile,
        include_optional=False,
    )
    step_contracts = workflow_step_contracts_by_name(settings=settings)
    step_names = tuple(step_contracts.keys())
    step_surfaces = {
        step_name: _build_step_runtime_surface(
            settings=settings,
            step_name=step_name,
            profile=profile,
            run_mode=run_mode,
            enabled_step_names=enabled_step_names,
        )
        for step_name in step_names
    }
    _validate_surface_invariants(
        step_surfaces=step_surfaces,
        enabled_step_names=enabled_step_names,
        profile=profile,
    )

    bootstrap_owned_artifact_keys = _bootstrap_owned_artifact_keys(settings)
    restart_frontier_contract = (
        _restart_frontier_contract(settings=settings, state=state)
        if state is not None
        else None
    )

    return EnabledWorkflowSurface(
        profile=profile,
        run_mode=run_mode,
        enabled_step_names=enabled_step_names,
        enabled_model_names=_enabled_model_names(settings),
        enabled_stage_names=_enabled_stage_names(profile),
        step_surfaces=step_surfaces,
        bootstrap_owned_artifact_keys=bootstrap_owned_artifact_keys,
        restart_prebootstrap_deferred_artifact_keys=bootstrap_owned_artifact_keys,
        restart_frontier_contract=restart_frontier_contract,
        _required_enabled_step_names=required_enabled_step_names,
    )
