from __future__ import annotations

"""
Workflow binding-layer data structures.

This module is intentionally narrow for the foundation batch: it defines the
runtime binding policy objects and the ``BindingPlan -> consist.BindingResult``
adapter used at the ``scenario.run(...)`` boundary.

Semantic workflow contracts remain owned by ``catalog.py``. Binding specs may
derive their artifact universe from the catalog by reference so runtime binding
does not become a second manually maintained semantic registry.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Literal

from consist.types import BindingResult

from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import resolve_input_precedence
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
)


def _ordered_unique(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


@dataclass(frozen=True)
class ArtifactBindingRule:
    """
    Runtime binding policy for one semantic workflow artifact.
    """

    semantic_key: str
    required: bool = True
    allow_explicit: bool = True
    allow_coupler: bool = True
    allow_fallback: bool = False
    preferred_keys: tuple[str, ...] = ()
    fallback_provider: Optional[str] = None
    pass_mode: Literal["auto", "input_key_only", "explicit_only"] = "auto"


@dataclass(frozen=True)
class StepBindingSpec:
    """
    Runtime binding policy for a workflow step.

    When ``derive_from_catalog`` is true, semantic input/output keys are pulled
    from ``catalog.py`` by reference. The foundation batch keeps the API small
    while giving the execution layer a first-class binding surface.
    """

    step_name: str
    derive_from_catalog: bool = True
    artifact_rules: tuple[ArtifactBindingRule, ...] = ()
    required_output_paths: tuple[str, ...] = ()
    optional_output_paths: tuple[str, ...] = ()
    notes: Optional[str] = None

    @classmethod
    def from_catalog(
        cls,
        step_name: str,
        *,
        notes: Optional[str] = None,
    ) -> "StepBindingSpec":
        from pilates.workflows.catalog import workflow_step_spec_for_step_name

        step_spec = workflow_step_spec_for_step_name(step_name)
        if step_spec is None:
            raise KeyError(f"Unknown workflow step '{step_name}'.")

        artifact_rules = tuple(
            [
                *(
                    ArtifactBindingRule(semantic_key=key, required=True)
                    for key in step_spec.input_keys
                ),
                *(
                    ArtifactBindingRule(semantic_key=key, required=False)
                    for key in step_spec.optional_input_keys
                ),
            ]
        )
        return cls(
            step_name=step_name,
            derive_from_catalog=True,
            artifact_rules=artifact_rules,
            required_output_paths=tuple(step_spec.output_keys),
            optional_output_paths=tuple(step_spec.optional_output_keys),
            notes=notes,
        )

    def with_rule_overrides(
        self,
        *overrides: ArtifactBindingRule,
        notes: Optional[str] = None,
    ) -> "StepBindingSpec":
        by_key = {rule.semantic_key: rule for rule in self.artifact_rules}
        for override in overrides:
            existing = by_key.get(override.semantic_key)
            if existing is None:
                by_key[override.semantic_key] = override
                continue
            by_key[override.semantic_key] = ArtifactBindingRule(
                semantic_key=existing.semantic_key,
                required=override.required,
                allow_explicit=override.allow_explicit,
                allow_coupler=override.allow_coupler,
                allow_fallback=override.allow_fallback,
                preferred_keys=override.preferred_keys or existing.preferred_keys,
                fallback_provider=(
                    override.fallback_provider
                    if override.fallback_provider is not None
                    else existing.fallback_provider
                ),
                pass_mode=override.pass_mode,
            )
        return StepBindingSpec(
            step_name=self.step_name,
            derive_from_catalog=self.derive_from_catalog,
            artifact_rules=tuple(by_key.values()),
            required_output_paths=self.required_output_paths,
            optional_output_paths=self.optional_output_paths,
            notes=notes if notes is not None else self.notes,
        )

    def semantic_input_keys(self) -> tuple[str, ...]:
        if self.derive_from_catalog:
            from pilates.workflows.catalog import workflow_step_spec_for_step_name

            step_spec = workflow_step_spec_for_step_name(self.step_name)
            if step_spec is not None:
                return _ordered_unique(
                    step_spec.input_keys,
                    step_spec.optional_input_keys,
                )
        return tuple(rule.semantic_key for rule in self.artifact_rules)

    def semantic_output_keys(self) -> tuple[str, ...]:
        if self.derive_from_catalog:
            from pilates.workflows.catalog import workflow_step_spec_for_step_name

            step_spec = workflow_step_spec_for_step_name(self.step_name)
            if step_spec is not None:
                return tuple(step_spec.output_keys)
        return tuple(self.required_output_paths)


@dataclass(frozen=True)
class BindingPlan:
    """
    PILATES-local binding plan for a resolved workflow step.
    """

    step_name: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = field(default_factory=dict)
    input_keys: Optional[list[str]] = field(default_factory=list)
    optional_input_keys: Optional[list[str]] = field(default_factory=list)
    source_by_key: Dict[str, str] = field(default_factory=dict)
    coupler_key_by_key: Dict[str, str] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    output_paths: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def stepref_inputs(self) -> Optional[Dict[str, Any]]:
        return dict(self.inputs or {}) or None

    def stepref_input_keys(self) -> Optional[list[str]]:
        return list(self.input_keys or []) or None

    def stepref_optional_input_keys(self) -> Optional[list[str]]:
        return list(self.optional_input_keys or []) or None

    def to_binding_result(self) -> BindingResult:
        return BindingResult(
            inputs=dict(self.inputs or {}) or None,
            input_keys=list(self.input_keys or []) or None,
            optional_input_keys=list(self.optional_input_keys or []) or None,
            metadata=dict(self.metadata) if self.metadata else None,
        )

    def to_scenario_run_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"binding": self.to_binding_result()}
        if self.output_paths is not None:
            kwargs["output_paths"] = dict(self.output_paths)
        return kwargs


BindingFallbackProvider = Callable[..., Optional[Mapping[str, Any]]]


def _urbansim_inputs_for_year(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    if settings is None or state is None or workspace is None or year is None:
        return None
    from pilates.urbansim.inputs import build_urbansim_inputs

    inputs, _ = build_urbansim_inputs(settings, state, workspace, year)
    return inputs


_FALLBACK_PROVIDERS: Dict[str, BindingFallbackProvider] = {
    "urbansim_inputs_for_year": _urbansim_inputs_for_year,
}


def _pilot_binding_overrides() -> Dict[str, tuple[ArtifactBindingRule, ...]]:
    return {
        "activitysim_preprocess": (
            ArtifactBindingRule(
                semantic_key=USIM_H5_UPDATED,
                required=True,
                allow_fallback=True,
                preferred_keys=(
                    USIM_H5_UPDATED,
                    USIM_DATASTORE_CURRENT_H5,
                    USIM_DATASTORE_BASE_H5,
                ),
                fallback_provider="urbansim_inputs_for_year",
            ),
            ArtifactBindingRule(
                semantic_key=FINAL_SKIMS_OMX,
                required=False,
            ),
        ),
        "activitysim_compile": (
            ArtifactBindingRule(
                semantic_key=ASIM_OMX_SKIMS,
                required=True,
            ),
        ),
        "beam_preprocess": (
            ArtifactBindingRule(
                semantic_key=BEAM_PLANS_IN,
                required=True,
                preferred_keys=(BEAM_PLANS_IN, "beam_plans_asim_out", BEAM_PLANS_OUT),
            ),
            ArtifactBindingRule(
                semantic_key=BEAM_HOUSEHOLDS_IN,
                required=True,
                preferred_keys=(BEAM_HOUSEHOLDS_IN, "households_asim_out"),
            ),
            ArtifactBindingRule(
                semantic_key=BEAM_PERSONS_IN,
                required=True,
                preferred_keys=(BEAM_PERSONS_IN, "persons_asim_out"),
            ),
            ArtifactBindingRule(
                semantic_key=LINKSTATS_WARMSTART,
                required=False,
                preferred_keys=(LINKSTATS_WARMSTART, LINKSTATS),
            ),
            ArtifactBindingRule(
                semantic_key=BEAM_CONFIG_FILE,
                required=True,
            ),
        ),
    }


def binding_spec_for_step_name(step_name: str) -> Optional[StepBindingSpec]:
    """
    Return a runtime binding spec for ``step_name``.

    The base binding surface derives from the semantic catalog. Pilot-step
    overrides only declare runtime resolution policy, not a second artifact
    registry.
    """

    try:
        spec = StepBindingSpec.from_catalog(step_name)
    except KeyError:
        return None
    overrides = _pilot_binding_overrides().get(step_name)
    if not overrides:
        return spec
    return spec.with_rule_overrides(*overrides)


def _binding_rule_lookup(
    spec: Optional[StepBindingSpec],
) -> Dict[str, ArtifactBindingRule]:
    if spec is None:
        return {}
    return {rule.semantic_key: rule for rule in spec.artifact_rules}


def _lookup_fallback_inputs(
    *,
    rule: ArtifactBindingRule,
    explicit_fallback_inputs: Optional[Mapping[str, Any]],
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: Optional[CouplerProtocol],
    year: Optional[int],
) -> Optional[Mapping[str, Any]]:
    combined: Dict[str, Any] = dict(explicit_fallback_inputs or {})
    if rule.fallback_provider:
        provider = _FALLBACK_PROVIDERS.get(rule.fallback_provider)
        if provider is None:
            raise KeyError(
                f"Unknown fallback provider '{rule.fallback_provider}' for binding rule "
                f"'{rule.semantic_key}'."
            )
        provided = provider(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            explicit_fallback_inputs=explicit_fallback_inputs,
            year=year,
        )
        if provided:
            combined.update(provided)
    return combined or None


def _resolve_rule_binding(
    *,
    rule: ArtifactBindingRule,
    coupler: Optional[CouplerProtocol],
    explicit_inputs: Optional[Mapping[str, Any]],
    fallback_inputs: Optional[Mapping[str, Any]],
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
) -> tuple[str, Optional[str], Optional[Any], Optional[str]]:
    candidates = rule.preferred_keys or (rule.semantic_key,)
    rule_fallback_inputs = (
        _lookup_fallback_inputs(
            rule=rule,
            explicit_fallback_inputs=fallback_inputs if rule.allow_fallback else None,
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            year=year,
        )
        if rule.allow_fallback or rule.fallback_provider
        else None
    )
    scoped_coupler = coupler if rule.allow_coupler else None

    for candidate in candidates:
        resolved = resolve_input_precedence(
            key=candidate,
            coupler=scoped_coupler,
            explicit_inputs=explicit_inputs if rule.allow_explicit else None,
            fallback_inputs=rule_fallback_inputs,
        )
        if resolved.source == "missing":
            continue
        selected_key = resolved.storage_key or candidate
        if rule.pass_mode == "explicit_only" and resolved.source == "coupler":
            continue
        if rule.pass_mode == "input_key_only" and resolved.source in {
            "explicit",
            "fallback",
        }:
            continue
        return resolved.source, selected_key, resolved.value, candidate
    return "missing", None, None, None


def build_binding_plan(
    *,
    step_name: str,
    coupler: Optional[CouplerProtocol] = None,
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
    required_keys: Optional[Iterable[str]] = None,
    optional_keys: Optional[Iterable[str]] = None,
    output_paths: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    settings: Any = None,
    state: Any = None,
    workspace: Any = None,
    year: Optional[int] = None,
) -> BindingPlan:
    spec = binding_spec_for_step_name(step_name)
    rule_lookup = _binding_rule_lookup(spec)
    if year is None and state is not None:
        year = getattr(state, "year", None)

    required_semantic_keys = tuple(
        required_keys
        if required_keys is not None
        else (
            rule.semantic_key
            for rule in (spec.artifact_rules if spec is not None else ())
            if rule.required
        )
    )
    optional_semantic_keys = tuple(
        optional_keys
        if optional_keys is not None
        else (
            rule.semantic_key
            for rule in (spec.artifact_rules if spec is not None else ())
            if not rule.required
        )
    )
    caller_scoped_fallback_inputs = fallback_inputs is not None and (
        required_keys is not None or optional_keys is not None
    )

    plan_inputs: Dict[str, Any] = {}
    plan_input_keys: list[str] = []
    source_by_key: Dict[str, str] = {}
    coupler_key_by_key: Dict[str, str] = {}
    missing_required: list[str] = []
    selected_key_by_semantic_key: Dict[str, str] = {}

    def _default_rule(semantic_key: str, *, required: bool) -> ArtifactBindingRule:
        return ArtifactBindingRule(semantic_key=semantic_key, required=required)

    for semantic_key, is_required in (
        [(key, True) for key in required_semantic_keys]
        + [(key, False) for key in optional_semantic_keys]
    ):
        rule = rule_lookup.get(semantic_key) or _default_rule(
            semantic_key, required=is_required
        )
        if caller_scoped_fallback_inputs and not rule.allow_fallback:
            rule = ArtifactBindingRule(
                semantic_key=rule.semantic_key,
                required=rule.required,
                allow_explicit=rule.allow_explicit,
                allow_coupler=rule.allow_coupler,
                allow_fallback=True,
                preferred_keys=rule.preferred_keys,
                fallback_provider=rule.fallback_provider,
                pass_mode=rule.pass_mode,
            )
        source, selected_key, value, matched_candidate = _resolve_rule_binding(
            rule=rule,
            coupler=coupler,
            explicit_inputs=explicit_inputs,
            fallback_inputs=fallback_inputs,
            settings=settings,
            state=state,
            workspace=workspace,
            year=year,
        )
        source_by_key[semantic_key] = source
        if selected_key is not None:
            coupler_key_by_key[semantic_key] = selected_key
            selected_key_by_semantic_key[semantic_key] = matched_candidate or selected_key
        if source == "coupler" and selected_key is not None:
            plan_input_keys.append(selected_key)
        elif source in {"explicit", "fallback"}:
            plan_inputs[semantic_key] = value
        elif is_required:
            missing_required.append(semantic_key)

    plan_metadata = dict(metadata or {})
    if selected_key_by_semantic_key:
        plan_metadata.setdefault("selected_key_by_semantic_key", {}).update(
            selected_key_by_semantic_key
        )
    if spec is not None and spec.notes and "notes" not in plan_metadata:
        plan_metadata["notes"] = spec.notes

    return BindingPlan(
        step_name=step_name,
        inputs=plan_inputs,
        input_keys=list(dict.fromkeys(plan_input_keys)),
        source_by_key=source_by_key,
        coupler_key_by_key=coupler_key_by_key,
        missing_required=missing_required,
        output_paths=dict(output_paths) if output_paths is not None else None,
        metadata=plan_metadata or None,
    )
