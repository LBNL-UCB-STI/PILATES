from __future__ import annotations

import logging
import os
from pathlib import Path

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
from pilates.utils.coupler_helpers import artifact_to_path, resolve_input_precedence
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.utils.io import get_traffic_assignment_model
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    ASIM_SHARROW_CACHE_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
)

logger = logging.getLogger(__name__)

_CANDIDATE_PATHS_METADATA_KEY = "candidate_paths_by_semantic_key"


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


@dataclass(frozen=True)
class StageBoundaryDurabilityRule:
    """
    Runtime policy for artifacts that must survive a stage boundary.
    """

    name: str
    semantic_keys: tuple[str, ...]
    resolve: Callable[..., Optional[Mapping[str, str]]]
    notes: Optional[str] = None


@dataclass(frozen=True)
class RestartArtifactRequirementRule:
    """
    Runtime policy for artifacts that restart preflight should keep present.
    """

    name: str
    semantic_keys: tuple[str, ...]
    resolve: Callable[..., Optional[Mapping[str, str]]]
    notes: Optional[str] = None


BindingFallbackProvider = Callable[..., Optional[Mapping[str, Any]]]


def activitysim_datastore_selection_rules() -> tuple[ArtifactBindingRule, ...]:
    """
    Shared current-vs-base datastore preference for ActivitySim input selection.
    """
    return (
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_CURRENT_H5,
            required=True,
            preferred_keys=(
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
        ),
    )


def beam_preprocess_binding_plan(
    *,
    coupler: Optional[CouplerProtocol],
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
    activity_demand_outputs: Optional[Mapping[str, Any]],
    previous_beam_outputs: Optional[Mapping[str, Any]],
) -> BindingPlan:
    """
    Build the BEAM preprocess binding plan from explicit upstream artifacts.

    The plan itself owns fallback selection for the BEAM-only exchange inputs,
    warm-start linkstats, and optional ATLAS vehicles2 resolution.
    """
    activity_demand_model = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(activity_demand_model, "activity_demand", None)
    if activity_demand_model is not None and activity_demand_outputs is None:
        if previous_beam_outputs is None:
            raise RuntimeError(
                "TrafficAssignment iteration 0 requires activity_demand_outputs "
                "or previous_beam_outputs. Ensure ActivityDemand completed or "
                "provide warm-start outputs before running BEAM."
            )

    explicit_inputs: Dict[str, Any] = {}
    if activity_demand_outputs is not None:
        activity_keys = {
            "beam_plans_asim_out",
            "beam_plans_out",
            "households_asim_out",
            "linkstats",
            "persons_asim_out",
        }
        for key, value in activity_demand_outputs.items():
            if key in activity_keys:
                explicit_inputs[key] = value
    if previous_beam_outputs is not None:
        for key, value in previous_beam_outputs.items():
            if key.startswith("linkstats"):
                explicit_inputs[key] = value

    if activity_demand_model is None:
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
                value = artifact_to_path(get_value(key), workspace)
                if value:
                    explicit_inputs.setdefault(key, value)
        exchange_inputs = _beam_preprocess_exchange_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
        )
        if exchange_inputs:
            for key, value in exchange_inputs.items():
                explicit_inputs.setdefault(key, value)

    explicit_linkstats_value = next(
        (
            value
            for key, value in explicit_inputs.items()
            if key.startswith("linkstats")
        ),
        None,
    )
    if explicit_linkstats_value is not None:
        explicit_inputs.setdefault(LINKSTATS_WARMSTART, explicit_linkstats_value)
    else:
        warmstart_inputs = _beam_preprocess_warmstart_inputs(
            settings=settings,
            coupler=coupler,
            workspace=workspace,
        )
        if warmstart_inputs:
            for key, value in warmstart_inputs.items():
                explicit_inputs.setdefault(key, value)

    atlas_inputs = _beam_preprocess_atlas_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if atlas_inputs:
        for key, value in atlas_inputs.items():
            explicit_inputs.setdefault(key, value)

    return build_binding_plan(
        step_name="beam_preprocess",
        coupler=coupler,
        explicit_inputs=explicit_inputs,
        settings=settings,
        state=state,
        workspace=workspace,
        year=year,
    )


def urbansim_datastore_selection_rules(
    *,
    fallback_provider: str = "urbansim_inputs_for_year",
) -> tuple[ArtifactBindingRule, ...]:
    """
    Shared current/base datastore selection policy for UrbanSim input assembly.
    """
    return (
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_BASE_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(
                USIM_DATASTORE_BASE_H5,
                USIM_DATASTORE_CURRENT_H5,
            ),
            fallback_provider=fallback_provider,
        ),
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_CURRENT_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
            fallback_provider=fallback_provider,
        ),
    )


def _archive_fallback_path(
    *,
    state: Any,
    workspace: Any,
    local_path: Path,
) -> Optional[Path]:
    run_info_path = getattr(state, "run_info_path", None)
    full_path = getattr(workspace, "full_path", None)
    if not run_info_path or full_path is None:
        return None
    archive_run_dir = Path(run_info_path).expanduser().resolve().parent
    local_root = Path(full_path).expanduser().resolve()
    try:
        rel = local_path.expanduser().resolve().relative_to(local_root)
    except Exception:
        return None
    return archive_run_dir / rel


def _first_existing_path(*paths: Optional[Path]) -> Optional[Path]:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _candidate_paths_metadata(
    *paths_by_semantic_key: tuple[str, Sequence[Optional[Path]]],
) -> Dict[str, list[str]]:
    metadata: Dict[str, list[str]] = {}
    for semantic_key, paths in paths_by_semantic_key:
        ordered_paths = [
            str(path)
            for path in paths
            if path is not None and str(path)
        ]
        if ordered_paths:
            metadata[semantic_key] = list(dict.fromkeys(ordered_paths))
    return metadata


def _urbansim_datastore_candidates_for_year(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
) -> Optional[Mapping[str, Any]]:
    if settings is None or state is None or workspace is None or year is None:
        return None
    get_usim_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    is_start_year = getattr(state, "is_start_year", None)
    if not callable(get_usim_dir) or not callable(is_start_year):
        return None

    from pilates.urbansim import postprocessor as usim_post

    usim_data_dir = Path(get_usim_dir())
    input_path = usim_data_dir / usim_post.get_usim_datastore_fname(settings, io="input")
    input_archive_path = _archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=input_path,
    )
    base_path = _first_existing_path(input_path, input_archive_path)
    candidate_paths = _candidate_paths_metadata(
        (USIM_DATASTORE_BASE_H5, (input_path, input_archive_path)),
    )

    mapping: Dict[str, Any] = {}
    if base_path is not None:
        mapping[USIM_DATASTORE_BASE_H5] = str(base_path)

    if is_start_year():
        candidate_paths.update(
            _candidate_paths_metadata(
                (USIM_DATASTORE_CURRENT_H5, (input_path, input_archive_path)),
            )
        )
        if base_path is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = str(base_path)
        if candidate_paths:
            mapping[_CANDIDATE_PATHS_METADATA_KEY] = candidate_paths
        return mapping or None

    output_path = usim_data_dir / usim_post.get_usim_datastore_fname(
        settings, io="output", year=year
    )
    output_archive_path = _archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=output_path,
    )
    current_path = _first_existing_path(output_path, output_archive_path)
    candidate_paths.update(
        _candidate_paths_metadata(
            (USIM_DATASTORE_CURRENT_H5, (output_path, output_archive_path)),
        )
    )
    if current_path is not None:
        mapping[USIM_DATASTORE_CURRENT_H5] = str(current_path)
    if candidate_paths:
        mapping[_CANDIDATE_PATHS_METADATA_KEY] = candidate_paths
    return mapping or None


def _urbansim_inputs_for_year(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    return _urbansim_datastore_candidates_for_year(
        settings=settings,
        state=state,
        workspace=workspace,
        year=year,
    )


def _activitysim_input_datastore(
    *,
    settings: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    if settings is None or workspace is None:
        return None
    get_usim_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    if not callable(get_usim_dir):
        return None

    from pilates.activitysim.postprocessor import get_usim_datastore_fname

    candidate = os.path.join(
        get_usim_dir(),
        get_usim_datastore_fname(settings, io="input"),
    )
    if not os.path.exists(candidate):
        return None
    return {USIM_DATASTORE_BASE_H5: candidate}


def _beam_preprocess_exchange_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None

    activity_demand_model = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(activity_demand_model, "activity_demand", None)
    if activity_demand_model is not None:
        return None

    from pilates.beam.beam_exchange import register_existing_beam_exchange_inputs

    try:
        record_store = register_existing_beam_exchange_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "BEAM preprocess could not seed default exchange inputs: %s",
            exc,
        )
        return None

    artifacts: Dict[str, Any] = {}
    workspace_root = getattr(workspace, "full_path", None)
    for record in record_store.all_records():
        key = getattr(record, "short_name", None)
        if key not in {BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN}:
            continue
        path = record.get_absolute_path(base_path=workspace_root)
        if path and os.path.exists(path):
            artifacts[key] = path
    return artifacts or None


def _beam_preprocess_warmstart_inputs(
    *,
    settings: Any,
    coupler: Optional[CouplerProtocol],
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None

    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        value = get_value(LINKSTATS_WARMSTART)
        warmstart_path = artifact_to_path(value, workspace)
        if warmstart_path and os.path.exists(warmstart_path):
            return {LINKSTATS_WARMSTART: warmstart_path}

    warmstart_path = resolve_initial_linkstats_path(settings, workspace)
    if warmstart_path:
        return {LINKSTATS_WARMSTART: warmstart_path}
    return None


def _beam_preprocess_atlas_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None
    if not getattr(settings, "vehicle_ownership_model_enabled", False):
        return None

    current_iter = getattr(state, "current_inner_iter", getattr(state, "iteration", 0))
    if current_iter != 0:
        return None

    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is None:
        return None

    atlas_output_dir = workspace.get_atlas_output_dir()
    candidates = [
        os.path.join(atlas_output_dir, f"vehicles2_{forecast_year}.csv"),
        os.path.join(atlas_output_dir, f"vehicles2_{forecast_year - 1}.csv"),
    ]
    for atlas_vehicle_path in candidates:
        if os.path.exists(atlas_vehicle_path):
            return {ATLAS_VEHICLES2_OUTPUT: atlas_vehicle_path}
    return None


_FALLBACK_PROVIDERS: Dict[str, BindingFallbackProvider] = {
    "urbansim_inputs_for_year": _urbansim_inputs_for_year,
    "activitysim_input_datastore": _activitysim_input_datastore,
    "beam_preprocess_exchange_inputs": _beam_preprocess_exchange_inputs,
    "beam_preprocess_warmstart_inputs": _beam_preprocess_warmstart_inputs,
    "beam_preprocess_atlas_inputs": _beam_preprocess_atlas_inputs,
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
        "activitysim_postprocess": (
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_BASE_H5,
                required=False,
                allow_fallback=True,
                fallback_provider="activitysim_input_datastore",
            ),
        ),
        "atlas_preprocess": (
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_CURRENT_H5,
                required=True,
                allow_fallback=True,
                preferred_keys=(USIM_DATASTORE_CURRENT_H5, USIM_H5_UPDATED),
            ),
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_BASE_H5,
                required=True,
                allow_fallback=True,
                preferred_keys=(
                    USIM_DATASTORE_BASE_H5,
                    USIM_DATASTORE_CURRENT_H5,
                    USIM_H5_UPDATED,
                ),
            ),
        ),
        "beam_preprocess": (
            ArtifactBindingRule(
                semantic_key=BEAM_PLANS_IN,
                required=True,
                preferred_keys=(BEAM_PLANS_IN, "beam_plans_asim_out", BEAM_PLANS_OUT),
                allow_fallback=True,
                fallback_provider="beam_preprocess_exchange_inputs",
            ),
            ArtifactBindingRule(
                semantic_key=BEAM_HOUSEHOLDS_IN,
                required=True,
                preferred_keys=(BEAM_HOUSEHOLDS_IN, "households_asim_out"),
                allow_fallback=True,
                fallback_provider="beam_preprocess_exchange_inputs",
            ),
            ArtifactBindingRule(
                semantic_key=BEAM_PERSONS_IN,
                required=True,
                preferred_keys=(BEAM_PERSONS_IN, "persons_asim_out"),
                allow_fallback=True,
                fallback_provider="beam_preprocess_exchange_inputs",
            ),
            ArtifactBindingRule(
                semantic_key=LINKSTATS_WARMSTART,
                required=False,
                preferred_keys=(LINKSTATS_WARMSTART, LINKSTATS),
                allow_fallback=True,
                fallback_provider="beam_preprocess_warmstart_inputs",
            ),
            ArtifactBindingRule(
                semantic_key=ATLAS_VEHICLES2_OUTPUT,
                required=False,
                allow_fallback=True,
                fallback_provider="beam_preprocess_atlas_inputs",
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


def _split_candidate_paths_metadata(
    fallback_inputs: Optional[Mapping[str, Any]],
) -> tuple[Optional[Mapping[str, Any]], Dict[str, list[str]]]:
    if not fallback_inputs:
        return None, {}
    raw_candidate_paths = fallback_inputs.get(_CANDIDATE_PATHS_METADATA_KEY)
    if not isinstance(raw_candidate_paths, Mapping):
        return fallback_inputs, {}

    cleaned = dict(fallback_inputs)
    cleaned.pop(_CANDIDATE_PATHS_METADATA_KEY, None)
    candidate_paths_by_semantic_key: Dict[str, list[str]] = {}
    for semantic_key, candidate_paths in raw_candidate_paths.items():
        if not isinstance(semantic_key, str):
            continue
        if not isinstance(candidate_paths, Sequence) or isinstance(
            candidate_paths, (str, bytes)
        ):
            continue
        ordered_paths = [
            str(path)
            for path in candidate_paths
            if path is not None and str(path)
        ]
        if ordered_paths:
            candidate_paths_by_semantic_key[semantic_key] = list(
                dict.fromkeys(ordered_paths)
            )
    return cleaned or None, candidate_paths_by_semantic_key


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
) -> tuple[str, Optional[str], Optional[Any], Optional[str], Dict[str, list[str]]]:
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
    rule_fallback_inputs, candidate_paths_by_semantic_key = _split_candidate_paths_metadata(
        rule_fallback_inputs
    )
    scoped_coupler = coupler if rule.allow_coupler else None

    fallback_passes = (None, rule_fallback_inputs) if rule_fallback_inputs else (None,)
    for pass_fallback_inputs in fallback_passes:
        for candidate in candidates:
            resolved = resolve_input_precedence(
                key=candidate,
                coupler=scoped_coupler,
                explicit_inputs=explicit_inputs if rule.allow_explicit else None,
                fallback_inputs=pass_fallback_inputs,
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
            return (
                resolved.source,
                selected_key,
                resolved.value,
                candidate,
                candidate_paths_by_semantic_key,
            )
    return "missing", None, None, None, candidate_paths_by_semantic_key


def build_binding_plan(
    *,
    step_name: str,
    coupler: Optional[CouplerProtocol] = None,
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
    artifact_rules: Optional[Iterable[ArtifactBindingRule]] = None,
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
    for rule in artifact_rules or ():
        rule_lookup[rule.semantic_key] = rule
    if year is None and state is not None:
        year = getattr(state, "year", None)

    required_semantic_keys = tuple(
        required_keys
        if required_keys is not None
        else (
            rule.semantic_key
            for rule in rule_lookup.values()
            if rule.required
        )
    )
    optional_semantic_keys = tuple(
        optional_keys
        if optional_keys is not None
        else (
            rule.semantic_key
            for rule in rule_lookup.values()
            if not rule.required
        )
    )
    caller_scoped_fallback_inputs = fallback_inputs is not None and (
        required_keys is not None or optional_keys is not None
    )

    plan_inputs: Dict[str, Any] = {}
    plan_input_keys: list[str] = []
    plan_optional_input_keys: list[str] = []
    source_by_key: Dict[str, str] = {}
    coupler_key_by_key: Dict[str, str] = {}
    missing_required: list[str] = []
    selected_key_by_semantic_key: Dict[str, str] = {}
    candidate_paths_by_semantic_key: Dict[str, list[str]] = {}

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
        source, selected_key, value, matched_candidate, candidate_paths = _resolve_rule_binding(
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
        if candidate_paths:
            candidate_paths_by_semantic_key.update(candidate_paths)
        if selected_key is not None:
            coupler_key_by_key[semantic_key] = selected_key
            selected_key_by_semantic_key[semantic_key] = matched_candidate or selected_key
        if source == "coupler" and selected_key is not None:
            if is_required:
                plan_input_keys.append(selected_key)
            else:
                plan_optional_input_keys.append(selected_key)
        elif source in {"explicit", "fallback"}:
            plan_inputs[semantic_key] = value
        elif is_required:
            missing_required.append(semantic_key)

    plan_metadata = dict(metadata or {})
    if selected_key_by_semantic_key:
        plan_metadata.setdefault("selected_key_by_semantic_key", {}).update(
            selected_key_by_semantic_key
        )
    if candidate_paths_by_semantic_key:
        plan_metadata.setdefault(_CANDIDATE_PATHS_METADATA_KEY, {}).update(
            candidate_paths_by_semantic_key
        )
    if spec is not None and spec.notes and "notes" not in plan_metadata:
        plan_metadata["notes"] = spec.notes

    return BindingPlan(
        step_name=step_name,
        inputs=plan_inputs,
        input_keys=list(dict.fromkeys(plan_input_keys)),
        optional_input_keys=list(dict.fromkeys(plan_optional_input_keys)),
        source_by_key=source_by_key,
        coupler_key_by_key=coupler_key_by_key,
        missing_required=missing_required,
        output_paths=dict(output_paths) if output_paths is not None else None,
        metadata=plan_metadata or None,
    )


def build_key_only_binding_plan(
    *,
    step_name: str,
    input_keys: Optional[Iterable[str]] = None,
    optional_input_keys: Optional[Iterable[str]] = None,
    coupler: Optional[CouplerProtocol] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    settings: Any = None,
    state: Any = None,
    workspace: Any = None,
    year: Optional[int] = None,
) -> BindingPlan:
    """
    Build a binding plan for steps that consume coupler-backed keys only.

    This keeps dynamic key lists on the shared binding path so stages no longer
    need to assemble raw ``BindingPlan(input_keys=...)`` envelopes by hand.
    """
    ordered_input_keys = list(dict.fromkeys(input_keys or ()))
    if not ordered_input_keys:
        return BindingPlan(
            step_name=step_name,
            metadata=dict(metadata) if metadata else None,
        )

    optional_key_set = set(optional_input_keys or ())
    required_keys = [key for key in ordered_input_keys if key not in optional_key_set]
    optional_keys = [key for key in ordered_input_keys if key in optional_key_set]
    return build_binding_plan(
        step_name=step_name,
        coupler=coupler,
        required_keys=required_keys,
        optional_keys=optional_keys or None,
        metadata=metadata,
        settings=settings,
        state=state,
        workspace=workspace,
        year=year,
    )


def _bootstrap_activitysim_durable_artifacts(
    *,
    settings: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    activity_demand_model = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(activity_demand_model, "activity_demand", None)
    if activity_demand_model != "activitysim":
        return None

    artifacts: Dict[str, str] = {}
    get_asim_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if callable(get_asim_output_dir):
        zarr_candidate = os.path.join(get_asim_output_dir(), "cache", "skims.zarr")
        if os.path.exists(zarr_candidate):
            artifacts[ZARR_SKIMS] = zarr_candidate

    sharrow_cache_dir = os.path.join(
        getattr(workspace, "full_path", ""),
        "shared_cache",
        "numba",
    )
    if os.path.isdir(sharrow_cache_dir):
        for _root, _dirs, files in os.walk(sharrow_cache_dir):
            if files:
                artifacts[ASIM_SHARROW_CACHE_DIR] = sharrow_cache_dir
                break
    return artifacts or None


def _bootstrap_beam_exchange_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    model_factory_cls: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None

    activity_demand_model = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(activity_demand_model, "activity_demand", None)
    if activity_demand_model is not None:
        return None

    model_factory = model_factory_cls()
    beam_preprocessor = model_factory.get_preprocessor("beam", state)
    existing_inputs = getattr(beam_preprocessor, "existing_beam_exchange_inputs", None)
    if not callable(existing_inputs):
        logger.debug(
            "BEAM preprocessor does not expose existing_beam_exchange_inputs(); "
            "skipping bootstrap coupler seeding."
        )
        return None

    try:
        record_store = existing_inputs(workspace)
    except FileNotFoundError as exc:
        logger.warning(
            "Bootstrap could not seed default BEAM inputs into coupler: %s",
            exc,
        )
        return None

    allowed_keys = {BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN}
    artifacts: Dict[str, str] = {}
    if record_store is not None:
        for record in record_store.all_records():
            key = getattr(record, "short_name", None)
            if key not in allowed_keys:
                continue
            path = record.get_absolute_path(base_path=getattr(workspace, "full_path", None))
            if not path or not os.path.exists(path):
                continue
            artifacts[key] = path
    return artifacts or None


def _bootstrap_beam_warmstart_artifacts(
    *,
    settings: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None

    activity_demand_model = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(activity_demand_model, "activity_demand", None)
    if activity_demand_model is not None:
        return None

    warmstart_path = resolve_initial_linkstats_path(settings, workspace)
    if not warmstart_path:
        return None
    return {LINKSTATS_WARMSTART: warmstart_path}


def bootstrap_stage_boundary_durability_policy() -> tuple[StageBoundaryDurabilityRule, ...]:
    """
    Stage-boundary durability policy for bootstrap seeding.

    The returned rules are consumable by runtime bootstrap code and keep the
    policy inspectable without hard-coding path inventories in the runtime.
    """

    return (
        StageBoundaryDurabilityRule(
            name="activitysim_bootstrap_artifacts",
            semantic_keys=(ZARR_SKIMS, ASIM_SHARROW_CACHE_DIR),
            resolve=_bootstrap_activitysim_durable_artifacts,
            notes=(
                "ActivitySim bootstrap should publish compiled skims and the "
                "persisted numba/sharrow cache when present."
            ),
        ),
        StageBoundaryDurabilityRule(
            name="beam_exchange_inputs",
            semantic_keys=(BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN),
            resolve=_bootstrap_beam_exchange_inputs,
            notes=(
                "BEAM-only bootstrap should seed the mutable exchange inputs "
                "already staged inside the BEAM workspace."
            ),
        ),
        StageBoundaryDurabilityRule(
            name="beam_warmstart",
            semantic_keys=(LINKSTATS_WARMSTART,),
            resolve=_bootstrap_beam_warmstart_artifacts,
            notes="BEAM-only bootstrap should seed the initial linkstats warmstart when configured.",
        ),
    )


def _restart_urbansim_required_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    get_usim_datastore_fname_fn: Callable[..., str],
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    current_stage = getattr(state, "current_major_stage", None)
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    requires_usim_base_h5 = (
        getattr(model_cfg, "land_use", None) == "urbansim"
        or getattr(model_cfg, "activity_demand", None) == "activitysim"
    )
    if not requires_usim_base_h5:
        return None

    usim_data_dir = workspace.get_usim_mutable_data_dir()
    usim_base_fname = get_usim_datastore_fname_fn(settings, io="input")
    required: Dict[str, str] = {
        USIM_DATASTORE_BASE_H5: os.path.join(usim_data_dir, usim_base_fname)
    }

    region = getattr(getattr(settings, "run", None), "region", None)
    urbansim_cfg = getattr(settings, "urbansim", None)
    if (
        current_stage in {
            None,
            workflow_stage.land_use,
        }
        and region
        and urbansim_cfg is not None
    ):
        region_id = (
            getattr(urbansim_cfg, "region_mappings", {})
            .get("region_to_region_id", {})
            .get(region)
        )
        if region_id:
            required.update(
                {
                    "omx_skims": os.path.join(usim_data_dir, f"skims_mpo_{region_id}.omx"),
                    "hh_size": os.path.join(usim_data_dir, f"hsize_ct_{region_id}.csv"),
                    "income_rates": os.path.join(
                        usim_data_dir, f"income_rates_{region_id}.csv"
                    ),
                    "relmap": os.path.join(usim_data_dir, f"relmap_{region_id}.csv"),
                    "schools": os.path.join(usim_data_dir, "schools_2010.csv"),
                    "school_districts": os.path.join(
                        usim_data_dir, "blocks_school_districts_2010.csv"
                    ),
                }
            )

    return required or None


def _restart_activitysim_required_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    required_asim_config_dirs_fn: Callable[[str], Sequence[str]],
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    current_stage = getattr(state, "current_major_stage", None)
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "activity_demand", None) != "activitysim":
        return None
    if current_stage not in {
        None,
        workflow_stage.supply_demand_loop,
        workflow_stage.activity_demand,
        workflow_stage.activity_demand_directly_from_land_use,
    }:
        return None

    asim_configs_dir = workspace.get_asim_mutable_configs_dir()
    main_configs_dir = (
        getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
        or "configs"
    )
    required: Dict[str, str] = {}
    for dirname in required_asim_config_dirs_fn(main_configs_dir):
        required[f"activitysim_config_settings_yaml_{dirname}"] = os.path.join(
            asim_configs_dir,
            dirname,
            "settings.yaml",
        )
    return required or None


def _restart_activitysim_iteration_outputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "activity_demand", None) != "activitysim":
        return None

    get_asim_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if not callable(get_asim_output_dir):
        return None

    if not (
        getattr(state, "current_major_stage", None) == workflow_stage.supply_demand_loop
        and getattr(state, "current_sub_stage", None) == workflow_stage.traffic_assignment
    ):
        return None

    return {
        "activitysim_iteration_beam_plans_parquet": os.path.join(
            get_asim_output_dir(),
            f"year-{getattr(state, 'current_year', 'unknown')}-iteration-{getattr(state, 'current_inner_iter', 0)}",
            "beam_plans.parquet",
        ),
        "activitysim_iteration_households_parquet": os.path.join(
            get_asim_output_dir(),
            f"year-{getattr(state, 'current_year', 'unknown')}-iteration-{getattr(state, 'current_inner_iter', 0)}",
            "households.parquet",
        ),
        "activitysim_iteration_persons_parquet": os.path.join(
            get_asim_output_dir(),
            f"year-{getattr(state, 'current_year', 'unknown')}-iteration-{getattr(state, 'current_inner_iter', 0)}",
            "persons.parquet",
        ),
    }


def _restart_beam_required_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    current_stage = getattr(state, "current_major_stage", None)
    if get_traffic_assignment_model(settings) != "beam":
        return None
    if current_stage not in {
        workflow_stage.supply_demand_loop,
        workflow_stage.traffic_assignment,
    }:
        return None

    get_beam_input_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
    region = getattr(getattr(settings, "run", None), "region", None)
    if not callable(get_beam_input_dir) or not region:
        return None

    beam_input_dir = get_beam_input_dir()
    beam_cfg = getattr(settings, "beam", None)
    beam_config_name = getattr(beam_cfg, "config", None)
    required: Dict[str, str] = {
        "beam_mutable_data_dir": beam_input_dir,
        "beam_region_input_dir": os.path.join(beam_input_dir, region),
    }
    if beam_config_name:
        required["beam_primary_config_file"] = os.path.join(
            beam_input_dir, region, beam_config_name
        )
    return required or None


def _restart_atlas_required_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    atlas_static_input_relpaths_fn: Callable[[Any], Sequence[str]],
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return None
    if getattr(state, "current_major_stage", None) != workflow_stage.vehicle_ownership_model:
        return None

    get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if not callable(get_atlas_input_dir):
        return None

    atlas_input_dir = get_atlas_input_dir()
    required: Dict[str, str] = {}
    for relpath in atlas_static_input_relpaths_fn(settings):
        required[f"atlas_static::{relpath}"] = os.path.join(atlas_input_dir, relpath)
    return required or None


def restart_required_local_artifact_policy() -> tuple[RestartArtifactRequirementRule, ...]:
    """
    Stage-aware restart artifact policy.

    Restart preflight consumes this policy to keep its required-local-artifact
    inventory inspectable without open-coding the stage branches in runtime
    logic.
    """

    return (
        RestartArtifactRequirementRule(
            name="urbansim_restart_artifacts",
            semantic_keys=(
                USIM_DATASTORE_BASE_H5,
                "omx_skims",
                "hh_size",
                "income_rates",
                "relmap",
                "schools",
                "school_districts",
            ),
            resolve=_restart_urbansim_required_artifacts,
            notes=(
                "UrbanSim restart should keep the mutable base datastore, "
                "land-use datastore handle, and local lookup inputs required "
                "to resume UrbanSim entry stages."
            ),
        ),
        RestartArtifactRequirementRule(
            name="activitysim_restart_configs",
            semantic_keys=(),
            resolve=_restart_activitysim_required_artifacts,
            notes=(
                "ActivitySim restart should keep the mutable config tree for "
                "stage entries that can re-enter ActivitySim directly."
            ),
        ),
        RestartArtifactRequirementRule(
            name="activitysim_iteration_outputs",
            semantic_keys=(
                "activitysim_iteration_beam_plans_parquet",
                "activitysim_iteration_households_parquet",
                "activitysim_iteration_persons_parquet",
            ),
            resolve=_restart_activitysim_iteration_outputs,
            notes=(
                "Traffic-assignment restart should preserve the ActivitySim "
                "iteration outputs that BEAM consumes for the resumed loop."
            ),
        ),
        RestartArtifactRequirementRule(
            name="beam_restart_inputs",
            semantic_keys=("beam_mutable_data_dir", "beam_region_input_dir", "beam_primary_config_file"),
            resolve=_restart_beam_required_artifacts,
            notes=(
                "BEAM restart should preserve the mutable input tree and primary "
                "config for resumed traffic assignment."
            ),
        ),
        RestartArtifactRequirementRule(
            name="atlas_restart_static_inputs",
            semantic_keys=(),
            resolve=_restart_atlas_required_artifacts,
            notes=(
                "ATLAS restart should keep the static mutable-input files needed "
                "for vehicle ownership resume."
            ),
        ),
    )
