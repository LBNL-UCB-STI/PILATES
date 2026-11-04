"""
Workflow binding-layer data structures.

This module defines runtime binding policy objects and freezes selected
artifacts into ``consist.BindingResult`` values for native step execution.

Semantic workflow contracts remain owned by ``catalog.py``. Binding specs may
derive their artifact universe from the catalog by reference so runtime binding
does not become a second manually maintained semantic registry.
"""

from __future__ import annotations

import logging
import os
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from consist import (
    AdmissionEvidence,
    Artifact,
    ResolvedBinding,
    ResolvedBindingBuilder,
    StepIdentity,
)
from consist.types import BindingResult

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.utils.archive_paths import WorkspaceRootLike
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.utils.io import (
    get_activity_demand_model,
    get_land_use_model,
    get_traffic_assignment_model,
    get_vehicle_ownership_model,
)
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import is_start_year, uses_input_datastore
from pilates.utils.usim_h5 import resolve_usim_population_table_paths
from pilates.workflows.coupler_namespace import resolve_input_precedence
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface

_RESOLVED_VALUES_METADATA_KEY = "resolved_values_by_semantic_key"


@runtime_checkable
class _HasIsEnabled(Protocol):
    def is_enabled(self, stage: Any) -> bool: ...


@runtime_checkable
class _HasStageEnum(Protocol):
    @property
    def Stage(self) -> Any: ...


@runtime_checkable
class _HasLandUseStage(Protocol):
    @property
    def land_use(self) -> Any: ...


@runtime_checkable
class _HasCurrentMajorStage(Protocol):
    @property
    def current_major_stage(self) -> Any: ...


@runtime_checkable
class _HasUsimMutableDataDir(Protocol):
    def get_usim_mutable_data_dir(self) -> str: ...


@runtime_checkable
class _HasBeamMutableDataDir(Protocol):
    def get_beam_mutable_data_dir(self) -> str: ...


def _current_major_stage(state: Any) -> Any:
    return (
        state.current_major_stage if isinstance(state, _HasCurrentMajorStage) else None
    )


def _usim_mutable_data_dir(workspace: Any) -> Optional[str]:
    if isinstance(workspace, _HasUsimMutableDataDir) and callable(
        workspace.get_usim_mutable_data_dir
    ):
        return workspace.get_usim_mutable_data_dir()
    return None


def _beam_mutable_data_dir(workspace: Any) -> Optional[str]:
    if isinstance(workspace, _HasBeamMutableDataDir) and callable(
        workspace.get_beam_mutable_data_dir
    ):
        return workspace.get_beam_mutable_data_dir()
    return None


def build_resolved_binding(
    *,
    step_name: str,
    function: Callable[..., Any],
    selected_artifacts: Mapping[str, Artifact],
    logical_destinations: Mapping[str, Path],
    selection_diagnostics: Mapping[str, Any],
    admission_evidence: Mapping[str, AdmissionEvidence] | None = None,
    source_by_parameter: Mapping[str, str] | None = None,
    step_identity: StepIdentity,
) -> ResolvedBinding:
    """Freeze locally tracked named inputs into one V1 strict binding."""

    parameters = inspect.signature(function).parameters
    destinations = {
        parameter: Path(destination)
        for parameter, destination in logical_destinations.items()
    }
    if set(selected_artifacts) != set(destinations):
        raise ValueError(
            "strict binding artifacts and destinations must have matching parameters"
        )
    if len(set(destinations.values())) != len(destinations):
        raise ValueError("strict binding destinations must be unique")

    builder = ResolvedBindingBuilder(
        step_name=step_identity.name,
        step_contract_identity=step_identity.step_contract_identity,
    ).with_diagnostics(selection_diagnostics)
    evidence_by_parameter = admission_evidence or {}
    sources = source_by_parameter or {}
    for parameter, artifact in selected_artifacts.items():
        if (
            parameter not in parameters
            or parameters[parameter].kind is inspect.Parameter.VAR_KEYWORD
        ):
            raise ValueError(
                f"strict binding requires named callable parameter: {parameter!r}"
            )
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"strict binding input {parameter!r} must be a tracked Artifact"
            )
        destination = destinations[parameter]
        if destination.is_absolute():
            raise ValueError("strict binding destinations must be relative")
        source = sources.get(parameter, "coupler")
        if source not in {
            "explicit",
            "coupler",
            "fallback",
            "pinned",
            "external_admitted",
        }:
            raise ValueError(
                f"invalid strict binding source for {parameter!r}: {source!r}"
            )
        builder.bind_tracked_artifact(
            parameter=parameter,
            artifact=artifact,
            destination=destination,
            source=source,
            selected_role=parameter,
        )
        evidence = evidence_by_parameter.get(parameter)
        if evidence is not None:
            builder.with_admission(parameter=parameter, evidence=evidence)
    unknown_evidence = set(evidence_by_parameter).difference(selected_artifacts)
    if unknown_evidence:
        raise ValueError(
            "admission evidence has no bound input: "
            + ", ".join(sorted(unknown_evidence))
        )
    return builder.freeze()


def _is_land_use_stage_enabled(state: Any) -> bool:
    if not isinstance(state, _HasIsEnabled) or not isinstance(state, _HasStageEnum):
        return False
    if not callable(state.is_enabled) or state.Stage is None:
        return False
    stage_enum = state.Stage
    stage_value: Any = None
    if isinstance(stage_enum, _HasLandUseStage):
        stage_value = stage_enum.land_use
    elif isinstance(stage_enum, Mapping):
        stage_value = stage_enum.get("land_use")
    if stage_value is None:
        return False
    try:
        return bool(state.is_enabled(stage_value))
    except Exception:
        return False


def _requires_exact_activitysim_population_year(state: Any) -> bool:
    if not _is_land_use_stage_enabled(state):
        return False
    try:
        return not is_start_year(state)
    except Exception:
        return False


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
    pass_mode: Literal["auto", "input_key_only", "explicit_only", "metadata_only"] = (
        "auto"
    )


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


def activitysim_population_source_selection_rules() -> tuple[ArtifactBindingRule, ...]:
    """
    Shared population-source datastore preference for ActivitySim input selection.
    All rules here resolve to forecast-year artifacts.
    """
    return (
        ArtifactBindingRule(
            semantic_key=USIM_POPULATION_SOURCE_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(USIM_POPULATION_SOURCE_H5,),
            fallback_provider="activitysim_population_source",
        ),
        ArtifactBindingRule(
            semantic_key=USIM_POPULATION_HOUSEHOLDS_TABLE,
            required=False,
            allow_fallback=True,
            preferred_keys=(USIM_POPULATION_HOUSEHOLDS_TABLE,),
            fallback_provider="activitysim_population_source",
            pass_mode="metadata_only",
        ),
        ArtifactBindingRule(
            semantic_key=USIM_POPULATION_PERSONS_TABLE,
            required=False,
            allow_fallback=True,
            preferred_keys=(USIM_POPULATION_PERSONS_TABLE,),
            fallback_provider="activitysim_population_source",
            pass_mode="metadata_only",
        ),
        ArtifactBindingRule(
            semantic_key=USIM_POPULATION_JOBS_TABLE,
            required=False,
            allow_fallback=True,
            preferred_keys=(USIM_POPULATION_JOBS_TABLE,),
            fallback_provider="activitysim_population_source",
            pass_mode="metadata_only",
        ),
        ArtifactBindingRule(
            semantic_key=USIM_POPULATION_BLOCKS_TABLE,
            required=False,
            allow_fallback=True,
            preferred_keys=(USIM_POPULATION_BLOCKS_TABLE,),
            fallback_provider="activitysim_population_source",
            pass_mode="metadata_only",
        ),
    )


def _population_source_snapshot_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_population_source{path.suffix}")


def _urbansim_datastore_candidates_for_year(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
) -> Optional[Mapping[str, Any]]:
    if settings is None or state is None or workspace is None or year is None:
        return None
    usim_dir = _usim_mutable_data_dir(workspace)
    if usim_dir is None:
        return None

    from pilates.urbansim import postprocessor as usim_post

    usim_data_dir = Path(usim_dir)
    input_path = usim_data_dir / usim_post.get_usim_datastore_fname(
        settings, io="input"
    )
    base_path = input_path if input_path.exists() else None
    land_use_enabled = _is_land_use_stage_enabled(state)

    mapping: Dict[str, Any] = {}
    if base_path is not None:
        mapping[USIM_DATASTORE_BASE_H5] = str(base_path)

    if not land_use_enabled:
        if base_path is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = str(base_path)
            mapping[USIM_POPULATION_SOURCE_H5] = str(base_path)
        return mapping or None

    start_year = get_setting(state, "start_year")
    is_base_year = (
        (year == start_year)
        if start_year is not None
        else is_start_year(state, settings)
    )

    if is_base_year:
        if base_path is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = str(base_path)
            mapping[USIM_POPULATION_SOURCE_H5] = str(base_path)
        return mapping or None

    output_path = usim_data_dir / usim_post.get_usim_datastore_fname(
        settings, io="output", year=year
    )
    current_path = output_path if output_path.exists() else None
    population_source_path = _population_source_snapshot_path(output_path)
    population_path = next(
        (
            p
            for p in (population_source_path, output_path)
            if p is not None and p.exists()
        ),
        None,
    )
    if population_path is not None and population_path == current_path:
        logger.warning(
            "ActivitySim population-source snapshot missing for year %s; "
            "falling back to the current UrbanSim datastore %s",
            year,
            current_path,
        )
    if current_path is not None:
        mapping[USIM_DATASTORE_CURRENT_H5] = str(current_path)
    if population_path is not None:
        mapping[USIM_FORECAST_OUTPUT] = str(population_path)
        mapping[USIM_POPULATION_SOURCE_H5] = str(population_path)
    return mapping or None


def _urbansim_inputs_for_year(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    year: Optional[int],
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Planner-year UrbanSim datastore selector.

    This is the one fallback provider that intentionally consumes the caller
    supplied ``year``; the rule contract is literally "for the requested
    year", so the binding layer forwards that year unchanged.
    """
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
    """Canonical yearless ActivitySim input datastore role."""
    if settings is None or workspace is None:
        return None
    usim_dir = _usim_mutable_data_dir(workspace)
    if usim_dir is None:
        return None

    from pilates.activitysim.postprocessor import get_usim_datastore_fname

    candidate = os.path.join(
        usim_dir,
        get_usim_datastore_fname(settings, io="input"),
    )
    if not os.path.exists(candidate):
        return None
    return {USIM_DATASTORE_BASE_H5: candidate}


def _activitysim_population_source(
    *,
    explicit_fallback_inputs: Optional[Mapping[str, Any]],
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Forecast-year ActivitySim population source.

    Derive the year from ``state.forecast_year`` and ignore any planner-supplied
    ``year`` kwarg so the datastore filename and HDF5 table prefix stay aligned.
    """
    target_year = resolve_forecast_year(state)
    explicit_inputs = dict(explicit_fallback_inputs or {})

    def _with_population_tables(mapping: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        selected = mapping.get(USIM_POPULATION_SOURCE_H5)
        if isinstance(selected, (str, os.PathLike)):
            selected_path = os.fspath(selected)
            if os.path.exists(selected_path):
                require_exact_year = _requires_exact_activitysim_population_year(state)
                try:
                    mapping.update(
                        resolve_usim_population_table_paths(
                            h5_path=selected_path,
                            year=target_year,
                            require_exact_year=require_exact_year,
                        )
                    )
                except Exception as exc:
                    if require_exact_year:
                        raise
                    logger.warning(
                        "Failed to resolve ActivitySim population-source table paths "
                        "for %s: %s",
                        selected_path,
                        exc,
                    )
        return mapping or None

    if USIM_POPULATION_SOURCE_H5 in explicit_inputs:
        candidate = explicit_inputs.get(USIM_POPULATION_SOURCE_H5)
        if candidate:
            return _with_population_tables({USIM_POPULATION_SOURCE_H5: candidate})

    if not _is_land_use_stage_enabled(state):
        base_candidate = explicit_inputs.get(USIM_DATASTORE_BASE_H5)
        current_candidate = explicit_inputs.get(USIM_DATASTORE_CURRENT_H5)
        selected = base_candidate or current_candidate
        if selected:
            return _with_population_tables({USIM_POPULATION_SOURCE_H5: selected})
        base_inputs = _activitysim_input_datastore(
            settings=settings,
            workspace=workspace,
        )
        base_path = None if not base_inputs else base_inputs.get(USIM_DATASTORE_BASE_H5)
        if base_path:
            return _with_population_tables({USIM_POPULATION_SOURCE_H5: base_path})
        return None

    candidates = _urbansim_datastore_candidates_for_year(
        settings=settings,
        state=state,
        workspace=workspace,
        year=target_year,
    )
    if not candidates:
        return None
    selected = candidates.get(USIM_POPULATION_SOURCE_H5)
    if selected:
        return _with_population_tables({USIM_POPULATION_SOURCE_H5: selected})
    return None


_FALLBACK_PROVIDERS: Dict[str, BindingFallbackProvider] = {
    "urbansim_inputs_for_year": _urbansim_inputs_for_year,
    "activitysim_input_datastore": _activitysim_input_datastore,
    "activitysim_population_source": _activitysim_population_source,
}


def workflow_step_spec_for_step_name(step_name: str) -> Any:
    from pilates.workflows.catalog import (
        workflow_step_spec_for_step_name as _catalog_step_spec,
    )

    return _catalog_step_spec(step_name)


def artifact_rules_for_step_name(
    step_name: str,
    *,
    settings: Any = None,
) -> tuple[ArtifactBindingRule, ...]:
    """Return catalog-derived native selection rules with step-defined overrides."""
    del settings
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None:
        return ()
    rules = {
        key: ArtifactBindingRule(semantic_key=key, required=True)
        for key in spec.input_keys
    }
    rules.update(
        {
            key: ArtifactBindingRule(semantic_key=key, required=False)
            for key in spec.optional_input_keys
        }
    )
    for override in spec.binding_rules:
        existing = rules.get(override.semantic_key)
        rules[override.semantic_key] = ArtifactBindingRule(
            semantic_key=override.semantic_key,
            required=override.required if existing is None else existing.required,
            allow_explicit=override.allow_explicit,
            allow_coupler=override.allow_coupler,
            allow_fallback=override.allow_fallback,
            preferred_keys=override.preferred_keys
            or (() if existing is None else existing.preferred_keys),
            fallback_provider=override.fallback_provider,
            pass_mode=override.pass_mode,
        )
    return tuple(rules.values())


def _lookup_fallback_inputs(
    *,
    rule: ArtifactBindingRule,
    explicit_fallback_inputs: Optional[Mapping[str, Any]],
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: Optional[CouplerProtocol],
    year: Optional[int],
    surface: Optional["EnabledWorkflowSurface"],
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
            surface=surface,
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
    surface: Optional["EnabledWorkflowSurface"],
) -> tuple[str, Optional[str], Optional[Any], Optional[str]]:
    candidates = rule.preferred_keys or (rule.semantic_key,)
    scoped_coupler = coupler if rule.allow_coupler else None
    for candidate in candidates:
        resolved = resolve_input_precedence(
            key=candidate,
            coupler=scoped_coupler,
            explicit_inputs=explicit_inputs if rule.allow_explicit else None,
        )
        if resolved.source == "missing":
            continue
        selected_key = resolved.storage_key or candidate
        if rule.pass_mode == "explicit_only" and resolved.source == "coupler":
            continue
        if rule.pass_mode == "input_key_only" and resolved.source == "explicit":
            continue
        return resolved.source, selected_key, resolved.value, candidate

    rule_fallback_inputs = (
        _lookup_fallback_inputs(
            rule=rule,
            explicit_fallback_inputs=fallback_inputs if rule.allow_fallback else None,
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            year=year,
            surface=surface,
        )
        if rule.allow_fallback or rule.fallback_provider
        else None
    )
    if rule_fallback_inputs is None:
        return "missing", None, None, None

    for candidate in candidates:
        resolved = resolve_input_precedence(
            key=candidate,
            coupler=None,
            fallback_inputs=rule_fallback_inputs,
        )
        if resolved.source == "missing":
            continue
        if rule.pass_mode == "input_key_only":
            continue
        return (
            resolved.source,
            candidate,
            resolved.value,
            candidate,
        )
    return "missing", None, None, None


def resolve_artifact_roles(
    *,
    step_name: str,
    required_roles: Iterable[str],
    optional_roles: Iterable[str],
    artifact_rules: Iterable[ArtifactBindingRule],
    logical_destinations: Mapping[str, Path],
    coupler: Optional[CouplerProtocol],
    settings: Any,
    state: Any,
    workspace: Any,
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
    year: Optional[int] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> ResolvedStepInputs:
    """Select semantic roles directly into one native Consist binding envelope."""

    required = tuple(required_roles)
    optional = tuple(optional_roles)
    rule_by_role = {rule.semantic_key: rule for rule in artifact_rules}
    inputs: Dict[str, Any] = {}
    source_by_role: Dict[str, str] = {}
    selected_key_by_role: Dict[str, str] = {}
    resolved_values_by_role: Dict[str, Any] = {}

    for role, is_required in [
        *((role, True) for role in required),
        *((role, False) for role in optional),
    ]:
        rule = rule_by_role.get(role) or ArtifactBindingRule(
            semantic_key=role, required=is_required
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
            surface=surface,
        )
        source_by_role[role] = source
        if selected_key is not None:
            selected_key_by_role[role] = matched_candidate or selected_key
        if source != "missing" and rule.pass_mode == "metadata_only":
            resolved_values_by_role[role] = value
        elif source != "missing":
            inputs[role] = value

    metadata: Dict[str, Any] = {}
    if resolved_values_by_role:
        metadata[_RESOLVED_VALUES_METADATA_KEY] = resolved_values_by_role
    return ResolvedStepInputs(
        step_name=step_name,
        binding=BindingResult(
            inputs=inputs or None,
            metadata=metadata or None,
        ),
        required_roles=required,
        optional_roles=optional,
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=logical_destinations,
        metadata=metadata,
    )


def _bootstrap_beam_exchange_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    if get_traffic_assignment_model(settings) != "beam":
        return None
    if get_activity_demand_model(settings) is not None:
        return None

    beam_preprocessor = BeamPreprocessor("beam", state)
    try:
        record_store = beam_preprocessor.existing_beam_exchange_inputs(workspace)
    except FileNotFoundError as exc:
        logger.warning(
            "Bootstrap could not seed default BEAM inputs into coupler: %s",
            exc,
        )
        return None

    allowed_keys = {BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN}
    artifacts: Dict[str, str] = {}
    base_path = (
        str(workspace.full_path)
        if isinstance(workspace, WorkspaceRootLike) and workspace.full_path is not None
        else None
    )
    if record_store is not None:
        for record in record_store.all_records():
            key = record.short_name
            if key not in allowed_keys:
                continue
            path = record.get_absolute_path(base_path=base_path)
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

    warmstart_path = resolve_initial_linkstats_path(settings, workspace)
    if not warmstart_path:
        return None
    return {LINKSTATS_WARMSTART: warmstart_path}


def _bootstrap_urbansim_initial_datastore(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    """Publish the bootstrap-staged UrbanSim input H5 at the initial frontier."""
    if not uses_input_datastore(state):
        return None

    requires_urbansim_datastore = (
        get_land_use_model(settings) == "urbansim"
        or get_activity_demand_model(settings) == "activitysim"
        or get_vehicle_ownership_model(settings) == "atlas"
    )
    if not requires_urbansim_datastore:
        return None

    usim_dir = _usim_mutable_data_dir(workspace)
    if usim_dir is None or get_setting(settings, "urbansim") is None:
        return None

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    staged_input_h5 = os.path.join(
        usim_dir,
        get_usim_datastore_fname(settings, io="input"),
    )
    if not os.path.exists(staged_input_h5):
        return None
    return {
        USIM_DATASTORE_BASE_H5: staged_input_h5,
        USIM_DATASTORE_CURRENT_H5: staged_input_h5,
    }


def bootstrap_stage_boundary_durability_policy() -> tuple[
    StageBoundaryDurabilityRule, ...
]:
    """
    Stage-boundary durability policy for bootstrap seeding.

    The returned rules are consumable by runtime bootstrap code and keep the
    policy inspectable without hard-coding path inventories in the runtime.
    """

    return (
        StageBoundaryDurabilityRule(
            name="urbansim_initial_datastore",
            semantic_keys=(USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5),
            resolve=_bootstrap_urbansim_initial_datastore,
            notes=(
                "The initial workflow frontier consumes the bootstrap-staged "
                "UrbanSim input datastore as both its immutable base and current role."
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
    current_stage = _current_major_stage(state)
    requires_usim_base_h5 = (
        get_land_use_model(settings) == "urbansim"
        or get_activity_demand_model(settings) == "activitysim"
    )
    if not requires_usim_base_h5:
        return None

    usim_data_dir = workspace.get_usim_mutable_data_dir()
    usim_base_fname = get_usim_datastore_fname_fn(settings, io="input")
    local_base_path = Path(usim_data_dir) / usim_base_fname
    required: Dict[str, str] = {USIM_DATASTORE_BASE_H5: str(local_base_path)}

    region = get_setting(settings, "run.region")
    urbansim_cfg = get_setting(settings, "urbansim")
    if (
        current_stage
        in {
            None,
            workflow_stage.land_use,
        }
        and region
        and urbansim_cfg is not None
    ):
        region_to_id = get_setting(
            settings, "urbansim.region_mappings.region_to_region_id", {}
        )
        region_id = (
            region_to_id.get(region) if isinstance(region_to_id, Mapping) else None
        )
        if region_id:
            required.update(
                {
                    "omx_skims": os.path.join(
                        usim_data_dir, f"skims_mpo_{region_id}.omx"
                    ),
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
    current_stage = _current_major_stage(state)
    if get_activity_demand_model(settings) != "activitysim":
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
        get_setting(settings, "activitysim.main_configs_dir") or "configs"
    )
    required: Dict[str, str] = {}
    for dirname in required_asim_config_dirs_fn(main_configs_dir):
        required[f"activitysim_config_settings_yaml_{dirname}"] = os.path.join(
            asim_configs_dir,
            dirname,
            "settings.yaml",
        )
    return required or None


def _restart_beam_required_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    workflow_stage: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    current_stage = _current_major_stage(state)
    if get_traffic_assignment_model(settings) != "beam":
        return None
    if current_stage not in {
        workflow_stage.supply_demand_loop,
        workflow_stage.traffic_assignment,
    }:
        return None

    region = get_setting(settings, "run.region")
    beam_input_dir = _beam_mutable_data_dir(workspace)
    if beam_input_dir is None or not region:
        return None

    beam_config_name = get_setting(settings, "beam.config")
    required: Dict[str, str] = {
        "beam_mutable_data_dir": beam_input_dir,
        "beam_region_input_dir": os.path.join(beam_input_dir, region),
    }
    if beam_config_name:
        required["beam_primary_config_file"] = os.path.join(
            beam_input_dir, region, beam_config_name
        )
    return required or None


def restart_required_local_artifact_policy() -> tuple[
    RestartArtifactRequirementRule, ...
]:
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
            name="beam_restart_inputs",
            semantic_keys=(
                "beam_mutable_data_dir",
                "beam_region_input_dir",
                "beam_primary_config_file",
            ),
            resolve=_restart_beam_required_artifacts,
            notes=(
                "BEAM restart should preserve the mutable input tree and primary "
                "config for resumed traffic assignment."
            ),
        ),
    )


def is_restart_prebootstrap_deferred_artifact_key(key: str) -> bool:
    """
    Return True when bootstrap is expected to recreate the missing artifact.

    These keys are still validated after bootstrap in strict restart mode, but
    pre-bootstrap diagnostics should treat them as deferred hydration work
    rather than immediate restart failures.
    """
    if not key:
        return False
    if key.startswith("activitysim_config_settings_yaml_"):
        return True
    return key in {
        "beam_mutable_data_dir",
        "beam_region_input_dir",
        "beam_primary_config_file",
        "usim_datastore_base_h5",
        "omx_skims",
        "hh_size",
        "income_rates",
        "relmap",
        "schools",
        "school_districts",
    }
