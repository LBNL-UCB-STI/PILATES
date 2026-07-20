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
    Sequence,
)

from consist import (
    AdmissionEvidence,
    Artifact,
    ResolvedBinding,
    ResolvedBindingBuilder,
    StepIdentity,
)
from consist.types import BindingResult

from pilates.beam.vehicle_source import resolve_atlas_vehicles2_source
from pilates.runtime.archive_paths import archive_fallback_path, first_existing_path
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_input_precedence,
)
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.utils.io import get_traffic_assignment_model
from pilates.utils.state_access import iteration_index, uses_input_datastore
from pilates.utils.usim_h5 import (
    ensure_usim_population_year_table_aliases,
    resolve_usim_population_table_paths,
)
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    OMX_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface

_CANDIDATE_PATHS_METADATA_KEY = "candidate_paths_by_semantic_key"
_RESOLVED_VALUES_METADATA_KEY = "resolved_values_by_semantic_key"


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


def _ordered_unique(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


def _workflow_stage_enabled(state: Any, stage_name: str) -> bool:
    is_enabled = getattr(state, "is_enabled", None)
    stage_enum = getattr(state, "Stage", None)
    if not callable(is_enabled) or stage_enum is None:
        return False
    stage_value = getattr(stage_enum, stage_name, None)
    if stage_value is None:
        return False
    try:
        return bool(is_enabled(stage_value))
    except Exception:
        return False


def _requires_exact_activitysim_population_year(state: Any) -> bool:
    if not _workflow_stage_enabled(state, "land_use"):
        return False
    is_start_year = getattr(state, "is_start_year", None)
    if not callable(is_start_year):
        return False
    try:
        return not bool(is_start_year())
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

FallbackPolicyClass = Literal[
    "bootstrap",
    "recovery",
    "format_selection",
    "legacy_compatibility",
]
FallbackPolicyEndState = Literal["retain", "replace_with_producer_handoff", "delete"]
FallbackIdentitySource = Literal[
    "tracked_artifact",
    "admitted_local_file",
    "pinned_run_member",
    "none",
]


@dataclass(frozen=True)
class FallbackProviderInventoryEntry:
    """Declared authority and retirement policy for one generic provider."""

    identifier: str
    consuming_steps: tuple[str, ...]
    semantic_roles: tuple[str, ...]
    trigger: str
    candidate_order: tuple[str, ...]
    identity_source: FallbackIdentitySource
    policy_class: FallbackPolicyClass
    intended_end_state: FallbackPolicyEndState
    focused_tests: tuple[str, ...]


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


def activitysim_datastore_selection_rules() -> tuple[ArtifactBindingRule, ...]:
    """
    Backward-compatible wrapper for callers that still use the old helper name.
    Resolves to the UrbanSim datastore for the requested planner-year.
    """
    return (
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_CURRENT_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(
                USIM_POPULATION_SOURCE_H5,
                USIM_FORECAST_OUTPUT,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
            fallback_provider="urbansim_inputs_for_year",
        ),
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


def _candidate_paths_metadata(
    *paths_by_semantic_key: tuple[str, Sequence[Optional[Path]]],
) -> Dict[str, list[str]]:
    metadata: Dict[str, list[str]] = {}
    for semantic_key, paths in paths_by_semantic_key:
        ordered_paths = [str(path) for path in paths if path is not None and str(path)]
        if ordered_paths:
            metadata[semantic_key] = list(dict.fromkeys(ordered_paths))
    return metadata


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
    get_usim_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    is_start_year = getattr(state, "is_start_year", None)
    if not callable(get_usim_dir) or not callable(is_start_year):
        return None

    from pilates.urbansim import postprocessor as usim_post

    usim_data_dir = Path(get_usim_dir())
    input_path = usim_data_dir / usim_post.get_usim_datastore_fname(
        settings, io="input"
    )
    input_archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=input_path,
    )
    base_path = first_existing_path(input_path, input_archive_path)
    candidate_paths = _candidate_paths_metadata(
        (USIM_DATASTORE_BASE_H5, (input_path, input_archive_path)),
    )
    land_use_enabled = _workflow_stage_enabled(state, "land_use")

    mapping: Dict[str, Any] = {}
    if base_path is not None:
        mapping[USIM_DATASTORE_BASE_H5] = str(base_path)

    if not land_use_enabled:
        candidate_paths.update(
            _candidate_paths_metadata(
                (USIM_DATASTORE_CURRENT_H5, (input_path, input_archive_path)),
                (USIM_POPULATION_SOURCE_H5, (input_path, input_archive_path)),
            )
        )
        if base_path is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = str(base_path)
            mapping[USIM_POPULATION_SOURCE_H5] = str(base_path)
        if candidate_paths:
            mapping[_CANDIDATE_PATHS_METADATA_KEY] = candidate_paths
        return mapping or None

    if is_start_year():
        candidate_paths.update(
            _candidate_paths_metadata(
                (USIM_DATASTORE_CURRENT_H5, (input_path, input_archive_path)),
                (USIM_POPULATION_SOURCE_H5, (input_path, input_archive_path)),
            )
        )
        if base_path is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = str(base_path)
            mapping[USIM_POPULATION_SOURCE_H5] = str(base_path)
        if candidate_paths:
            mapping[_CANDIDATE_PATHS_METADATA_KEY] = candidate_paths
        return mapping or None

    output_path = usim_data_dir / usim_post.get_usim_datastore_fname(
        settings, io="output", year=year
    )
    output_archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=output_path,
    )
    current_path = first_existing_path(output_path, output_archive_path)
    population_source_path = _population_source_snapshot_path(output_path)
    population_source_archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=population_source_path,
    )
    population_path = first_existing_path(
        population_source_path,
        population_source_archive_path,
        output_path,
        output_archive_path,
    )
    if population_path is not None and population_path == current_path:
        logger.warning(
            "ActivitySim population-source snapshot missing for year %s; "
            "falling back to the current UrbanSim datastore %s",
            year,
            current_path,
        )
    candidate_paths.update(
        _candidate_paths_metadata(
            (USIM_DATASTORE_CURRENT_H5, (output_path, output_archive_path)),
            (
                USIM_FORECAST_OUTPUT,
                (
                    population_source_path,
                    population_source_archive_path,
                    output_path,
                    output_archive_path,
                ),
            ),
            (
                USIM_POPULATION_SOURCE_H5,
                (
                    population_source_path,
                    population_source_archive_path,
                    output_path,
                    output_archive_path,
                ),
            ),
        )
    )
    if current_path is not None:
        mapping[USIM_DATASTORE_CURRENT_H5] = str(current_path)
    if population_path is not None:
        mapping[USIM_FORECAST_OUTPUT] = str(population_path)
        mapping[USIM_POPULATION_SOURCE_H5] = str(population_path)
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
    candidate_paths: Dict[str, list[str]] = {}

    def _with_metadata(mapping: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        if candidate_paths:
            mapping[_CANDIDATE_PATHS_METADATA_KEY] = dict(candidate_paths)
        return mapping or None

    def _with_population_tables(mapping: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        selected = mapping.get(USIM_POPULATION_SOURCE_H5)
        if isinstance(selected, (str, os.PathLike)):
            selected_path = os.fspath(selected)
            if os.path.exists(selected_path):
                require_exact_year = _requires_exact_activitysim_population_year(state)
                if (
                    require_exact_year
                    and target_year is not None
                    and Path(selected_path).stem.endswith("_population_source")
                ):
                    try:
                        alias_result = ensure_usim_population_year_table_aliases(
                            h5_path=selected_path,
                            year=target_year,
                        )
                    except Exception:
                        logger.debug(
                            "Failed to repair ActivitySim population-source year "
                            "aliases for %s",
                            selected_path,
                            exc_info=True,
                        )
                    else:
                        created = alias_result.get("created") or []
                        if created:
                            logger.info(
                                "Added exact-year population table aliases for %s: %s",
                                selected_path,
                                created,
                            )
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
        return _with_metadata(mapping)

    if USIM_POPULATION_SOURCE_H5 in explicit_inputs:
        candidate = explicit_inputs.get(USIM_POPULATION_SOURCE_H5)
        if candidate:
            return _with_population_tables({USIM_POPULATION_SOURCE_H5: candidate})

    if not _workflow_stage_enabled(state, "land_use"):
        base_candidate = explicit_inputs.get(USIM_DATASTORE_BASE_H5)
        current_candidate = explicit_inputs.get(USIM_DATASTORE_CURRENT_H5)
        ordered_candidates = [
            str(path)
            for path in (base_candidate, current_candidate)
            if path is not None and str(path)
        ]
        if ordered_candidates:
            candidate_paths[USIM_POPULATION_SOURCE_H5] = list(
                dict.fromkeys(ordered_candidates)
            )
        selected = base_candidate or current_candidate
        if selected:
            return _with_population_tables({USIM_POPULATION_SOURCE_H5: selected})
        base_inputs = _activitysim_input_datastore(
            settings=settings,
            workspace=workspace,
        )
        base_path = None if not base_inputs else base_inputs.get(USIM_DATASTORE_BASE_H5)
        if base_path:
            candidate_paths[USIM_POPULATION_SOURCE_H5] = [str(base_path)]
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
    raw_candidate_paths = candidates.get(_CANDIDATE_PATHS_METADATA_KEY)
    if isinstance(raw_candidate_paths, Mapping):
        pop_paths = raw_candidate_paths.get(USIM_POPULATION_SOURCE_H5)
        if isinstance(pop_paths, Sequence) and not isinstance(pop_paths, (str, bytes)):
            ordered = [
                str(path) for path in pop_paths if path is not None and str(path)
            ]
            if ordered:
                candidate_paths[USIM_POPULATION_SOURCE_H5] = list(
                    dict.fromkeys(ordered)
                )
    selected = candidates.get(USIM_POPULATION_SOURCE_H5)
    if selected:
        return _with_population_tables({USIM_POPULATION_SOURCE_H5: selected})
    return None


def _beam_preprocess_exchange_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    surface: "EnabledWorkflowSurface",
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Yearless BEAM exchange-input fallback delegated to model state."""
    if get_traffic_assignment_model(settings) != "beam":
        return None

    resolved_profile = surface.profile
    if resolved_profile.activity_demand_enabled:
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
    surface: Optional["EnabledWorkflowSurface"] = None,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Yearless BEAM warmstart fallback resolved from coupler or workspace."""
    if get_traffic_assignment_model(settings) != "beam":
        return None

    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        value = get_value(LINKSTATS_WARMSTART)
        warmstart_path = artifact_to_existing_path(
            value,
            workspace,
        )
        if warmstart_path:
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
    surface: "EnabledWorkflowSurface",
    require_exact_year: bool = False,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Forecast-year ATLAS vehicles2 fallback.

    The provider derives candidate filenames from ``state.forecast_year`` and
    ``state.forecast_year - 1`` and intentionally ignores planner ``year``.
    """
    if get_traffic_assignment_model(settings) != "beam":
        return None
    resolved_profile = surface.profile
    if not resolved_profile.vehicle_ownership_model_enabled:
        return None

    current_iter = iteration_index(state, default=0)
    if current_iter != 0:
        return None

    resolved = resolve_atlas_vehicles2_source(
        state=state,
        workspace=workspace,
        require_exact_year=require_exact_year,
    )
    if resolved is not None:
        return {ATLAS_VEHICLES2_OUTPUT: str(resolved.selected_path)}
    return None


def _beam_preprocess_config_input(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, Any]]:
    """Yearless BEAM config fallback with state-aware archive resolution."""
    if get_traffic_assignment_model(settings) != "beam":
        return None

    try:
        from pilates.beam.config_hocon import beam_primary_config_path

        local_path = beam_primary_config_path(settings, workspace=workspace)
    except Exception:
        return None

    archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=local_path,
    )
    selected = first_existing_path(local_path, archive_path)
    if selected is None:
        return None
    return {BEAM_CONFIG_FILE: str(selected)}


_FALLBACK_PROVIDERS: Dict[str, BindingFallbackProvider] = {
    "urbansim_inputs_for_year": _urbansim_inputs_for_year,
    "activitysim_input_datastore": _activitysim_input_datastore,
    "activitysim_population_source": _activitysim_population_source,
    "beam_preprocess_exchange_inputs": _beam_preprocess_exchange_inputs,
    "beam_preprocess_warmstart_inputs": _beam_preprocess_warmstart_inputs,
    "beam_preprocess_atlas_inputs": _beam_preprocess_atlas_inputs,
    "beam_preprocess_config_input": _beam_preprocess_config_input,
}


# This inventory classifies the authority of the current generic providers.  It
# deliberately does not alter their selection behavior; subsequent slices use
# these entries to replace raw-path fallback with explicit producer handoffs.
FALLBACK_PROVIDER_INVENTORY: tuple[FallbackProviderInventoryEntry, ...] = (
    FallbackProviderInventoryEntry(
        identifier="urbansim_inputs_for_year",
        consuming_steps=("activitysim_preprocess", "atlas_preprocess"),
        semantic_roles=(
            USIM_DATASTORE_BASE_H5,
            USIM_DATASTORE_CURRENT_H5,
            USIM_FORECAST_OUTPUT,
            USIM_POPULATION_SOURCE_H5,
        ),
        trigger=(
            "the requested UrbanSim datastore is absent from explicit inputs "
            "and the coupler"
        ),
        candidate_order=(
            "configured local input datastore",
            "archive copy of the input datastore",
            "local forecast-year population-source snapshot",
            "archive copy of the population-source snapshot",
            "configured local forecast output",
            "archive copy of the forecast output",
        ),
        identity_source="none",
        policy_class="legacy_compatibility",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_workflow_binding.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="activitysim_input_datastore",
        consuming_steps=("activitysim_postprocess",),
        semantic_roles=(USIM_DATASTORE_BASE_H5,),
        trigger="no explicit or coupler-backed ActivitySim input datastore exists",
        candidate_order=("configured mutable UrbanSim input datastore",),
        identity_source="none",
        policy_class="bootstrap",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_activitysim_step_definitions.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="activitysim_population_source",
        consuming_steps=("activitysim_preprocess", "activitysim_postprocess"),
        semantic_roles=(
            USIM_POPULATION_SOURCE_H5,
            USIM_POPULATION_HOUSEHOLDS_TABLE,
            USIM_POPULATION_PERSONS_TABLE,
            USIM_POPULATION_JOBS_TABLE,
            USIM_POPULATION_BLOCKS_TABLE,
        ),
        trigger=(
            "no explicit or coupler-backed forecast-year population source exists"
        ),
        candidate_order=(
            "explicit population source",
            "explicit base or current datastore outside land-use",
            "configured mutable UrbanSim input datastore outside land-use",
            "forecast-year UrbanSim datastore candidates",
        ),
        identity_source="none",
        policy_class="legacy_compatibility",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_activitysim_step_definitions.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="beam_preprocess_exchange_inputs",
        consuming_steps=("beam_preprocess",),
        semantic_roles=(BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN),
        trigger="BEAM runs without ActivitySim-provided exchange inputs",
        candidate_order=(
            "registered existing BEAM exchange record",
            "existing workspace exchange file",
        ),
        identity_source="none",
        policy_class="bootstrap",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_beam_preprocessor_exchange_folder.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="beam_preprocess_warmstart_inputs",
        consuming_steps=("beam_preprocess",),
        semantic_roles=(LINKSTATS_WARMSTART,),
        trigger="no normal BEAM linkstats warmstart binding exists",
        candidate_order=(
            "coupler linkstats warmstart materialization",
            "configured initial linkstats path",
        ),
        identity_source="none",
        policy_class="recovery",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_beam_preprocessor_linkstats_warmstart.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="beam_preprocess_atlas_inputs",
        consuming_steps=("beam_preprocess",),
        semantic_roles=(ATLAS_VEHICLES2_OUTPUT,),
        trigger=(
            "BEAM traffic assignment and vehicle ownership are enabled on the "
            "first inner iteration, with no explicit or coupler-backed vehicles2 "
            "handoff"
        ),
        candidate_order=(
            "forecast-year local ATLAS vehicles2",
            "forecast-year archive ATLAS vehicles2",
            "artifact or environment materialization of the selected exact-year path",
            "previous-year local/archive vehicles2 compatibility path when exact year is not required",
        ),
        identity_source="none",
        policy_class="legacy_compatibility",
        intended_end_state="replace_with_producer_handoff",
        focused_tests=("tests/test_vehicle_ownership_usim_selection.py",),
    ),
    FallbackProviderInventoryEntry(
        identifier="beam_preprocess_config_input",
        consuming_steps=("beam_preprocess",),
        semantic_roles=(BEAM_CONFIG_FILE,),
        trigger="the configured BEAM config is absent from explicit inputs and coupler",
        candidate_order=(
            "configured local primary BEAM config",
            "archive copy of the primary BEAM config",
        ),
        identity_source="none",
        policy_class="legacy_compatibility",
        intended_end_state="delete",
        focused_tests=("tests/test_beam_native_step_definitions.py",),
    ),
)


def _pilot_binding_overrides() -> Dict[str, tuple[ArtifactBindingRule, ...]]:
    return {
        "urbansim_preprocess": (
            ArtifactBindingRule(
                semantic_key=OMX_SKIMS,
                required=False,
                allow_fallback=True,
                preferred_keys=(FINAL_SKIMS_OMX, OMX_SKIMS),
            ),
        ),
        "activitysim_preprocess": (
            *activitysim_population_source_selection_rules(),
            ArtifactBindingRule(
                semantic_key=FINAL_SKIMS_OMX,
                required=False,
            ),
        ),
        "activitysim_postprocess": (
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_CURRENT_H5,
                required=False,
                preferred_keys=(USIM_DATASTORE_CURRENT_H5,),
            ),
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_BASE_H5,
                required=False,
                allow_fallback=True,
                fallback_provider="activitysim_input_datastore",
            ),
            ArtifactBindingRule(
                semantic_key=USIM_POPULATION_SOURCE_H5,
                required=False,
                allow_fallback=True,
                preferred_keys=(USIM_POPULATION_SOURCE_H5,),
                fallback_provider="activitysim_population_source",
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
            ArtifactBindingRule(
                semantic_key=OMX_SKIMS,
                required=False,
                allow_fallback=True,
                preferred_keys=(FINAL_SKIMS_OMX, OMX_SKIMS),
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
                allow_fallback=True,
                fallback_provider="beam_preprocess_config_input",
            ),
        ),
    }


def artifact_rules_for_step_name(
    step_name: str,
    *,
    settings: Any = None,
) -> tuple[ArtifactBindingRule, ...]:
    """Return catalog-derived native selection rules with local overrides."""
    from pilates.workflows.catalog import workflow_step_contracts_by_name

    contract = workflow_step_contracts_by_name(settings=settings).get(step_name)
    if contract is None:
        return ()
    rules = {
        key: ArtifactBindingRule(semantic_key=key, required=True)
        for key in contract.get("input_keys", ())
    }
    rules.update(
        {
            key: ArtifactBindingRule(semantic_key=key, required=False)
            for key in contract.get("optional_input_keys", ())
        }
    )
    for override in _pilot_binding_overrides().get(step_name, ()):
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
            str(path) for path in candidate_paths if path is not None and str(path)
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
    surface: Optional["EnabledWorkflowSurface"],
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
            surface=surface,
        )
        if rule.allow_fallback or rule.fallback_provider
        else None
    )
    rule_fallback_inputs, candidate_paths_by_semantic_key = (
        _split_candidate_paths_metadata(rule_fallback_inputs)
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
) -> ResolvedStepInputs:
    """Select semantic roles directly into one native Consist binding envelope."""

    required = tuple(required_roles)
    optional = tuple(optional_roles)
    rule_by_role = {rule.semantic_key: rule for rule in artifact_rules}
    inputs: Dict[str, Any] = {}
    source_by_role: Dict[str, str] = {}
    selected_key_by_role: Dict[str, str] = {}
    candidate_paths_by_role: Dict[str, list[str]] = {}
    resolved_values_by_role: Dict[str, Any] = {}

    for role, is_required in [
        *((role, True) for role in required),
        *((role, False) for role in optional),
    ]:
        rule = rule_by_role.get(role) or ArtifactBindingRule(
            semantic_key=role, required=is_required
        )
        source, selected_key, value, matched_candidate, candidate_paths = (
            _resolve_rule_binding(
                rule=rule,
                coupler=coupler,
                explicit_inputs=explicit_inputs,
                fallback_inputs=fallback_inputs,
                settings=settings,
                state=state,
                workspace=workspace,
                year=year,
                surface=None,
            )
        )
        source_by_role[role] = source
        candidate_paths_by_role.update(candidate_paths)
        if selected_key is not None:
            selected_key_by_role[role] = matched_candidate or selected_key
        if source != "missing" and rule.pass_mode == "metadata_only":
            resolved_values_by_role[role] = value
        elif source != "missing":
            inputs[role] = value

    metadata: Dict[str, Any] = {}
    if candidate_paths_by_role:
        metadata[_CANDIDATE_PATHS_METADATA_KEY] = candidate_paths_by_role
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
            path = record.get_absolute_path(
                base_path=getattr(workspace, "full_path", None)
            )
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


def _bootstrap_urbansim_initial_datastore(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **_: Any,
) -> Optional[Mapping[str, str]]:
    """Publish the bootstrap-staged UrbanSim input H5 at the initial frontier."""
    if getattr(state, "data_initialized", None) is not False:
        return None
    if not uses_input_datastore(state):
        return None

    models = getattr(getattr(settings, "run", None), "models", None)
    requires_urbansim_datastore = (
        getattr(models, "land_use", None) == "urbansim"
        or getattr(models, "activity_demand", None) == "activitysim"
        or getattr(models, "vehicle_ownership", None) == "atlas"
    )
    if not requires_urbansim_datastore:
        return None

    get_usim_data_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    urbansim_cfg = getattr(settings, "urbansim", None)
    if not callable(get_usim_data_dir) or urbansim_cfg is None:
        return None

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    staged_input_h5 = os.path.join(
        get_usim_data_dir(),
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
        current_stage
        in {
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
    if _workflow_stage_enabled(state, "land_use"):
        datastore_candidates = _urbansim_datastore_candidates_for_year(
            settings=settings,
            state=state,
            workspace=workspace,
            year=getattr(state, "year", getattr(state, "forecast_year", None)),
        )
        if datastore_candidates:
            for semantic_key in (
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
            ):
                path = datastore_candidates.get(semantic_key)
                if path:
                    required[semantic_key] = str(path)
    return required or None


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
    if (
        getattr(state, "current_major_stage", None)
        != workflow_stage.vehicle_ownership_model
    ):
        return None

    get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if not callable(get_atlas_input_dir):
        return None

    atlas_input_dir = get_atlas_input_dir()
    required: Dict[str, str] = {}
    for relpath in atlas_static_input_relpaths_fn(settings):
        required[f"atlas_static::{relpath}"] = os.path.join(atlas_input_dir, relpath)
    start_year = getattr(state, "start_year", None)
    atlas_year = getattr(state, "year", getattr(state, "current_year", None))
    if start_year is not None and atlas_year is not None:
        from pilates.atlas.preprocessor import restart_required_atlas_input_paths

        required.update(
            restart_required_atlas_input_paths(
                atlas_input_root=atlas_input_dir,
                start_year=int(start_year),
                atlas_year=int(atlas_year),
            )
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
            semantic_keys=(
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
            ),
            resolve=_restart_activitysim_required_artifacts,
            notes=(
                "ActivitySim restart should keep the mutable config tree for "
                "stage entries that can re-enter ActivitySim directly, along "
                "with the explicit UrbanSim H5 roles needed after the "
                "population-source role split."
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
        RestartArtifactRequirementRule(
            name="atlas_restart_inputs",
            semantic_keys=(),
            resolve=_restart_atlas_required_artifacts,
            notes=(
                "ATLAS restart should keep both static mutable-input files and "
                "restart-critical year directories needed for vehicle ownership resume."
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
    }
