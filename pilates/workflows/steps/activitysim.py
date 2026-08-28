from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping

from consist import (
    Artifact,
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    StepIdentity,
    define_step,
    require_runtime_kwargs,
)
from consist.types import OutputArtifactSpec

from pilates.activitysim.preprocessor import (
    ActivitysimPreprocessor,
)
from pilates.activitysim.runner import (
    ActivitySimSkimDecision,
    ActivitysimSkimMode,
    ActivitysimRunner,
    ActivitySimConfigStagingPlan,
    ActivitySimLaunchContext,
    asim_runtime_zarr_path,
    stage_activitysim_config_roots,
    validate_activitysim_zarr_skims,
)
from pilates.activitysim.outputs import (
    ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    configured_asim_output_keys,
)
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.config.models import PilatesConfig
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import (
    ArtifactBindingRule,
    build_resolved_binding,
    resolve_artifact_roles,
)
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff
from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.runtime_overlays import resolve_runtime_input_output_overrides
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.step_consist_meta import (
    activitysim_config_root_dirs,
    consist_step_meta,
)
from pilates.workflows.step_definition import (
    ConfigContract,
    InputContract,
    StepDefinition,
)
from pilates.workflows.outputs_base import ValidationContext
from pilates.workspace import Workspace
from pilates.generic.model_factory import ModelFactory
from pilates.utils.coupler_helpers import artifact_to_path

# Model-specific step factories for ActivitySim.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    ZARR_SKIMS,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    CouplerProtocol,
    WorkflowState,
    _activitysim_output_facet_meta,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

_ACTIVITYSIM_POPULATION_TABLE_KEYS = (
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_BLOCKS_TABLE,
)


def _land_use_enabled(state: WorkflowState) -> bool:
    """Return whether this invocation has a preceding UrbanSim producer."""

    try:
        return state.is_enabled(WorkflowState.Stage.land_use)
    except AttributeError:
        # Lightweight resolver tests and legacy callers may use a state facade
        # without enabled-stage metadata.  Those retain bootstrap semantics.
        return False


_ACTIVITYSIM_CONFIG_REFERENCES_ARCHIVED_KEY = "activitysim_config_references_archived"
_ACTIVITYSIM_CONFIG_REFERENCE_ARCHIVE_CACHE_LOCK = threading.Lock()
_ACTIVITYSIM_CONFIG_REFERENCE_ARCHIVE_CACHE: Dict[
    tuple[str, str, tuple[str, ...]], Dict[str, str]
] = {}


def activitysim_run_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    produces_zarr: bool,
) -> Dict[str, Any]:
    """Return cache-recoverable ActivitySim outputs and their logging metadata."""
    expected_outputs = ActivitysimRunner.expected_outputs(settings, state, workspace)
    output_keys = set(configured_asim_output_keys(settings))
    if produces_zarr:
        expected_outputs[ZARR_SKIMS] = asim_runtime_zarr_path(workspace)
        output_keys.add(ZARR_SKIMS)

    forecast_year = resolve_forecast_year(state)
    iteration = getattr(state, "iteration", None)
    if forecast_year is None or iteration is None:
        return {
            key: path for key, path in expected_outputs.items() if key in output_keys
        }

    return {
        key: OutputArtifactSpec(
            path=path,
            **_activitysim_output_facet_meta(
                key,
                year=forecast_year,
                iteration=iteration,
            ),
        )
        for key, path in expected_outputs.items()
        if key in output_keys
    }


def activitysim_preprocess_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    """Declare cache-recoverable file outputs from ActivitySim preprocessing."""
    expected_outputs = ActivitysimPreprocessor.expected_outputs(
        settings, state, workspace
    )
    profiled_keys = {ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN}
    output_keys = (
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        *(
            ()
            if requires_prior_beam_skim_handoff(settings=settings, state=state)
            else (ASIM_OMX_SKIMS,)
        ),
    )
    return {
        key: OutputArtifactSpec(
            path=expected_outputs[key], profile_file_schema=key in profiled_keys
        )
        for key in output_keys
        if expected_outputs.get(key) is not None
    }


def activitysim_postprocess_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    """Declare non-H5 ActivitySim postprocess outputs for one-link logging."""
    expected_outputs = ActivitysimPostprocessor.expected_outputs(
        settings, state, workspace
    )
    profile_schema_keys = {
        "persons_asim_out",
        "trips_asim_out",
        "tours_asim_out",
        "beam_plans_asim_out",
        "households_asim_out",
    }
    forecast_year = resolve_forecast_year(state)
    iteration = getattr(state, "iteration", None)
    return {
        key: (
            (
                OutputArtifactSpec(
                    path=path,
                    profile_file_schema=key in profile_schema_keys,
                    **_activitysim_output_facet_meta(
                        key,
                        year=forecast_year,
                        iteration=iteration,
                    ),
                )
                if forecast_year is not None and iteration is not None
                else path
            )
            if key != "asim_output_dir" and path is not None
            else path
        )
        for key, path in expected_outputs.items()
        if key != "asim_output_dir" and path is not None
    }


# Native Consist definitions -------------------------------------------------

#
# The factories above remain only until the coordinated stage cutover deletes the
# legacy holder path.  These definitions intentionally do not use a holder: the
# resolver decides one BindingResult, Consist materializes those paths, and the
# callable constructs the model's existing typed boundary values from them.

_ACTIVITYSIM_PREPROCESS_REQUIRED_ROLES = (USIM_POPULATION_SOURCE_H5,)
_ACTIVITYSIM_PREPROCESS_OPTIONAL_ROLES = (FINAL_SKIMS_OMX,)
_ACTIVITYSIM_RUN_REQUIRED_ROLES = (
    ASIM_LAND_USE_IN,
    ASIM_HOUSEHOLDS_IN,
    ASIM_PERSONS_IN,
)

_ACTIVITYSIM_PREPROCESS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="mutable ActivitySim config and data roots with selected-skim mode",
    config_contract=ConfigContract.adapter("activitysim"),
)
_ACTIVITYSIM_RUN_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="runtime cache, warmup, and model output closure",
    config_contract=ConfigContract.adapter("activitysim"),
)
_ACTIVITYSIM_POSTPROCESS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="mixed aliases, direct skims, and model-owned roots",
    config_contract=ConfigContract.adapter("activitysim"),
)
_ACTIVITYSIM_RUN_SKIM_ROLES = (ZARR_SKIMS, ASIM_OMX_SKIMS)
_ACTIVITYSIM_SKIM_DECISION_METADATA_KEY = "activitysim_skim_decision"
_ACTIVITYSIM_SKIM_MODE_METADATA_KEY = "activitysim_skim_mode"
_ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY = "activitysim_produces_zarr"
_ACTIVITYSIM_POSTPROCESS_BASE_REQUIRED_ROLES = (
    ASIM_HOUSEHOLDS_IN,
    ASIM_PERSONS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ZARR_SKIMS,
)
_ACTIVITYSIM_POSTPROCESS_OPTIONAL_ROLES = (
    USIM_POPULATION_SOURCE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_BASE_H5,
)


def _activitysim_launch_context(
    workspace: Workspace,
    *,
    staged_data_dir: Path,
    staged_configs_dir: Path,
) -> ActivitySimLaunchContext:
    """Build one launch tree from resolver-owned model input destinations."""

    workspace_root = Path(workspace.full_path)
    output_dir = Path(workspace.get_asim_output_dir())
    runtime_cache_dir = Path(workspace.get_asim_runtime_cache_dir())
    return ActivitySimLaunchContext(
        workspace_root=workspace_root,
        mutable_data_dir=staged_data_dir,
        output_dir=output_dir,
        compile_output_dir=output_dir.parent / "compile-output",
        mutable_configs_dir=staged_configs_dir,
        runtime_cache_dir=runtime_cache_dir,
        runtime_zarr_path=runtime_cache_dir / "skims.zarr",
        shared_cache_dir=workspace_root / "shared_cache",
        shared_tmp_dir=workspace_root / "tmp",
        requires_staged_config_dirs=True,
    )


def _native_activitysim_resolved_inputs(
    *,
    step_name: str,
    required_roles: tuple[str, ...],
    optional_roles: tuple[str, ...],
    logical_destinations: Mapping[str, Any],
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    """Freeze the one ActivitySim semantic selection for a native invocation."""

    allow_population_bootstrap_fallback = not _land_use_enabled(state)
    workspace_root = Path(getattr(workspace, "full_path", "."))
    destinations = {
        key: Path(path)
        if path is not None
        else workspace_root / "activitysim" / "native-inputs" / key
        for key, path in logical_destinations.items()
        if key in (*required_roles, *optional_roles)
    }
    for key in (*required_roles, *optional_roles):
        destinations.setdefault(
            key, workspace_root / "activitysim" / "native-inputs" / key
        )

    rules = {
        role: ArtifactBindingRule(semantic_key=role, required=True)
        for role in required_roles
    }
    rules.update(
        {
            role: ArtifactBindingRule(semantic_key=role, required=False)
            for role in optional_roles
        }
    )
    if step_name == "activitysim_preprocess":
        rules[USIM_POPULATION_SOURCE_H5] = ArtifactBindingRule(
            semantic_key=USIM_POPULATION_SOURCE_H5,
            required=True,
            allow_fallback=allow_population_bootstrap_fallback,
            preferred_keys=(USIM_POPULATION_SOURCE_H5,),
            fallback_provider=(
                "activitysim_population_source"
                if allow_population_bootstrap_fallback
                else None
            ),
        )
    elif step_name == "activitysim_postprocess":
        rules[USIM_DATASTORE_CURRENT_H5] = ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_CURRENT_H5,
            required=False,
            preferred_keys=(USIM_DATASTORE_CURRENT_H5,),
        )
        rules[USIM_DATASTORE_BASE_H5] = ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_BASE_H5,
            required=False,
            allow_fallback=True,
            fallback_provider="activitysim_input_datastore",
        )
        rules[USIM_POPULATION_SOURCE_H5] = ArtifactBindingRule(
            semantic_key=USIM_POPULATION_SOURCE_H5,
            required=False,
            allow_fallback=allow_population_bootstrap_fallback,
            preferred_keys=(USIM_POPULATION_SOURCE_H5,),
            fallback_provider=(
                "activitysim_population_source"
                if allow_population_bootstrap_fallback
                else None
            ),
        )

    return resolve_artifact_roles(
        step_name=step_name,
        required_roles=required_roles,
        optional_roles=optional_roles,
        artifact_rules=tuple(rules.values()),
        logical_destinations=destinations,
        coupler=coupler,
        settings=settings,
        state=state,
        workspace=workspace,
        year=getattr(state, "year", None),
    )


def _activitysim_preprocess_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    return _native_activitysim_resolved_inputs(
        step_name="activitysim_preprocess",
        required_roles=_ACTIVITYSIM_PREPROCESS_REQUIRED_ROLES,
        optional_roles=(
            () if requires_beam_skim else _ACTIVITYSIM_PREPROCESS_OPTIONAL_ROLES
        ),
        logical_destinations=ActivitysimPreprocessor.declared_expected_inputs(
            settings, state, workspace
        ),
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )


def _activitysim_run_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    required_roles = (
        *_ACTIVITYSIM_RUN_REQUIRED_ROLES,
        *((ZARR_SKIMS,) if requires_beam_skim else ()),
    )
    optional_roles = () if requires_beam_skim else _ACTIVITYSIM_RUN_SKIM_ROLES
    resolved = _native_activitysim_resolved_inputs(
        step_name="activitysim_run",
        required_roles=required_roles,
        optional_roles=optional_roles,
        logical_destinations=ActivitysimRunner.declared_expected_inputs(
            settings, state, workspace
        ),
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    zarr_source = resolved.source_by_role.get(ZARR_SKIMS, "missing")
    omx_source = resolved.source_by_role.get(ASIM_OMX_SKIMS, "missing")
    zarr_rejection: str | None = None
    if zarr_source != "missing":
        zarr_input = (resolved.binding.inputs or {}).get(ZARR_SKIMS)
        zarr_path = artifact_to_path(zarr_input, workspace=workspace)
        zarr_rejection = (
            "could not resolve a concrete path"
            if zarr_path is None
            else validate_activitysim_zarr_skims(zarr_path)
        )

    # A valid published Zarr is authoritative. Otherwise the preprocessor's
    # OMX artifact is the first-run source that this invocation converts and
    # publishes as Zarr. Keep the unselected source out of the binding so it
    # cannot affect materialization or cache identity.
    if zarr_source != "missing" and zarr_rejection is None:
        decision = ActivitySimSkimDecision(
            role=ZARR_SKIMS,
            mode="zarr",
            produces_zarr=False,
            selected_source=zarr_source,
        )
    elif not requires_beam_skim and omx_source != "missing":
        if zarr_rejection is not None:
            logger.warning(
                "activitysim_run rejected %s before binding selection: %s; "
                "falling back to %s",
                ZARR_SKIMS,
                zarr_rejection,
                ASIM_OMX_SKIMS,
            )
        decision = ActivitySimSkimDecision(
            role=ASIM_OMX_SKIMS,
            mode="omx",
            produces_zarr=True,
            selected_source=omx_source,
        )
    elif zarr_rejection is not None:
        raise RuntimeError(
            f"activitysim_run rejected {ZARR_SKIMS}: {zarr_rejection}"
        )
    else:
        expected = (
            ZARR_SKIMS if requires_beam_skim else f"{ZARR_SKIMS} or {ASIM_OMX_SKIMS}"
        )
        raise RuntimeError(
            f"activitysim_run requires one published skim role: {expected}"
        )

    selected_roles = (*_ACTIVITYSIM_RUN_REQUIRED_ROLES, decision.role)
    # The native selector has already frozen each selected artifact under its
    # semantic role.  Subset that immutable selection so unselected skim
    # alternatives cannot affect this run's materialization or cache identity.
    binding_inputs: dict[str, Any] = {}
    for role in selected_roles:
        value = (resolved.binding.inputs or {}).get(role)
        if value is None:
            raise RuntimeError(
                f"activitysim_run resolved role has no concrete binding value: {role}"
            )
        binding_inputs[role] = value
    metadata = {
        **resolved.metadata,
        _ACTIVITYSIM_SKIM_DECISION_METADATA_KEY: decision,
        _ACTIVITYSIM_SKIM_MODE_METADATA_KEY: decision.mode,
        _ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY: decision.produces_zarr,
    }
    selected = ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=BindingResult(
            inputs=binding_inputs,
            metadata=metadata,
        ),
        required_roles=_ACTIVITYSIM_RUN_REQUIRED_ROLES,
        optional_roles=(decision.role,),
        source_by_role={
            key: value
            for key, value in resolved.source_by_role.items()
            if key in selected_roles
        },
        selected_key_by_role={
            key: value
            for key, value in resolved.selected_key_by_role.items()
            if key in selected_roles
        },
        logical_destinations={
            key: value
            for key, value in resolved.logical_destinations.items()
            if key in selected_roles
        },
        metadata=metadata,
    )
    return selected


def _activitysim_postprocess_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    step_identity: StepIdentity | None = None,
) -> ResolvedStepInputs:
    required_roles, optional_roles = resolve_runtime_input_output_overrides(
        step_name="activitysim_postprocess",
        land_use_enabled=_land_use_enabled(state),
        # Postprocess H5 requiredness is mode-independent. Restart-only
        # preprocess guards remain part of prebootstrap policy.
        run_mode=None,
        required_inputs=(
            *_ACTIVITYSIM_POSTPROCESS_BASE_REQUIRED_ROLES,
            *configured_asim_output_keys(settings),
        ),
        optional_inputs=_ACTIVITYSIM_POSTPROCESS_OPTIONAL_ROLES,
    )
    resolved = _native_activitysim_resolved_inputs(
        step_name="activitysim_postprocess",
        required_roles=required_roles,
        optional_roles=optional_roles,
        logical_destinations=ActivitysimPostprocessor.declared_expected_inputs(
            settings, state, workspace
        ),
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )
    inputs = dict(resolved.binding.inputs or {})
    if step_identity is not None and all(
        isinstance(value, Artifact) for value in inputs.values()
    ):
        # This callable intentionally gives three named parameters to one
        # datastore in restart/base-only cases.  A normal BindingResult loses
        # that parameter-to-artifact relation during requested staging; freeze
        # it so Consist stages each named parameter by its tracked identity.
        return ResolvedStepInputs(
            step_name=resolved.step_name,
            binding=build_resolved_binding(
                step_name=resolved.step_name,
                function=_activitysim_postprocess_callable,
                selected_artifacts=inputs,
                logical_destinations={role: Path("inputs") / role for role in inputs},
                selection_diagnostics={
                    "source_by_role": resolved.source_by_role,
                    "selected_key_by_role": resolved.selected_key_by_role,
                },
                source_by_parameter={
                    role: resolved.source_by_role[role] for role in inputs
                },
                step_identity=step_identity,
            ),
            required_roles=resolved.required_roles,
            optional_roles=resolved.optional_roles,
            source_by_role=resolved.source_by_role,
            selected_key_by_role=resolved.selected_key_by_role,
            logical_destinations=resolved.logical_destinations,
            metadata=resolved.metadata,
        )

    population_source = inputs.get(USIM_POPULATION_SOURCE_H5)
    current_datastore = inputs.get(USIM_DATASTORE_CURRENT_H5)
    if not _is_population_source_alias(population_source, current_datastore):
        return resolved

    # Vehicle ownership intentionally aliases the current datastore role to
    # the immutable population snapshot.  The ActivitySim postprocessor can
    # use that snapshot for both optional parameters, but Consist must receive
    # the artifact once so requested staging has one unambiguous input key.
    inputs.pop(USIM_DATASTORE_CURRENT_H5)
    return ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=BindingResult(
            inputs=inputs or None,
            input_keys=resolved.binding.input_keys,
            optional_input_keys=resolved.binding.optional_input_keys,
            metadata=resolved.binding.metadata,
        ),
        required_roles=resolved.required_roles,
        optional_roles=resolved.optional_roles,
        source_by_role=resolved.source_by_role,
        selected_key_by_role=resolved.selected_key_by_role,
        logical_destinations=resolved.logical_destinations,
        metadata=resolved.metadata,
    )


def _is_population_source_alias(population_source: Any, current_datastore: Any) -> bool:
    """Return whether current datastore is the population-source alias."""

    if population_source is current_datastore:
        return population_source is not None
    if not isinstance(population_source, Artifact) or not isinstance(
        current_datastore, Artifact
    ):
        return False
    return population_source.id == current_datastore.id or (
        population_source.key == USIM_POPULATION_SOURCE_H5
        and current_datastore.key == USIM_POPULATION_SOURCE_H5
    )


def _activitysim_execution_options(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> ExecutionOptions:
    """Request Consist materialization at the one deterministic path per role."""

    del state
    selected_roles = set(resolved_inputs.selected_roles())
    runtime_kwargs: dict[str, Any] = {}
    if resolved_inputs.step_name == "activitysim_run":
        staged_destinations = {
            Path(resolved_inputs.logical_destinations[role]).parent
            for role in selected_roles
        }
        if len(staged_destinations) != 1:
            raise RuntimeError(
                "activitysim_run materialized inputs must share one staged data root"
            )
        activitysim_settings = settings.activitysim
        if activitysim_settings is None:
            raise RuntimeError("activitysim_run requires ActivitySim settings")
        staged_data_dir = staged_destinations.pop()
        workspace_root = Path(workspace.full_path)
        launch_context = _activitysim_launch_context(
            workspace,
            staged_data_dir=staged_data_dir,
            staged_configs_dir=workspace_root / "activitysim" / "native-launch" / "configs",
        )
        runtime_kwargs["activitysim_launch_context"] = launch_context
        runtime_kwargs["activitysim_config_staging_plan"] = ActivitySimConfigStagingPlan(
            source_roots=activitysim_config_root_dirs(
                activitysim_settings,
                Path(workspace.get_asim_mutable_configs_dir()),
            )
        )
    return ExecutionOptions(
        input_binding="paths",
        input_paths={
            key: destination
            for key, destination in resolved_inputs.logical_destinations.items()
            if key in selected_roles
        },
        input_materialization="requested",
        input_materialization_mode="copy",
        runtime_kwargs=runtime_kwargs,
    )


def _activitysim_cache_options(
    *, settings: PilatesConfig, state: WorkflowState, workspace: Workspace
) -> CacheOptions:
    """Admit a cache candidate only after all requested ActivitySim outputs land."""

    del settings, state, workspace
    return CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
    )


def _activitysim_run_produces_zarr(
    resolved_inputs: ResolvedStepInputs,
) -> bool:
    """Return the immutable native invocation decision for Zarr publication."""

    value = resolved_inputs.metadata.get(_ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY)
    if not isinstance(value, bool):
        raise RuntimeError(
            "activitysim_run requires resolver metadata "
            f"{_ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY!r}"
        )
    return value


def _activitysim_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs,
) -> Dict[str, Any]:
    """Declare only the immutable invocation-specific native output set."""

    return activitysim_run_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        produces_zarr=_activitysim_run_produces_zarr(resolved_inputs),
    )


def _persisted_output_path(
    outputs: Mapping[str, Any],
    *,
    step_name: str,
    key: str,
    declared_outputs: Mapping[str, Any],
    workspace: Workspace,
) -> Path:
    """Return one output only when it exists at its declared current destination.

    A projected result represents the current step execution.  It must not fall
    back to an artifact's mounted source, archive location, or historical
    workspace when the declared destination is absent.
    """

    require_output(outputs, step_name=step_name, key=key)
    try:
        output_spec = declared_outputs[key]
    except KeyError as exc:
        raise RuntimeError(
            f"{step_name} output {key!r} has no declared destination"
        ) from exc

    declared_path = getattr(output_spec, "path", output_spec)
    if isinstance(declared_path, os.PathLike):
        path = Path(declared_path)
    elif isinstance(declared_path, str):
        if declared_path.startswith("workspace://"):
            workspace_root = getattr(workspace, "full_path", None)
            if workspace_root is None:
                raise RuntimeError(
                    f"{step_name} output {key!r} cannot resolve its workspace destination"
                )
            path = Path(workspace_root) / declared_path[len("workspace://") :].lstrip(
                "/"
            )
        elif "://" in declared_path:
            raise RuntimeError(
                f"{step_name} output {key!r} has a non-local declared destination"
            )
        else:
            path = Path(declared_path)
            if not path.is_absolute():
                workspace_root = getattr(workspace, "full_path", None)
                if workspace_root is None:
                    raise RuntimeError(
                        f"{step_name} output {key!r} cannot resolve its relative destination"
                    )
                path = Path(workspace_root) / path
    else:
        raise RuntimeError(
            f"{step_name} output {key!r} has an invalid declared destination"
        )

    if not path.exists():
        raise RuntimeError(
            f"{step_name} output {key!r} is missing at declared destination {path}"
        )
    return path


def _project_activitysim_preprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> ActivitySimPreprocessOutputs:
    declared_outputs = activitysim_preprocess_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = ActivitySimPreprocessOutputs(
        mutable_data_dir=Path(workspace.get_asim_mutable_data_dir()),
        land_use_table=_persisted_output_path(
            outputs,
            step_name="activitysim_preprocess",
            key=ASIM_LAND_USE_IN,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        households_table=_persisted_output_path(
            outputs,
            step_name="activitysim_preprocess",
            key=ASIM_HOUSEHOLDS_IN,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        persons_table=_persisted_output_path(
            outputs,
            step_name="activitysim_preprocess",
            key=ASIM_PERSONS_IN,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        omx_skims=(
            _persisted_output_path(
                outputs,
                step_name="activitysim_preprocess",
                key=ASIM_OMX_SKIMS,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if ASIM_OMX_SKIMS in outputs
            else None
        ),
    )
    projected.validate(
        ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_preprocess",
        )
    )
    return projected


def _project_activitysim_run_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> ActivitySimRunOutputs:
    if resolved_inputs is None:
        raise RuntimeError(
            "activitysim_run projection requires the resolved native skim decision"
        )
    produces_zarr = _activitysim_run_produces_zarr(resolved_inputs)
    if (ZARR_SKIMS in outputs) != produces_zarr:
        expected = "include" if produces_zarr else "omit"
        raise RuntimeError(
            "activitysim_run output parity violation: resolved skim decision requires "
            f"outputs to {expected} {ZARR_SKIMS!r}"
        )
    declared_outputs = _activitysim_run_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )
    projected = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        raw_outputs={
            key: _persisted_output_path(
                outputs,
                step_name="activitysim_run",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in declared_outputs
            if key.endswith("_asim_out")
        },
        zarr_skims=(
            _persisted_output_path(
                outputs,
                step_name="activitysim_run",
                key=ZARR_SKIMS,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if produces_zarr
            else None
        ),
    )
    projected.validate(
        ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_run",
        )
    )
    return projected


def _project_activitysim_postprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> ActivitySimPostprocessOutputs:
    declared_outputs = activitysim_postprocess_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    for key in declared_outputs:
        if not key.endswith("_asim_out"):
            continue
        require_output(outputs, step_name="activitysim_postprocess", key=key)
    projected = ActivitySimPostprocessOutputs(
        usim_datastore_h5=(
            _persisted_output_path(
                outputs,
                step_name="activitysim_postprocess",
                key=USIM_DATASTORE_H5,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if USIM_DATASTORE_H5 in outputs
            else None
        ),
        asim_output_dir=Path(workspace.get_asim_output_dir()),
        processed_outputs={
            key: _persisted_output_path(
                outputs,
                step_name="activitysim_postprocess",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key != USIM_DATASTORE_H5
        },
        usim_datastore_key=(
            USIM_DATASTORE_H5 if USIM_DATASTORE_H5 in outputs else None
        ),
    )
    projected.validate(
        ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_postprocess",
        )
    )
    return projected


@define_step(
    model="activitysim",
    name_template="activitysim_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_POPULATION_SOURCE_H5: None},
    optional_input_keys=(FINAL_SKIMS_OMX,),
    outputs=[ASIM_LAND_USE_IN, ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN],
    schema_outputs=[
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_OMX_SKIMS,
    ],
    input_binding="paths",
    tags=["activitysim", "preprocess"],
    **consist_step_meta(
        "activitysim_preprocess", input_contract=_ACTIVITYSIM_PREPROCESS_INPUT_CONTRACT
    ),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _activitysim_preprocess_callable(
    usim_population_source_h5: Path,
    final_skims_omx: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    """Run ActivitySim preprocessing from the resolved semantic input paths."""

    prepare_omx_skims = not requires_prior_beam_skim_handoff(
        settings=settings, state=state
    )
    preprocess_kwargs: dict[str, Any] = {
        "population_source_h5_path": str(usim_population_source_h5),
        "prepare_omx_skims": prepare_omx_skims,
    }
    if prepare_omx_skims and final_skims_omx is not None:
        preprocess_kwargs["final_skims_omx"] = final_skims_omx
    ModelFactory().get_preprocessor("activitysim", state).preprocess(
        workspace, **preprocess_kwargs
    )


@define_step(
    model="activitysim",
    name_template="activitysim_run__y{year}__i{iteration}__phase_{phase}",
    inputs={
        ASIM_LAND_USE_IN: None,
        ASIM_HOUSEHOLDS_IN: None,
        ASIM_PERSONS_IN: None,
    },
    optional_input_keys=_ACTIVITYSIM_RUN_SKIM_ROLES,
    outputs=list(ASIM_REQUIRED_RUN_OUTPUT_KEYS),
    schema_outputs=[
        *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
        *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
        ZARR_SKIMS,
    ],
    input_binding="paths",
    tags=["activitysim", "run"],
    **consist_step_meta(
        "activitysim_run", input_contract=_ACTIVITYSIM_RUN_INPUT_CONTRACT
    ),
)
@require_runtime_kwargs(
    "settings",
    "state",
    "workspace",
    "activitysim_launch_context",
    "activitysim_config_staging_plan",
)
def _activitysim_run_callable(
    land_use_asim_in: Path,
    households_asim_in: Path,
    persons_asim_in: Path,
    zarr_skims: Path | None = None,
    omx_skims: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    activitysim_launch_context: ActivitySimLaunchContext | None = None,
    activitysim_config_staging_plan: ActivitySimConfigStagingPlan | None = None,
) -> None:
    """Run ActivitySim from exactly one materialized native skim source."""

    del settings
    if zarr_skims is not None and omx_skims is None:
        skim_mode: ActivitysimSkimMode = "zarr"
        extra_inputs: Mapping[str, Path] = {ZARR_SKIMS: Path(zarr_skims)}
    elif omx_skims is not None and zarr_skims is None:
        skim_mode = "omx"
        extra_inputs = {}
    else:
        raise RuntimeError(
            "activitysim_run requires exactly one materialized skim input: "
            f"{ZARR_SKIMS} or {ASIM_OMX_SKIMS}"
        )
    if not isinstance(activitysim_launch_context, ActivitySimLaunchContext):
        raise RuntimeError("activitysim_run is missing its resolved launch context")
    if not isinstance(activitysim_config_staging_plan, ActivitySimConfigStagingPlan):
        raise RuntimeError("activitysim_run is missing its configuration staging plan")
    stage_activitysim_config_roots(
        activitysim_launch_context,
        activitysim_config_staging_plan,
    )
    inputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=activitysim_launch_context.mutable_data_dir,
        land_use_table=Path(land_use_asim_in),
        households_table=Path(households_asim_in),
        persons_table=Path(persons_asim_in),
        omx_skims=Path(omx_skims) if omx_skims is not None else None,
    )
    ModelFactory().get_runner("activitysim", state).run(
        inputs,
        activitysim_launch_context,
        skim_mode=skim_mode,
        extra_inputs=extra_inputs,
        workspace=workspace,
    )


@define_step(
    model="activitysim",
    name_template="activitysim_postprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={
        ASIM_HOUSEHOLDS_IN: None,
        ASIM_PERSONS_IN: None,
        ASIM_LAND_USE_IN: None,
        ASIM_OMX_SKIMS: None,
        ZARR_SKIMS: None,
    },
    optional_input_keys=(
        *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
        *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
        USIM_POPULATION_SOURCE_H5,
        USIM_DATASTORE_CURRENT_H5,
        USIM_DATASTORE_BASE_H5,
    ),
    outputs=[USIM_DATASTORE_H5, *ASIM_REQUIRED_RUN_OUTPUT_KEYS],
    schema_outputs=[
        USIM_DATASTORE_H5,
        *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
        *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    ],
    input_binding="paths",
    tags=["activitysim", "postprocess"],
    **consist_step_meta(
        "activitysim_postprocess",
        input_contract=_ACTIVITYSIM_POSTPROCESS_INPUT_CONTRACT,
    ),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _activitysim_postprocess_callable(
    households_asim_in: Path,
    persons_asim_in: Path,
    land_use_asim_in: Path,
    omx_skims: Path,
    zarr_skims: Path,
    accessibility_asim_out: Path | None = None,
    beam_plans_asim_out: Path | None = None,
    disaggregate_accessibility_asim_out: Path | None = None,
    households_asim_out: Path | None = None,
    joint_tour_participants_asim_out: Path | None = None,
    land_use_asim_out: Path | None = None,
    non_mandatory_tour_destination_accessibility_asim_out: Path | None = None,
    persons_asim_out: Path | None = None,
    tours_asim_out: Path | None = None,
    trips_asim_out: Path | None = None,
    usim_population_source_h5: Path | None = None,
    usim_datastore_h5: Path | None = None,
    usim_datastore_base_h5: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    """Postprocess persisted ActivitySim raw outputs without a holder lookup."""

    del persons_asim_in, land_use_asim_in, omx_skims, zarr_skims
    del settings
    raw_outputs = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        raw_outputs={
            key: Path(path)
            for key, path in {
                "accessibility_asim_out": accessibility_asim_out,
                "beam_plans_asim_out": beam_plans_asim_out,
                "disaggregate_accessibility_asim_out": disaggregate_accessibility_asim_out,
                "households_asim_out": households_asim_out,
                "joint_tour_participants_asim_out": joint_tour_participants_asim_out,
                "land_use_asim_out": land_use_asim_out,
                "non_mandatory_tour_destination_accessibility_asim_out": non_mandatory_tour_destination_accessibility_asim_out,
                "persons_asim_out": persons_asim_out,
                "tours_asim_out": tours_asim_out,
                "trips_asim_out": trips_asim_out,
            }.items()
            if path is not None
        },
    )
    ModelFactory().get_postprocessor("activitysim", state).postprocess(
        raw_outputs,
        workspace,
        households_asim_input_path=str(households_asim_in),
        population_source_h5_path=(
            str(usim_population_source_h5)
            if usim_population_source_h5 is not None
            else None
        ),
        current_input_h5_path=str(
            usim_datastore_h5 or usim_population_source_h5 or usim_datastore_base_h5
        )
        if (
            usim_datastore_h5 is not None
            or usim_population_source_h5 is not None
            or usim_datastore_base_h5 is not None
        )
        else None,
    )


activitysim_preprocess = StepDefinition(
    name="activitysim_preprocess",
    function=_activitysim_preprocess_callable,
    resolve_inputs=_activitysim_preprocess_resolver,
    project_outputs=_project_activitysim_preprocess_outputs,
    input_contract=_ACTIVITYSIM_PREPROCESS_INPUT_CONTRACT,
    output_paths=activitysim_preprocess_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_activitysim_cache_options,
)
activitysim_run = StepDefinition(
    name="activitysim_run",
    function=_activitysim_run_callable,
    resolve_inputs=_activitysim_run_resolver,
    project_outputs=_project_activitysim_run_outputs,
    input_contract=_ACTIVITYSIM_RUN_INPUT_CONTRACT,
    output_paths=_activitysim_run_native_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_activitysim_cache_options,
)
activitysim_postprocess = StepDefinition(
    name="activitysim_postprocess",
    function=_activitysim_postprocess_callable,
    resolve_inputs=_activitysim_postprocess_resolver,
    project_outputs=_project_activitysim_postprocess_outputs,
    input_contract=_ACTIVITYSIM_POSTPROCESS_INPUT_CONTRACT,
    output_paths=activitysim_postprocess_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_activitysim_cache_options,
    preflight_identity=True,
)
