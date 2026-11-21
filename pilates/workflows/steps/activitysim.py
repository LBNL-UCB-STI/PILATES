from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from consist import (
    Artifact,
    ExecutionOptions,
    define_step,
    require_runtime_kwargs,
)
from consist.types import OutputArtifactSpec

from pilates.activitysim.preprocessor import (
    ActivitysimPreprocessor,
    ensure_activitysim_mutable_configs,
)
from pilates.activitysim.config_roots import (
    ActivitySimConfigRoot,
    required_activitysim_config_roots,
)
from pilates.activitysim.runner import (
    ActivitySimSkimDecision,
    ActivitysimSkimMode,
    ActivitysimRunner,
    ActivitySimConfigStagingPlan,
    ActivitySimLaunchContext,
    activitysim_persists_zarr_skims,
    asim_required_run_output_paths,
    asim_runtime_zarr_path,
    stage_activitysim_config_roots,
    validate_activitysim_zarr_skims,
)
from pilates.activitysim.outputs import (
    ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    configured_asim_output_keys,
)
from pilates.activitysim.postprocessor import (
    ActivitysimPostprocessor,
    _activitysim_archived_input_paths,
    _activitysim_iteration_output_paths,
    _next_iter_usim_input_store_path,
)
from pilates.config.models import PilatesConfig
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import iteration_index
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.runtime_overlays import resolve_runtime_input_output_overrides
from pilates.workflows.state_helpers import (
    requires_prior_beam_skim_handoff,
    resolve_forecast_year,
)
from pilates.workflows.consist_metadata import consist_step_meta
from pilates.workflows.contracts import (
    ConfigContract,
    InputContract,
    StepCallInputs,
    StepDefinition,
    require_output,
)
from pilates.workflows.sim_state import ArtifactRef, SimState
from pilates.workspace import Workspace
from pilates.utils.coupler_helpers import artifact_to_path, resolve_existing_path
from pilates.workflows.state import WorkflowState

from .shared import (
    _activitysim_output_facet_meta,
    _build_step_call_inputs,
    _metadata_cache_options,
    _native_contract_output_paths,
    _persisted_output_path,
)

logger = logging.getLogger(__name__)


def _land_use_enabled(state: WorkflowState) -> bool:
    """Return whether this invocation has a preceding UrbanSim producer."""

    try:
        return state.is_enabled(WorkflowState.Stage.land_use)
    except AttributeError:
        # Lightweight resolver tests and legacy callers may use a state facade
        # without enabled-stage metadata.  Those retain bootstrap semantics.
        return False


def activitysim_run_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    produces_zarr: bool,
) -> Dict[str, Any]:
    """
    Return cache-recoverable ActivitySim run outputs and logging metadata.

    Filters the required ActivitySim output map to ``configured_asim_output_keys``
    and includes ``ZARR_SKIMS`` when ``produces_zarr`` is enabled. When both
    ``forecast_year`` and ``iteration`` resolve, wraps each entry in an
    ``OutputArtifactSpec`` with year/iteration facet metadata; otherwise returns
    plain paths (with a Zarr directory spec for ``ZARR_SKIMS``).
    """
    expected_outputs: Dict[str, Any] = dict(
        asim_required_run_output_paths(settings, workspace)
    )
    output_keys = set(configured_asim_output_keys(settings))
    if produces_zarr:
        expected_outputs[ZARR_SKIMS] = asim_runtime_zarr_path(workspace)
        output_keys.add(ZARR_SKIMS)

    forecast_year = resolve_forecast_year(state)
    iteration = iteration_index(state)
    if forecast_year is None or iteration is None:
        return {
            key: (
                OutputArtifactSpec.for_directory(
                    path,
                    driver="zarr",
                )
                if key == ZARR_SKIMS
                else path
            )
            for key, path in expected_outputs.items()
            if key in output_keys
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
    step_inputs: StepCallInputs | None = None,
) -> Dict[str, Any]:
    del step_inputs
    """
    Declare cache-recoverable file outputs from ActivitySim preprocessing.

    Always declares ``land_use.csv``, ``households.csv``, and ``persons.csv``
    with file-schema profiling enabled. Omits ``ASIM_OMX_SKIMS`` when
    ``requires_prior_beam_skim_handoff`` is true because ActivitySim consumes
    BEAM's Zarr skims directly instead of staging an OMX file.
    """
    asim_data_dir = Path(workspace.get_asim_mutable_data_dir())
    candidate_paths = {
        ASIM_LAND_USE_IN: asim_data_dir / "land_use.csv",
        ASIM_HOUSEHOLDS_IN: asim_data_dir / "households.csv",
        ASIM_PERSONS_IN: asim_data_dir / "persons.csv",
        ASIM_OMX_SKIMS: asim_data_dir / "skims.omx",
    }
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
            path=candidate_paths[key], profile_file_schema=key in profiled_keys
        )
        for key in output_keys
        if candidate_paths.get(key) is not None
    }


def activitysim_postprocess_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> Dict[str, Any]:
    del step_inputs
    """
    Declare ActivitySim postprocess outputs for Consist logging and hydration.

    Collects the next-iteration UrbanSim datastore (``USIM_DATASTORE_H5`` when
    writeback is enabled), per-iteration ActivitySim output parquets, and
    archived ActivitySim inputs. Attaches file-schema profiling to core tabular
    outputs and year/iteration facet metadata whenever state context resolves.
    """
    raw_paths: Dict[str, Any] = {}
    next_usim_store = _next_iter_usim_input_store_path(
        settings,
        workspace,
        state,
    )
    if next_usim_store is not None:
        raw_paths[USIM_DATASTORE_H5] = next_usim_store
    raw_paths.update(_activitysim_iteration_output_paths(settings, state, workspace))
    raw_paths.update(_activitysim_archived_input_paths(state, workspace))

    profile_schema_keys = {
        "persons_asim_out",
        "trips_asim_out",
        "tours_asim_out",
        "beam_plans_asim_out",
        "households_asim_out",
    }
    forecast_year = resolve_forecast_year(state)
    iteration = iteration_index(state)
    return {
        key: (
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
        for key, path in raw_paths.items()
        if path is not None
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
    config_roots: tuple[ActivitySimConfigRoot, ...],
    runtime_zarr_path: Path | None = None,
) -> ActivitySimLaunchContext:
    """Build one launch tree from resolver-owned model input destinations."""

    workspace_root = Path(workspace.full_path)
    output_dir = Path(workspace.get_asim_output_dir())
    default_runtime_cache_dir = Path(workspace.get_asim_runtime_cache_dir())
    resolved_zarr_path = (
        Path(runtime_zarr_path)
        if runtime_zarr_path is not None
        else default_runtime_cache_dir / "skims.zarr"
    )
    runtime_cache_dir = (
        resolved_zarr_path.parent
        if resolved_zarr_path.name == "skims.zarr"
        else default_runtime_cache_dir
    )
    return ActivitySimLaunchContext(
        workspace_root=workspace_root,
        mutable_data_dir=staged_data_dir,
        output_dir=output_dir,
        compile_output_dir=output_dir.parent / "compile-output",
        mutable_configs_dir=staged_configs_dir,
        runtime_cache_dir=runtime_cache_dir,
        runtime_zarr_path=resolved_zarr_path,
        shared_cache_dir=workspace_root / "shared_cache",
        shared_tmp_dir=workspace_root / "tmp",
        requires_staged_config_dirs=True,
        config_roots=config_roots,
    )


def _sim_state_role_candidates(
    sim_state: SimState,
    *,
    land_use_enabled: bool,
) -> dict[str, ArtifactRef | None]:
    """Map typed ``SimState`` fields to ActivitySim role names."""
    candidates: dict[str, ArtifactRef | None] = {
        USIM_POPULATION_SOURCE_H5: (
            sim_state.land_use.population_source
            if sim_state.land_use.population_source is not None
            else (sim_state.land_use.current_h5 if not land_use_enabled else None)
        ),
        USIM_DATASTORE_CURRENT_H5: sim_state.land_use.current_h5,
        USIM_DATASTORE_BASE_H5: sim_state.land_use.prior_h5,
        FINAL_SKIMS_OMX: sim_state.network.skims_omx,
        ASIM_OMX_SKIMS: (
            sim_state.network.skims_omx
            if sim_state.network.skims_omx is not None
            else sim_state.demand.skims_omx_in
        ),
        ZARR_SKIMS: (
            sim_state.network.skims_zarr
            if sim_state.network.skims_zarr is not None
            else sim_state.demand.skims_zarr
        ),
        ASIM_LAND_USE_IN: sim_state.demand.land_use_csv,
        ASIM_HOUSEHOLDS_IN: sim_state.demand.households_in,
        ASIM_PERSONS_IN: sim_state.demand.persons_in,
        "households_asim_out": sim_state.demand.households,
        "persons_asim_out": sim_state.demand.persons,
        "tours_asim_out": sim_state.demand.tours,
        "trips_asim_out": sim_state.demand.trips,
        "beam_plans_asim_out": sim_state.demand.plans,
    }
    for key, value in sim_state.extras.items():
        if key not in candidates or candidates[key] is None:
            candidates[key] = value
    return candidates


def _native_activitysim_step_inputs(
    *,
    step_name: str,
    required_roles: tuple[str, ...],
    optional_roles: tuple[str, ...],
    logical_destinations: Mapping[str, Any],
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
) -> StepCallInputs:
    """Build typed ``StepCallInputs`` for an ActivitySim step from ``SimState``."""

    workspace_root = Path(get_setting(workspace, "full_path", "."))
    destinations: dict[str, Path] = {
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
    candidates = _sim_state_role_candidates(
        sim_state,
        land_use_enabled=_land_use_enabled(state),
    )
    if prior_outputs is not None:
        for key, value in prior_outputs.items():
            if value is not None:
                candidates[key] = value
    return _build_step_call_inputs(
        candidates=candidates,
        required_roles=required_roles,
        optional_roles=optional_roles,
        logical_destinations=destinations,
        metadata={"step_name": step_name},
    )


def _activitysim_preprocess_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    ensure_activitysim_mutable_configs(settings, workspace)
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    return _native_activitysim_step_inputs(
        step_name="activitysim_preprocess",
        required_roles=_ACTIVITYSIM_PREPROCESS_REQUIRED_ROLES,
        optional_roles=(
            () if requires_beam_skim else _ACTIVITYSIM_PREPROCESS_OPTIONAL_ROLES
        ),
        logical_destinations=ActivitysimPreprocessor.declared_expected_inputs(
            settings, state, workspace
        ),
        state=state,
        workspace=workspace,
        sim_state=active_sim_state,
    )


def _activitysim_run_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    ensure_activitysim_mutable_configs(settings, workspace)
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    required_roles = _ACTIVITYSIM_RUN_REQUIRED_ROLES
    optional_roles = _ACTIVITYSIM_RUN_SKIM_ROLES
    resolved = _native_activitysim_step_inputs(
        step_name="activitysim_run",
        required_roles=required_roles,
        optional_roles=optional_roles,
        logical_destinations=ActivitysimRunner.declared_expected_inputs(
            settings, state, workspace
        ),
        state=state,
        workspace=workspace,
        sim_state=active_sim_state,
    )

    from pilates.workspace import is_dry_run_workspace

    zarr_input = resolved.inputs.get(ZARR_SKIMS)
    omx_input = resolved.inputs.get(ASIM_OMX_SKIMS)
    zarr_rejection: str | None = None
    if zarr_input is not None and not is_dry_run_workspace(workspace):
        zarr_path = artifact_to_path(zarr_input, workspace=workspace)
        existing_zarr_path = (
            resolve_existing_path(zarr_path, workspace=workspace)
            if zarr_path is not None
            else None
        )
        if existing_zarr_path is not None:
            zarr_rejection = validate_activitysim_zarr_skims(existing_zarr_path)
        elif isinstance(zarr_input, Artifact):
            zarr_rejection = None
        else:
            zarr_rejection = (
                "could not resolve a concrete path"
                if zarr_path is None
                else validate_activitysim_zarr_skims(zarr_path)
            )

    if zarr_input is not None and zarr_rejection is None:
        decision = ActivitySimSkimDecision(
            role=ZARR_SKIMS,
            mode="zarr",
            produces_zarr=False,
            selected_source="sim_state",
        )
    elif omx_input is not None:
        if zarr_rejection is not None:
            logger.warning(
                "activitysim_run rejected %s before binding selection: %s; "
                "falling back to %s",
                ZARR_SKIMS,
                zarr_rejection,
                ASIM_OMX_SKIMS,
            )
        get_asim_mutable_configs_dir = getattr(
            workspace, "get_asim_mutable_configs_dir", None
        )
        configs_root = (
            get_asim_mutable_configs_dir()
            if callable(get_asim_mutable_configs_dir)
            else None
        )
        produces_zarr = activitysim_persists_zarr_skims(
            settings,
            configs_root=configs_root,
        )
        decision = ActivitySimSkimDecision(
            role=ASIM_OMX_SKIMS,
            mode="omx",
            produces_zarr=produces_zarr,
            selected_source="sim_state",
        )
    elif zarr_rejection is not None:
        raise RuntimeError(f"activitysim_run rejected {ZARR_SKIMS}: {zarr_rejection}")
    else:
        expected = (
            f"{ZARR_SKIMS} or {ASIM_OMX_SKIMS}"
            if not requires_beam_skim
            else f"prior BEAM skim handoff ({ZARR_SKIMS} or {ASIM_OMX_SKIMS})"
        )
        raise RuntimeError(
            f"activitysim_run requires one published skim role: {expected}"
        )

    selected_roles = (*_ACTIVITYSIM_RUN_REQUIRED_ROLES, decision.role)
    binding_inputs: dict[str, ArtifactRef] = {}
    for role in selected_roles:
        value = resolved.inputs.get(role)
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
    logical_destinations = {
        key: value
        for key, value in resolved.logical_destinations.items()
        if key in selected_roles
    }
    if decision.role == ZARR_SKIMS:
        zarr_path = artifact_to_path(
            binding_inputs.get(ZARR_SKIMS), workspace=workspace
        )
        zarr_existing = (
            resolve_existing_path(zarr_path, workspace=workspace)
            if zarr_path is not None
            else None
        ) or zarr_path
        if zarr_existing is not None:
            logical_destinations[ZARR_SKIMS] = Path(zarr_existing)
    return StepCallInputs(
        inputs=binding_inputs,
        required_keys=_ACTIVITYSIM_RUN_REQUIRED_ROLES,
        optional_keys=(decision.role,),
        logical_destinations=logical_destinations,
        metadata=metadata,
    )


def _activitysim_postprocess_resolver(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    ensure_activitysim_mutable_configs(settings, workspace)
    required_roles, optional_roles = resolve_runtime_input_output_overrides(
        step_name="activitysim_postprocess",
        land_use_enabled=_land_use_enabled(state),
        run_mode=None,
        required_inputs=(
            *_ACTIVITYSIM_POSTPROCESS_BASE_REQUIRED_ROLES,
            *configured_asim_output_keys(settings),
        ),
        optional_inputs=_ACTIVITYSIM_POSTPROCESS_OPTIONAL_ROLES,
    )
    resolved = _native_activitysim_step_inputs(
        step_name="activitysim_postprocess",
        required_roles=required_roles,
        optional_roles=optional_roles,
        logical_destinations=ActivitysimPostprocessor.declared_expected_inputs(
            settings, state, workspace
        ),
        state=state,
        workspace=workspace,
        sim_state=active_sim_state,
        prior_outputs=prior_outputs,
    )
    inputs = dict(resolved.inputs)
    removed_roles: set[str] = set()
    if (
        USIM_DATASTORE_BASE_H5 in inputs
        and not isinstance(inputs[USIM_DATASTORE_BASE_H5], Artifact)
        and (
            isinstance(inputs.get(USIM_POPULATION_SOURCE_H5), Artifact)
            or isinstance(inputs.get(USIM_DATASTORE_CURRENT_H5), Artifact)
        )
    ):
        inputs.pop(USIM_DATASTORE_BASE_H5, None)
        removed_roles.add(USIM_DATASTORE_BASE_H5)
    elif (
        isinstance(inputs.get(USIM_DATASTORE_BASE_H5), Artifact)
        and isinstance(inputs.get(USIM_DATASTORE_CURRENT_H5), Artifact)
        and inputs[USIM_DATASTORE_BASE_H5].id != inputs[USIM_DATASTORE_CURRENT_H5].id
    ):
        base_local = _h5_input_local_path(inputs[USIM_DATASTORE_BASE_H5])
        curr_local = _h5_input_local_path(inputs[USIM_DATASTORE_CURRENT_H5])
        if base_local is not None and base_local == curr_local:
            inputs[USIM_DATASTORE_BASE_H5] = inputs[USIM_DATASTORE_CURRENT_H5]

    population_source = inputs.get(USIM_POPULATION_SOURCE_H5)
    current_datastore = inputs.get(USIM_DATASTORE_CURRENT_H5)
    base_datastore = inputs.get(USIM_DATASTORE_BASE_H5)
    if _is_population_source_alias(population_source, current_datastore):
        inputs.pop(USIM_DATASTORE_CURRENT_H5, None)
        current_datastore = None
        removed_roles.add(USIM_DATASTORE_CURRENT_H5)
    if _is_population_source_alias(
        population_source, base_datastore
    ) or _is_population_source_alias(current_datastore, base_datastore):
        inputs.pop(USIM_DATASTORE_BASE_H5, None)
        removed_roles.add(USIM_DATASTORE_BASE_H5)

    workspace_root = Path(get_setting(workspace, "full_path", ".")).resolve()
    filtered_destinations = {
        role: dest
        for role, dest in resolved.logical_destinations.items()
        if role not in removed_roles
    }
    used_destinations: set[Path] = {
        Path(dest).resolve()
        for role, dest in filtered_destinations.items()
        if role not in _ACTIVITYSIM_POSTPROCESS_OPTIONAL_ROLES
    }
    destinations_updated = False
    for role in _ACTIVITYSIM_POSTPROCESS_OPTIONAL_ROLES:
        if role in removed_roles or role not in inputs:
            continue
        existing_local = _h5_input_local_path(inputs[role], workspace=workspace)
        if (
            existing_local is not None
            and existing_local.is_relative_to(workspace_root)
            and existing_local not in used_destinations
        ):
            if filtered_destinations.get(role) != existing_local:
                filtered_destinations[role] = existing_local
                destinations_updated = True
            used_destinations.add(existing_local)

    if not removed_roles and not destinations_updated:
        return resolved

    return StepCallInputs(
        inputs=inputs,
        required_keys=tuple(
            k for k in resolved.required_keys if k not in removed_roles
        ),
        optional_keys=tuple(
            k for k in resolved.optional_keys if k not in removed_roles
        ),
        logical_destinations=filtered_destinations,
        metadata=resolved.metadata,
    )


def _h5_input_local_path(
    value: Any,
    *,
    workspace: Workspace | None = None,
) -> Path | None:
    """Resolve an H5 input role value (Artifact or path) to its canonical local path."""

    if isinstance(value, Artifact):
        candidate = (
            artifact_to_path(value, workspace=workspace) or value.abs_path or value.path
        )
        return Path(str(candidate)).resolve() if candidate else None
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return Path(text).resolve() if text else None
    return None


def _is_population_source_alias(population_source: Any, current_datastore: Any) -> bool:
    """Return whether current datastore is the population-source alias."""

    if population_source is current_datastore:
        return population_source is not None
    if isinstance(population_source, Artifact) and isinstance(
        current_datastore, Artifact
    ):
        if population_source.id == current_datastore.id or (
            population_source.key == USIM_POPULATION_SOURCE_H5
            and current_datastore.key == USIM_POPULATION_SOURCE_H5
        ):
            return True
    primary_path = _h5_input_local_path(population_source)
    candidate_path = _h5_input_local_path(current_datastore)
    return (
        primary_path is not None
        and candidate_path is not None
        and primary_path == candidate_path
    )


def _activitysim_execution_options(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> ExecutionOptions:
    """Request Consist materialization at the one deterministic path per role."""

    del state
    if step_inputs is None:
        return ExecutionOptions(input_binding="paths")
    selected_roles = set(step_inputs.inputs.keys())
    runtime_kwargs: dict[str, Any] = {}
    if step_inputs.metadata.get("step_name") == "activitysim_run":
        data_roles = [role for role in selected_roles if role != ZARR_SKIMS]
        staged_destinations = {
            Path(step_inputs.logical_destinations[role]).parent for role in data_roles
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
        config_roots = required_activitysim_config_roots(
            activitysim_settings.main_configs_dir
        )
        launch_context = _activitysim_launch_context(
            workspace,
            staged_data_dir=staged_data_dir,
            staged_configs_dir=workspace_root
            / "activitysim"
            / "native-launch"
            / "configs",
            config_roots=config_roots,
            runtime_zarr_path=(
                Path(step_inputs.logical_destinations[ZARR_SKIMS])
                if ZARR_SKIMS in selected_roles
                and ZARR_SKIMS in step_inputs.logical_destinations
                else None
            ),
        )
        runtime_kwargs["activitysim_launch_context"] = launch_context
        runtime_kwargs["activitysim_config_staging_plan"] = (
            ActivitySimConfigStagingPlan(
                source_root=Path(workspace.get_asim_mutable_configs_dir()),
                config_roots=config_roots,
            )
        )
    return ExecutionOptions(
        input_binding="paths",
        input_paths={
            key: destination
            for key, destination in step_inputs.logical_destinations.items()
            if key in selected_roles
        },
        input_materialization="requested",
        input_materialization_mode="copy",
        runtime_kwargs=runtime_kwargs,
    )


def _activitysim_run_produces_zarr(
    step_inputs: StepCallInputs,
) -> bool:
    """Return the immutable native invocation decision for Zarr publication."""

    value = step_inputs.metadata.get(_ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY)
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
    step_inputs: StepCallInputs,
) -> Dict[str, Any]:
    """Declare only the immutable invocation-specific native output set."""

    return activitysim_run_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        produces_zarr=_activitysim_run_produces_zarr(step_inputs),
    )


def _project_activitysim_preprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> ActivitySimPreprocessOutputs:
    del step_inputs
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
    return projected


def _project_activitysim_run_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> ActivitySimRunOutputs:
    if step_inputs is None:
        raise RuntimeError(
            "activitysim_run projection requires the resolved native skim decision"
        )
    produces_zarr = _activitysim_run_produces_zarr(step_inputs)
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
        step_inputs=step_inputs,
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
    return projected


def _project_activitysim_postprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> ActivitySimPostprocessOutputs:
    del step_inputs
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
    return projected


@define_step(
    model="activitysim",
    name_template="activitysim_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_POPULATION_SOURCE_H5: None},
    optional_input_keys=(FINAL_SKIMS_OMX,),
    outputs=_native_contract_output_paths(activitysim_preprocess_output_paths),
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
    if prepare_omx_skims:
        from pilates.activitysim.admission import (
            has_initial_skims_admission_policy,
            preflight_activitysim_initial_skims_admission,
        )
        from pilates.utils import consist_runtime as cr
        from pilates.utils.admission_helpers import (
            extract_admission_reports,
            resolve_tracker_report_dir,
        )

        if has_initial_skims_admission_policy(settings):
            active_tracker = cr.current_tracker()
            existing_reports = extract_admission_reports(cr.current_run())
            report_dir = resolve_tracker_report_dir(
                active_tracker,
                fallback=workspace.full_path,
            )
            preflight_activitysim_initial_skims_admission(
                settings=settings,
                metadata_logger=active_tracker,
                tracker=active_tracker,
                workspace_path=Path(workspace.full_path),
                report_dir=report_dir,
                existing_admission_reports=existing_reports,
                explicit_skims_path=(
                    Path(final_skims_omx) if final_skims_omx is not None else None
                ),
            )
    preprocess_kwargs: dict[str, Any] = {
        "population_source_h5_path": str(usim_population_source_h5),
        "prepare_omx_skims": prepare_omx_skims,
    }
    if prepare_omx_skims and final_skims_omx is not None:
        preprocess_kwargs["final_skims_omx"] = final_skims_omx
    ActivitysimPreprocessor("activitysim", state).preprocess(
        workspace, **preprocess_kwargs
    )
    declared_outputs = activitysim_preprocess_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    from pilates.workflows.steps.shared import _validation_context

    projected = _project_activitysim_preprocess_outputs(
        {key: None for key in declared_outputs},
        settings=settings,
        state=state,
        workspace=workspace,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_preprocess",
        )
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

    # del settings
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
    ActivitysimRunner("activitysim", state).run(
        inputs,
        activitysim_launch_context,
        skim_mode=skim_mode,
        extra_inputs=extra_inputs,
        workspace=workspace,
    )

    step_inputs_stub = StepCallInputs(
        metadata={_ACTIVITYSIM_PRODUCES_ZARR_METADATA_KEY: skim_mode == "zarr"}
    )
    declared_outputs = _activitysim_run_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        step_inputs=step_inputs_stub,
    )
    outputs_mock = {k: None for k in declared_outputs}
    from pilates.workflows.steps.shared import _validation_context

    projected = _project_activitysim_run_outputs(
        outputs_mock,
        settings=settings,
        state=state,
        workspace=workspace,
        step_inputs=step_inputs_stub,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_run",
        )
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
    # del settings
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
    ActivitysimPostprocessor("activitysim", state).postprocess(
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
        base_input_h5_path=(
            str(usim_datastore_base_h5) if usim_datastore_base_h5 is not None else None
        ),
    )
    declared_outputs = activitysim_postprocess_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    from pilates.workflows.steps.shared import _validation_context

    has_usim_input = any(
        x is not None
        for x in (
            usim_datastore_h5,
            usim_population_source_h5,
            usim_datastore_base_h5,
        )
    )
    projected = _project_activitysim_postprocess_outputs(
        {
            key: None
            for key in declared_outputs
            if key != USIM_DATASTORE_H5 or has_usim_input
        },
        settings=settings,
        state=state,
        workspace=workspace,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="activitysim_postprocess",
        )
    )


activitysim_preprocess = StepDefinition(
    name="activitysim_preprocess",
    function=_activitysim_preprocess_callable,
    resolve_inputs=_activitysim_preprocess_resolver,
    project_outputs=_project_activitysim_preprocess_outputs,
    input_contract=_ACTIVITYSIM_PREPROCESS_INPUT_CONTRACT,
    outputs=activitysim_preprocess_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_metadata_cache_options(),
    archive_outputs=True,
)
activitysim_run = StepDefinition(
    name="activitysim_run",
    function=_activitysim_run_callable,
    resolve_inputs=_activitysim_run_resolver,
    project_outputs=_project_activitysim_run_outputs,
    input_contract=_ACTIVITYSIM_RUN_INPUT_CONTRACT,
    output_paths=_activitysim_run_native_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_metadata_cache_options(),
    optional_output_keys=ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    archive_outputs=True,
)
activitysim_postprocess = StepDefinition(
    name="activitysim_postprocess",
    function=_activitysim_postprocess_callable,
    resolve_inputs=_activitysim_postprocess_resolver,
    project_outputs=_project_activitysim_postprocess_outputs,
    input_contract=_ACTIVITYSIM_POSTPROCESS_INPUT_CONTRACT,
    output_paths=activitysim_postprocess_output_paths,
    execution_options=_activitysim_execution_options,
    cache_options=_metadata_cache_options(),
    archive_outputs=True,
    optional_output_keys=ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
)
