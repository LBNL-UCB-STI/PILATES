from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.workflows.binding import (
    build_binding_plan,
    build_key_only_binding_plan,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StageRunner,
    StepRef,
    run_workflow,
)
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
)
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
    ASIM_OMX_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.activitysim.runner import persist_sharrow_cache_enabled
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

_ACTIVITYSIM_PILOT_H5_ROLE_KEYS = (
    USIM_POPULATION_SOURCE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_BASE_H5,
)


@dataclass
class ActivityDemandPhaseInputs:
    """
    Inputs for one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Pre-resolved UrbanSim datastore inputs, if land use already ran or
        fallback inputs were provided.
    """

    year: int
    iteration: int
    usim_inputs: Mapping[str, Union[str, os.PathLike]]


@dataclass
class ActivityDemandPhaseOutputs:
    """
    Outputs from one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    activity_demand_outputs : Optional[dict[str, Any]]
        Mapping containing ActivitySim outputs needed downstream
        (e.g., households, persons, plans). None if not produced.
    """

    activity_demand_outputs: Optional[Dict[str, Any]]


def _should_force_restart_activitysim_compile(state: WorkflowState) -> bool:
    if not bool(getattr(state, "is_restart_run", False)):
        return False
    return not bool(getattr(state, "_restart_activitysim_compile_done", False))


def _activitysim_exact_rewind_restore(
    workspace: Workspace,
    *,
    year: int,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    metadata = getattr(workspace, "_activitysim_exact_rewind_restore", None)
    if not isinstance(metadata, dict):
        return None
    if metadata.get("year") != year or metadata.get("iteration") != iteration:
        return None
    return metadata


def _seed_exact_rewind_activitysim_preprocess_outputs(
    *,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
) -> ActivitySimPreprocessOutputs:
    input_dir = workspace.get_asim_mutable_data_dir()
    preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=Path(input_dir),
        land_use_table=Path(input_dir) / "land_use.csv",
        households_table=Path(input_dir) / "households.csv",
        persons_table=Path(input_dir) / "persons.csv",
        omx_skims=(
            (Path(input_dir) / "skims.omx")
            if (Path(input_dir) / "skims.omx").exists()
            else None
        ),
    )
    outputs_holder.activitysim_preprocess = preprocess_outputs
    return preprocess_outputs


def _resolve_activitysim_surface(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Optional["EnabledWorkflowSurface"]:
    if surface is not None:
        return surface
    from pilates.workflows.surface import build_enabled_workflow_surface

    return build_enabled_workflow_surface(settings, state=state)


def _surface_restart_missing_explicit_roles(
    *,
    coupler: CouplerProtocol,
    resolved_usim_inputs: Mapping[str, Union[str, os.PathLike]],
    surface: Optional["EnabledWorkflowSurface"],
) -> tuple[str, ...]:
    if surface is None:
        return ()
    runtime_surface = surface.step_surface("activitysim_preprocess")
    if runtime_surface is None or not runtime_surface.enabled:
        return ()
    get_value = getattr(coupler, "get", None)
    return tuple(
        key
        for key in (
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
        )
        if bool(
            getattr(
                runtime_surface.input_role_policies.get(key),
                "restart_requires_explicit_before_execution",
                False,
            )
        )
        and key not in resolved_usim_inputs
        and not (callable(get_value) and get_value(key) is not None)
    )


def _seed_postprocess_role_fallbacks_from_coupler(
    *,
    coupler: CouplerProtocol,
    resolved_usim_inputs: Dict[str, Union[str, os.PathLike]],
) -> None:
    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return
    coupler_population = get_value(USIM_POPULATION_SOURCE_H5)
    coupler_current = get_value(USIM_DATASTORE_CURRENT_H5)
    if (
        USIM_DATASTORE_CURRENT_H5 not in resolved_usim_inputs
        and coupler_current is not None
    ):
        resolved_usim_inputs[USIM_DATASTORE_CURRENT_H5] = coupler_current
    elif (
        USIM_DATASTORE_CURRENT_H5 not in resolved_usim_inputs
        and coupler_population is not None
    ):
        resolved_usim_inputs[USIM_DATASTORE_CURRENT_H5] = coupler_population
    if (
        USIM_POPULATION_SOURCE_H5 not in resolved_usim_inputs
        and coupler_population is not None
    ):
        resolved_usim_inputs[USIM_POPULATION_SOURCE_H5] = coupler_population
    elif (
        USIM_POPULATION_SOURCE_H5 not in resolved_usim_inputs
        and coupler_current is not None
    ):
        resolved_usim_inputs[USIM_POPULATION_SOURCE_H5] = coupler_current


def _run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    inputs: ActivityDemandPhaseInputs,
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> ActivityDemandPhaseOutputs:
    """
    Run ActivitySim for a single supply-demand iteration.

    This executes the ActivitySim preprocess, compile (once per year),
    and run/postprocess steps. It also assembles the required inputs
    from UrbanSim outputs or fallbacks and ensures skims are available
    when resuming after compilation.
    """
    formatted_print("ACTIVITY DEMAND MODEL")
    stage_runner = StageRunner(
        stage_name="activity_demand",
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{inputs.year}_iter{inputs.iteration}",
        iteration=inputs.iteration,
        manifest_config=manifest_config,
        run_workflow_fn=run_workflow,
    )
    compile_runner = StageRunner(
        stage_name="activity_demand_compile",
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=str(inputs.year),
        iteration=-1,
        run_workflow_fn=run_workflow,
    )
    exact_rewind_restore = _activitysim_exact_rewind_restore(
        workspace,
        year=inputs.year,
        iteration=inputs.iteration,
    )
    runtime_surface = _resolve_activitysim_surface(
        settings=settings,
        state=state,
        surface=surface,
    )
    profile = runtime_surface.profile
    resolved_usim_inputs = dict(inputs.usim_inputs)
    missing_restart_roles = _surface_restart_missing_explicit_roles(
        coupler=coupler,
        resolved_usim_inputs=resolved_usim_inputs,
        surface=runtime_surface,
    )
    if missing_restart_roles:
        raise RuntimeError(
            "Restart metadata is missing required post-land-use UrbanSim H5 roles "
            f"for ActivitySim: {', '.join(missing_restart_roles)}. "
            "This restart likely predates the explicit population-source H5 role split."
        )
    if (
        bool(getattr(state, "is_restart_run", False))
        and profile.land_use_enabled
        and not resolved_usim_inputs
    ):
        from .supply_demand_resume import (
            _restore_supply_demand_usim_inputs_for_resume,
        )

        for key, value in _restore_supply_demand_usim_inputs_for_resume(
            coupler=coupler,
            workspace=workspace,
            state=state,
            settings=settings,
        ).items():
            resolved_usim_inputs.setdefault(key, value)

    # ActivitySim runs in two manifest-checkpointed phases:
    # 1) Preprocess (per-iteration) to prepare compile inputs.
    # 2) Compile (per-year) outside manifest checkpointing.
    # 3) Run/Postprocess (per-iteration) for demand outputs.
    if exact_rewind_restore is not None:
        _seed_exact_rewind_activitysim_preprocess_outputs(
            workspace=workspace,
            outputs_holder=outputs_holder,
        )
    else:
        preprocess_explicit_inputs: Optional[Dict[str, Union[str, os.PathLike]]] = None
        if not profile.land_use_enabled:
            population_source = (
                resolved_usim_inputs.get(USIM_DATASTORE_BASE_H5)
                or resolved_usim_inputs.get(USIM_DATASTORE_CURRENT_H5)
            )
            if population_source is not None:
                preprocess_explicit_inputs = {
                    USIM_POPULATION_SOURCE_H5: population_source,
                }

        preprocess_binding = build_binding_plan(
            step_name="activitysim_preprocess",
            coupler=coupler,
            explicit_inputs=preprocess_explicit_inputs,
            fallback_inputs=resolved_usim_inputs,
            settings=settings,
            state=state,
            workspace=workspace,
            year=inputs.year,
            profile=profile,
            surface=runtime_surface,
        )

        if preprocess_binding.missing_required:
            raise RuntimeError(
                "ActivitySim preprocess requires a resolved population-source UrbanSim datastore "
                "(population_source, forecast output, current, or base), but none were available."
            )

        stage_runner.run_step(
            stage_name="activity_demand_preprocess",
            step=StepRef(
                name="activitysim_preprocess",
                step_func=make_activitysim_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                binding=preprocess_binding,
                year=state.forecast_year,
            ),
        )

    def _resolved_existing_zarr_skims_path() -> Optional[str]:
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_path = artifact_to_existing_path(
                resolve_artifact_from_value(
                    get_value(ZARR_SKIMS),
                    key=ZARR_SKIMS,
                    workspace=workspace,
                ),
                workspace=workspace,
                materialize_from_archive=True,
            )
            if zarr_path:
                return zarr_path
        candidate = os.path.join(workspace.get_asim_runtime_cache_dir(), "skims.zarr")
        return resolve_existing_path(
            candidate,
            workspace=workspace,
            materialize_from_archive=True,
        )

    def _resolved_existing_numba_cache_path() -> Optional[str]:
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            cache_path = artifact_to_existing_path(
                resolve_artifact_from_value(
                    get_value(ASIM_SHARROW_CACHE_DIR),
                    key=ASIM_SHARROW_CACHE_DIR,
                    workspace=workspace,
                ),
                workspace=workspace,
                materialize_from_archive=True,
            )
            if cache_path and os.path.isdir(cache_path):
                for _root, _dirs, files in os.walk(cache_path):
                    if files:
                        return cache_path
        cache_path = resolve_existing_path(
            os.path.join(workspace.full_path, "shared_cache", "numba"),
            workspace=workspace,
            materialize_from_archive=True,
        )
        if cache_path and os.path.isdir(cache_path):
            for _root, _dirs, files in os.walk(cache_path):
                if files:
                    return cache_path
        return None

    def _republish_existing_compile_artifacts() -> None:
        zarr_path = _resolved_existing_zarr_skims_path()
        if zarr_path:
            set_coupler_from_artifact(
                coupler,
                ZARR_SKIMS,
                None,
                fallback=zarr_path,
            )

        if requires_numba_cache:
            cache_path = _resolved_existing_numba_cache_path()
            if cache_path:
                set_coupler_from_artifact(
                    coupler,
                    ASIM_SHARROW_CACHE_DIR,
                    None,
                    fallback=cache_path,
                )

    activitysim_cfg = getattr(settings, "activitysim", None)
    requires_numba_cache = bool(
        getattr(activitysim_cfg, "num_processes", 1) > 1
        and persist_sharrow_cache_enabled(settings)
    )

    # ActivitySim compilation is effectively a run-level one-time step because
    # WorkflowState.asim_compiled is not reset on year advance. On restart, we
    # intentionally force one recompile before the first resumed ActivitySim run.
    # If compiled artifacts are missing on local ephemeral storage, also force
    # recompile instead of hard-failing.
    force_restart_compile = _should_force_restart_activitysim_compile(state)
    needs_compile = force_restart_compile or not state.asim_compiled
    if force_restart_compile:
        logger.info(
            "Restart detected; forcing ActivitySim compile before first resumed "
            "ActivitySim run (year=%s iteration=%s).",
            state.current_year,
            state.current_inner_iter,
        )
    if not needs_compile:
        existing_zarr_path = _resolved_existing_zarr_skims_path()
        existing_cache_path = (
            _resolved_existing_numba_cache_path() if requires_numba_cache else "n/a"
        )
        if not existing_zarr_path or (requires_numba_cache and not existing_cache_path):
            missing_parts = []
            if not existing_zarr_path:
                missing_parts.append("zarr_skims")
            if requires_numba_cache and not existing_cache_path:
                missing_parts.append("numba_cache")
            logger.warning(
                "ActivitySim marked compiled for year %s but compiled artifacts were "
                "missing (%s); forcing recompilation.",
                state.current_year,
                ",".join(missing_parts),
            )
            needs_compile = True

    if exact_rewind_restore is not None:
        needs_compile = False
    if needs_compile:
        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim compile requires preprocess outputs.")
        compile_store_inputs = step_output_handoff_mapping(upstream, coupler=coupler)
        compile_explicit_inputs: Dict[str, Any] = {}
        if ASIM_OMX_SKIMS in compile_store_inputs:
            compile_explicit_inputs[ASIM_OMX_SKIMS] = compile_store_inputs[
                ASIM_OMX_SKIMS
            ]
        compile_binding = build_binding_plan(
            step_name="activitysim_compile",
            coupler=coupler,
            explicit_inputs=compile_explicit_inputs or None,
            settings=settings,
            state=state,
            workspace=workspace,
            year=inputs.year,
            surface=runtime_surface,
        )
        if compile_binding.missing_required:
            raise RuntimeError(
                "ActivitySim compile requires omx_skims input, but it could not be "
                "resolved from explicit preprocess outputs or coupler keys."
            )
        activitysim_compile_step = make_activitysim_compile_step(
            coupler=coupler,
            outputs_holder=outputs_holder,
        )
        compile_runner.run_step(
            stage_name="activity_demand_compile",
            step=StepRef(
                name="activitysim_compile",
                step_func=activitysim_compile_step,
                binding=compile_binding,
                phase="compile",
                year=state.forecast_year,
                iteration=-1,
            ),
        )
        setattr(state, "_restart_activitysim_compile_done", True)
    elif exact_rewind_restore is None:
        _republish_existing_compile_artifacts()
    final_zarr_path = _resolved_existing_zarr_skims_path()
    final_cache_path = (
        _resolved_existing_numba_cache_path() if requires_numba_cache else "n/a"
    )
    if (
        exact_rewind_restore is None
        and (
            not final_zarr_path or (requires_numba_cache and not final_cache_path)
        )
    ):
        missing_parts = []
        if not final_zarr_path:
            missing_parts.append("zarr_skims")
        if requires_numba_cache and not final_cache_path:
            missing_parts.append("numba_cache")
        raise RuntimeError(
            "ActivitySim run requires compiled artifacts before execution, but "
            f"missing after compile check: {','.join(missing_parts)}. "
            "This guard prevents multi-process runtime re-compilation races."
        )

    upstream_preprocess = outputs_holder.activitysim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    asim_run_input_keys = [
        short_name for short_name, _, _ in upstream_preprocess._iter_record_items()
    ]
    asim_run_input_keys = [key for key in asim_run_input_keys if key != ASIM_OMX_SKIMS]
    if exact_rewind_restore is None:
        asim_run_input_keys.append(ZARR_SKIMS)
    elif exact_rewind_restore.get("zarr_available"):
        asim_run_input_keys.append(ZARR_SKIMS)
    if requires_numba_cache and exact_rewind_restore is None:
        asim_run_input_keys.append(ASIM_SHARROW_CACHE_DIR)

    optional_run_keys = (
        [ASIM_SHARROW_CACHE_DIR]
        if requires_numba_cache and exact_rewind_restore is None
        else []
    )

    stage_runner.run_step(
        stage_name="activity_demand_run",
        step=StepRef(
            name="activitysim_run",
            step_func=make_activitysim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=build_key_only_binding_plan(
                step_name="activitysim_run",
                input_keys=asim_run_input_keys,
                optional_input_keys=optional_run_keys,
                coupler=coupler,
                settings=settings,
                state=state,
                workspace=workspace,
                year=inputs.year,
                surface=runtime_surface,
            ),
            year=state.forecast_year,
        ),
    )

    if outputs_holder.activitysim_run is None:
        raise RuntimeError("ActivitySim run must complete first")

    postprocess_required_keys: tuple[str, ...] = ()
    postprocess_optional_keys: tuple[str, ...] = ()
    if runtime_surface is not None:
        postprocess_surface = runtime_surface.step_surface("activitysim_postprocess")
        if postprocess_surface is not None:
            postprocess_required_keys = tuple(
                key
                for key in postprocess_surface.required_input_keys
                if key in _ACTIVITYSIM_PILOT_H5_ROLE_KEYS
            )
            postprocess_optional_keys = tuple(
                key
                for key in postprocess_surface.optional_input_keys
                if key in _ACTIVITYSIM_PILOT_H5_ROLE_KEYS
            )
    if profile.land_use_enabled:
        _seed_postprocess_role_fallbacks_from_coupler(
            coupler=coupler,
            resolved_usim_inputs=resolved_usim_inputs,
        )
    activitysim_postprocess_binding = build_binding_plan(
        step_name="activitysim_postprocess",
        coupler=coupler,
        fallback_inputs=resolved_usim_inputs,
        required_keys=postprocess_required_keys,
        optional_keys=postprocess_optional_keys,
        settings=settings,
        state=state,
        workspace=workspace,
        year=inputs.year,
        profile=profile,
        surface=runtime_surface,
    )
    if activitysim_postprocess_binding.missing_required:
        get_value = getattr(coupler, "get", None)
        coupler_current = get_value(USIM_DATASTORE_CURRENT_H5) if callable(get_value) else None
        coupler_population = get_value(USIM_POPULATION_SOURCE_H5) if callable(get_value) else None
        raise RuntimeError(
            "ActivitySim postprocess could not resolve its required UrbanSim H5 roles: "
            f"{', '.join(activitysim_postprocess_binding.missing_required)}; "
            f"resolved_usim_inputs_keys={sorted(resolved_usim_inputs.keys())}; "
            f"resolved_current={resolved_usim_inputs.get(USIM_DATASTORE_CURRENT_H5)!r}; "
            f"resolved_population={resolved_usim_inputs.get(USIM_POPULATION_SOURCE_H5)!r}; "
            f"coupler_current={coupler_current!r}; "
            f"coupler_population={coupler_population!r}"
        )

    stage_runner.run_step(
        stage_name="activity_demand_postprocess",
        step=StepRef(
            name="activitysim_postprocess",
            step_func=make_activitysim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=activitysim_postprocess_binding,
            year=state.forecast_year,
        ),
    )

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.activity_demand,
    )

    postprocess_outputs = outputs_holder.activitysim_postprocess
    activity_demand_outputs = (
        step_output_handoff_mapping(postprocess_outputs, coupler=coupler)
        if postprocess_outputs is not None
        else None
    )
    if exact_rewind_restore is not None:
        try:
            delattr(workspace, "_activitysim_exact_rewind_restore")
        except Exception:
            logger.debug(
                "Failed clearing ActivitySim exact rewind metadata after use",
                exc_info=True,
            )

    return ActivityDemandPhaseOutputs(activity_demand_outputs=activity_demand_outputs)
