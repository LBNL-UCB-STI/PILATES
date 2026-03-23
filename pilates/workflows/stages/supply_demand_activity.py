from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    clean_expected_outputs,
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
    StepRef,
    run_workflow,
    run_manifested_steps,
)
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.step_io import build_outputs
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
    ZARR_SKIMS,
)
from pilates.activitysim.runner import persist_sharrow_cache_enabled
from pilates.activitysim.outputs import normalize_asim_output_key
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


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


def _run_supply_demand_manifested_steps(
    *,
    stage_name: str,
    steps: list[StepRef],
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    year: int,
    iteration: int,
) -> None:
    """Run manifest-backed steps with shared supply-demand stage context."""
    run_manifested_steps(
        stage_name=stage_name,
        steps=steps,
        outputs_holder=outputs_holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
    )


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
) -> ActivityDemandPhaseOutputs:
    """
    Run ActivitySim for a single supply-demand iteration.

    This executes the ActivitySim preprocess, compile (once per year),
    and run/postprocess steps. It also assembles the required inputs
    from UrbanSim outputs or fallbacks and ensures skims are available
    when resuming after compilation.
    """
    formatted_print("ACTIVITY DEMAND MODEL")

    # ActivitySim runs in two manifest-checkpointed phases:
    # 1) Preprocess (per-iteration) to prepare compile inputs.
    # 2) Compile (per-year) outside manifest checkpointing.
    # 3) Run/Postprocess (per-iteration) for demand outputs.
    preprocess_binding = build_binding_plan(
        step_name="activitysim_preprocess",
        coupler=coupler,
        explicit_inputs=inputs.usim_inputs,
        settings=settings,
        state=state,
        workspace=workspace,
        year=inputs.year,
    )

    if preprocess_binding.missing_required:
        raise RuntimeError(
            "ActivitySim preprocess requires a resolved UrbanSim datastore input "
            "(explicit, coupler, or fallback), but none were available."
        )

    preprocess_specs = [
        StepRef(
            name="activitysim_preprocess",
            step_func=make_activitysim_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=preprocess_binding,
            year=state.forecast_year,
        )
    ]
    _run_supply_demand_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=preprocess_specs,
        outputs_holder=outputs_holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=inputs.year,
        iteration=inputs.iteration,
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
        candidate = os.path.join(
            workspace.get_asim_output_dir(),
            "cache",
            "skims.zarr",
        )
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
        )
        if compile_binding.missing_required:
            raise RuntimeError(
                "ActivitySim compile requires omx_skims input, but it could not be "
                "resolved from explicit preprocess outputs or coupler keys."
            )
        expected_compile_outputs = clean_expected_outputs(
            build_outputs(
                "activitysim_compile",
                settings,
                state,
                workspace,
                components=("runner",),
            )
        )
        activitysim_compile_step = make_activitysim_compile_step(
            coupler=coupler,
            outputs_holder=outputs_holder,
        )
        run_workflow(
            stage_name="activity_demand_compile",
            steps=[
            StepRef(
                name="activitysim_compile",
                step_func=activitysim_compile_step,
                binding=compile_binding,
                output_paths=expected_compile_outputs or None,
                cache_mode="overwrite",
                load_inputs=False,
                    phase="compile",
                    model="activitysim_compile",
                    year=state.forecast_year,
                    iteration=-1,
                )
            ],
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix=str(inputs.year),
            iteration=-1,
            runtime_kwargs_extra={
                "expected_outputs": expected_compile_outputs,
            },
        )
        setattr(state, "_restart_activitysim_compile_done", True)
    else:
        _republish_existing_compile_artifacts()
    final_zarr_path = _resolved_existing_zarr_skims_path()
    final_cache_path = (
        _resolved_existing_numba_cache_path() if requires_numba_cache else "n/a"
    )
    if not final_zarr_path or (requires_numba_cache and not final_cache_path):
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
    asim_run_input_keys.append(ZARR_SKIMS)
    if requires_numba_cache:
        asim_run_input_keys.append(ASIM_SHARROW_CACHE_DIR)

    optional_run_keys = [ASIM_SHARROW_CACHE_DIR] if requires_numba_cache else []

    activitysim_run_specs = [
        StepRef(
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
            ),
            year=state.forecast_year,
        ),
    ]
    _run_supply_demand_manifested_steps(
        stage_name="activity_demand_run",
        steps=activitysim_run_specs,
        outputs_holder=outputs_holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=inputs.year,
        iteration=inputs.iteration,
    )

    upstream_run = outputs_holder.activitysim_run
    if upstream_run is None:
        raise RuntimeError("ActivitySim run must complete first")
    postprocess_input_keys = list(
        dict.fromkeys(
            normalize_asim_output_key(
                short_name[: -len("_temp")]
                if short_name.endswith("_asim_out_temp")
                else short_name
            )
            for short_name, _, _ in upstream_run._iter_record_items()
        )
    )
    if not postprocess_input_keys:
        postprocess_input_keys = None
    activitysim_postprocess_binding = build_binding_plan(
        step_name="activitysim_postprocess",
        coupler=coupler,
        explicit_inputs=inputs.usim_inputs,
        required_keys=postprocess_input_keys or [],
        optional_keys=[USIM_DATASTORE_BASE_H5],
        settings=settings,
        state=state,
        workspace=workspace,
        year=inputs.year,
    )

    activitysim_postprocess_specs = [
        StepRef(
            name="activitysim_postprocess",
            step_func=make_activitysim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=activitysim_postprocess_binding,
            year=state.forecast_year,
        )
    ]
    _run_supply_demand_manifested_steps(
        stage_name="activity_demand_postprocess",
        steps=activitysim_postprocess_specs,
        outputs_holder=outputs_holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=inputs.year,
        iteration=inputs.iteration,
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

    return ActivityDemandPhaseOutputs(activity_demand_outputs=activity_demand_outputs)
