from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.io import locate_beam_file
from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    artifact_to_path,
    clean_expected_outputs,
    enqueue_archive_copy,
    flush_archive_queue,
    resolve_existing_path,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.workflows.input_resolution import (
    ResolvedStepInputs,
    resolved_value_for_key,
    resolve_preferred_step_input,
    resolve_step_inputs,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    run_workflow,
    run_manifested_steps,
)
from pilates.workflows.outputs_base import (
    step_output_handoff_mapping,
    step_output_mapping,
)
from pilates.workflows.step_io import build_outputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
    ASIM_OMX_SKIMS,
    ATLAS_VEHICLES2_INPUT,
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
    ZARR_SKIMS,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.activitysim.runner import persist_sharrow_cache_enabled
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


_TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS = (
    "beam_plans_asim_out",
    "households_asim_out",
    "persons_asim_out",
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


@dataclass
class TrafficAssignmentPhaseInputs:
    """
    Inputs for one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    activity_demand_outputs : Optional[dict[str, Any]]
        ActivitySim outputs used to seed BEAM inputs for this iteration.
    previous_beam_outputs : Optional[dict[str, Any]]
        Prior BEAM outputs (e.g., linkstats) used for warm-starting.
    """

    year: int
    iteration: int
    activity_demand_outputs: Optional[Dict[str, Any]]
    previous_beam_outputs: Optional[Dict[str, Any]]


@dataclass
class TrafficAssignmentPhaseOutputs:
    """
    Outputs from one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    previous_beam_outputs : Optional[dict[str, Any]]
        Combined BEAM run + postprocess outputs for warm-starting the
        next iteration, if available.
    """

    previous_beam_outputs: Optional[Dict[str, Any]]


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


def _run_supply_demand_workflow(
    *,
    stage_name: str,
    steps: list[StepRef],
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    runtime_kwargs_extra: Optional[dict[str, Any]] = None,
) -> None:
    """Run workflow steps with shared supply-demand stage context."""
    run_workflow(
        stage_name=stage_name,
        steps=steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
    )


def _find_initial_linkstats_warmstart(
    settings: PilatesConfig, workspace: Workspace
) -> Optional[str]:
    beam_settings = settings.beam
    if beam_settings is None:
        return None
    base_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        beam_settings.router_directory,
    )
    candidates = [
        os.path.join(base_dir, "init.linkstats.parquet"),
        os.path.join(base_dir, "init.linkstats.csv.gz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = getattr(settings, "beam", None)
    skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg else None
    if skim_cfg is None:
        return "disabled"
    return getattr(skim_cfg, "run_schedule", "standalone")


def _should_run_full_skim(settings: PilatesConfig, iteration: int) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        return True
    if schedule == "after_each_iteration":
        return True
    if schedule == "after_final_iteration":
        total_iters = settings.run.supply_demand_iters
        return iteration == total_iters - 1
    return False


def _is_iteration_scoped_artifact_key(
    key: str, *, prefix: str, year: int, iteration: int
) -> bool:
    base = f"{prefix}_{year}_{iteration}"
    return key == base or key.startswith(f"{base}_sub")


def _build_beam_postprocess_input_keys(
    *,
    upstream_keys: Iterable[str],
    year: int,
    iteration: int,
    include_zarr_skims: bool,
) -> Optional[list[str]]:
    """
    Select BEAM postprocess coupler inputs from BEAM run outputs.

    BEAM postprocess only consumes BEAM events parquet and OD skims artifacts
    from the run output store, plus upstream ActivitySim ``zarr_skims`` when
    available. Trimming input keys to this set keeps run identity aligned with
    actual behavior while avoiding unnecessary cache invalidation from unrelated
    BEAM outputs.
    """
    selected: list[str] = []
    keys = list(upstream_keys)

    for key in keys:
        if _is_iteration_scoped_artifact_key(
            key, prefix="events_parquet", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims_zarr", year=year, iteration=iteration
        ):
            selected.append(key)

    # Conservative fallback for naming drift: keep skim/event dependencies if
    # exact iteration-scoped keys are absent.
    if not any(key.startswith("raw_od_skims") for key in selected):
        selected.extend(key for key in keys if key.startswith("raw_od_skims"))
    if not any(key.startswith("events_parquet_") for key in selected):
        selected.extend(key for key in keys if key.startswith("events_parquet_"))

    if include_zarr_skims:
        selected.append(ZARR_SKIMS)

    deduped = list(dict.fromkeys(selected))
    return deduped or None


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

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    inputs : ActivityDemandPhaseInputs
        Inputs required for this iteration.
    outputs_holder : StepOutputsHolder
        Accumulator for step outputs within the iteration.
    manifest_config : ManifestConfig
        Manifest checkpointing configuration for ActivitySim steps.

    Returns
    -------
    ActivityDemandPhaseOutputs
        Mapping of ActivitySim outputs for downstream BEAM inputs.
    """
    formatted_print("ACTIVITY DEMAND MODEL")

    # ActivitySim runs in two manifest-checkpointed phases:
    # 1) Preprocess (per-iteration) to prepare compile inputs.
    # 2) Compile (per-year) outside manifest checkpointing.
    # 3) Run/Postprocess (per-iteration) for demand outputs.
    preprocess_resolution = resolve_step_inputs(
        keys=[USIM_DATASTORE_CURRENT_H5, FINAL_SKIMS_OMX],
        coupler=coupler,
        explicit_inputs=inputs.usim_inputs,
    )
    if preprocess_resolution.source_by_key.get(USIM_DATASTORE_CURRENT_H5) == "missing":
        usim_resolution = resolve_preferred_step_input(
            preferred_keys=[
                USIM_H5_UPDATED,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ],
            coupler=coupler,
            explicit_inputs=inputs.usim_inputs,
            required=False,
        )
        final_skims_resolution = resolve_step_inputs(
            keys=[FINAL_SKIMS_OMX],
            coupler=coupler,
        )
        preprocess_resolution = ResolvedStepInputs(
            inputs={
                **usim_resolution.inputs,
                **final_skims_resolution.inputs,
            },
            input_keys=usim_resolution.input_keys + final_skims_resolution.input_keys,
            source_by_key={
                **usim_resolution.source_by_key,
                **final_skims_resolution.source_by_key,
            },
            coupler_key_by_key={
                **usim_resolution.coupler_key_by_key,
                **final_skims_resolution.coupler_key_by_key,
            },
            missing_required=usim_resolution.missing_required,
        )

    preferred_sources = {"explicit", "coupler", "fallback"}
    if not any(
        source in preferred_sources
        for source in preprocess_resolution.source_by_key.values()
    ):
        fallback_inputs, _ = build_urbansim_inputs(
            settings, state, workspace, inputs.year
        )
        usim_resolution = resolve_preferred_step_input(
            preferred_keys=[
                USIM_H5_UPDATED,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ],
            coupler=coupler,
            explicit_inputs=inputs.usim_inputs,
            fallback_inputs=fallback_inputs,
            required=True,
        )
        final_skims_resolution = resolve_step_inputs(
            keys=[FINAL_SKIMS_OMX],
            coupler=coupler,
        )
        preprocess_resolution = ResolvedStepInputs(
            inputs={
                **usim_resolution.inputs,
                **final_skims_resolution.inputs,
            },
            input_keys=usim_resolution.input_keys + final_skims_resolution.input_keys,
            source_by_key={
                **usim_resolution.source_by_key,
                **final_skims_resolution.source_by_key,
            },
            coupler_key_by_key={
                **usim_resolution.coupler_key_by_key,
                **final_skims_resolution.coupler_key_by_key,
            },
            missing_required=usim_resolution.missing_required,
        )

    if preprocess_resolution.missing_required:
        raise RuntimeError(
            "ActivitySim preprocess requires a resolved UrbanSim datastore input "
            "(explicit, coupler, or fallback), but none were available."
        )

    if not preprocess_resolution.inputs and not preprocess_resolution.input_keys:
        # Keep existing behavior for provenance/cache identity compatibility:
        # require the canonical current datastore key when no concrete source
        # could be selected from the preferred chain.
        preprocess_resolution = ResolvedStepInputs(
            inputs={},
            input_keys=[USIM_DATASTORE_CURRENT_H5],
            source_by_key={USIM_DATASTORE_CURRENT_H5: "missing"},
            coupler_key_by_key={},
            missing_required=[USIM_DATASTORE_CURRENT_H5],
        )

    preprocess_specs = [
        StepRef(
            name="activitysim_preprocess",
            step_func=make_activitysim_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=preprocess_resolution.stepref_input_keys(),
            inputs=preprocess_resolution.stepref_inputs(),
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

    # ActivitySim Compilation: run once per year after preprocess.
    # Restart safety: if state says compiled but zarr skims are missing on local
    # ephemeral storage, force recompile instead of hard-failing.
    needs_compile = not state.asim_compiled
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
        compile_resolution = resolve_step_inputs(
            keys=[ASIM_OMX_SKIMS],
            coupler=coupler,
            explicit_inputs=compile_explicit_inputs or None,
            required_keys=[ASIM_OMX_SKIMS],
        )
        if compile_resolution.missing_required:
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
                    inputs=compile_resolution.stepref_inputs(),
                    input_keys=compile_resolution.stepref_input_keys(),
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

    activitysim_postprocess_inputs: Dict[str, str] = {}
    usim_base_fallback = None
    usim_input_fname = get_usim_datastore_fname(settings, io="input")
    usim_input_path = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_input_fname
    )
    if os.path.exists(usim_input_path):
        usim_base_fallback = usim_input_path
    usim_base_resolution = resolve_step_inputs(
        keys=[USIM_DATASTORE_BASE_H5],
        coupler=coupler,
        explicit_inputs=inputs.usim_inputs,
        fallback_inputs={USIM_DATASTORE_BASE_H5: usim_base_fallback}
        if usim_base_fallback is not None
        else None,
    )
    usim_base_input = resolved_value_for_key(
        resolved=usim_base_resolution,
        key=USIM_DATASTORE_BASE_H5,
        coupler=coupler,
    )
    usim_base_path = artifact_to_path(usim_base_input, workspace)
    if usim_base_path and os.path.exists(usim_base_path):
        activitysim_postprocess_inputs[USIM_DATASTORE_BASE_H5] = usim_base_path

    activitysim_run_specs = [
        StepRef(
            name="activitysim_run",
            step_func=make_activitysim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=asim_run_input_keys or None,
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
    postprocess_input_keys = [
        short_name for short_name, _, _ in upstream_run._iter_record_items()
    ]
    if not postprocess_input_keys:
        postprocess_input_keys = None

    activitysim_postprocess_specs = [
        StepRef(
            name="activitysim_postprocess",
            step_func=make_activitysim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=postprocess_input_keys,
            inputs=activitysim_postprocess_inputs or None,
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


def _find_input_scenario_dir(
    settings: PilatesConfig,
    workspace: Workspace,
    filename: str,
    filetype: str = "parquet",
) -> str:
    beam_settings = settings.beam
    if beam_settings is None:
        raise RuntimeError("BEAM config is required for traffic-assignment inputs.")
    scenario_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        beam_settings.scenario_folder,
    )
    return locate_beam_file(scenario_dir, filename, filetype)


def _collect_previous_beam_outputs(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    state: WorkflowState,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Resolve previous BEAM outputs for warm-starting.

    When explicit previous outputs are unavailable, this attempts to hydrate
    a minimal promoted store from coupler keys written by BEAM postprocess.
    """
    if previous_beam_outputs is not None:
        return previous_beam_outputs

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None

    promoted_outputs: Dict[str, Any] = {}
    for key in (LINKSTATS_WARMSTART, LINKSTATS, BEAM_PLANS_OUT):
        value = get_value(key)
        if value is None:
            continue
        path = artifact_to_path(value, workspace)
        if path and os.path.exists(path):
            promoted_outputs[key] = value
    return promoted_outputs or None


def _collect_beam_preprocess_inputs(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    state: WorkflowState,
    iteration: int,
    activity_demand_outputs: Optional[Dict[str, Any]],
    previous_beam_outputs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build preprocess inputs for BEAM from available upstream sources.

    Source precedence:
    1) ActivitySim outputs (when enabled and available)
    2) Default BEAM scenario files (when ActivitySim is disabled)
    3) Prior BEAM outputs and warm-start linkstats
    4) ATLAS vehicles2 (iteration 0 only, when vehicle ownership is enabled)
    """
    beam_preprocess_inputs: Dict[str, Any] = {}
    forecast_year = state.forecast_year
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before building BEAM inputs."
        )

    if activity_demand_outputs is not None:
        asim_input_keys = {
            "beam_plans_asim_out",
            "beam_plans_out",
            "households_asim_out",
            "linkstats",
            "persons_asim_out",
        }
        for key, value in activity_demand_outputs.items():
            if key in asim_input_keys:
                beam_preprocess_inputs[key] = value
    elif settings.run.models.activity_demand is None:
        logger.info("Falling back on default inputs to BEAM")
        # In BEAM-only mode the copied mutable scenario already contains the
        # canonical plans/households/persons files. Let beam_preprocess resolve
        # and publish those inputs from the staged scenario directory instead of
        # passing filesystem paths through Consist ahead of time.
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
                value = get_value(key)
                if value is not None:
                    beam_preprocess_inputs[key] = value
    elif previous_beam_outputs is None:
        raise RuntimeError(
            "TrafficAssignment iteration 0 requires activity_demand_outputs "
            "or previous_beam_outputs. Ensure ActivityDemand completed or "
            "provide warm-start outputs before running BEAM."
        )

    if previous_beam_outputs is not None:
        for key, value in previous_beam_outputs.items():
            if key.startswith("linkstats"):
                beam_preprocess_inputs[key] = value

    if previous_beam_outputs is None or not any(
        key.startswith("linkstats") for key in previous_beam_outputs.keys()
    ):
        get_value = getattr(coupler, "get", None)
        coupler_warmstart = None
        if callable(get_value):
            value = get_value(LINKSTATS_WARMSTART)
            coupler_warmstart = artifact_to_path(value, workspace) if value is not None else None
        if coupler_warmstart and os.path.exists(coupler_warmstart):
            beam_preprocess_inputs.setdefault(
                LINKSTATS_WARMSTART, coupler_warmstart
            )
        else:
            warmstart_path = _find_initial_linkstats_warmstart(settings, workspace)
            if warmstart_path:
                beam_preprocess_inputs.setdefault(LINKSTATS_WARMSTART, warmstart_path)

    if getattr(settings, "vehicle_ownership_model_enabled", False) and iteration == 0:
        if state.run_info_path and os.path.exists(state.run_info_path):
            previous_run_dir = os.path.dirname(state.run_info_path)
            atlas_output_dir = os.path.join(previous_run_dir, "atlas", "atlas_output")
        else:
            atlas_output_dir = workspace.get_atlas_output_dir()
        atlas_vehicle_path = os.path.join(
            atlas_output_dir,
            f"vehicles2_{forecast_year}.csv",
        )
        if not os.path.exists(atlas_vehicle_path):
            atlas_vehicle_path = os.path.join(
                atlas_output_dir,
                f"vehicles2_{forecast_year - 1}.csv",
            )
        if os.path.exists(atlas_vehicle_path):
            beam_preprocess_inputs.setdefault(ATLAS_VEHICLES2_INPUT, atlas_vehicle_path)

    return beam_preprocess_inputs


def _restore_activity_demand_outputs_for_resume(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    state: WorkflowState,
) -> Optional[Dict[str, Any]]:
    """
    Rehydrate ActivitySim outputs for BEAM when resuming after a skipped substage.

    On restart directly into ``traffic_assignment``, the current iteration's
    ``StepOutputsHolder`` starts empty even though restart recovery may already
    have restored the required ActivitySim artifacts into the coupler. Promote
    the narrow BEAM-facing subset back into a plain mapping so BEAM preprocess
    sees the same inputs it would have received after a live ActivitySim run.
    """

    def _require_complete_restore(
        restored_outputs: Dict[str, Any], source: str
    ) -> Optional[Dict[str, Any]]:
        if not restored_outputs:
            return None
        missing_keys = sorted(
            key
            for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS
            if key not in restored_outputs
            or not artifact_to_path(restored_outputs.get(key), workspace)
            or not os.path.exists(artifact_to_path(restored_outputs.get(key), workspace))
        )
        if missing_keys:
            raise RuntimeError(
                "Restart into traffic_assignment found incomplete ActivitySim "
                f"outputs from {source}; missing {missing_keys}"
            )
        return restored_outputs

    postprocess_outputs = outputs_holder.activitysim_postprocess
    if postprocess_outputs is not None:
        restored_outputs = step_output_handoff_mapping(
            postprocess_outputs,
            coupler=coupler,
        )
        return _require_complete_restore(restored_outputs, "step outputs")

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None

    restored_outputs: Dict[str, Any] = {}
    for key in (
        "beam_plans_asim_out",
        "beam_plans_out",
        "households_asim_out",
        "linkstats",
        "persons_asim_out",
    ):
        value = get_value(key)
        if value is None:
            continue
        path = artifact_to_path(value, workspace)
        if path and os.path.exists(path):
            restored_outputs[key] = value
    if restored_outputs:
        outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=None,
            processed_outputs={
                key: Path(path)
                for key, value in restored_outputs.items()
                for path in [artifact_to_path(value, workspace)]
                if path is not None
            },
        )
        return _require_complete_restore(restored_outputs, "coupler artifacts")

    iter_dir = Path(workspace.get_asim_output_dir()) / (
        f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    filesystem_candidates = {
        "beam_plans_asim_out": iter_dir / "beam_plans.parquet",
        "households_asim_out": iter_dir / "households.parquet",
        "persons_asim_out": iter_dir / "persons.parquet",
    }
    restored_outputs = {
        key: str(path)
        for key, path in filesystem_candidates.items()
        if path.exists()
    }
    restored_outputs = _require_complete_restore(
        restored_outputs,
        "filesystem iteration outputs",
    )
    if restored_outputs:
        outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=iter_dir,
            processed_outputs={
                key: Path(path) for key, path in restored_outputs.items()
            },
        )
    return restored_outputs


def _derive_beam_run_input_keys(
    *,
    beam_preprocess_inputs: Mapping[str, Any],
    activity_demand_outputs: Optional[Dict[str, Any]],
) -> list[str]:
    """
    Derive BEAM run input keys from preprocess outputs and warm-start signals.

    beam_preprocess always publishes the canonical plans/households/persons trio,
    regardless of whether they came from ActivitySim outputs or from existing
    default files in the copied BEAM scenario directory.
    """
    _ = activity_demand_outputs
    run_input_keys = [
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    ]

    # Only require LINKSTATS_WARMSTART at BEAM run time when that explicit key
    # is provided to preprocess. Other linkstats* artifacts may exist for
    # bookkeeping/history but do not guarantee a warm-start input artifact.
    if LINKSTATS_WARMSTART in beam_preprocess_inputs:
        run_input_keys.append(LINKSTATS_WARMSTART)
    else:
        logger.debug(
            "[BEAM] linkstats warmstart not available; omitting %s from inputs",
            LINKSTATS_WARMSTART,
        )

    return run_input_keys


def _finalize_beam_run_input_keys(
    *,
    beam_run_input_keys: Optional[list[str]],
    outputs_holder: StepOutputsHolder,
) -> list[str]:
    """
    Reconcile BEAM run inputs with the artifacts actually published by preprocess.

    The pre-run key derivation happens before BEAM preprocess executes, but
    preprocess may decide to publish ``linkstats_warmstart`` after resolving
    previous outputs. Use the realized preprocess outputs as the final contract.
    """
    finalized_keys = list(
        beam_run_input_keys
        or [
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ]
    )
    preprocess_outputs = outputs_holder.beam_preprocess
    prepared_inputs = (
        preprocess_outputs.prepared_inputs if preprocess_outputs is not None else {}
    )
    has_warmstart = LINKSTATS_WARMSTART in prepared_inputs
    if has_warmstart and LINKSTATS_WARMSTART not in finalized_keys:
        finalized_keys.append(LINKSTATS_WARMSTART)
    if not has_warmstart and LINKSTATS_WARMSTART in finalized_keys:
        finalized_keys = [
            key for key in finalized_keys if key != LINKSTATS_WARMSTART
        ]
    return finalized_keys


def _run_beam_preprocess_step(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_inputs: Mapping[str, Any],
    runtime_kwargs_extra: Mapping[str, Any],
) -> None:
    """
    Execute BEAM preprocess with explicit resolved inputs.
    """
    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_preprocess",
                step_func=make_beam_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=None,
                inputs=resolve_step_inputs(
                    keys=beam_preprocess_inputs.keys(),
                    explicit_inputs=beam_preprocess_inputs,
                ).stepref_inputs(),
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_inputs: Mapping[str, Any],
    beam_run_input_keys: Optional[list[str]],
    include_zarr_skims: bool,
    runtime_kwargs_extra: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Execute BEAM preprocess/run/postprocess and return combined outputs.
    """
    _run_beam_preprocess_step(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        beam_preprocess_inputs=beam_preprocess_inputs,
        runtime_kwargs_extra=runtime_kwargs_extra,
    )
    beam_run_input_keys = _finalize_beam_run_input_keys(
        beam_run_input_keys=beam_run_input_keys,
        outputs_holder=outputs_holder,
    )

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_run",
                step_func=make_beam_run_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=beam_run_input_keys,
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )

    upstream_run = outputs_holder.beam_run
    if upstream_run is None:
        raise RuntimeError("BEAM run must complete first")
    beam_postprocess_input_keys = _build_beam_postprocess_input_keys(
        upstream_keys=[
            short_name for short_name, _, _ in upstream_run._iter_record_items()
        ],
        year=year,
        iteration=iteration,
        include_zarr_skims=include_zarr_skims,
    )
    beam_postprocess_resolution = None
    if beam_postprocess_input_keys:
        beam_postprocess_resolution = resolve_step_inputs(
            keys=beam_postprocess_input_keys,
            coupler=coupler,
            explicit_inputs=step_output_handoff_mapping(upstream_run, coupler=coupler),
        )

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_postprocess",
                step_func=make_beam_postprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=(
                    beam_postprocess_resolution.stepref_input_keys()
                    if beam_postprocess_resolution is not None
                    else None
                ),
                inputs=(
                    beam_postprocess_resolution.stepref_inputs()
                    if beam_postprocess_resolution is not None
                    else None
                ),
                year=year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=dict(runtime_kwargs_extra),
    )

    if outputs_holder.beam_run is None and outputs_holder.beam_postprocess is None:
        return None

    combined_beam_outputs: Dict[str, Any] = {}
    if outputs_holder.beam_run is not None:
        combined_beam_outputs.update(
            step_output_handoff_mapping(outputs_holder.beam_run, coupler=coupler)
        )
    if outputs_holder.beam_postprocess is not None:
        combined_beam_outputs.update(
            step_output_handoff_mapping(
                outputs_holder.beam_postprocess,
                coupler=coupler,
            )
        )
    return combined_beam_outputs


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
    runtime_kwargs_extra: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Execute dedicated BEAM full-skim step and return its outputs.
    """
    runtime_kwargs = dict(runtime_kwargs_extra)
    runtime_kwargs["previous_beam_outputs"] = previous_beam_outputs

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=[
            StepRef(
                name="beam_full_skim",
                step_func=make_beam_full_skim_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                input_keys=None,
                year=state.forecast_year,
            )
        ],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs,
    )

    if outputs_holder.beam_full_skim is None:
        return None
    return step_output_handoff_mapping(outputs_holder.beam_full_skim, coupler=coupler)


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    inputs: TrafficAssignmentPhaseInputs,
    outputs_holder: StepOutputsHolder,
) -> TrafficAssignmentPhaseOutputs:
    """
    Run BEAM for a single supply-demand iteration.

    This prepares BEAM inputs from ActivitySim outputs, warm-starts
    linkstats when available, executes preprocess/run/postprocess,
    and updates the coupler with BEAM artifacts for subsequent steps.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    inputs : TrafficAssignmentPhaseInputs
        Inputs required for this iteration.
    outputs_holder : StepOutputsHolder
        Accumulator for step outputs within the iteration.

    Returns
    -------
    TrafficAssignmentPhaseOutputs
        Combined BEAM outputs for warm-starting the next iteration.
    """
    formatted_print("TRAFFIC ASSIGNMENT MODEL")

    previous_beam_outputs = _collect_previous_beam_outputs(
        coupler=coupler,
        workspace=workspace,
        state=state,
        iteration=inputs.iteration,
        previous_beam_outputs=inputs.previous_beam_outputs,
    )
    beam_preprocess_inputs = _collect_beam_preprocess_inputs(
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        state=state,
        iteration=inputs.iteration,
        activity_demand_outputs=inputs.activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
    )
    beam_run_input_keys = _derive_beam_run_input_keys(
        beam_preprocess_inputs=beam_preprocess_inputs,
        activity_demand_outputs=inputs.activity_demand_outputs,
    )

    traffic_runtime_kwargs = {
        "activity_demand_outputs": inputs.activity_demand_outputs,
        "previous_beam_outputs": previous_beam_outputs,
        "beam_preprocess_inputs": beam_preprocess_inputs,
    }
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        _run_beam_preprocess_step(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_inputs=beam_preprocess_inputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            previous_beam_outputs=previous_beam_outputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_inputs=beam_preprocess_inputs,
            beam_run_input_keys=beam_run_input_keys,
            include_zarr_skims=bool(inputs.activity_demand_outputs),
            runtime_kwargs_extra=traffic_runtime_kwargs,
        )
        if _should_run_full_skim(settings, inputs.iteration):
            full_skim_outputs = _run_beam_full_skim_step(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                outputs_holder=outputs_holder,
                year=inputs.year,
                iteration=inputs.iteration,
                previous_beam_outputs=combined_beam_outputs,
                runtime_kwargs_extra=traffic_runtime_kwargs,
            )
            if full_skim_outputs is not None:
                if combined_beam_outputs is None:
                    combined_beam_outputs = {}
                combined_beam_outputs.update(full_skim_outputs)

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return TrafficAssignmentPhaseOutputs(previous_beam_outputs=combined_beam_outputs)


def run_supply_demand_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    year: int,
    usim_inputs: Mapping[str, Union[str, os.PathLike]],
    build_manifest_path: Callable[[Workspace, int, int], os.PathLike],
    on_iteration_boundary: Optional[Callable[[int], None]] = None,
) -> None:
    """
    Run the supply-demand loop (ActivitySim + BEAM) for the year.

    This stage iterates the activity-demand and traffic-assignment sub-stages:
    ActivitySim preprocess -> compile (once per year) -> run/postprocess produces
    household/person/activity outputs, which feed BEAM. BEAM then runs traffic
    assignment and postprocessing, producing skims and other artifacts that can
    be fed into the next iteration. Manifest checkpointing is used around
    ActivitySim preprocess and run/postprocess to support restart/resume without
    re-running completed steps.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Input mapping (including any UrbanSim datastore paths) used to seed
        ActivitySim preprocessing when land use was not run.
    build_manifest_path : Callable[[Workspace, int, int], os.PathLike]
        Factory for per-year/per-iteration manifest file locations.
    on_iteration_boundary : Optional[Callable[[int], None]], optional
        Callback invoked after each outer iteration completes. Intended for
        orchestration-level safe-point actions such as DB snapshots.

    State Machine & Resume Behavior
    -------------------------------
    The stage maintains iteration state via ``state.iteration`` and records
    per-step completion in a manifest under ``.workflow/``. On resume:

    1. **ActivityDemand Phase**:
       - If ``should_run(...)`` returns True, preprocess → compile (once per year)
         → run/postprocess executes and updates coupler outputs.
       - If it returns False, ActivitySim steps are skipped and downstream
         phases must rely on previously-produced artifacts.

    2. **TrafficAssignment Phase**:
       - Requires ActivitySim outputs or prior BEAM outputs to seed inputs.
         If neither is available on iteration 0, the stage raises an error.
       - For later iterations, warm-start linkstats may be pulled from prior
         BEAM outputs or from an initial warm-start file.

    3. **Convergence Check**:
       - Checked at each iteration boundary. If convergence is detected, the
         loop exits early and the stage marks completion.

    Warnings
    --------
    - Removing ``.workflow/`` manifests forces all iterations to re-run.
    - If resuming mid-iteration, ensure coupler artifacts for ActivitySim
      outputs are available before running BEAM.
    - Do not mutate coupler keys between iterations; they carry warm-start
      state to the next iteration.
    """
    total_iters = settings.run.supply_demand_iters
    if settings.run.models.activity_demand is None and total_iters > 1:
        resumed_iteration = int(getattr(state, "iteration", 0) or 0)
        clamped_total_iters = max(1, resumed_iteration + 1)
        logger.warning(
            "BEAM-only supply_demand_iters=%d. Clamping outer supply-demand "
            "iterations to %d because BEAM already manages its own internal "
            "iterations.",
            total_iters,
            clamped_total_iters,
        )
        total_iters = clamped_total_iters
    previous_beam_outputs: Optional[Dict[str, Any]] = None

    for i in range(state.iteration, total_iters):
        state.iteration = i
        formatted_print(f"SUPPLY/DEMAND ITERATION {i + 1}/{total_iters}")
        activity_demand_outputs = None
        outputs_holder = StepOutputsHolder()
        manifest_path = build_manifest_path(workspace, year, i)
        manifest_config = ManifestConfig(path=Path(manifest_path))

        # C1. ACTIVITY DEMAND
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.activity_demand,
        ):
            activity_demand_inputs = ActivityDemandPhaseInputs(
                year=year,
                iteration=i,
                usim_inputs=usim_inputs,
            )
            activity_demand_outputs = _run_activity_demand_phase(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                inputs=activity_demand_inputs,
                outputs_holder=outputs_holder,
                manifest_config=manifest_config,
            ).activity_demand_outputs
        elif (
            settings.run.models.activity_demand is None
            and outputs_holder.activitysim_postprocess is None
        ):
            # Satisfy BEAM preprocess dependencies when ActivitySim is disabled.
            outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
                usim_datastore_h5=None,
                asim_output_dir=None,
            )
        elif outputs_holder.activitysim_postprocess is None:
            activity_demand_outputs = _restore_activity_demand_outputs_for_resume(
                coupler=coupler,
                workspace=workspace,
                outputs_holder=outputs_holder,
                state=state,
            )

        # C2. TRAFFIC ASSIGNMENT
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.traffic_assignment,
        ):
            traffic_inputs = TrafficAssignmentPhaseInputs(
                year=year,
                iteration=i,
                activity_demand_outputs=activity_demand_outputs,
                previous_beam_outputs=previous_beam_outputs,
            )
            previous_beam_outputs = _run_traffic_assignment_phase(
                scenario=scenario,
                state=state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                inputs=traffic_inputs,
                outputs_holder=outputs_holder,
            ).previous_beam_outputs

        if os.path.exists(manifest_path):
            enqueue_archive_copy(
                key="workflow_manifest",
                path=manifest_path,
            )
        # Year/iteration boundary durability checkpoint for restart artifacts.
        flush_archive_queue(timeout=300, fail_on_timeout=True)

        if on_iteration_boundary is not None:
            on_iteration_boundary(i)

    # The final substage completion may already have advanced the workflow
    # into the next year/major stage. Only complete the major stage here when
    # the state is still inside the supply-demand loop.
    if state.current_major_stage == state.Stage.supply_demand_loop:
        state.complete_step(state.Stage.supply_demand_loop)
