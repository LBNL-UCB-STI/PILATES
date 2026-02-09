from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.generic.records import FileRecord, RecordStore
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.io import locate_beam_file
from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    clean_expected_outputs,
    resolve_artifact_from_value,
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
from pilates.workflows.step_io import build_outputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    ATLAS_VEHICLES2_INPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
    ZARR_SKIMS,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.urbansim.inputs import build_urbansim_inputs
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
    activity_demand_outputs : Optional[RecordStore]
        RecordStore containing ActivitySim outputs needed downstream
        (e.g., households, persons, plans). None if not produced.
    """

    activity_demand_outputs: Optional[RecordStore]


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
    activity_demand_outputs : Optional[RecordStore]
        ActivitySim outputs used to seed BEAM inputs for this iteration.
    previous_beam_outputs : Optional[RecordStore]
        Prior BEAM outputs (e.g., linkstats) used for warm-starting.
    """

    year: int
    iteration: int
    activity_demand_outputs: Optional[RecordStore]
    previous_beam_outputs: Optional[RecordStore]


@dataclass
class TrafficAssignmentPhaseOutputs:
    """
    Outputs from one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    previous_beam_outputs : Optional[RecordStore]
        Combined BEAM run + postprocess outputs for warm-starting the
        next iteration, if available.
    """

    previous_beam_outputs: Optional[RecordStore]


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
    base_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        settings.beam.router_directory,
    )
    candidates = [
        os.path.join(base_dir, "init.linkstats.parquet"),
        os.path.join(base_dir, "init.linkstats.csv.gz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


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
        RecordStore of ActivitySim outputs for downstream BEAM inputs.
    """
    formatted_print("ACTIVITY DEMAND MODEL")

    # ActivitySim runs in two manifest-checkpointed phases:
    # 1) Preprocess (per-iteration) to prepare compile inputs.
    # 2) Compile (per-year) outside manifest checkpointing.
    # 3) Run/Postprocess (per-iteration) for demand outputs.
    preprocess_resolution = resolve_step_inputs(
        keys=[USIM_DATASTORE_CURRENT_H5],
        explicit_inputs=inputs.usim_inputs,
    )
    if preprocess_resolution.source_by_key.get(USIM_DATASTORE_CURRENT_H5) == "missing":
        preprocess_resolution = resolve_preferred_step_input(
            preferred_keys=[
                USIM_H5_UPDATED,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ],
            coupler=coupler,
            explicit_inputs=inputs.usim_inputs,
            required=False,
        )

    preferred_sources = {"explicit", "coupler", "fallback"}
    if not any(
        source in preferred_sources
        for source in preprocess_resolution.source_by_key.values()
    ):
        fallback_inputs, _ = build_urbansim_inputs(
            settings, state, workspace, inputs.year
        )
        preprocess_resolution = resolve_preferred_step_input(
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

    if preprocess_resolution.missing_required:
        raise RuntimeError(
            "ActivitySim preprocess requires a resolved UrbanSim datastore input "
            "(explicit, coupler, or fallback), but none were available."
        )

    if (
        not preprocess_resolution.inputs
        and not preprocess_resolution.input_keys
    ):
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

    # ActivitySim Compilation: run once per year after preprocess.
    if not state.asim_compiled:
        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim compile requires preprocess outputs.")
        compile_store_inputs = upstream.to_record_store().to_mapping()
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
                    year=inputs.year,
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
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value(ZARR_SKIMS),
                key=ZARR_SKIMS,
                workspace=workspace,
            )
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(),
                "cache",
                "skims.zarr",
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if not zarr_path or not os.path.exists(zarr_path):
            raise RuntimeError(
                "ActivitySim run requires zarr_skims, "
                "but none were found while asim_compiled=True."
            )

    upstream_preprocess = outputs_holder.activitysim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    asim_run_input_keys = [
        short_name for short_name, _, _ in upstream_preprocess._iter_record_items()
    ]
    asim_run_input_keys = [
        key for key in asim_run_input_keys if key != ASIM_OMX_SKIMS
    ]
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
        postprocess_outputs.to_record_store()
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
    scenario_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        settings.beam.scenario_folder,
    )
    return locate_beam_file(scenario_dir, filename, filetype)


def _collect_previous_beam_outputs(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    state: WorkflowState,
    iteration: int,
    previous_beam_outputs: Optional[RecordStore],
) -> Optional[RecordStore]:
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

    promoted_store = RecordStore()
    for key in (LINKSTATS, BEAM_PLANS_OUT):
        value = get_value(key)
        if value is None:
            continue
        path = artifact_to_path(value, workspace)
        if path and os.path.exists(path):
            promoted_store.add_record(
                FileRecord(
                    file_path=path,
                    short_name=key,
                    description=f"Promoted BEAM output: {key}",
                    year=state.forecast_year,
                    iteration=iteration,
                )
            )
    return promoted_store if promoted_store.all_records() else None


def _collect_beam_preprocess_inputs(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    state: WorkflowState,
    iteration: int,
    activity_demand_outputs: Optional[RecordStore],
    previous_beam_outputs: Optional[RecordStore],
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

    if activity_demand_outputs is not None:
        asim_input_keys = {
            "beam_plans_asim_out",
            "beam_plans_out",
            "households_asim_out",
            "linkstats",
            "persons_asim_out",
        }
        for key, value in activity_demand_outputs.to_mapping().items():
            if key in asim_input_keys:
                beam_preprocess_inputs[key] = value
    elif settings.run.models.activity_demand is None:
        logger.info("Falling back on default inputs to BEAM")
        default_inputs = {
            BEAM_PLANS_IN: "plans",
            BEAM_HOUSEHOLDS_IN: "households",
            BEAM_PERSONS_IN: "persons",
        }
        for key, filename in default_inputs.items():
            beam_preprocess_inputs[key] = _find_input_scenario_dir(
                settings,
                workspace,
                filename,
            )
    elif previous_beam_outputs is None:
        raise RuntimeError(
            "TrafficAssignment iteration 0 requires activity_demand_outputs "
            "or previous_beam_outputs. Ensure ActivityDemand completed or "
            "provide warm-start outputs before running BEAM."
        )

    if previous_beam_outputs is not None:
        for key, value in previous_beam_outputs.to_mapping().items():
            if key.startswith("linkstats"):
                beam_preprocess_inputs[key] = value

    if previous_beam_outputs is None or not any(
        key.startswith("linkstats")
        for key in previous_beam_outputs.to_mapping().keys()
    ):
        warmstart_path = _find_initial_linkstats_warmstart(settings, workspace)
        if warmstart_path:
            beam_preprocess_inputs.setdefault(LINKSTATS_WARMSTART, warmstart_path)

    if (
        getattr(settings, "vehicle_ownership_model_enabled", False)
        and iteration == 0
    ):
        if state.run_info_path and os.path.exists(state.run_info_path):
            previous_run_dir = os.path.dirname(state.run_info_path)
            atlas_output_dir = os.path.join(previous_run_dir, "atlas", "atlas_output")
        else:
            atlas_output_dir = workspace.get_atlas_output_dir()
        atlas_vehicle_path = os.path.join(
            atlas_output_dir,
            f"vehicles2_{state.forecast_year}.csv",
        )
        if not os.path.exists(atlas_vehicle_path):
            atlas_vehicle_path = os.path.join(
                atlas_output_dir,
                f"vehicles2_{state.forecast_year - 1}.csv",
            )
        if os.path.exists(atlas_vehicle_path):
            beam_preprocess_inputs.setdefault(ATLAS_VEHICLES2_INPUT, atlas_vehicle_path)

    return beam_preprocess_inputs


def _derive_beam_run_input_keys(
    *,
    beam_preprocess_inputs: Mapping[str, Any],
    activity_demand_outputs: Optional[RecordStore],
) -> list[str]:
    """
    Derive BEAM run input keys from preprocess outputs and warm-start signals.

    The BEAM runner always consumes plans/households/persons from the mutable
    scenario directory, regardless of whether ActivitySim produced them in this
    workflow. Keep these as explicit dependencies so run identity/provenance
    captures both ActivitySim-driven and default-scenario BEAM runs.
    """
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

    if activity_demand_outputs is None:
        logger.debug(
            "[BEAM] ActivitySim disabled/unavailable; using default scenario inputs "
            "for %s, %s, %s",
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        )
    return run_input_keys


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
) -> Optional[RecordStore]:
    """
    Execute BEAM preprocess/run/postprocess and return combined outputs.
    """
    beam_pre_run_steps = [
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
        ),
        StepRef(
            name="beam_run",
            step_func=make_beam_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=beam_run_input_keys,
        ),
    ]

    _run_supply_demand_workflow(
        stage_name="beam",
        steps=beam_pre_run_steps,
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
        year=state.forecast_year,
        iteration=iteration,
        include_zarr_skims=include_zarr_skims,
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
                input_keys=beam_postprocess_input_keys,
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

    combined_beam_outputs = RecordStore()
    if outputs_holder.beam_run is not None:
        combined_beam_outputs += outputs_holder.beam_run.to_record_store()
    if outputs_holder.beam_postprocess is not None:
        combined_beam_outputs += outputs_holder.beam_postprocess.to_record_store()
    return combined_beam_outputs


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
    previous_beam_outputs: Optional[RecordStore] = None

    for i in range(state.iteration, total_iters):
        state.iteration = i
        formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
        activity_demand_outputs = None
        outputs_holder = StepOutputsHolder()
        manifest_path = build_manifest_path(workspace, year, i)
        manifest_config = ManifestConfig(path=manifest_path)

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
                asim_output_dir=Path(workspace.get_asim_output_dir()),
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

    state.complete_step(state.Stage.supply_demand_loop)
