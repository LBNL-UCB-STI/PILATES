from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.runtime.restart import restart_target_for_step
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import (
    _emit_artifact_lifecycle_event,
    artifact_to_existing_path,
    set_coupler_from_artifact,
)
from pilates.utils.formatting import formatted_print
from pilates.beam.outputs import BeamRunOutputs
from pilates.workflows.tracker_outputs import load_tracker_run_outputs
from pilates.workflows.binding import (
    BindingPlan,
    beam_preprocess_binding_plan,
    build_binding_plan,
    build_key_only_binding_plan,
)
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.orchestration import StageRunner
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    ATLAS_VEHICLES2_OUTPUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface


_FAIL_AFTER_BEAM_RUN_ENV = "PILATES_FAIL_AFTER_BEAM_RUN"
_BEAM_VEHICLES_IN = "vehicles_beam_in"


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


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = getattr(settings, "beam", None)
    skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg else None
    if skim_cfg is None:
        return "disabled"
    return getattr(skim_cfg, "run_schedule", "standalone")


def _stringify_mapping_values(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in dict(mapping or {}).items()}


def _mapping_from_runtime_attr(owner: Any, name: str) -> Optional[Mapping[str, Any]]:
    value = getattr(owner, name, None)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    if isinstance(value, Mapping):
        return value
    return None


def _beam_restart_identity_context(
    *,
    scenario: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Read optional original-vs-resumed cache identity diagnostics from runtime state.

    Consist already exposes cache-miss and identity summaries on run metadata.
    When a restart harness or scenario wrapper has captured those summaries,
    include them here instead of inventing a new Consist API.
    """
    context: Dict[str, Any] = {}
    for owner in (scenario, state):
        if owner is None:
            continue
        for attr_name in (
            "beam_restart_binding_context",
            "beam_restart_identity_context",
        ):
            attr_context = _mapping_from_runtime_attr(owner, attr_name)
            if attr_context:
                context.update(attr_context)
        for attr_name, context_key in (
            ("cache_miss_explanation", "cache_miss_explanation"),
            ("latest_cache_miss_explanation", "cache_miss_explanation"),
            ("beam_cache_miss_explanation", "cache_miss_explanation"),
            ("identity_summary", "identity_summary"),
            ("latest_identity_summary", "identity_summary"),
            ("beam_identity_summary", "identity_summary"),
        ):
            attr_context = _mapping_from_runtime_attr(owner, attr_name)
            if attr_context and context_key not in context:
                context[context_key] = attr_context
    return context


def beam_preprocess_binding_diagnostic_payload(
    *,
    binding: BindingPlan,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    scenario: Optional[Any] = None,
    identity_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize the resumed BEAM binding surface before cache hashes are trusted."""
    beam_region_input_dir = None
    beam_primary_config_file = None
    try:
        beam_region_input_dir = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            settings.run.region,
        )
        beam_config = getattr(getattr(settings, "beam", None), "config", None)
        if beam_config:
            beam_primary_config_file = os.path.join(
                beam_region_input_dir,
                beam_config,
            )
    except Exception:
        pass

    required_local_inputs = {
        "beam_region_input_dir": beam_region_input_dir,
        "beam_primary_config_file": beam_primary_config_file,
    }
    missing_local_inputs = sorted(
        key
        for key, path in required_local_inputs.items()
        if path and not os.path.exists(path)
    )
    missing_required = (
        binding.missing_required if binding.missing_required is not None else []
    )
    missing_restart_inputs = sorted(set(missing_required) | set(missing_local_inputs))
    resolved_identity_context = dict(
        identity_context
        if identity_context is not None
        else _beam_restart_identity_context(scenario=scenario, state=state)
    )
    cache_miss_explanation = resolved_identity_context.get("cache_miss_explanation")
    identity_summary = resolved_identity_context.get("identity_summary")
    if not isinstance(cache_miss_explanation, Mapping):
        cache_miss_explanation = None
    if not isinstance(identity_summary, Mapping):
        identity_summary = None
    drift_components: Dict[str, Any] = {}
    if cache_miss_explanation:
        for key in (
            "mismatched_components",
            "config_keys_changed",
            "adapter_identity_changed",
            "identity_inputs_changed",
            "input_keys_changed",
            "missing_input_keys",
        ):
            value = cache_miss_explanation.get(key)
            if value:
                drift_components[key] = value
    if missing_restart_inputs:
        drift_classification = "missing_restart_inputs"
    elif drift_components:
        drift_classification = "content_or_config_drift"
    elif cache_miss_explanation:
        drift_classification = "cache_miss_without_binding_gap"
    else:
        drift_classification = "binding_surface_complete"
    return {
        "key": "beam_restart_binding",
        "artifact_family": "beam_restart_diagnostic",
        "diagnostic": "beam_restart_binding",
        "restart_run": bool(getattr(state, "is_restart_run", False)),
        "workflow_year": getattr(state, "year", getattr(state, "current_year", None)),
        "forecast_year": getattr(state, "forecast_year", None),
        "iteration": getattr(
            state,
            "iteration",
            getattr(state, "current_inner_iter", None),
        ),
        "input_keys": sorted(
            binding.input_keys if binding.input_keys is not None else []
        ),
        "optional_input_keys": sorted(
            binding.optional_input_keys
            if binding.optional_input_keys is not None
            else []
        ),
        "bound_input_keys": sorted(
            (binding.inputs if binding.inputs is not None else {}).keys()
        ),
        "missing_required": sorted(missing_required),
        "missing_restart_inputs": missing_restart_inputs,
        "source_by_key": dict(
            sorted(
                (
                    binding.source_by_key if binding.source_by_key is not None else {}
                ).items()
            )
        ),
        "coupler_key_by_key": dict(
            sorted(
                (
                    binding.coupler_key_by_key
                    if binding.coupler_key_by_key is not None
                    else {}
                ).items()
            )
        ),
        "required_local_inputs": _stringify_mapping_values(required_local_inputs),
        "missing_local_inputs": missing_local_inputs,
        "identity_summary": dict(identity_summary or {}),
        "cache_miss_explanation": dict(cache_miss_explanation or {}),
        "cache_miss_reason": (
            cache_miss_explanation.get("reason") if cache_miss_explanation else None
        ),
        "identity_drift_components": drift_components,
        "drift_classification": drift_classification,
    }


def _emit_beam_preprocess_binding_diagnostic(
    *,
    binding: BindingPlan,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    scenario: Optional[Any] = None,
) -> None:
    if not bool(getattr(state, "is_restart_run", False)):
        return
    payload = beam_preprocess_binding_diagnostic_payload(
        binding=binding,
        state=state,
        settings=settings,
        workspace=workspace,
        scenario=scenario,
    )
    logger.info(
        "[BEAM][restart] preprocess binding diagnostic: classification=%s missing_restart_inputs=%s bound_input_keys=%s",
        payload["drift_classification"],
        payload["missing_restart_inputs"],
        payload["bound_input_keys"],
    )
    _emit_artifact_lifecycle_event("beam_restart_binding", **payload)


def _raise_if_restart_beam_config_missing(
    *,
    binding: BindingPlan,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
) -> None:
    if not bool(getattr(state, "is_restart_run", False)):
        return

    try:
        from pilates.beam.config_hocon import beam_primary_config_path

        expected_path = Path(beam_primary_config_path(settings, workspace=workspace))
        expected = str(expected_path)
    except Exception:
        expected_path = None
        expected = "<unresolved>"
    if expected_path is not None and expected_path.exists():
        return

    binding_inputs = binding.inputs if binding.inputs is not None else {}
    config_value = binding_inputs.get(BEAM_CONFIG_FILE)
    config_hint = f" Resolved binding value: {config_value}." if config_value else ""
    raise RuntimeError(
        "BEAM restart cannot continue because beam_config_file is missing. "
        f"Expected primary config at {expected}. This must be restored or "
        f"bootstrapped before BEAM can run.{config_hint}"
    )


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
    _ = state, iteration
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
        if artifact_to_existing_path(value, workspace):
            promoted_outputs[key] = value
    return promoted_outputs or None


def _archive_run_dir_for_restart(state: WorkflowState) -> Optional[Path]:
    run_info_path = getattr(state, "run_info_path", None)
    if not run_info_path:
        return None
    try:
        return Path(run_info_path).expanduser().resolve().parent
    except Exception:
        return None


def _find_completed_beam_run_for_restart(
    *,
    tracker: Any,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    year: int,
    iteration: int,
) -> Optional[Any]:
    find_matching_run = getattr(tracker, "find_matching_run", None)
    if not callable(find_matching_run):
        raise AttributeError("tracker does not support find_matching_run")

    target = restart_target_for_step(
        settings=settings,
        step_name="beam_run",
        year=year,
        iteration=iteration,
        state=state,
        workspace=workspace,
    )
    return find_matching_run(**target)


def _output_path_from_hydration_item(item: Any) -> Optional[Path]:
    path = getattr(item, "path", None)
    if path is not None and bool(getattr(item, "resolvable", True)):
        return Path(path)
    artifact = getattr(item, "artifact", None)
    as_path = getattr(artifact, "as_path", None)
    if callable(as_path):
        try:
            resolved = as_path()
        except Exception:
            resolved = None
        if resolved is not None:
            return Path(resolved)
    return None


def _hydrate_completed_beam_run_outputs(
    *,
    tracker: Any,
    run_id: str,
    workspace: Workspace,
    state: WorkflowState,
    year: int,
    iteration: int,
) -> Optional[BeamRunOutputs]:
    tracker_outputs = load_tracker_run_outputs(
        run_id,
        tracker=tracker,
        logger=logger,
        log_context="completed BEAM run restart recovery",
    )
    keys = list(tracker_outputs.keys())
    if not keys:
        logger.info(
            "[BEAM][restart] completed beam_run run_id=%s has no linked outputs to hydrate",
            run_id,
        )
        return None

    critical_keys = _build_beam_postprocess_input_keys(
        upstream_keys=keys,
        year=year,
        iteration=iteration,
        include_zarr_skims=False,
    )
    if not critical_keys:
        logger.info(
            "[BEAM][restart] completed beam_run run_id=%s has no postprocess-critical outputs to hydrate",
            run_id,
        )
        return None
    publication_keys = [
        key for key in (LINKSTATS, BEAM_PLANS_OUT) if key in tracker_outputs
    ]
    required_keys = list(dict.fromkeys([*critical_keys, *publication_keys]))

    target_root = os.path.realpath(str(workspace.full_path))
    archive_run_dir = _archive_run_dir_for_restart(state)
    source_root = os.path.realpath(str(archive_run_dir)) if archive_run_dir else None
    raw_outputs: Dict[str, Path] = {}
    hydration_complete: Optional[bool] = None
    hydration_summary: Optional[str] = None

    hydrate_run_outputs = getattr(tracker, "hydrate_run_outputs", None)
    if callable(hydrate_run_outputs):
        try:
            hydrated = hydrate_run_outputs(
                run_id=run_id,
                target_root=target_root,
                source_root=source_root,
                preserve_existing=True,
                keys=keys,
            )
        except Exception:
            logger.debug(
                "[BEAM][restart] hydrate_run_outputs failed for run_id=%s",
                run_id,
                exc_info=True,
            )
            hydrated = None
        if hydrated is not None:
            hydration_complete = bool(getattr(hydrated, "complete", True))
            hydration_summary = str(getattr(hydrated, "summary", ""))
            for key, item in hydrated.items():
                path = _output_path_from_hydration_item(item)
                if path is not None and path.exists():
                    raw_outputs[str(key)] = path

    for key, value in tracker_outputs.items():
        if str(key) in raw_outputs:
            continue
        path = artifact_to_existing_path(
            value,
            workspace=workspace,
            materialize_from_archive=True,
        )
        if path is not None:
            raw_outputs[str(key)] = Path(path)

    if not raw_outputs:
        materialize_run_outputs = getattr(tracker, "materialize_run_outputs", None)
        if callable(materialize_run_outputs):
            try:
                materialize_run_outputs(
                    run_id=run_id,
                    target_root=target_root,
                    source_root=source_root,
                    preserve_existing=True,
                    keys=keys,
                )
            except Exception:
                logger.debug(
                    "[BEAM][restart] materialize_run_outputs failed for run_id=%s",
                    run_id,
                    exc_info=True,
                )
        for key, value in tracker_outputs.items():
            if str(key) in raw_outputs:
                continue
            path = artifact_to_existing_path(
                value,
                workspace=workspace,
                materialize_from_archive=True,
            )
            if path is not None:
                raw_outputs[str(key)] = Path(path)

    if not raw_outputs:
        logger.info(
            "[BEAM][restart] completed beam_run run_id=%s could not hydrate usable output paths",
            run_id,
        )
        return None

    missing_required = [key for key in required_keys if key not in raw_outputs]
    if missing_required:
        logger.warning(
            "[BEAM][restart] completed beam_run run_id=%s hydration missing required keys=%s summary=%s",
            run_id,
            missing_required,
            hydration_summary,
        )
        return None
    if hydration_complete is False:
        logger.warning(
            "[BEAM][restart] completed beam_run run_id=%s hydration was partial but all postprocess-critical keys resolved summary=%s",
            run_id,
            hydration_summary,
        )

    return BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs=raw_outputs,
    )


def _publish_recovered_beam_run_outputs(
    *,
    outputs: BeamRunOutputs,
    coupler: CouplerProtocol,
) -> None:
    for key, path, _description in outputs._iter_record_items():
        set_coupler_from_artifact(coupler, key, None, fallback=str(path))
        if key == LINKSTATS:
            set_coupler_from_artifact(
                coupler,
                LINKSTATS_WARMSTART,
                None,
                fallback=str(path),
            )


def _restored_beam_parent_years(
    *,
    state: WorkflowState,
    run_year: int,
) -> list[int]:
    years: list[int] = []
    for value in (run_year, getattr(state, "forecast_year", None)):
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if year not in years:
            years.append(year)
    return years


def _try_restore_completed_beam_run_for_restart(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
) -> Optional[BeamRunOutputs]:
    if not bool(getattr(state, "is_restart_run", False)):
        return None
    if outputs_holder.beam_run is not None:
        return outputs_holder.beam_run

    tracker = cr.current_tracker()
    if tracker is None:
        return None
    run = _find_completed_beam_run_for_restart(
        tracker=tracker,
        settings=settings,
        state=state,
        workspace=workspace,
        year=year,
        iteration=iteration,
    )
    if run is None:
        return None

    run_id = str(getattr(run, "id", "") or "")
    if not run_id:
        return None
    outputs = _hydrate_completed_beam_run_outputs(
        tracker=tracker,
        run_id=run_id,
        workspace=workspace,
        state=state,
        year=year,
        iteration=iteration,
    )
    if outputs is None:
        raise RuntimeError(
            "BEAM restart found a completed same-run beam_run but could not "
            "hydrate the required outputs for postprocess. Refusing to rerun "
            f"BEAM from restart because the completed run is authoritative. "
            f"run_id={run_id}"
        )

    outputs_holder.beam_run = outputs
    _publish_recovered_beam_run_outputs(outputs=outputs, coupler=coupler)
    remember_restored_run_id = getattr(scenario, "remember_restored_run_id", None)
    if callable(remember_restored_run_id):
        for parent_year in _restored_beam_parent_years(state=state, run_year=year):
            remember_restored_run_id(
                model_name="beam_run",
                year=parent_year,
                iteration=iteration,
                run_id=run_id,
            )
    restored_keys = sorted(outputs.raw_outputs.keys())
    logger.info(
        "[BEAM][restart] restored completed beam_run from Consist run_id=%s hydrated_keys=%s",
        run_id,
        restored_keys,
    )
    _emit_artifact_lifecycle_event(
        "beam_restart_binding",
        key="beam_restart_binding",
        artifact_family="beam_restart_diagnostic",
        diagnostic="beam_restart_binding",
        restart_run=True,
        workflow_year=getattr(state, "year", getattr(state, "current_year", None)),
        forecast_year=getattr(state, "forecast_year", None),
        iteration=iteration,
        recovery_mode="consist_completed_run_hydration",
        recovered_run_id=run_id,
        hydrated_output_keys=restored_keys,
        drift_classification="completed_beam_run_recovered",
    )
    return outputs


def _beam_run_output_keys(outputs: BeamRunOutputs) -> list[str]:
    raw_outputs = getattr(outputs, "raw_outputs", None)
    if isinstance(raw_outputs, Mapping):
        return sorted(str(key) for key in raw_outputs.keys())
    iter_record_items = getattr(outputs, "_iter_record_items", None)
    if callable(iter_record_items):
        return sorted(
            str(short_name) for short_name, _path, _desc in iter_record_items()
        )
    return []


def _emit_beam_restart_recovery_readiness_diagnostic(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    outputs: BeamRunOutputs,
    year: int,
    iteration: int,
    include_zarr_skims: bool,
) -> None:
    """Log whether a completed BEAM run is ready for restart-side recovery."""
    output_keys = _beam_run_output_keys(outputs)
    postprocess_keys = (
        _build_beam_postprocess_input_keys(
            upstream_keys=output_keys,
            year=year,
            iteration=iteration,
            include_zarr_skims=include_zarr_skims,
        )
        or []
    )
    publication_keys = [
        key for key in (LINKSTATS, BEAM_PLANS_OUT) if key in output_keys
    ]
    required_keys = list(dict.fromkeys([*postprocess_keys, *publication_keys]))

    matched_run_id: Optional[str] = None
    matched_output_keys: list[str] = []
    error: Optional[str] = None
    target: Dict[str, Any] = {}
    hydration_api_available = False

    try:
        tracker = cr.current_tracker()
        hydrate_run_outputs = getattr(tracker, "hydrate_run_outputs", None)
        hydration_api_available = callable(hydrate_run_outputs)
        find_matching_run = getattr(tracker, "find_matching_run", None)
        if not callable(find_matching_run):
            raise AttributeError("tracker does not support find_matching_run")
        target = dict(
            restart_target_for_step(
                settings=settings,
                step_name="beam_run",
                year=year,
                iteration=iteration,
                state=state,
                workspace=workspace,
            )
        )
        run = find_matching_run(**target)
        matched_run_id = str(getattr(run, "id", "") or "") if run is not None else None
        if matched_run_id:
            try:
                matched_outputs = load_tracker_run_outputs(
                    matched_run_id,
                    tracker=tracker,
                    logger=logger,
                    log_context="BEAM restart readiness diagnostic",
                )
                matched_output_keys = sorted(str(key) for key in matched_outputs.keys())
            except Exception as exc:
                error = f"get_run_outputs_failed:{type(exc).__name__}:{exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"

    output_key_source = matched_output_keys or output_keys
    missing_required = [
        key for key in required_keys if key not in set(output_key_source)
    ]
    matchable = bool(matched_run_id)
    logger.info(
        "[BEAM][restart] recovery readiness diagnostic: matchable=%s run_id=%s "
        "missing_required=%s required_keys=%s output_keys=%s",
        matchable,
        matched_run_id,
        missing_required,
        required_keys,
        output_key_source,
    )
    _emit_artifact_lifecycle_event(
        "beam_restart_recovery_readiness",
        key="beam_restart_recovery_readiness",
        artifact_family="beam_restart_diagnostic",
        diagnostic="beam_restart_recovery_readiness",
        restart_run=bool(getattr(state, "is_restart_run", False)),
        workflow_year=getattr(state, "year", getattr(state, "current_year", None)),
        forecast_year=getattr(state, "forecast_year", None),
        iteration=iteration,
        run_scope=str(target.get("run_scope")) if target else None,
        query_status=target.get("status") if target else None,
        matched_run_id=matched_run_id,
        matchable=matchable,
        output_keys=output_keys,
        matched_output_keys=matched_output_keys,
        required_postprocess_keys=required_keys,
        missing_required_keys=missing_required,
        hydration_api_available=hydration_api_available,
        drift_classification=(
            "completed_beam_run_recovery_ready"
            if matchable and not missing_required and error is None
            else "completed_beam_run_recovery_not_ready"
        ),
        diagnostic_error=error,
    )


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
        BEAM_CONFIG_FILE,
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
    has_vehicles = _BEAM_VEHICLES_IN in prepared_inputs
    if has_warmstart and LINKSTATS_WARMSTART not in finalized_keys:
        finalized_keys.append(LINKSTATS_WARMSTART)
    if not has_warmstart and LINKSTATS_WARMSTART in finalized_keys:
        finalized_keys = [key for key in finalized_keys if key != LINKSTATS_WARMSTART]
    if has_vehicles and _BEAM_VEHICLES_IN not in finalized_keys:
        finalized_keys.append(_BEAM_VEHICLES_IN)
    return finalized_keys


def _make_beam_stage_runner(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    runtime_kwargs_extra: Optional[Mapping[str, Any]] = None,
    context: WorkflowRuntimeContext,
) -> StageRunner:
    """Build the execution context shared by BEAM stage slices."""
    return StageRunner(
        stage_name="beam",
        scenario=scenario,
        state=context.state,
        settings=context.settings,
        workspace=context.workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
        run_workflow_fn=run_workflow,
    )


def _maybe_fail_after_beam_run_for_canary(*, year: int, iteration: int) -> None:
    """Inject a controlled restart-canary failure after BEAM run completion."""
    if os.environ.get(_FAIL_AFTER_BEAM_RUN_ENV) != "1":
        return

    message = (
        "Injected failure after completed beam_run for restart canary "
        f"({_FAIL_AFTER_BEAM_RUN_ENV}=1, year={year}, iteration={iteration}). "
        "Unset the environment variable before restarting from run_state.yaml."
    )
    logger.error(message)
    raise RuntimeError(message)


def _run_beam_preprocess_step(
    *,
    stage_runner: StageRunner,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    beam_preprocess_binding: BindingPlan,
) -> None:
    """
    Execute BEAM preprocess with explicit resolved inputs.
    """
    stage_runner.run_step(
        step=StepRef(
            name="beam_preprocess",
            step_func=make_beam_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=beam_preprocess_binding,
            year=year,
        )
    )


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    beam_preprocess_binding: BindingPlan,
    beam_run_input_keys: Optional[list[str]],
    include_zarr_skims: bool,
    runtime_kwargs_extra: Mapping[str, Any],
    context: WorkflowRuntimeContext,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute BEAM preprocess/run/postprocess and return combined outputs.
    """
    surface = surface or context.surface
    stage_runner = _make_beam_stage_runner(
        scenario=scenario,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
        context=context,
    )
    recovered_run = _try_restore_completed_beam_run_for_restart(
        scenario=scenario,
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
    )
    if recovered_run is None:
        _raise_if_restart_beam_config_missing(
            binding=beam_preprocess_binding,
            state=context.state,
            settings=context.settings,
            workspace=context.workspace,
        )
        _run_beam_preprocess_step(
            stage_runner=stage_runner,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=year,
            beam_preprocess_binding=beam_preprocess_binding,
        )
        beam_preprocess_inputs = (
            beam_preprocess_binding.inputs
            if beam_preprocess_binding.inputs is not None
            else {}
        )
        config_value = beam_preprocess_inputs.get(BEAM_CONFIG_FILE)
        if config_value is not None:
            set_coupler_from_artifact(
                coupler,
                BEAM_CONFIG_FILE,
                None,
                fallback=str(config_value),
            )
        vehicles_value = beam_preprocess_inputs.get(ATLAS_VEHICLES2_OUTPUT)
        if vehicles_value is not None:
            set_coupler_from_artifact(
                coupler,
                _BEAM_VEHICLES_IN,
                None,
                fallback=str(vehicles_value),
            )
            set_coupler_from_artifact(
                coupler,
                ATLAS_VEHICLES2_OUTPUT,
                None,
                fallback=str(vehicles_value),
            )
        beam_run_input_keys = _finalize_beam_run_input_keys(
            beam_run_input_keys=beam_run_input_keys,
            outputs_holder=outputs_holder,
        )

        stage_runner.run_step(
            step=StepRef(
                name="beam_run",
                step_func=make_beam_run_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                ),
                binding=build_key_only_binding_plan(
                    step_name="beam_run",
                    input_keys=beam_run_input_keys,
                    optional_input_keys=[LINKSTATS_WARMSTART, _BEAM_VEHICLES_IN],
                    coupler=coupler,
                    settings=context.settings,
                    state=context.state,
                    workspace=context.workspace,
                    year=year,
                ),
                year=year,
            )
        )

    upstream_run = outputs_holder.beam_run
    if upstream_run is None:
        raise RuntimeError("BEAM run must complete first")
    if recovered_run is None:
        _emit_beam_restart_recovery_readiness_diagnostic(
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            outputs=upstream_run,
            year=year,
            iteration=iteration,
            include_zarr_skims=include_zarr_skims,
        )
        _maybe_fail_after_beam_run_for_canary(year=year, iteration=iteration)
    beam_postprocess_input_keys = _build_beam_postprocess_input_keys(
        upstream_keys=[
            short_name for short_name, _, _ in upstream_run._iter_record_items()
        ],
        year=year,
        iteration=iteration,
        include_zarr_skims=include_zarr_skims,
    )
    beam_postprocess_binding = None
    if beam_postprocess_input_keys:
        optional_keys = (
            [ZARR_SKIMS] if ZARR_SKIMS in beam_postprocess_input_keys else []
        )
        required_keys = [
            key for key in beam_postprocess_input_keys if key not in optional_keys
        ]
        beam_postprocess_binding = build_binding_plan(
            step_name="beam_postprocess",
            coupler=coupler,
            explicit_inputs=step_output_handoff_mapping(upstream_run, coupler=coupler),
            required_keys=required_keys,
            optional_keys=optional_keys,
            settings=context.settings,
            state=context.state,
            workspace=context.workspace,
            year=year,
            surface=surface,
        )

    stage_runner.run_step(
        step=StepRef(
            name="beam_postprocess",
            step_func=make_beam_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=beam_postprocess_binding,
            year=year,
        )
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
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
    runtime_kwargs_extra: Mapping[str, Any],
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """
    Execute dedicated BEAM full-skim step and return its outputs.
    """
    runtime_kwargs = dict(runtime_kwargs_extra)
    runtime_kwargs["previous_beam_outputs"] = previous_beam_outputs
    stage_runner = _make_beam_stage_runner(
        scenario=scenario,
        coupler=coupler,
        outputs_holder=outputs_holder,
        year=year,
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs,
        context=context,
    )
    stage_runner.run_step(
        step=StepRef(
            name="beam_full_skim",
            step_func=make_beam_full_skim_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            binding=BindingPlan(),
            year=context.state.forecast_year,
        )
    )

    if outputs_holder.beam_full_skim is None:
        return None
    return step_output_handoff_mapping(outputs_holder.beam_full_skim, coupler=coupler)


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    inputs: TrafficAssignmentPhaseInputs,
    outputs_holder: StepOutputsHolder,
    context: Optional[WorkflowRuntimeContext] = None,
    state: Optional[WorkflowState] = None,
    settings: Optional[PilatesConfig] = None,
    workspace: Optional[Workspace] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
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
    runtime_context = ensure_workflow_runtime_context(
        context=context,
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    settings = runtime_context.settings
    state = runtime_context.state
    workspace = runtime_context.workspace
    surface = runtime_context.surface

    formatted_print("TRAFFIC ASSIGNMENT MODEL")

    previous_beam_outputs = _collect_previous_beam_outputs(
        coupler=coupler,
        workspace=workspace,
        state=state,
        iteration=inputs.iteration,
        previous_beam_outputs=inputs.previous_beam_outputs,
    )
    beam_preprocess_binding = beam_preprocess_binding_plan(
        coupler=coupler,
        settings=settings,
        state=state,
        workspace=workspace,
        year=inputs.year,
        activity_demand_outputs=inputs.activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
        surface=surface,
    )
    _emit_beam_preprocess_binding_diagnostic(
        binding=beam_preprocess_binding,
        state=state,
        settings=settings,
        workspace=workspace,
        scenario=scenario,
    )
    beam_preprocess_inputs = (
        dict(beam_preprocess_binding.inputs)
        if beam_preprocess_binding.inputs is not None
        else {}
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
        _raise_if_restart_beam_config_missing(
            binding=beam_preprocess_binding,
            state=state,
            settings=settings,
            workspace=workspace,
        )
        standalone_runner = _make_beam_stage_runner(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
        )
        _run_beam_preprocess_step(
            stage_runner=standalone_runner,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            beam_preprocess_binding=beam_preprocess_binding,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            previous_beam_outputs=previous_beam_outputs,
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            coupler=coupler,
            outputs_holder=outputs_holder,
            year=inputs.year,
            iteration=inputs.iteration,
            beam_preprocess_binding=beam_preprocess_binding,
            beam_run_input_keys=beam_run_input_keys,
            include_zarr_skims=bool(inputs.activity_demand_outputs),
            runtime_kwargs_extra=traffic_runtime_kwargs,
            context=runtime_context,
            surface=surface,
        )
        if _should_run_full_skim(settings, inputs.iteration):
            full_skim_outputs = _run_beam_full_skim_step(
                scenario=scenario,
                coupler=coupler,
                outputs_holder=outputs_holder,
                year=inputs.year,
                iteration=inputs.iteration,
                previous_beam_outputs=combined_beam_outputs,
                runtime_kwargs_extra=traffic_runtime_kwargs,
                context=runtime_context,
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
