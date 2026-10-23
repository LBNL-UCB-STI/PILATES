from __future__ import annotations

import logging
import os
from bisect import bisect_left
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
)

from pilates.runtime.promote_run_archive import _open_archive_tracker
from pilates.runtime.scenario_runtime import resolve_cache_epoch
from pilates.utils.coupler_helpers import (
    resolve_existing_path,
)
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import current_year, is_start_year, iteration_index
from pilates.workflows.artifact_keys import (
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.binding import restart_required_local_artifact_policy

logger = logging.getLogger(__name__)

WARMSTART_CONVERGENCE_ROLES: frozenset[str] = frozenset(
    {
        LINKSTATS,
        LINKSTATS_WARMSTART,
        ZARR_SKIMS,
    }
)
RESTART_REMAPPABLE_SCHEMES: Mapping[str, str] = {
    "workspace": ".",
    "beam_input": "beam/input",
}


class WorkflowStageLike(Protocol):
    supply_demand_loop: Any
    traffic_assignment: Any


class RestoredSnapshotLike(Protocol):
    pruned_keys: Tuple[str, ...]


class RestartArtifactDiagnostic(TypedDict):
    key: str
    path: str
    reason: str


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def restart_required_local_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    surface: Any = None,
    get_usim_datastore_fname_fn: Callable[..., str],
    required_asim_config_dirs_fn: Callable[[str], Sequence[str]],
    atlas_static_input_relpaths_fn: Callable[[Any], Sequence[str]],
    workflow_stage: Any,
) -> List[RestartArtifactDiagnostic]:
    """Build the local restart artifact inventory used by preflight checks.

    This stays operational rather than semantic: the surface decides which
    frontier/bootstrap classifications are active, while this function keeps the
    existing path resolution and local-materialization checks.
    """
    required: List[RestartArtifactDiagnostic] = []
    for rule in restart_required_local_artifact_policy():
        resolved = rule.resolve(
            settings=settings,
            state=state,
            workspace=workspace,
            get_usim_datastore_fname_fn=get_usim_datastore_fname_fn,
            required_asim_config_dirs_fn=required_asim_config_dirs_fn,
            atlas_static_input_relpaths_fn=atlas_static_input_relpaths_fn,
            workflow_stage=workflow_stage,
        )
        if not resolved:
            continue
        for key, path in resolved.items():
            if path is None:
                continue
            required.append(
                {
                    "key": key,
                    "path": path,
                    "reason": (
                        f"Restart policy '{rule.name}' requires {key}"
                        + (f" ({rule.notes})" if rule.notes else "")
                    ),
                }
            )
    return required


def find_missing_restart_local_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    surface: Any = None,
    restart_required_local_artifacts_fn: Callable[..., List[RestartArtifactDiagnostic]],
) -> List[RestartArtifactDiagnostic]:
    """Resolve the restart inventory against local/archive materialization state."""
    missing: List[RestartArtifactDiagnostic] = []
    kwargs = {
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    if surface is not None:
        kwargs["surface"] = surface
    for artifact in restart_required_local_artifacts_fn(**kwargs):
        path = os.path.realpath(artifact["path"])
        resolved_path = resolve_existing_path(
            path,
            workspace=workspace,
        )
        if resolved_path is None or not os.path.exists(resolved_path):
            missing.append(
                {
                    "key": artifact["key"],
                    "path": path,
                    "reason": artifact["reason"],
                }
            )
    return missing


def hydrate_restart_atlas_continuation_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    workflow_stage: Any,
) -> None:
    """Hydrate the selected prior ATLAS subyear before strict restart checks."""

    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return
    if (
        getattr(state, "current_major_stage", None)
        != workflow_stage.vehicle_ownership_model
    ):
        return

    start_year = _coerce_int(getattr(state, "start_year", None))
    atlas_year = _coerce_int(
        getattr(state, "year", getattr(state, "current_year", None))
    )
    run_info_path = getattr(state, "run_info_path", None)
    if (
        start_year is None
        or atlas_year is None
        or atlas_year <= start_year
        or not run_info_path
    ):
        return

    prior_atlas_year = atlas_year - 2
    archive_run_dir = Path(os.fspath(run_info_path)).parent
    tracker = _open_archive_tracker(
        settings,
        archive_run_dir=archive_run_dir,
    )
    if tracker is None:
        raise RuntimeError(
            f"ATLAS restart requires an archive Consist tracker for {archive_run_dir}"
        )

    iteration = _coerce_int(getattr(state, "iteration", None))
    facet = _restart_atlas_identity_facet(
        settings=settings,
        state=state,
        atlas_subyear=prior_atlas_year,
        workspace=workspace,
    )
    cache_epoch = resolve_cache_epoch(settings)
    preprocess_target = {
        "model": "atlas_preprocess",
        "stage": "vehicle_ownership",
        "phase": "preprocess",
        "status": "completed",
        "year": prior_atlas_year,
        "cache_epoch": cache_epoch,
        "facet": facet,
        "run_scope": archive_run_dir.name,
    }
    if iteration is not None:
        preprocess_target["iteration"] = iteration
    preprocess_run = _select_exact_completed_restart_run(
        tracker=tracker,
        target=preprocess_target,
        step_name="atlas_preprocess",
    )

    atlas_target = {
        "model": "atlas_run",
        "stage": "vehicle_ownership",
        "phase": "run",
        "status": "completed",
        "year": prior_atlas_year,
        "cache_epoch": cache_epoch,
        "facet": facet,
        "run_scope": archive_run_dir.name,
    }
    if iteration is not None:
        atlas_target["iteration"] = iteration
    atlas_run = _select_exact_completed_restart_run(
        tracker=tracker,
        target=atlas_target,
        step_name="atlas_run",
    )

    from pilates.workflows.steps.atlas import (
        _atlas_preprocess_prepared_input_paths,
    )

    prior_state = type(
        "RestartAtlasSubState",
        (),
        {
            "year": prior_atlas_year,
            "forecast_year": prior_atlas_year,
            "start_year": start_year,
        },
    )()
    prepared_destinations = _atlas_preprocess_prepared_input_paths(
        settings=settings,
        state=prior_state,
        workspace=workspace,
    )
    tracker.hydrate_run_outputs(
        str(atlas_run.id),
        target_root=workspace.full_path,
        keys=[f"atlas_continuation_{prior_atlas_year}"],
        preserve_existing=False,
        on_missing="raise",
    )
    tracker.hydrate_run_outputs_to_destinations(
        str(preprocess_run.id),
        destinations_by_key=prepared_destinations,
        preserve_existing=False,
        on_missing="raise",
    )


def _restart_atlas_identity_facet(
    *,
    settings: Any,
    state: Any,
    atlas_subyear: int,
    workspace: Any,
) -> dict[str, Any]:
    """Reconstruct the native ATLAS producer facet for one archived subyear."""

    from pilates.utils.consist_config import build_step_consist_kwargs

    workspace_path = getattr(workspace, "full_path", None)
    metadata = build_step_consist_kwargs(
        model="atlas_preprocess",
        settings=settings,
        workspace_path=str(workspace_path) if workspace_path is not None else None,
    )
    facet = dict(metadata.get("facet") or {})
    facet["atlas_subyear"] = atlas_subyear
    facet["main_forecast_year"] = _atlas_parent_forecast_year_for_subyear(
        state=state,
        atlas_subyear=atlas_subyear,
    )
    return facet


def _atlas_parent_forecast_year_for_subyear(*, state: Any, atlas_subyear: int) -> int:
    """Return the scheduler boundary that ended the selected ATLAS interval."""

    schedule = tuple(int(year) for year in getattr(state, "_year_schedule", ()))
    boundary_index = bisect_left(schedule, atlas_subyear)
    if boundary_index == len(schedule) or (
        boundary_index == 0 and atlas_subyear < schedule[0]
    ):
        raise RuntimeError(
            "Cannot reconstruct the native ATLAS producer facet: selected subyear "
            f"{atlas_subyear} is outside WorkflowState._year_schedule={schedule}"
        )
    return schedule[boundary_index]


def _select_exact_completed_restart_run(
    *,
    tracker: Any,
    target: Mapping[str, Any],
    step_name: str,
) -> Any:
    """Select one completed producer within the archive's scoped run catalog."""

    query = {
        key: target[key]
        for key in (
            "year",
            "model",
            "stage",
            "phase",
            "status",
            "iteration",
            "cache_epoch",
            "run_scope",
            "facet",
        )
        if key in target
    }
    candidates = list(tracker.find_matching_runs(**query))
    if len(candidates) != 1:
        raise RuntimeError(
            "ATLAS restart requires exactly one completed "
            f"{step_name} run for target {query}; found {len(candidates)}"
        )
    return candidates[0]


def format_missing_artifact_summary(
    artifacts: Sequence[RestartArtifactDiagnostic],
) -> str:
    if not artifacts:
        return "none"
    return ", ".join(f"{item.get('key')}:{item.get('path')}" for item in artifacts)


def split_prebootstrap_missing_artifacts(
    artifacts: Sequence[RestartArtifactDiagnostic],
    *,
    surface: Any,
) -> Tuple[List[RestartArtifactDiagnostic], List[RestartArtifactDiagnostic]]:
    blocking_missing = [
        item
        for item in artifacts
        if not surface.is_restart_prebootstrap_deferred_artifact_key(
            item.get("key", "")
        )
    ]
    deferred_missing = [
        item
        for item in artifacts
        if surface.is_restart_prebootstrap_deferred_artifact_key(item.get("key", ""))
    ]
    return blocking_missing, deferred_missing


def log_prebootstrap_missing_artifacts(
    artifacts: Sequence[RestartArtifactDiagnostic],
    *,
    surface: Any,
) -> None:
    if not artifacts:
        return
    blocking_missing, deferred_missing = split_prebootstrap_missing_artifacts(
        artifacts,
        surface=surface,
    )
    if blocking_missing:
        logger.warning(
            "Restart diagnostic found missing local workspace inputs while "
            "data_initialized=True: %s",
            format_missing_artifact_summary(blocking_missing),
        )
    if deferred_missing:
        logger.info(
            "Restart diagnostic deferring bootstrap-owned workspace inputs "
            "until bootstrap hydration: %s",
            format_missing_artifact_summary(deferred_missing),
        )


def enforce_postbootstrap_missing_artifacts(
    artifacts: Sequence[RestartArtifactDiagnostic],
    *,
    settings: Any,
) -> None:
    if artifacts:
        logger.warning(
            "Restart diagnostic still sees missing local workspace inputs "
            "after restart bootstrap: %s",
            format_missing_artifact_summary(artifacts),
        )
    if artifacts and bool(
        getattr(getattr(settings, "run", None), "restart_strict", False)
    ):
        raise RuntimeError(
            "Strict restart preflight failed; required restart artifacts are "
            "still missing after restart bootstrap. missing="
            + format_missing_artifact_summary(artifacts)
        )


def read_archive_run_state_year(
    state_path: str,
    *,
    read_current_stage_fn: Callable[[str], Tuple[Any, ...]],
) -> Optional[int]:
    if not state_path:
        return None
    try:
        year, *_ = read_current_stage_fn(state_path)
    except Exception as exc:
        logger.warning(
            "Failed reading archive run_state year from %s: %s", state_path, exc
        )
        return None
    return _coerce_int(year)


def enforce_resume_rewind_guardrail(
    *,
    state: Any,
    archive_state_path: str,
    allow_rewind_resume: bool,
    read_archive_run_state_year_fn: Callable[[str], Optional[int]],
) -> None:
    resume_year = _coerce_int(getattr(state, "current_year", None))
    archive_year = read_archive_run_state_year_fn(archive_state_path)
    if resume_year is None or archive_year is None:
        return
    if resume_year >= archive_year:
        return

    message = (
        "Refusing rewind resume: requested resume year "
        f"{resume_year} is lower than archive run_state year {archive_year} "
        f"(archive={os.path.realpath(archive_state_path)})."
    )
    if allow_rewind_resume:
        logger.warning("%s Proceeding because --allow-rewind-resume was set.", message)
        return
    raise RuntimeError(message + " Use --allow-rewind-resume to override.")


def restore_restart_coupler_snapshot(
    *,
    scenario: Any,
    tracker: Any,
    settings: Any,
    state: Any,
    workspace: Any,
) -> Mapping[str, Any]:
    """Restore coupler state from DuckDB, materializing available artifacts and enforcing warmstart policy."""
    del tracker
    candidate_ids: List[str] = []
    output_run_name = get_setting(settings, "run.output_run_name", None)
    for cand in (
        getattr(state, "scenario_id", None),
        (
            Path(state.run_info_path).resolve().parent.name
            if getattr(state, "run_info_path", None)
            else None
        ),
        getattr(state, "run_id", None),
        output_run_name,
    ):
        if cand and str(cand) not in candidate_ids:
            candidate_ids.append(str(cand))

    workspace_full_path = getattr(workspace, "full_path", None)
    target_root = Path(workspace_full_path) if workspace_full_path else None
    archive_run_dir_val = (
        getattr(workspace, "archive_run_dir", None)
        or get_setting(settings, "run.archive_run_dir", None)
        or (
            Path(state.run_info_path).parent
            if getattr(state, "run_info_path", None)
            else None
        )
    )
    source_root = Path(archive_run_dir_val).resolve() if archive_run_dir_val else None

    coupler_obj = getattr(scenario, "coupler", None)
    baseline_coupler: Optional[dict[str, Any]] = (
        dict(coupler_obj) if isinstance(coupler_obj, Mapping) else None
    )

    restored: Any = {}
    fallback_pruned_result: Any = None
    if candidate_ids:
        for sid in candidate_ids:
            if baseline_coupler is not None and coupler_obj is not None:
                coupler_obj.clear()
                coupler_obj.update(baseline_coupler)
            try:
                restored = scenario.restore_coupler_snapshot(
                    phase="post",
                    scenario_run_id=sid,
                    clear_existing=False,
                    fallback_to_completed_outputs=True,
                    target_root=target_root,
                    source_root=source_root,
                    remappable_schemes=RESTART_REMAPPABLE_SCHEMES,
                    prune_unrecoverable=True,
                )
            except TypeError:
                try:
                    restored = scenario.restore_coupler_snapshot(
                        phase="post",
                        scenario_run_id=sid,
                        clear_existing=False,
                        fallback_to_completed_outputs=True,
                    )
                except (TypeError, AttributeError):
                    try:
                        restored = scenario.restore_coupler_snapshot(phase="post")
                    except AttributeError:
                        restored = {}
            except AttributeError:
                restored = {}
            if restored:
                break
            pruned_for_sid = (
                restored.pruned_keys
                if isinstance(restored, Mapping) and hasattr(restored, "pruned_keys")
                else ()
            )
            if pruned_for_sid and fallback_pruned_result is None:
                fallback_pruned_result = restored
        if not restored and fallback_pruned_result is not None:
            restored = fallback_pruned_result
            if coupler_obj is not None:
                for pruned_key in getattr(restored, "pruned_keys", ()):
                    coupler_obj.pop(pruned_key, None)
    else:
        try:
            restored = scenario.restore_coupler_snapshot(
                phase="post",
                target_root=target_root,
                source_root=source_root,
                remappable_schemes=RESTART_REMAPPABLE_SCHEMES,
                prune_unrecoverable=True,
            )
        except TypeError:
            try:
                restored = scenario.restore_coupler_snapshot(phase="post")
            except AttributeError:
                restored = {}
        except AttributeError:
            restored = {}

    logger.info(
        "Restored restart coupler snapshot with %d artifacts: %s",
        len(restored) if restored else 0,
        sorted(restored.keys()) if restored else [],
    )

    pruned_keys: Sequence[str] = (
        restored.pruned_keys
        if isinstance(restored, Mapping) and hasattr(restored, "pruned_keys")
        else ()
    )
    current_iteration = _coerce_int(iteration_index(state, default=0)) or 0
    restart_strict = bool(
        getattr(getattr(settings, "run", None), "restart_strict", False)
    )
    beam_settings = getattr(settings, "beam", None)
    replan_pct = float(getattr(beam_settings, "replan_PercentOfIter", 0.0) or 0.0)
    discard_plans = bool(getattr(beam_settings, "discard_plans_every_year", False))
    has_resolved_year = (
        current_year(state) is not None
        or getattr(state, "forecast_year", None) is not None
        or callable(getattr(state, "is_start_year", None))
    )
    requires_cross_year_plans = (
        has_resolved_year and not is_start_year(state, settings) and not discard_plans
    )
    critical_roles = (
        set(WARMSTART_CONVERGENCE_ROLES) if current_iteration > 0 else set()
    )
    if (current_iteration > 0 and replan_pct > 0.0) or requires_cross_year_plans:
        critical_roles.add(BEAM_PLANS_OUT)

    for role_key in pruned_keys:
        if role_key in critical_roles and restart_strict:
            raise RuntimeError(
                f"Strict restart failed: convergence artifact '{role_key}' from DuckDB "
                f"snapshot is unrecoverable at iteration {current_iteration}."
            )
        logger.warning(
            "[Restart Coupler] Pruned unrecoverable DB snapshot artifact '%s' "
            "(iteration=%s). Backing file was not archived to persistent storage.",
            role_key,
            current_iteration,
        )

    return restored if restored is not None else {}
