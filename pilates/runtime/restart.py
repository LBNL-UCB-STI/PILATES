from __future__ import annotations

import logging
import os
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
)

from pilates.runtime.scenario_runtime import resolve_cache_epoch
from pilates.runtime.promote_run_archive import _open_archive_tracker
from pilates.utils.coupler_helpers import (
    resolve_existing_path,
)
from pilates.workflows.artifact_keys import (
    ZARR_SKIMS,
)
from pilates.workflows.binding import restart_required_local_artifact_policy
from pilates.workflows.catalog import (
    restart_query_scope_for_step,
)
from pilates.workflows.surface import RestartFrontierContract

logger = logging.getLogger(__name__)


class WorkflowStageLike(Protocol):
    supply_demand_loop: Any
    traffic_assignment: Any


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
    preprocess_run = _select_exact_completed_restart_run(
        tracker=tracker,
        target=restart_target_for_step(
            settings=settings,
            step_name="atlas_preprocess",
            year=prior_atlas_year,
            iteration=iteration,
            facet=facet,
            state=state,
            archive_run_dir=archive_run_dir,
        ),
        step_name="atlas_preprocess",
    )
    atlas_run = _select_exact_completed_restart_run(
        tracker=tracker,
        target=restart_target_for_step(
            settings=settings,
            step_name="atlas_run",
            year=prior_atlas_year,
            iteration=iteration,
            facet=facet,
            state=state,
            archive_run_dir=archive_run_dir,
        ),
        step_name="atlas_run",
    )

    from pilates.workflows.steps.urbansim_atlas import (
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


def read_archive_run_state_snapshot(
    state_path: str,
    *,
    read_current_stage_fn: Callable[[str], Tuple[Any, ...]],
) -> RestartStateSnapshot:
    if not state_path:
        return RestartStateSnapshot(year=None, stage_name=None, iteration=None)
    try:
        year, stage, iteration, *_ = read_current_stage_fn(state_path)
    except Exception as exc:
        logger.warning(
            "Failed reading archive run_state snapshot from %s: %s", state_path, exc
        )
        return RestartStateSnapshot(year=None, stage_name=None, iteration=None)
    return RestartStateSnapshot(
        year=_coerce_int(year),
        stage_name=getattr(stage, "name", None) if stage is not None else None,
        iteration=_coerce_int(iteration),
    )


def _runtime_state_snapshot(state: Any) -> RestartStateSnapshot:
    current_stage = getattr(state, "current_sub_stage", None) or getattr(
        state, "current_major_stage", None
    )
    return RestartStateSnapshot(
        year=_coerce_int(getattr(state, "current_year", None)),
        stage_name=getattr(current_stage, "name", None)
        if current_stage is not None
        else None,
        iteration=_coerce_int(getattr(state, "current_inner_iter", None)),
    )


def _stage_progress_rank(stage_name: Optional[str]) -> int:
    order = {
        "initialize_data": 0,
        "land_use": 10,
        "vehicle_ownership_model": 20,
        "activity_demand": 30,
        "activity_demand_directly_from_land_use": 30,
        "traffic_assignment": 40,
        "postprocessing": 50,
    }
    return order.get(str(stage_name), -1)


def _progress_tuple(snapshot: RestartStateSnapshot) -> Tuple[int, int, int]:
    return (
        snapshot.year if snapshot.year is not None else -1,
        snapshot.iteration if snapshot.iteration is not None else 0,
        _stage_progress_rank(snapshot.stage_name),
    )


def is_rewind_resume_request(
    *,
    state: Any,
    archive_state_path: str,
    read_current_stage_fn: Callable[[str], Tuple[Any, ...]],
) -> bool:
    requested = _runtime_state_snapshot(state)
    archive = read_archive_run_state_snapshot(
        archive_state_path,
        read_current_stage_fn=read_current_stage_fn,
    )
    if requested.year is None or archive.year is None:
        return False
    return _progress_tuple(requested) < _progress_tuple(archive)


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


@dataclass(frozen=True)
class RestartStateSnapshot:
    year: Optional[int]
    stage_name: Optional[str]
    iteration: Optional[int]


def _surface_restart_frontier_contract(
    surface: Any,
) -> Optional[RestartFrontierContract]:
    if surface is None:
        return None
    getter = getattr(surface, "restart_frontier", None)
    contract = (
        getter()
        if callable(getter)
        else getattr(surface, "restart_frontier_contract", None)
    )
    if contract is None:
        return None
    return RestartFrontierContract(
        frontier_stage=str(contract.frontier_stage),
        frontier_step=str(contract.frontier_step),
        required_keys=tuple(contract.required_keys),
    )


def restart_frontier_contract(
    *,
    settings: Any,
    state: Any,
    workflow_stage: WorkflowStageLike,
    surface: Any = None,
) -> Optional[RestartFrontierContract]:
    """Return the effective restart frontier, preferring the shared surface.

    Keeping this bridge lets older restart callers continue using the legacy
    module API while the runtime authority moves into `EnabledWorkflowSurface`.
    """
    surface_contract = _surface_restart_frontier_contract(surface)
    if surface_contract is not None:
        return surface_contract

    if getattr(state, "current_major_stage", None) != workflow_stage.supply_demand_loop:
        return None
    if getattr(state, "current_sub_stage", None) != workflow_stage.traffic_assignment:
        return None

    models = getattr(getattr(settings, "run", None), "models", None)
    if models is None:
        return None
    if getattr(models, "activity_demand", None) != "activitysim":
        return None
    if getattr(models, "traffic_assignment", None) != "beam":
        return None

    return RestartFrontierContract(
        frontier_stage="traffic_assignment",
        frontier_step="beam_preprocess",
        required_keys=(
            "beam_plans_asim_out",
            "households_asim_out",
            "persons_asim_out",
            ZARR_SKIMS,
        ),
    )


def restart_run_scope(
    *,
    state: Any = None,
    workspace: Any = None,
    archive_run_dir: Optional[str | os.PathLike[str]] = None,
    local_run_dir: Optional[str | os.PathLike[str]] = None,
) -> Optional[str]:
    """Return the Consist run-scope prefix for restart lookups."""

    candidate_paths: List[Any] = []
    if archive_run_dir:
        candidate_paths.append(archive_run_dir)
    if state is not None:
        run_info_path = getattr(state, "run_info_path", None)
        if run_info_path:
            try:
                candidate_paths.append(Path(run_info_path).expanduser().parent)
            except Exception:
                pass
    if workspace is not None:
        full_path = getattr(workspace, "full_path", None)
        if full_path:
            candidate_paths.append(full_path)
    if local_run_dir:
        candidate_paths.append(local_run_dir)

    for value in candidate_paths:
        try:
            name = Path(value).expanduser().resolve().name
        except Exception:
            name = Path(str(value)).name
        if name:
            return name
    return None


def restart_target_for_step(
    *,
    settings: Any,
    step_name: str,
    year: int,
    iteration: Optional[int] = None,
    facet: Optional[Mapping[str, Any]] = None,
    state: Any = None,
    workspace: Any = None,
    archive_run_dir: Optional[str | os.PathLike[str]] = None,
    local_run_dir: Optional[str | os.PathLike[str]] = None,
    include_iteration: bool = True,
) -> Dict[str, Any]:
    """Build the shared semantic Consist target for a completed restart run."""

    scope = restart_query_scope_for_step(step_name)
    target: Dict[str, Any] = {
        "year": year,
        "model": scope["model"],
        "stage": scope["stage"],
        "status": "completed",
        "cache_epoch": resolve_cache_epoch(settings),
    }
    phase = scope.get("phase")
    if phase is not None:
        target["phase"] = phase
    if include_iteration and iteration is not None:
        target["iteration"] = iteration
    if facet is not None:
        target["facet"] = dict(facet)
    run_scope = restart_run_scope(
        state=state,
        workspace=workspace,
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
    )
    if run_scope is not None:
        target["run_scope"] = run_scope
    return target
