from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict

from pilates.runtime.scenario_runtime import resolve_cache_epoch
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
    set_coupler_from_artifact,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_SHARROW_CACHE_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_HOUSEHOLDS_IN,
    BEAM_INPUT_CONFIG_ARCHIVED,
    BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_VEHICLES_ARCHIVED,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.binding import restart_required_local_artifact_policy
from pilates.workflows.catalog import (
    RestartProducerCandidate,
    restart_artifact_producers,
    restart_query_scope_for_step,
)
from pilates.workflows.surface import RestartFrontierContract
from pilates.workflows.tracker_outputs import load_tracker_run_outputs

logger = logging.getLogger(__name__)


class WorkflowStageLike(Protocol):
    supply_demand_loop: Any
    traffic_assignment: Any


class RestartArtifactDiagnostic(TypedDict):
    key: str
    path: str
    reason: str


class RestartHydrationSummary(TypedDict):
    frontier_stage: Optional[str]
    frontier_step: Optional[str]
    success: bool
    hydrated_keys: List[str]
    missing_keys: List[str]
    producer_steps_by_key: Dict[str, str]
    fallback_reason: Optional[str]
    rewind_restore: bool
    overlay_root: Optional[str]


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
            materialize_from_archive=True,
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


def _hydrate_archive_workflow_manifests(
    *,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    if not local_run_dir or not archive_run_dir:
        return
    source_workflow_dir = Path(archive_run_dir) / ".workflow"
    if not source_workflow_dir.exists():
        return
    target_workflow_dir = Path(local_run_dir) / ".workflow"
    for manifest_path in source_workflow_dir.rglob("*.yaml"):
        relative_path = manifest_path.relative_to(source_workflow_dir)
        target_path = target_workflow_dir / relative_path
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, target_path)


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


class RestartHydrationError(RuntimeError):
    def __init__(self, message: str, *, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


@dataclass(frozen=True)
class RestartStateSnapshot:
    year: Optional[int]
    stage_name: Optional[str]
    iteration: Optional[int]


@dataclass(frozen=True)
class RestartExactRewindContract:
    stage_name: str
    target_step: str
    producer_step: str
    overlay_family: str
    required_snapshot_keys: Tuple[str, ...]
    optional_snapshot_keys: Tuple[str, ...] = ()


def _surface_restart_frontier_contract(surface: Any) -> Optional[RestartFrontierContract]:
    if surface is None:
        return None
    getter = getattr(surface, "restart_frontier", None)
    contract = getter() if callable(getter) else getattr(surface, "restart_frontier_contract", None)
    if contract is None:
        return None
    return RestartFrontierContract(
        frontier_stage=str(contract.frontier_stage),
        frontier_step=str(contract.frontier_step),
        required_keys=tuple(contract.required_keys),
    )


def _enabled_restart_models(settings: Any) -> Tuple[str, ...]:
    models = getattr(getattr(settings, "run", None), "models", None)
    if models is None:
        return ()

    enabled: List[str] = []
    for attr_name in (
        "land_use",
        "vehicle_ownership",
        "activity_demand",
        "traffic_assignment",
        "postprocessing",
    ):
        model_name = getattr(models, attr_name, None)
        if model_name is None:
            continue
        text = str(model_name).strip()
        if text:
            enabled.append(text)
    return tuple(dict.fromkeys(enabled))


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


def restart_exact_rewind_contract(
    *,
    settings: Any,
    state: Any,
    workflow_stage: WorkflowStageLike,
) -> Optional[RestartExactRewindContract]:
    current_stage = getattr(state, "current_sub_stage", None) or getattr(
        state, "current_major_stage", None
    )
    models = getattr(getattr(settings, "run", None), "models", None)
    if models is None or current_stage is None:
        return None

    if (
        current_stage == workflow_stage.activity_demand
        and getattr(models, "activity_demand", None) == "activitysim"
    ):
        return RestartExactRewindContract(
            stage_name="activity_demand",
            target_step="activitysim_run",
            producer_step="activitysim_postprocess",
            overlay_family="activitysim",
            required_snapshot_keys=(
                "asim_input_households_csv_archived",
                "asim_input_persons_csv_archived",
                "asim_input_land_use_csv_archived",
            ),
            optional_snapshot_keys=(
                "asim_input_skims_zarr_archived",
                "asim_input_skims_omx_archived",
            ),
        )

    if (
        current_stage == workflow_stage.traffic_assignment
        and getattr(models, "traffic_assignment", None) == "beam"
    ):
        return RestartExactRewindContract(
            stage_name="traffic_assignment",
            target_step="beam_run",
            producer_step="beam_run",
            overlay_family="beam",
            required_snapshot_keys=(
                BEAM_INPUT_PLANS_ARCHIVED,
                BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
                BEAM_INPUT_PERSONS_ARCHIVED,
                BEAM_INPUT_CONFIG_ARCHIVED,
            ),
            optional_snapshot_keys=(
                BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
                BEAM_INPUT_VEHICLES_ARCHIVED,
                BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
                BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
                BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
            ),
        )

    return None


def _select_latest_run_from_candidates(runs: Sequence[Any]) -> Any:
    if not runs:
        return None

    def _latest_key(run: Any) -> Tuple[int, float, str]:
        created_at_value = float("-inf")
        created_at = getattr(run, "created_at", None)
        if hasattr(created_at, "timestamp"):
            try:
                created_at_value = float(created_at.timestamp())
            except Exception:
                created_at_value = float("-inf")
        return (
            _coerce_int(getattr(run, "iteration", None)) or -1,
            created_at_value,
            str(getattr(run, "id", "")),
        )

    return max(runs, key=_latest_key)


def _find_latest_run_for_restart_target(
    *,
    tracker: Any,
    target: Mapping[str, Any],
) -> Any:
    find_latest_run = getattr(tracker, "find_latest_run", None)
    if callable(find_latest_run):
        return find_latest_run(**dict(target))

    find_runs = getattr(tracker, "find_runs", None)
    if callable(find_runs):
        runs = find_runs(limit=10_000, **dict(target))
        if isinstance(runs, dict):
            runs = list(runs.values())
        run = _select_latest_run_from_candidates(list(runs or ()))
        if run is None:
            raise ValueError(
                f"No runs found matching criteria for restart target: {dict(target)}"
            )
        return run

    raise AttributeError("tracker does not support find_latest_run/find_runs")


def _materialization_failures(result: Any) -> List[str]:
    failures: List[str] = []
    for field_name in ("failed", "skipped_unmapped", "skipped_missing_source"):
        entries = list(getattr(result, field_name, []) or [])
        for entry in entries:
            failures.append(f"{field_name}={entry}")
    return failures


def _remap_workspace_local_path(
    path_value: str,
    *,
    workspace: Any,
) -> Optional[str]:
    current_root_raw = getattr(workspace, "full_path", None)
    if not current_root_raw:
        return None
    current_root = Path(str(current_root_raw))
    current_run_dir_name = current_root.name
    if not current_run_dir_name:
        return None

    candidate_path = Path(path_value)
    matching_indices = [
        index
        for index, part in enumerate(candidate_path.parts)
        if part == current_run_dir_name
    ]
    for index in reversed(matching_indices):
        suffix = candidate_path.parts[index + 1 :]
        remapped = current_root.joinpath(*suffix)
        resolved = resolve_existing_path(
            str(remapped),
            workspace=workspace,
            materialize_from_archive=False,
        )
        if resolved is not None:
            return resolved
    return None


def _resolve_restart_hydrated_path(
    *,
    artifact: Any,
    workspace: Any,
    materialized_path: Optional[str],
) -> Optional[str]:
    resolved = artifact_to_existing_path(artifact, workspace)
    if resolved is not None:
        return resolved

    raw_artifact_path = (
        str(artifact) if isinstance(artifact, (str, os.PathLike)) else None
    )
    if raw_artifact_path is not None:
        remapped = _remap_workspace_local_path(
            raw_artifact_path,
            workspace=workspace,
        )
        if remapped is not None:
            return remapped

    if materialized_path is None:
        return None

    resolved_materialized = resolve_existing_path(
        str(materialized_path),
        workspace=workspace,
        materialize_from_archive=False,
    )
    if resolved_materialized is not None:
        if os.path.isdir(resolved_materialized) and raw_artifact_path is not None:
            basename = os.path.basename(raw_artifact_path.rstrip(os.sep))
            if basename:
                candidate = os.path.join(resolved_materialized, basename)
                remapped_candidate = resolve_existing_path(
                    candidate,
                    workspace=workspace,
                    materialize_from_archive=False,
                )
                if remapped_candidate is not None:
                    return remapped_candidate
        return resolved_materialized
    return None


def _build_restart_query_target(
    *,
    settings: Any,
    year: int,
    iteration: Optional[int],
    producer: RestartProducerCandidate,
    query_facet: Optional[Mapping[str, Any]],
    include_iteration: bool,
) -> Dict[str, Any]:
    scope = restart_query_scope_for_step(producer.step_name)
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
    if query_facet is not None:
        target["facet"] = dict(query_facet)
    return target


def _raise_restart_hydration_error(
    *,
    summary: RestartHydrationSummary,
    missing_key: str,
    producer_step: Optional[str],
    reason: str,
) -> None:
    message = (
        "Restart hydration failed for "
        f"frontier_stage={summary.get('frontier_stage')} "
        f"frontier_step={summary.get('frontier_step')} "
        f"missing_key={missing_key} "
        f"producer_step={producer_step or 'unresolved'} "
        f"reason={reason}"
    )
    raise RestartHydrationError(message, summary=summary)


def _copy_to_target(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def _merge_tree_into_target(source_root: Path, target_root: Path) -> None:
    if not source_root.exists() or not source_root.is_dir():
        return
    for child in sorted(source_root.iterdir(), key=lambda path: path.name):
        if child.name == "__archive_manifest.json":
            continue
        _copy_to_target(child, target_root / child.name)


def _set_coupler_path(coupler: CouplerProtocol, key: str, path: Path) -> None:
    set_coupler_from_artifact(
        coupler,
        key,
        None,
        fallback=str(path),
    )


def _clear_coupler_key(coupler: CouplerProtocol, key: str) -> None:
    view_fn = getattr(coupler, "view", None)
    if callable(view_fn):
        try:
            from pilates.workflows.coupler_namespace import namespaced_view_target

            target = namespaced_view_target(key)
            if target is not None:
                namespace, local_key = target
                view = view_fn(namespace)
                set_value = getattr(view, "set", None)
                if callable(set_value):
                    set_value(local_key, None)
        except Exception:
            logger.debug("Failed clearing namespaced coupler key %s", key, exc_info=True)
    set_value = getattr(coupler, "set", None)
    if callable(set_value):
        set_value(key, None)


def _materialize_run_output_paths(
    *,
    tracker: Any,
    run_id: str,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
    requested_keys: Sequence[str],
) -> Dict[str, str]:
    outputs = load_tracker_run_outputs(
        run_id,
        tracker=tracker,
        logger=logger,
        log_context=f"rewind restore output lookup for {run_id}",
    )
    available_keys = [key for key in requested_keys if outputs.get(key) is not None]
    materialize_run_outputs = getattr(tracker, "materialize_run_outputs", None)
    if not callable(materialize_run_outputs):
        raise RuntimeError("tracker_missing_materialize_run_outputs")
    if not available_keys:
        return {}

    result = materialize_run_outputs(
        run_id=run_id,
        target_root=os.path.realpath(local_run_dir),
        source_root=os.path.realpath(archive_run_dir) if archive_run_dir else None,
        preserve_existing=True,
        keys=list(available_keys),
    )
    failures = _materialization_failures(result)
    if failures:
        raise RuntimeError("materialization_incomplete:" + ";".join(failures))

    materialized_path = None
    materialized_from_filesystem = getattr(
        result, "materialized_from_filesystem", {}
    ) or {}
    if isinstance(materialized_from_filesystem, Mapping):
        materialized_path = materialized_from_filesystem.get(run_id)

    resolved: Dict[str, str] = {}
    for key in available_keys:
        artifact = outputs.get(key)
        restored = _resolve_restart_hydrated_path(
            artifact=artifact,
            workspace=workspace,
            materialized_path=(
                str(materialized_path) if materialized_path is not None else None
            ),
        )
        if restored is not None and os.path.exists(restored):
            resolved[key] = restored
    return resolved


def _find_exact_rewind_source_run(
    *,
    tracker: Any,
    settings: Any,
    contract: RestartExactRewindContract,
    year: int,
    iteration: int,
    query_facet: Optional[Mapping[str, Any]],
) -> Any:
    scope = restart_query_scope_for_step(contract.producer_step)
    target: Dict[str, Any] = {
        "year": year,
        "iteration": iteration,
        "model": scope["model"],
        "stage": scope["stage"],
        "status": "completed",
        "cache_epoch": resolve_cache_epoch(settings),
    }
    if scope.get("phase") is not None:
        target["phase"] = scope["phase"]
    if query_facet is not None:
        target["facet"] = dict(query_facet)
    return _find_latest_run_for_restart_target(
        tracker=tracker,
        target=target,
    )


def _restore_activitysim_rewind_overlay(
    *,
    workspace: Any,
    coupler: CouplerProtocol,
    overlay_root: Path,
    restored_snapshot_paths: Mapping[str, str],
    year: int,
    iteration: int,
) -> None:
    data_dir = overlay_root / "data"
    cache_dir = overlay_root / "cache"

    required_targets = {
        "asim_input_households_csv_archived": data_dir / "households.csv",
        "asim_input_persons_csv_archived": data_dir / "persons.csv",
        "asim_input_land_use_csv_archived": data_dir / "land_use.csv",
    }
    for key, target in required_targets.items():
        _copy_to_target(Path(restored_snapshot_paths[key]), target)

    if "asim_input_skims_omx_archived" in restored_snapshot_paths:
        _copy_to_target(
            Path(restored_snapshot_paths["asim_input_skims_omx_archived"]),
            data_dir / "skims.omx",
        )
    if "asim_input_skims_zarr_archived" in restored_snapshot_paths:
        _copy_to_target(
            Path(restored_snapshot_paths["asim_input_skims_zarr_archived"]),
            cache_dir / "skims.zarr",
        )

    workspace.set_asim_mutable_data_dir_override(str(data_dir))
    workspace.set_asim_runtime_cache_dir_override(str(cache_dir))

    _set_coupler_path(coupler, ASIM_HOUSEHOLDS_IN, data_dir / "households.csv")
    _set_coupler_path(coupler, ASIM_PERSONS_IN, data_dir / "persons.csv")
    _set_coupler_path(coupler, ASIM_LAND_USE_IN, data_dir / "land_use.csv")
    omx_path = data_dir / "skims.omx"
    if omx_path.exists():
        _set_coupler_path(coupler, ASIM_OMX_SKIMS, omx_path)
    zarr_path = cache_dir / "skims.zarr"
    if zarr_path.exists():
        _set_coupler_path(coupler, ZARR_SKIMS, zarr_path)
    else:
        _clear_coupler_key(coupler, ZARR_SKIMS)
    _clear_coupler_key(coupler, ASIM_SHARROW_CACHE_DIR)
    setattr(
        workspace,
        "_activitysim_exact_rewind_restore",
        {
            "overlay_root": str(overlay_root),
            "mutable_data_dir": str(data_dir),
            "runtime_cache_dir": str(cache_dir),
            "zarr_available": zarr_path.exists(),
            "omx_path": str(omx_path) if omx_path.exists() else None,
            "year": year,
            "iteration": iteration,
        },
    )


def _restore_beam_rewind_overlay(
    *,
    settings: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    overlay_root: Path,
    restored_snapshot_paths: Mapping[str, str],
    year: int,
    iteration: int,
) -> None:
    from pilates.beam import beam_exchange

    beam_input_root = overlay_root / "input"
    workspace.set_beam_mutable_data_dir_override(str(beam_input_root))

    region_dir = Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    config_reference_dir = restored_snapshot_paths.get(
        BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED
    )
    if config_reference_dir is not None:
        _merge_tree_into_target(Path(config_reference_dir), region_dir)
    config_target = region_dir / str(getattr(settings.beam, "config", "beam.conf"))
    _copy_to_target(
        Path(restored_snapshot_paths[BEAM_INPUT_CONFIG_ARCHIVED]),
        config_target,
    )

    scenario_dir = Path(
        beam_exchange.resolve_beam_exchange_scenario_folder(settings, workspace)
    )
    runtime_targets = {
        BEAM_INPUT_PLANS_ARCHIVED: "plans",
        BEAM_INPUT_HOUSEHOLDS_ARCHIVED: "households",
        BEAM_INPUT_PERSONS_ARCHIVED: "persons",
        BEAM_INPUT_VEHICLES_ARCHIVED: "vehicles",
        BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED: "linkstats",
    }
    restored_runtime_paths: Dict[str, Path] = {}
    for archive_key, stem in runtime_targets.items():
        snapshot_path = restored_snapshot_paths.get(archive_key)
        if snapshot_path is None:
            continue
        source = Path(snapshot_path)
        target = scenario_dir / f"{stem}{''.join(source.suffixes)}"
        restored_runtime_paths[archive_key] = _copy_to_target(source, target)

    warmstart_dir = overlay_root / "warmstart"
    warmstart_restored: Dict[str, Path] = {}
    for archive_key, stem in (
        (BEAM_INPUT_PLANS_WARMSTART_ARCHIVED, "beam_warmstart_plans"),
        (
            BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
            "beam_warmstart_experienced_plans",
        ),
    ):
        snapshot_path = restored_snapshot_paths.get(archive_key)
        if snapshot_path is None:
            continue
        source = Path(snapshot_path)
        warmstart_restored[archive_key] = _copy_to_target(
            source,
            warmstart_dir / f"{stem}{''.join(source.suffixes)}",
        )

    _set_coupler_path(coupler, BEAM_CONFIG_FILE, config_target)
    _set_coupler_path(
        coupler,
        BEAM_PLANS_IN,
        restored_runtime_paths[BEAM_INPUT_PLANS_ARCHIVED],
    )
    _set_coupler_path(
        coupler,
        BEAM_HOUSEHOLDS_IN,
        restored_runtime_paths[BEAM_INPUT_HOUSEHOLDS_ARCHIVED],
    )
    _set_coupler_path(
        coupler,
        BEAM_PERSONS_IN,
        restored_runtime_paths[BEAM_INPUT_PERSONS_ARCHIVED],
    )
    if BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED in restored_runtime_paths:
        _set_coupler_path(
            coupler,
            LINKSTATS_WARMSTART,
            restored_runtime_paths[BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED],
        )
    if BEAM_INPUT_VEHICLES_ARCHIVED in restored_runtime_paths:
        _set_coupler_path(
            coupler,
            "vehicles_beam_in",
            restored_runtime_paths[BEAM_INPUT_VEHICLES_ARCHIVED],
        )
        _set_coupler_path(
            coupler,
            ATLAS_VEHICLES2_OUTPUT,
            restored_runtime_paths[BEAM_INPUT_VEHICLES_ARCHIVED],
        )
    warmstart_plans = warmstart_restored.get(BEAM_INPUT_PLANS_WARMSTART_ARCHIVED)
    if warmstart_plans is not None:
        if warmstart_plans.suffixes[-2:] == [".xml", ".gz"] or warmstart_plans.suffix == ".xml":
            _set_coupler_path(coupler, BEAM_OUTPUT_PLANS_XML, warmstart_plans)
        else:
            _set_coupler_path(coupler, BEAM_PLANS_OUT, warmstart_plans)
    warmstart_experienced = warmstart_restored.get(
        BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED
    )
    if warmstart_experienced is not None:
        _set_coupler_path(coupler, BEAM_EXPERIENCED_PLANS_XML, warmstart_experienced)
        _set_coupler_path(
            coupler,
            BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            warmstart_experienced,
        )
    setattr(
        workspace,
        "_beam_exact_rewind_restore",
        {
            "overlay_root": str(overlay_root),
            "beam_input_root": str(beam_input_root),
            "vehicles_path": str(restored_runtime_paths[BEAM_INPUT_VEHICLES_ARCHIVED])
            if BEAM_INPUT_VEHICLES_ARCHIVED in restored_runtime_paths
            else None,
            "year": year,
            "iteration": iteration,
        },
    )


def hydrate_rewind_runner_inputs(
    *,
    tracker: Any,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    local_run_dir: str,
    archive_run_dir: str,
    archive_state_path: str,
    allow_rewind_resume: bool,
    workflow_stage: WorkflowStageLike,
    read_current_stage_fn: Callable[[str], Tuple[Any, ...]],
    query_facet: Optional[Mapping[str, Any]] = None,
) -> Optional[RestartHydrationSummary]:
    """
    Legacy manual recovery helper for exact-rewind resume overlays.

    The normal launcher path no longer calls this; replay-first restarts should
    rely on scenario replay plus cache hits instead.
    """
    contract = restart_exact_rewind_contract(
        settings=settings,
        state=state,
        workflow_stage=workflow_stage,
    )
    if contract is None or not allow_rewind_resume:
        return None
    if not is_rewind_resume_request(
        state=state,
        archive_state_path=archive_state_path,
        read_current_stage_fn=read_current_stage_fn,
    ):
        return None

    year = _coerce_int(getattr(state, "current_year", None))
    iteration = _coerce_int(getattr(state, "current_inner_iter", None))
    summary: RestartHydrationSummary = {
        "frontier_stage": contract.stage_name,
        "frontier_step": contract.target_step,
        "success": False,
        "hydrated_keys": [],
        "missing_keys": [],
        "producer_steps_by_key": {},
        "fallback_reason": None,
        "rewind_restore": True,
        "overlay_root": None,
    }
    if year is None or iteration is None:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=contract.required_snapshot_keys[0],
            producer_step=contract.producer_step,
            reason="missing_resume_year_or_iteration",
        )

    try:
        run = _find_exact_rewind_source_run(
            tracker=tracker,
            settings=settings,
            contract=contract,
            year=year,
            iteration=iteration,
            query_facet=query_facet,
        )
    except Exception as exc:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=contract.required_snapshot_keys[0],
            producer_step=contract.producer_step,
            reason=f"no_completed_run_found:{exc}",
        )

    run_id = str(getattr(run, "id", "")).strip()
    if not run_id:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=contract.required_snapshot_keys[0],
            producer_step=contract.producer_step,
            reason="matched_run_missing_id",
        )

    requested_snapshot_keys = (
        *contract.required_snapshot_keys,
        *contract.optional_snapshot_keys,
    )
    try:
        restored_snapshot_paths = _materialize_run_output_paths(
            tracker=tracker,
            run_id=run_id,
            workspace=workspace,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
            requested_keys=requested_snapshot_keys,
        )
    except Exception as exc:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=contract.required_snapshot_keys[0],
            producer_step=contract.producer_step,
            reason=str(exc),
        )

    for key in contract.required_snapshot_keys:
        summary["producer_steps_by_key"][key] = contract.producer_step
        if key not in restored_snapshot_paths:
            summary["missing_keys"].append(key)
    for key in contract.optional_snapshot_keys:
        if key in restored_snapshot_paths:
            summary["producer_steps_by_key"][key] = contract.producer_step

    if contract.overlay_family == "activitysim" and not any(
        key in restored_snapshot_paths
        for key in ("asim_input_skims_zarr_archived", "asim_input_skims_omx_archived")
    ):
        summary["missing_keys"].append("asim_input_skims_{zarr|omx}_archived")
    if summary["missing_keys"]:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=summary["missing_keys"][0],
            producer_step=contract.producer_step,
            reason="producer_run_missing_declared_output",
        )

    overlay_root = (
        Path(workspace.full_path)
        / ".restart_overlays"
        / contract.overlay_family
        / f"year-{year}-iteration-{iteration}"
    )
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)

    if contract.overlay_family == "activitysim":
        _restore_activitysim_rewind_overlay(
            workspace=workspace,
            coupler=coupler,
            overlay_root=overlay_root,
            restored_snapshot_paths=restored_snapshot_paths,
            year=year,
            iteration=iteration,
        )
    else:
        _restore_beam_rewind_overlay(
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            overlay_root=overlay_root,
            restored_snapshot_paths=restored_snapshot_paths,
            year=year,
            iteration=iteration,
        )

    summary["success"] = True
    summary["overlay_root"] = str(overlay_root)
    summary["hydrated_keys"] = list(dict.fromkeys(restored_snapshot_paths.keys()))
    return summary


def hydrate_missing_restart_artifacts(
    *,
    tracker: Any,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    local_run_dir: str,
    archive_run_dir: str,
    workflow_stage: WorkflowStageLike,
    query_facet: Optional[Mapping[str, Any]] = None,
    surface: Any = None,
) -> RestartHydrationSummary:
    """
    Legacy manual recovery helper for explicit frontier artifact hydration.

    The default launcher path is replay-first and does not invoke this helper.
    It remains available as narrow operator tooling and for focused tests while
    the legacy restart subsystem is retired incrementally.
    """
    contract = restart_frontier_contract(
        settings=settings,
        state=state,
        workflow_stage=workflow_stage,
        surface=surface,
    )
    if contract is None:
        return {
            "frontier_stage": None,
            "frontier_step": None,
            "success": True,
            "hydrated_keys": [],
            "missing_keys": [],
            "producer_steps_by_key": {},
            "fallback_reason": None,
            "rewind_restore": False,
            "overlay_root": None,
        }

    year = _coerce_int(getattr(state, "current_year", None))
    iteration = _coerce_int(getattr(state, "current_inner_iter", None))
    get_value = getattr(coupler, "get", None)
    missing_keys: List[str] = []
    if callable(get_value):
        for key in contract.required_keys:
            value = get_value(key)
            if value is None or artifact_to_existing_path(value, workspace) is None:
                missing_keys.append(key)
    else:
        missing_keys.extend(contract.required_keys)

    summary: RestartHydrationSummary = {
        "frontier_stage": contract.frontier_stage,
        "frontier_step": contract.frontier_step,
        "success": False,
        "hydrated_keys": [],
        "missing_keys": list(missing_keys),
        "producer_steps_by_key": {},
        "fallback_reason": None,
        "rewind_restore": False,
        "overlay_root": None,
    }
    if year is None:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=missing_keys[0] if missing_keys else "unknown",
            producer_step=None,
            reason="missing_resume_year",
        )
    if not missing_keys:
        _hydrate_archive_workflow_manifests(
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )
        summary["success"] = True
        return summary

    producers_by_key = restart_artifact_producers(
        frontier_stage=contract.frontier_stage,
        enabled_models=_enabled_restart_models(settings),
    )
    materialize_run_outputs = getattr(tracker, "materialize_run_outputs", None)
    if not callable(materialize_run_outputs):
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=missing_keys[0],
            producer_step=None,
            reason="tracker_missing_materialize_run_outputs",
        )

    target_root = os.path.realpath(local_run_dir)
    source_root = os.path.realpath(archive_run_dir) if archive_run_dir else None

    for key in missing_keys:
        candidates = tuple(producers_by_key.get(key, ()))
        if not candidates:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=None,
                reason="no_registered_producer",
            )
        producer = candidates[0]
        summary["producer_steps_by_key"][key] = producer.step_name

        query_target = _build_restart_query_target(
            settings=settings,
            year=year,
            iteration=iteration,
            producer=producer,
            query_facet=query_facet,
            include_iteration=True,
        )
        used_iteration_fallback = False
        try:
            run = _find_latest_run_for_restart_target(
                tracker=tracker,
                target=query_target,
            )
        except ValueError:
            query_target = _build_restart_query_target(
                settings=settings,
                year=year,
                iteration=iteration,
                producer=producer,
                query_facet=query_facet,
                include_iteration=False,
            )
            try:
                run = _find_latest_run_for_restart_target(
                    tracker=tracker,
                    target=query_target,
                )
            except ValueError:
                _raise_restart_hydration_error(
                    summary=summary,
                    missing_key=key,
                    producer_step=producer.step_name,
                    reason="no_completed_run_found",
                )
            used_iteration_fallback = True
        except Exception as exc:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason=f"tracker_query_failed:{exc}",
            )

        run_id = str(getattr(run, "id", "")).strip()
        if not run_id:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason="matched_run_missing_id",
            )

        outputs = load_tracker_run_outputs(
            run_id,
            tracker=tracker,
            logger=logger,
            log_context=f"restart hydration output lookup for {key}",
        )
        artifact = outputs.get(key)
        if artifact is None:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason="producer_run_missing_declared_output",
            )

        try:
            result = materialize_run_outputs(
                run_id=run_id,
                target_root=target_root,
                source_root=source_root,
                preserve_existing=True,
                keys=[key],
            )
        except Exception as exc:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason=f"materialize_run_outputs_failed:{exc}",
            )

        failures = _materialization_failures(result)
        if failures:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason="materialization_incomplete:" + ";".join(failures),
            )

        materialized_path = None
        materialized_from_filesystem = getattr(
            result, "materialized_from_filesystem", {}
        ) or {}
        if isinstance(materialized_from_filesystem, Mapping):
            materialized_path = materialized_from_filesystem.get(run_id)

        resolved_path = _resolve_restart_hydrated_path(
            artifact=artifact,
            workspace=workspace,
            materialized_path=str(materialized_path) if materialized_path is not None else None,
        )
        artifact_for_coupler = artifact
        if isinstance(artifact, (str, os.PathLike)):
            artifact_for_coupler = None
        set_coupler_from_artifact(
            coupler,
            key,
            artifact_for_coupler,
            fallback=resolved_path,
        )
        coupler_value = get_value(key) if callable(get_value) else artifact
        if artifact_to_existing_path(coupler_value, workspace) is None:
            _raise_restart_hydration_error(
                summary=summary,
                missing_key=key,
                producer_step=producer.step_name,
                reason="hydrated_value_not_resolved_in_workspace",
            )

        summary["hydrated_keys"].append(key)
        if used_iteration_fallback:
            summary["fallback_reason"] = "iteration_agnostic_retry"

    _hydrate_archive_workflow_manifests(
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )
    summary["success"] = True
    return summary
