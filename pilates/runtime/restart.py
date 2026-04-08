from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict

from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
    set_coupler_from_artifact,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.workflows.artifact_keys import ZARR_SKIMS
from pilates.workflows.binding import restart_required_local_artifact_policy
from pilates.workflows.catalog import (
    RestartProducerCandidate,
    restart_artifact_producers,
    restart_query_scope_for_step,
)
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
    get_usim_datastore_fname_fn: Callable[..., str],
    required_asim_config_dirs_fn: Callable[[str], Sequence[str]],
    atlas_static_input_relpaths_fn: Callable[[Any], Sequence[str]],
    workflow_stage: Any,
) -> List[RestartArtifactDiagnostic]:
    """
    Build a pragmatic set of local artifacts that must exist to safely skip bootstrap.
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
    restart_required_local_artifacts_fn: Callable[..., List[RestartArtifactDiagnostic]],
) -> List[RestartArtifactDiagnostic]:
    missing: List[RestartArtifactDiagnostic] = []
    for artifact in restart_required_local_artifacts_fn(
        settings=settings, state=state, workspace=workspace
    ):
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


@dataclass(frozen=True)
class RestartFrontierContract:
    frontier_stage: str
    frontier_step: str
    required_keys: Tuple[str, ...]


class RestartHydrationError(RuntimeError):
    def __init__(self, message: str, *, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


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
) -> Optional[RestartFrontierContract]:
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
) -> RestartHydrationSummary:
    contract = restart_frontier_contract(
        settings=settings,
        state=state,
        workflow_stage=workflow_stage,
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
    }
    if year is None:
        _raise_restart_hydration_error(
            summary=summary,
            missing_key=missing_keys[0] if missing_keys else "unknown",
            producer_step=None,
            reason="missing_resume_year",
        )
    if not missing_keys:
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

    summary["success"] = True
    return summary
