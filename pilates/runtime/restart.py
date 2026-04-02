from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from consist import MaterializationResult

from pilates.runtime.cache_recovery import materialize_cached_runs
from pilates.utils.coupler_helpers import resolve_existing_path
from pilates.workflows.binding import restart_required_local_artifact_policy

logger = logging.getLogger(__name__)

_OPTIONAL_RESTART_MISSING_SOURCE_SUFFIXES = ("_asim_out_temp",)
_ACTIVITYSIM_QUERY_TARGETS = (
    ("activitysim_preprocess", "activity_demand_preprocess", "preprocess"),
    ("activitysim_run", "activity_demand_run", "run"),
    ("activitysim_postprocess", "activity_demand_postprocess", "postprocess"),
)
_LAND_USE_QUERY_TARGETS = (
    ("urbansim_preprocess", "land_use", "preprocess"),
    ("urbansim_run", "land_use", "run"),
    ("urbansim_postprocess", "land_use", "postprocess"),
)
_ATLAS_QUERY_TARGETS = (
    ("atlas_preprocess", "atlas", "preprocess"),
    ("atlas_run", "atlas", "run"),
    ("atlas_postprocess", "atlas", "postprocess"),
)
_BEAM_QUERY_TARGETS = (
    ("beam_preprocess", "beam", "preprocess"),
    ("beam_run", "beam", "run"),
    ("beam_postprocess", "beam", "postprocess"),
)

def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _materialization_entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    if isinstance(entry, dict):
        for key in ("key", "name", "short_name"):
            value = entry.get(key)
            if value:
                return str(value)
    return str(entry)


def _is_optional_restart_missing_source(entry: Any) -> bool:
    name = _materialization_entry_name(entry)
    return any(
        name.endswith(suffix) for suffix in _OPTIONAL_RESTART_MISSING_SOURCE_SUFFIXES
    )


def _prune_optional_restart_missing_sources(
    result: MaterializationResult,
) -> MaterializationResult:
    tolerated = [
        entry
        for entry in list(getattr(result, "skipped_missing_source", []) or [])
        if _is_optional_restart_missing_source(entry)
    ]
    if tolerated:
        result.skipped_missing_source = [
            entry
            for entry in list(getattr(result, "skipped_missing_source", []) or [])
            if not _is_optional_restart_missing_source(entry)
        ]
        logger.info(
            "Restart reconstruction ignoring optional missing-source artifacts: %s",
            [_materialization_entry_name(entry) for entry in tolerated],
        )
    return result


def _activitysim_iteration_output_requirements(
    *, asim_output_dir: str, year: Any, iteration: Any
) -> List[Dict[str, str]]:
    iter_dir = os.path.join(
        asim_output_dir,
        f"year-{year}-iteration-{iteration}",
    )
    return [
        {
            "key": "activitysim_iteration_beam_plans_parquet",
            "path": os.path.join(iter_dir, "beam_plans.parquet"),
            "reason": (
                "ActivitySim beam plans required to resume BEAM from "
                "traffic assignment"
            ),
        },
        {
            "key": "activitysim_iteration_households_parquet",
            "path": os.path.join(iter_dir, "households.parquet"),
            "reason": (
                "ActivitySim households required to resume BEAM from "
                "traffic assignment"
            ),
        },
        {
            "key": "activitysim_iteration_persons_parquet",
            "path": os.path.join(iter_dir, "persons.parquet"),
            "reason": (
                "ActivitySim persons required to resume BEAM from "
                "traffic assignment"
            ),
        },
    ]


def restart_required_local_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    get_usim_datastore_fname_fn: Callable[..., str],
    required_asim_config_dirs_fn: Callable[[str], Sequence[str]],
    atlas_static_input_relpaths_fn: Callable[[Any], Sequence[str]],
    workflow_stage: Any,
) -> List[Dict[str, str]]:
    """
    Build a pragmatic set of local artifacts that must exist to safely skip bootstrap.
    """
    required: List[Dict[str, str]] = []
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
                        + (
                            f" ({rule.notes})"
                            if rule.notes
                            else ""
                        )
                    ),
                }
            )
    return required


def find_missing_restart_local_artifacts(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    restart_required_local_artifacts_fn: Callable[..., List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    missing: List[Dict[str, str]] = []
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


def format_missing_artifact_summary(artifacts: List[Dict[str, str]]) -> str:
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
        logger.warning("Failed reading archive run_state year from %s: %s", state_path, exc)
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


def _supply_demand_manifest_path(
    *,
    run_dir: str,
    year: int,
    iteration: int,
) -> Path:
    return Path(os.path.realpath(run_dir)) / ".workflow" / f"year_{year}_iteration_{iteration}.yaml"


def _is_stage_explicitly_enabled(
    *,
    state: Any,
    stage: Any,
) -> Optional[bool]:
    enabled_stages = getattr(state, "enabled_stages", None)
    if isinstance(enabled_stages, (set, list, tuple)):
        return stage in enabled_stages
    return None


def _atlas_sub_years(
    *,
    current_year: int,
    forecast_year: Optional[int],
) -> List[int]:
    sub_years = [current_year]
    if forecast_year is None or forecast_year <= current_year:
        return sub_years
    sub_years.extend(range(current_year + 2, forecast_year + 1, 2))
    return sub_years


def _stage_query_target(
    *,
    year: int,
    model: str,
    stage: str,
    phase: Optional[str],
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    target = {
        "year": year,
        "model": model,
        "stage": stage,
        "status": "completed",
    }
    if phase is not None:
        target["phase"] = phase
    if iteration is not None:
        target["iteration"] = iteration
    return target


def _restart_query_targets(
    *,
    state: Any,
    year: int,
    workflow_stage: Any,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    current_stage = getattr(state, "current_major_stage", None)
    current_sub_stage = getattr(state, "current_sub_stage", None)
    resume_iter = max(_coerce_int(getattr(state, "current_inner_iter", 0)) or 0, 0)
    forecast_year = _coerce_int(getattr(state, "forecast_year", None))

    land_use_enabled = _is_stage_explicitly_enabled(
        state=state,
        stage=workflow_stage.land_use,
    )
    if current_stage in {
        workflow_stage.vehicle_ownership_model,
        workflow_stage.supply_demand_loop,
        workflow_stage.postprocessing,
    } and land_use_enabled is not False:
        for model, stage_name, phase in _LAND_USE_QUERY_TARGETS:
            targets.append(
                _stage_query_target(
                    year=year,
                    iteration=0,
                    model=model,
                    stage=stage_name,
                    phase=phase,
                )
            )

    vehicle_enabled = _is_stage_explicitly_enabled(
        state=state,
        stage=workflow_stage.vehicle_ownership_model,
    )
    if (
        current_stage
        in {
            workflow_stage.vehicle_ownership_model,
            workflow_stage.supply_demand_loop,
            workflow_stage.postprocessing,
        }
        and vehicle_enabled is not False
    ):
        atlas_targets = _ATLAS_QUERY_TARGETS
        if current_stage in {
            workflow_stage.supply_demand_loop,
            workflow_stage.postprocessing,
        }:
            atlas_targets = (("atlas_postprocess", "atlas", "postprocess"),)
        for sub_year in _atlas_sub_years(
            current_year=year,
            forecast_year=forecast_year,
        ):
            for model, stage_name, phase in atlas_targets:
                targets.append(
                    _stage_query_target(
                        year=sub_year,
                        iteration=0,
                        model=model,
                        stage=stage_name,
                        phase=phase,
                    )
                )

    if current_stage in {workflow_stage.supply_demand_loop, workflow_stage.postprocessing}:
        total_iters = max(
            _coerce_int(getattr(state, "_settings", {}).get("supply_demand_iters", None))
            or 0,
            0,
        )
        completed_iterations = range(0, total_iters) if current_stage == workflow_stage.postprocessing and total_iters > 0 else range(0, resume_iter)
        for iteration in completed_iterations:
            for model, stage_name, phase in (*_ACTIVITYSIM_QUERY_TARGETS, *_BEAM_QUERY_TARGETS):
                targets.append(
                    _stage_query_target(
                        year=year,
                        iteration=iteration,
                        model=model,
                        stage=stage_name,
                        phase=phase,
                    )
                )
        if current_stage == workflow_stage.supply_demand_loop and current_sub_stage == workflow_stage.traffic_assignment:
            for model, stage_name, phase in _ACTIVITYSIM_QUERY_TARGETS:
                targets.append(
                    _stage_query_target(
                        year=year,
                        iteration=resume_iter,
                        model=model,
                        stage=stage_name,
                        phase=phase,
                    )
                )
        if current_stage == workflow_stage.postprocessing:
            postprocessing_enabled = _is_stage_explicitly_enabled(
                state=state,
                stage=workflow_stage.postprocessing,
            )
            if postprocessing_enabled is not False:
                targets.append(
                    _stage_query_target(
                        year=year,
                        model="postprocessing",
                        stage="postprocessing",
                        phase=None,
                    )
                )

    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    for target in targets:
        key = (
            target.get("year"),
            target.get("iteration"),
            target.get("model"),
            target.get("stage"),
            target.get("phase"),
            target.get("status"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped


def _select_latest_run_from_candidates(runs: Sequence[Any]) -> Any:
    if not runs:
        return None

    def _latest_key(run: Any) -> Tuple[int, float, str]:
        created_at = getattr(run, "created_at", None)
        if isinstance(created_at, datetime):
            created_at_value = created_at.timestamp()
        else:
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
    target: Dict[str, Any],
) -> Any:
    find_latest_run = getattr(tracker, "find_latest_run", None)
    if callable(find_latest_run):
        return find_latest_run(**target)

    find_runs = getattr(tracker, "find_runs", None)
    if callable(find_runs):
        runs = find_runs(limit=10_000, **target)
        if isinstance(runs, dict):
            runs = list(runs.values())
        run = _select_latest_run_from_candidates(runs)
        if run is None:
            raise ValueError(
                f"No runs found matching criteria for restart target: {target}"
            )
        return run

    raise AttributeError("tracker does not support find_latest_run/find_runs")


def _collect_restart_completed_run_ids_from_tracker(
    *,
    tracker: Any,
    state: Any,
    workflow_stage: Any,
    query_facet: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    year = _coerce_int(getattr(state, "current_year", None))
    if year is None:
        return {
            "run_ids": [],
            "issues": [("state.current_year", "missing year")],
            "manifest_paths": [],
            "query_targets": [],
            "discovery_mode": "tracker",
        }

    targets = _restart_query_targets(
        state=state,
        year=year,
        workflow_stage=workflow_stage,
    )
    run_ids: List[str] = []
    seen: Set[str] = set()
    issues: List[Tuple[str, str]] = []
    matched_targets: List[Dict[str, Any]] = []
    unmatched_targets: List[Dict[str, Any]] = []
    current_stage = getattr(state, "current_major_stage", None)
    vehicle_enabled = _is_stage_explicitly_enabled(
        state=state,
        stage=workflow_stage.vehicle_ownership_model,
    )
    atlas_require_all_subyears = (
        current_stage in {workflow_stage.supply_demand_loop, workflow_stage.postprocessing}
        and vehicle_enabled is True
    )
    atlas_gap_detected = False

    for target in targets:
        query_target = dict(target)
        if query_facet is not None:
            query_target["facet"] = dict(query_facet)
        is_atlas_target = target.get("stage") == "atlas"
        logger.info("[RestartQuery] target=%s status=pending", query_target)
        if atlas_gap_detected and is_atlas_target:
            if atlas_require_all_subyears:
                issues.append(
                    (
                        repr(query_target),
                        "required atlas restart target missing after contiguous-prefix gap",
                    )
                )
            unmatched_targets.append(query_target)
            logger.info(
                "[RestartQuery] target=%s status=skipped reason=contiguous_prefix_gap",
                query_target,
            )
            continue
        try:
            run = _find_latest_run_for_restart_target(
                tracker=tracker,
                target=query_target,
            )
        except ValueError:
            unmatched_targets.append(query_target)
            logger.info(
                "[RestartQuery] target=%s status=unmatched reason=no_completed_run",
                query_target,
            )
            if is_atlas_target:
                atlas_gap_detected = True
                if atlas_require_all_subyears:
                    issues.append(
                        (
                            repr(query_target),
                            "no completed run found for required atlas restart target",
                        )
                    )
                continue
            issues.append(
                (repr(query_target), "no completed run found for restart target")
            )
            continue
        except Exception as exc:
            logger.info(
                "[RestartQuery] target=%s status=error reason=%s",
                query_target,
                exc,
            )
            issues.append(
                (
                    repr(query_target),
                    f"tracker query failed: {exc}",
                )
            )
            continue
        run_id = str(getattr(run, "id", "")).strip()
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        run_ids.append(run_id)
        matched_targets.append({**query_target, "run_id": run_id})
        logger.info(
            "[RestartQuery] target=%s status=matched run_id=%s",
            query_target,
            run_id,
        )

    return {
        "run_ids": run_ids,
        "issues": issues,
        "manifest_paths": [],
        "query_targets": [
            (
                {**dict(target), "facet": dict(query_facet)}
                if query_facet is not None
                else dict(target)
            )
            for target in targets
        ],
        "matched_query_targets": matched_targets,
        "unmatched_query_targets": unmatched_targets,
        "atlas_gap_detected": atlas_gap_detected,
        "fallback_reason": None,
        "discovery_mode": "tracker",
    }


def collect_restart_completed_run_ids(
    *,
    state: Any,
    archive_run_dir: str,
    workflow_stage: Any,
    tracker: Any = None,
    query_facet: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if tracker is None:
        return {
            "run_ids": [],
            "issues": [("tracker", "tracker is required for restart discovery")],
            "manifest_paths": [],
            "query_targets": [],
            "matched_query_targets": [],
            "unmatched_query_targets": [],
            "atlas_gap_detected": False,
            "fallback_reason": "tracker unavailable",
            "discovery_mode": "tracker",
        }

    tracker_discovery = _collect_restart_completed_run_ids_from_tracker(
        tracker=tracker,
        state=state,
        workflow_stage=workflow_stage,
        query_facet=query_facet,
    )
    if not tracker_discovery["run_ids"] and tracker_discovery["issues"]:
        logger.warning(
            "Restart completed-run discovery returned no tracker runs issues=%s run_ids=%s",
            tracker_discovery["issues"],
            tracker_discovery["run_ids"],
        )
    return tracker_discovery


def reconstruct_restart_completed_run_outputs(
    *,
    tracker: Any,
    state: Any,
    local_run_dir: str,
    archive_run_dir: str,
    workflow_stage: Any,
    query_facet: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    discovery = collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=archive_run_dir,
        workflow_stage=workflow_stage,
        tracker=tracker,
        query_facet=query_facet,
    )
    run_ids = list(discovery["run_ids"])
    issues = list(discovery["issues"])
    source_root = os.path.realpath(archive_run_dir) if archive_run_dir else None
    target_root = os.path.realpath(local_run_dir)

    aggregate = materialize_cached_runs(
        tracker=tracker,
        run_ids=run_ids,
        target_root=target_root,
        source_root=source_root,
        preserve_existing=True,
        initial_failures=issues,
        missing_api_context="restart_reconstruction",
    )
    aggregate = _prune_optional_restart_missing_sources(aggregate)
    restored_run_diagnostics = []
    matched_query_targets = list(discovery.get("matched_query_targets", []))
    if matched_query_targets:
        for query_target in matched_query_targets:
            if not isinstance(query_target, Mapping):
                continue
            restored_run_diagnostics.append(
                {
                    "model": query_target.get("model"),
                    "year": query_target.get("year"),
                    "iteration": query_target.get("iteration"),
                    "run_id": query_target.get("run_id"),
                    "parent_run_id": None,
                    "source": "restore_reconstruction",
                }
            )
    else:
        restored_run_diagnostics.extend(
            {
                "model": None,
                "year": None,
                "iteration": None,
                "run_id": run_id,
                "parent_run_id": None,
                "source": "restore_reconstruction",
            }
            for run_id in run_ids
        )

    return {
        "run_ids": run_ids,
        "source_root": source_root,
        "target_root": target_root,
        "manifest_paths": list(discovery["manifest_paths"]),
        "query_targets": list(discovery.get("query_targets", [])),
        "matched_query_targets": list(discovery.get("matched_query_targets", [])),
        "unmatched_query_targets": list(discovery.get("unmatched_query_targets", [])),
        "discovery_mode": discovery.get("discovery_mode"),
        "fallback_reason": discovery.get("fallback_reason"),
        "atlas_gap_detected": bool(discovery.get("atlas_gap_detected", False)),
        "restored_run_diagnostics": restored_run_diagnostics,
        "materialization_result": aggregate,
    }
