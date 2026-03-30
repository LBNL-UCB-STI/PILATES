from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from consist import MaterializationResult

from pilates.runtime.cache_recovery import materialize_cached_runs
from pilates.utils.coupler_helpers import resolve_existing_path
from pilates.utils.io import get_traffic_assignment_model
from pilates.workflows.binding import restart_required_local_artifact_policy

logger = logging.getLogger(__name__)

_ACTIVITYSIM_MANIFEST_STEPS = (
    "activitysim_preprocess",
    "activitysim_run",
    "activitysim_postprocess",
)
_LAND_USE_MANIFEST_STEPS = (
    "urbansim_preprocess",
    "urbansim_run",
    "urbansim_postprocess",
)
_POSTPROCESSING_MANIFEST_STEPS = ("postprocessing",)
_ATLAS_MANIFEST_STEPS = (
    "atlas_preprocess",
    "atlas_run",
    "atlas_postprocess",
)
_ATLAS_SUBYEAR_MANIFEST_RE = re.compile(
    r"^forecast_year_(?P<forecast_year>-?\d+)_subyear_(?P<sub_year>-?\d+)\.yaml$"
)
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


def _copy_restart_bookkeeping_file(
    *,
    source: Path,
    target: Path,
) -> str:
    if target.exists():
        return "skipped_existing"
    if not source.exists():
        return "missing_source"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "copied"


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


def resolve_restart_rehydrate_mode(settings: Any) -> str:
    run_cfg = getattr(settings, "run", None)
    raw = getattr(run_cfg, "restart_rehydrate_mode", "native")
    mode = str(raw).strip().lower() if raw is not None else "native"
    if mode in {"native", "off"}:
        return mode
    logger.warning(
        "Unknown run.restart_rehydrate_mode=%r; defaulting to 'native'.",
        raw,
    )
    return "native"


def is_restart_strict(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "restart_strict", False))


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


def repair_restart_state_for_incomplete_atlas_outputs(
    *,
    settings: Any,
    state: Any,
    archive_run_dir: str,
) -> bool:
    """Rewind stale supply-demand resumes back to ATLAS when atlas outputs are incomplete."""
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return False

    stage_enum = getattr(state, "Stage", None)
    current_major_stage = getattr(state, "current_major_stage", None)
    if stage_enum is None or current_major_stage != stage_enum.supply_demand_loop:
        return False

    current_year = _coerce_int(getattr(state, "current_year", None))
    forecast_year = _coerce_int(getattr(state, "forecast_year", None))
    if current_year is None or forecast_year is None or forecast_year < current_year:
        return False

    atlas_output_dir = os.path.join(
        os.path.realpath(archive_run_dir), "atlas", "atlas_output"
    )
    sub_years = [current_year]
    if forecast_year > current_year:
        sub_years.extend(range(current_year + 2, forecast_year + 1, 2))

    incomplete_years: List[int] = []
    for atlas_year in sub_years:
        required_outputs = (
            os.path.join(atlas_output_dir, f"householdv_{atlas_year}.csv"),
            os.path.join(atlas_output_dir, f"vehicles_{atlas_year}.csv"),
            os.path.join(atlas_output_dir, f"vehicles2_{atlas_year}.csv"),
        )
        if not all(os.path.exists(path) for path in required_outputs):
            incomplete_years.append(atlas_year)

    if not incomplete_years:
        return False

    state.current_major_stage = stage_enum.vehicle_ownership_model
    state.current_sub_stage = None
    state.current_inner_iter = 0
    if hasattr(state, "sub_stage_progress"):
        state.sub_stage_progress = None
    write_state = getattr(state, "write_state", None)
    if callable(write_state):
        write_state()
    logger.warning(
        "[RestartRepair] Rewinding stale restart state from supply_demand_loop to "
        "vehicle_ownership_model because archived ATLAS outputs are incomplete for "
        "years=%s in %s",
        incomplete_years,
        atlas_output_dir,
    )
    return True


def repair_restart_atlas_inputs_from_archive(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    archive_run_dir: str,
) -> bool:
    """
    Rehydrate restart-critical ATLAS year-input directories into the local workspace.

    Strict restart preflight can require these directories before any ATLAS step
    executes, so startup needs a repair path independent from the later
    preprocess restore/rerun logic.
    """
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return False

    stage_enum = getattr(state, "Stage", None)
    current_major_stage = getattr(state, "current_major_stage", None)
    if stage_enum is None or current_major_stage != stage_enum.vehicle_ownership_model:
        return False

    get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if not callable(get_atlas_input_dir):
        return False

    start_year = _coerce_int(getattr(state, "start_year", None))
    atlas_year = _coerce_int(getattr(state, "year", getattr(state, "current_year", None)))
    if start_year is None or atlas_year is None:
        return False

    archive_root = os.path.realpath(archive_run_dir)
    if not os.path.isdir(archive_root):
        return False

    from pilates.atlas.preprocessor import (
        _restore_restart_atlas_year_inputs,
        restart_required_atlas_input_paths,
    )

    atlas_input_dir = os.path.realpath(get_atlas_input_dir())
    required_paths = restart_required_atlas_input_paths(
        atlas_input_root=atlas_input_dir,
        start_year=start_year,
        atlas_year=atlas_year,
    )
    missing_before = {
        key: path for key, path in required_paths.items() if not os.path.exists(path)
    }
    if not missing_before:
        return False

    logger.warning(
        "[RestartRepair] Missing restart-critical ATLAS inputs %s under %s; "
        "rehydrating from archive %s.",
        sorted(missing_before),
        atlas_input_dir,
        archive_root,
    )
    _restore_restart_atlas_year_inputs(
        previous_run_dir=archive_root,
        workspace=workspace,
        start_year=start_year,
        atlas_year=atlas_year,
    )

    missing_after = {
        key: path for key, path in required_paths.items() if not os.path.exists(path)
    }
    repaired_keys = [
        key for key in missing_before.keys() if key not in missing_after
    ]
    if repaired_keys:
        logger.info(
            "[RestartRepair] Restored restart-critical ATLAS inputs from archive: %s",
            repaired_keys,
        )
    if missing_after:
        logger.warning(
            "[RestartRepair] ATLAS restart year inputs still missing after archive "
            "rehydration: %s",
            sorted(missing_after),
        )
    return bool(repaired_keys)


def _supply_demand_manifest_path(
    *,
    run_dir: str,
    year: int,
    iteration: int,
) -> Path:
    return Path(os.path.realpath(run_dir)) / ".workflow" / f"year_{year}_iteration_{iteration}.yaml"


def _land_use_manifest_path(
    *,
    run_dir: str,
    year: int,
) -> Path:
    return Path(os.path.realpath(run_dir)) / ".workflow" / f"land_use_year_{year}.yaml"


def _atlas_subyear_manifest_path(
    *,
    run_dir: str,
    forecast_year: int,
    sub_year: int,
) -> Path:
    return (
        Path(os.path.realpath(run_dir))
        / ".workflow"
        / "vehicle_ownership"
        / f"forecast_year_{forecast_year}_subyear_{sub_year}.yaml"
    )


def _postprocessing_manifest_path(
    *,
    run_dir: str,
    year: int,
) -> Path:
    return Path(os.path.realpath(run_dir)) / ".workflow" / f"postprocessing_year_{year}.yaml"


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


def _atlas_manifest_targets(
    *,
    archive_run_dir: str,
    year: int,
    forecast_year: Optional[int],
    require_all_subyears: bool,
    require_complete_steps: bool,
    step_filter_override: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    sub_years = _atlas_sub_years(current_year=year, forecast_year=forecast_year)
    manifest_scope_year = forecast_year if forecast_year is not None else year

    atlas_dir = Path(os.path.realpath(archive_run_dir)) / ".workflow" / "vehicle_ownership"
    matched_by_sub_year: Dict[int, List[Path]] = {sub_year: [] for sub_year in sub_years}
    if atlas_dir.exists():
        for manifest_path in sorted(atlas_dir.glob("forecast_year_*_subyear_*.yaml")):
            match = _ATLAS_SUBYEAR_MANIFEST_RE.match(manifest_path.name)
            if match is None:
                continue
            manifest_forecast_year = _coerce_int(match.group("forecast_year"))
            manifest_sub_year = _coerce_int(match.group("sub_year"))
            if manifest_forecast_year is None or manifest_sub_year is None:
                continue
            if manifest_sub_year not in matched_by_sub_year:
                continue
            if manifest_forecast_year != manifest_scope_year:
                continue
            matched_by_sub_year[manifest_sub_year].append(manifest_path)

    targets: List[Dict[str, Any]] = []
    issues: List[Tuple[str, str]] = []
    if step_filter_override is not None:
        step_filter = set(step_filter_override)
    else:
        step_filter = set(_ATLAS_MANIFEST_STEPS) if require_complete_steps else None
    gap_detected = False
    for sub_year in sub_years:
        matching_paths = matched_by_sub_year.get(sub_year, [])
        manifest_path = sorted(matching_paths)[0] if matching_paths else None
        if manifest_path is not None and not gap_detected:
            targets.append(
                {
                    "path": manifest_path,
                    "steps": step_filter,
                }
            )
            continue
        gap_detected = True
        if not require_all_subyears:
            continue
        missing_path = _atlas_subyear_manifest_path(
            run_dir=archive_run_dir,
            forecast_year=manifest_scope_year,
            sub_year=sub_year,
        )
        issues.append((str(missing_path), "workflow manifest is missing"))

    return targets, issues


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
        is_atlas_target = target.get("stage") == "atlas"
        if atlas_gap_detected and is_atlas_target:
            if atlas_require_all_subyears:
                issues.append(
                    (
                        repr(target),
                        "required atlas restart target missing after contiguous-prefix gap",
                    )
                )
            unmatched_targets.append(dict(target))
            continue
        try:
            run = _find_latest_run_for_restart_target(tracker=tracker, target=target)
        except ValueError:
            unmatched_targets.append(dict(target))
            if is_atlas_target:
                atlas_gap_detected = True
                if atlas_require_all_subyears:
                    issues.append(
                        (
                            repr(target),
                            "no completed run found for required atlas restart target",
                        )
                    )
                continue
            issues.append((repr(target), "no completed run found for restart target"))
            continue
        except Exception as exc:
            issues.append(
                (
                    repr(target),
                    f"tracker query failed: {exc}",
                )
            )
            continue
        run_id = str(getattr(run, "id", "")).strip()
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        run_ids.append(run_id)
        matched_targets.append({**dict(target), "run_id": run_id})

    return {
        "run_ids": run_ids,
        "issues": issues,
        "manifest_paths": [],
        "query_targets": targets,
        "matched_query_targets": matched_targets,
        "unmatched_query_targets": unmatched_targets,
        "atlas_gap_detected": atlas_gap_detected,
        "fallback_reason": None,
        "discovery_mode": "tracker",
    }


def _restart_manifest_targets(
    *,
    state: Any,
    archive_run_dir: str,
    year: int,
    workflow_stage: Any,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    targets: List[Dict[str, Any]] = []
    issues: List[Tuple[str, str]] = []
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
    }:
        land_use_manifest_path = _land_use_manifest_path(
            run_dir=archive_run_dir,
            year=year,
        )
        if land_use_enabled is not False and (
            land_use_enabled is True or land_use_manifest_path.exists()
        ):
            targets.append(
                {
                    "path": land_use_manifest_path,
                    "steps": set(_LAND_USE_MANIFEST_STEPS),
                }
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
        atlas_step_filter: Optional[Set[str]] = None
        if current_stage in {
            workflow_stage.supply_demand_loop,
            workflow_stage.postprocessing,
        }:
            # Supply-demand only consumes the durable ATLAS handoff products
            # emitted by atlas_postprocess (for example vehicles2_* and the
            # updated UrbanSim datastore). Reconstructing atlas_preprocess/run
            # runs adds restart noise without improving recovery.
            atlas_step_filter = {"atlas_postprocess"}
        atlas_targets, atlas_issues = _atlas_manifest_targets(
            archive_run_dir=archive_run_dir,
            year=year,
            forecast_year=forecast_year,
            require_all_subyears=(
                current_stage in {workflow_stage.supply_demand_loop, workflow_stage.postprocessing}
                and vehicle_enabled is True
            ),
            require_complete_steps=current_stage
            in {workflow_stage.supply_demand_loop, workflow_stage.postprocessing},
            step_filter_override=atlas_step_filter,
        )
        targets.extend(atlas_targets)
        issues.extend(atlas_issues)

    if current_stage in {workflow_stage.supply_demand_loop, workflow_stage.postprocessing}:
        total_iters = max(
            _coerce_int(getattr(state, "_settings", {}).get("supply_demand_iters", None))
            or 0,
            0,
        )
        if current_stage == workflow_stage.postprocessing and total_iters > 0:
            for iteration in range(0, total_iters):
                targets.append(
                    {
                        "path": _supply_demand_manifest_path(
                            run_dir=archive_run_dir,
                            year=year,
                            iteration=iteration,
                        ),
                        "steps": None,
                    }
                )
        else:
            for iteration in range(0, resume_iter):
                targets.append(
                    {
                        "path": _supply_demand_manifest_path(
                            run_dir=archive_run_dir,
                            year=year,
                            iteration=iteration,
                        ),
                        "steps": None,
                    }
                )
            if current_sub_stage == workflow_stage.traffic_assignment:
                targets.append(
                    {
                        "path": _supply_demand_manifest_path(
                            run_dir=archive_run_dir,
                            year=year,
                            iteration=resume_iter,
                        ),
                        "steps": set(_ACTIVITYSIM_MANIFEST_STEPS),
                    }
                )
        if current_stage == workflow_stage.postprocessing:
            postprocessing_enabled = _is_stage_explicitly_enabled(
                state=state,
                stage=workflow_stage.postprocessing,
            )
            manifest_path = _postprocessing_manifest_path(
                run_dir=archive_run_dir,
                year=year,
            )
            if postprocessing_enabled is not False and (
                postprocessing_enabled is True or manifest_path.exists()
            ):
                targets.append(
                    {
                        "path": manifest_path,
                        "steps": set(_POSTPROCESSING_MANIFEST_STEPS),
                    }
                )

    return targets, issues


def _load_manifest_run_ids(
    *,
    manifest_path: Path,
    step_filter: Optional[Set[str]],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    issues: List[Tuple[str, str]] = []
    if not manifest_path.exists():
        issues.append((str(manifest_path), "workflow manifest is missing"))
        return [], issues
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_data = yaml.safe_load(handle) or {}
    except Exception as exc:
        issues.append((str(manifest_path), f"failed reading workflow manifest: {exc}"))
        return [], issues
    if not isinstance(manifest_data, dict):
        issues.append((str(manifest_path), "workflow manifest is not a mapping"))
        return [], issues

    run_ids: List[str] = []
    seen: Set[str] = set()
    for step_name, step_info in manifest_data.items():
        if step_filter and step_name not in step_filter:
            continue
        if not isinstance(step_info, dict):
            continue
        raw_run_id = step_info.get("run_id")
        run_id = str(raw_run_id).strip() if raw_run_id is not None else ""
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        run_ids.append(run_id)

    if step_filter:
        for step_name in sorted(step_filter):
            step_info = manifest_data.get(step_name)
            run_id = (
                str(step_info.get("run_id")).strip()
                if isinstance(step_info, dict) and step_info.get("run_id") is not None
                else ""
            )
            if not run_id:
                issues.append(
                    (
                        str(manifest_path),
                        f"required step '{step_name}' is missing a run_id",
                    )
                )
    return run_ids, issues


def collect_restart_completed_run_ids(
    *,
    state: Any,
    archive_run_dir: str,
    workflow_stage: Any,
    tracker: Any = None,
) -> Dict[str, Any]:
    tracker_discovery: Optional[Dict[str, Any]] = None
    if tracker is not None:
        tracker_discovery = _collect_restart_completed_run_ids_from_tracker(
            tracker=tracker,
            state=state,
            workflow_stage=workflow_stage,
        )
        if tracker_discovery["run_ids"] and not tracker_discovery["issues"]:
            shadow_discovery: Optional[Dict[str, Any]] = None
            year = _coerce_int(getattr(state, "current_year", None))
            if year is not None:
                targets, target_issues = _restart_manifest_targets(
                    state=state,
                    archive_run_dir=archive_run_dir,
                    year=year,
                    workflow_stage=workflow_stage,
                )
                all_run_ids: List[str] = []
                seen: Set[str] = set()
                issues: List[Tuple[str, str]] = list(target_issues)
                manifest_paths: List[str] = []
                seen_manifest_paths: Set[str] = set()
                for target in targets:
                    manifest_path = target["path"]
                    step_filter = target["steps"]
                    manifest_path_str = str(manifest_path)
                    if manifest_path_str in seen_manifest_paths:
                        continue
                    seen_manifest_paths.add(manifest_path_str)
                    manifest_paths.append(manifest_path_str)
                    run_ids, manifest_issues = _load_manifest_run_ids(
                        manifest_path=manifest_path,
                        step_filter=step_filter,
                    )
                    issues.extend(manifest_issues)
                    for run_id in run_ids:
                        if run_id in seen:
                            continue
                        seen.add(run_id)
                        all_run_ids.append(run_id)
                shadow_discovery = {
                    "run_ids": all_run_ids,
                    "issues": issues,
                    "manifest_paths": manifest_paths,
                }
            tracker_only_run_ids: List[str] = []
            manifest_only_run_ids: List[str] = []
            if shadow_discovery is not None:
                tracker_only_run_ids = sorted(
                    set(tracker_discovery["run_ids"]) - set(shadow_discovery["run_ids"])
                )
                manifest_only_run_ids = sorted(
                    set(shadow_discovery["run_ids"]) - set(tracker_discovery["run_ids"])
                )
                tracker_discovery["shadow_compare"] = {
                    "enabled": True,
                    "parity": not tracker_only_run_ids and not manifest_only_run_ids,
                    "manifest_run_ids": list(shadow_discovery["run_ids"]),
                    "tracker_only_run_ids": tracker_only_run_ids,
                    "manifest_only_run_ids": manifest_only_run_ids,
                    "manifest_issue_count": len(shadow_discovery["issues"]),
                    "manifest_path_count": len(shadow_discovery["manifest_paths"]),
                }
            else:
                tracker_discovery["shadow_compare"] = {
                    "enabled": False,
                    "parity": None,
                    "manifest_run_ids": [],
                    "tracker_only_run_ids": [],
                    "manifest_only_run_ids": [],
                    "manifest_issue_count": 0,
                    "manifest_path_count": 0,
                }
            return tracker_discovery
        logger.warning(
            "Restart completed-run discovery falling back to manifests after tracker query issues=%s run_ids=%s",
            tracker_discovery["issues"],
            tracker_discovery["run_ids"],
        )

    year = _coerce_int(getattr(state, "current_year", None))
    if year is None:
        return {
            "run_ids": [],
            "issues": [("state.current_year", "missing year")],
            "manifest_paths": [],
            "query_targets": [],
            "matched_query_targets": [],
            "unmatched_query_targets": [],
            "atlas_gap_detected": False,
            "fallback_reason": "state.current_year missing",
            "shadow_compare": {
                "enabled": False,
                "parity": None,
                "manifest_run_ids": [],
                "tracker_only_run_ids": [],
                "manifest_only_run_ids": [],
                "manifest_issue_count": 0,
                "manifest_path_count": 0,
            },
            "discovery_mode": "manifest",
        }

    targets, target_issues = _restart_manifest_targets(
        state=state,
        archive_run_dir=archive_run_dir,
        year=year,
        workflow_stage=workflow_stage,
    )

    all_run_ids: List[str] = []
    seen: Set[str] = set()
    issues: List[Tuple[str, str]] = list(target_issues)
    manifest_paths: List[str] = []
    seen_manifest_paths: Set[str] = set()
    for target in targets:
        manifest_path = target["path"]
        step_filter = target["steps"]
        manifest_path_str = str(manifest_path)
        if manifest_path_str in seen_manifest_paths:
            continue
        seen_manifest_paths.add(manifest_path_str)
        manifest_paths.append(manifest_path_str)
        run_ids, manifest_issues = _load_manifest_run_ids(
            manifest_path=manifest_path,
            step_filter=step_filter,
        )
        issues.extend(manifest_issues)
        for run_id in run_ids:
            if run_id in seen:
                continue
            seen.add(run_id)
            all_run_ids.append(run_id)

    return {
        "run_ids": all_run_ids,
        "issues": issues,
        "manifest_paths": manifest_paths,
        "query_targets": [],
        "matched_query_targets": [],
        "unmatched_query_targets": [],
        "atlas_gap_detected": bool(
            tracker_discovery and tracker_discovery.get("atlas_gap_detected")
        ),
        "fallback_reason": (
            None
            if tracker_discovery is None
            else (
                "tracker returned no run_ids"
                if not tracker_discovery.get("run_ids")
                else "tracker discovery had issues"
            )
        ),
        "shadow_compare": {
            "enabled": False,
            "parity": None,
            "manifest_run_ids": [],
            "tracker_only_run_ids": [],
            "manifest_only_run_ids": [],
            "manifest_issue_count": 0,
            "manifest_path_count": len(manifest_paths),
        },
        "discovery_mode": "manifest",
    }


def reconstruct_restart_completed_run_outputs(
    *,
    tracker: Any,
    state: Any,
    local_run_dir: str,
    archive_run_dir: str,
    workflow_stage: Any,
) -> Dict[str, Any]:
    discovery = collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=archive_run_dir,
        workflow_stage=workflow_stage,
        tracker=tracker,
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
        "shadow_compare": dict(discovery.get("shadow_compare", {}) or {}),
        "materialization_result": aggregate,
    }


def hydrate_restart_local_bookkeeping(
    *,
    archive_run_dir: str,
    local_run_dir: str,
    archive_state_path: str,
    local_state_path: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "state_mirror": "skipped_existing",
        "workflow_files_copied": 0,
        "skipped_existing": 0,
        "missing_source": 0,
        "failed": [],
    }

    archive_state = Path(os.path.realpath(archive_state_path))
    local_state = Path(os.path.realpath(local_state_path))
    try:
        summary["state_mirror"] = _copy_restart_bookkeeping_file(
            source=archive_state,
            target=local_state,
        )
    except Exception as exc:
        summary["failed"].append(
            {
                "source": str(archive_state),
                "target": str(local_state),
                "error": str(exc),
            }
        )
        logger.warning(
            "Restart bookkeeping hydration failed copying local state mirror %s -> %s: %s",
            archive_state,
            local_state,
            exc,
        )

    workflow_source_root = Path(os.path.realpath(archive_run_dir)) / ".workflow"
    workflow_target_root = Path(os.path.realpath(local_run_dir)) / ".workflow"
    if not workflow_source_root.exists():
        summary["missing_source"] += 1
        return summary

    for source_path in sorted(workflow_source_root.rglob("*")):
        if not source_path.is_file():
            continue
        target_path = workflow_target_root / source_path.relative_to(workflow_source_root)
        try:
            outcome = _copy_restart_bookkeeping_file(
                source=source_path,
                target=target_path,
            )
        except Exception as exc:
            summary["failed"].append(
                {
                    "source": str(source_path),
                    "target": str(target_path),
                    "error": str(exc),
                }
            )
            logger.warning(
                "Restart bookkeeping hydration failed copying workflow file %s -> %s: %s",
                source_path,
                target_path,
                exc,
            )
            continue
        if outcome == "copied":
            summary["workflow_files_copied"] += 1
        elif outcome == "skipped_existing":
            summary["skipped_existing"] += 1
        elif outcome == "missing_source":
            summary["missing_source"] += 1

    return summary


def log_resume_doctor_check(
    *,
    check: str,
    ok: bool,
    detail: str,
) -> None:
    status = "ok" if ok else "missing"
    log_fn = logger.info if ok else logger.warning
    log_fn("[ResumeDoctor] check=%s status=%s %s", check, status, detail)


def run_resume_doctor_diagnostics(
    *,
    state: Any,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
    archive_state_path: str,
    local_state_path: str,
    local_consist_db_path: Optional[str],
    restart_missing_artifacts_initial: List[Dict[str, str]],
    restart_missing_artifacts_after_recovery: List[Dict[str, str]],
    snapshot_latest_dir_fn: Callable[[str], Path],
    build_manifest_path_fn: Callable[..., Path],
    format_missing_artifact_summary_fn: Callable[[List[Dict[str, str]]], str] = format_missing_artifact_summary,
    log_resume_doctor_check_fn: Callable[..., None] = log_resume_doctor_check,
    restart_reconstruction: Optional[Dict[str, Any]] = None,
) -> None:
    degraded_checks: List[str] = []

    def record(check: str, ok: bool, detail: str, *, required: bool = True) -> None:
        log_resume_doctor_check_fn(check=check, ok=ok, detail=detail)
        if required and not ok:
            degraded_checks.append(check)

    logger.info(
        "[ResumeDoctor] start year=%s iteration=%s local_run_dir=%s archive_run_dir=%s",
        state.current_year,
        state.current_inner_iter,
        local_run_dir,
        archive_run_dir,
    )

    archive_state_real = os.path.realpath(archive_state_path)
    local_state_real = os.path.realpath(local_state_path)
    record("archive_run_state", os.path.exists(archive_state_real), f"path={archive_state_real}")
    record("local_run_state_mirror", os.path.exists(local_state_real), f"path={local_state_real}")

    if local_consist_db_path:
        local_consist_db_real = os.path.realpath(local_consist_db_path)
        record("local_consist_db", os.path.exists(local_consist_db_real), f"path={local_consist_db_real}")
        latest_snapshot_db = snapshot_latest_dir_fn(archive_run_dir) / Path(local_consist_db_real).name
        latest_snapshot_real = os.path.realpath(str(latest_snapshot_db))
        record("archive_latest_consist_db_snapshot", os.path.exists(latest_snapshot_real), f"path={latest_snapshot_real}")
    else:
        record("local_consist_db", True, "path=none reason=disabled_or_unconfigured", required=False)
        record("archive_latest_consist_db_snapshot", True, "path=none reason=disabled_or_unconfigured", required=False)

    if state.data_initialized:
        missing_summary = format_missing_artifact_summary_fn(restart_missing_artifacts_after_recovery)
        record(
            "required_restart_local_artifacts",
            not restart_missing_artifacts_after_recovery,
            "data_initialized=true "
            f"initial_missing={len(restart_missing_artifacts_initial)} "
            f"remaining_missing={len(restart_missing_artifacts_after_recovery)} "
            f"missing={missing_summary}",
        )
    else:
        record(
            "required_restart_local_artifacts",
            True,
            "data_initialized=false reason=bootstrap_required",
            required=False,
        )

    if restart_reconstruction:
        reconstruction_result = restart_reconstruction.get("materialization_result")
        run_ids = restart_reconstruction.get("run_ids", [])
        complete = bool(getattr(reconstruction_result, "complete", False))
        summary = (
            getattr(reconstruction_result, "summary", "unavailable")
            if reconstruction_result is not None
            else "missing result"
        )
        record(
            "completed_run_reconstruction",
            complete,
            f"run_ids={len(run_ids)} summary={summary}",
            required=False,
        )
    else:
        record(
            "completed_run_reconstruction",
            True,
            "disabled_or_not_applicable",
            required=False,
        )

    year = state.current_year
    iteration = state.current_inner_iter
    local_manifest_path: Optional[Path] = None
    try:
        local_manifest_path = build_manifest_path_fn(
            workspace=workspace,
            year=int(year),
            iteration=int(iteration),
        )
    except Exception as exc:
        record("supply_demand_manifest_local", False, f"year={year} iteration={iteration} error={exc}")
        record("supply_demand_manifest_archive", False, f"year={year} iteration={iteration} error=local_manifest_path_unavailable")

    if local_manifest_path is not None:
        local_manifest_real = os.path.realpath(str(local_manifest_path))
        record(
            "supply_demand_manifest_local",
            os.path.exists(local_manifest_real),
            f"year={year} iteration={iteration} path={local_manifest_real}",
        )
        archive_manifest_real = os.path.realpath(
            str(
                _supply_demand_manifest_path(
                    run_dir=archive_run_dir,
                    year=int(year),
                    iteration=int(iteration),
                )
            )
        )
        record(
            "supply_demand_manifest_archive",
            os.path.exists(archive_manifest_real),
            f"year={year} iteration={iteration} path={archive_manifest_real}",
        )

    if degraded_checks:
        logger.warning(
            "[ResumeDoctor] summary status=degraded reason=missing_checks:%s",
            ",".join(degraded_checks),
        )
    else:
        logger.info("[ResumeDoctor] summary status=ready reason=all_checks_ok")
