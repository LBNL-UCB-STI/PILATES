from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from consist import MaterializationResult

from pilates.runtime.cache_recovery import materialize_cached_runs
from pilates.utils.io import get_traffic_assignment_model

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
_ATLAS_MANIFEST_STEPS = (
    "atlas_preprocess",
    "atlas_run",
    "atlas_postprocess",
)
_ATLAS_SUBYEAR_MANIFEST_RE = re.compile(
    r"^forecast_year_(?P<forecast_year>-?\d+)_subyear_(?P<sub_year>-?\d+)\.yaml$"
)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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

    current_stage = getattr(state, "current_major_stage", None)
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    requires_usim_base_h5 = (
        getattr(model_cfg, "land_use", None) == "urbansim"
        or getattr(model_cfg, "activity_demand", None) == "activitysim"
    )
    if requires_usim_base_h5:
        usim_data_dir = workspace.get_usim_mutable_data_dir()
        usim_base_fname = get_usim_datastore_fname_fn(settings, io="input")
        required.append(
            {
                "key": "usim_datastore_base_h5",
                "path": os.path.join(usim_data_dir, usim_base_fname),
                "reason": "UrbanSim base datastore required for downstream restart inputs",
            }
        )

    region = getattr(getattr(settings, "run", None), "region", None)
    urbansim_cfg = getattr(settings, "urbansim", None)
    requires_urbansim_run_locals = current_stage in {
        None,
        workflow_stage.land_use,
    }
    if requires_urbansim_run_locals and region and urbansim_cfg is not None:
        region_id = (
            getattr(urbansim_cfg, "region_mappings", {})
            .get("region_to_region_id", {})
            .get(region)
        )
        if region_id:
            required.extend(
                [
                    {
                        "key": "omx_skims",
                        "path": os.path.join(usim_data_dir, f"skims_mpo_{region_id}.omx"),
                        "reason": "UrbanSim run requires mutable OMX skims in the local workspace",
                    },
                    {
                        "key": "hh_size",
                        "path": os.path.join(usim_data_dir, f"hsize_ct_{region_id}.csv"),
                        "reason": "UrbanSim run requires household-size lookup data in the local workspace",
                    },
                    {
                        "key": "income_rates",
                        "path": os.path.join(usim_data_dir, f"income_rates_{region_id}.csv"),
                        "reason": "UrbanSim run requires income-rate lookup data in the local workspace",
                    },
                    {
                        "key": "relmap",
                        "path": os.path.join(usim_data_dir, f"relmap_{region_id}.csv"),
                        "reason": "UrbanSim run requires relationship-mapping data in the local workspace",
                    },
                    {
                        "key": "schools",
                        "path": os.path.join(usim_data_dir, "schools_2010.csv"),
                        "reason": "UrbanSim run requires schools lookup data in the local workspace",
                    },
                    {
                        "key": "school_districts",
                        "path": os.path.join(usim_data_dir, "blocks_school_districts_2010.csv"),
                        "reason": "UrbanSim run requires school-district lookup data in the local workspace",
                    },
                ]
            )

    requires_activitysim_locals = (
        getattr(model_cfg, "activity_demand", None) == "activitysim"
        and (
            current_stage is None
            or current_stage
            in {
                workflow_stage.supply_demand_loop,
                workflow_stage.activity_demand,
                workflow_stage.activity_demand_directly_from_land_use,
            }
        )
    )
    if requires_activitysim_locals:
        asim_configs_dir = workspace.get_asim_mutable_configs_dir()
        main_configs_dir = (
            getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
            or "configs"
        )
        for dirname in required_asim_config_dirs_fn(main_configs_dir):
            required.append(
                {
                    "key": f"activitysim_config_settings_yaml_{dirname}",
                    "path": os.path.join(asim_configs_dir, dirname, "settings.yaml"),
                    "reason": (
                        "ActivitySim mutable config tree required on restart "
                        f"(config_dir={dirname})"
                    ),
                }
            )

    requires_activitysim_zarr = (
        getattr(model_cfg, "activity_demand", None) == "activitysim"
        and bool(getattr(state, "asim_compiled", False))
        and current_stage
        in {
            workflow_stage.supply_demand_loop,
            workflow_stage.activity_demand,
            workflow_stage.activity_demand_directly_from_land_use,
            workflow_stage.traffic_assignment,
        }
    )
    get_asim_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if requires_activitysim_zarr and callable(get_asim_output_dir):
        required.append(
            {
                "key": "zarr_skims",
                "path": os.path.join(get_asim_output_dir(), "cache", "skims.zarr"),
                "reason": "ActivitySim compiled skims required for resumed supply-demand loop",
            }
        )
        current_sub_stage = getattr(state, "current_sub_stage", None)
        if current_stage == workflow_stage.supply_demand_loop and (
            current_sub_stage == workflow_stage.traffic_assignment
        ):
            required.extend(
                _activitysim_iteration_output_requirements(
                    asim_output_dir=get_asim_output_dir(),
                    year=getattr(state, "current_year", "unknown"),
                    iteration=getattr(state, "current_inner_iter", 0),
                )
            )

    requires_beam_locals = (
        get_traffic_assignment_model(settings) == "beam"
        and current_stage
        in {
            workflow_stage.supply_demand_loop,
            workflow_stage.traffic_assignment,
        }
    )
    get_beam_input_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
    if requires_beam_locals and callable(get_beam_input_dir) and region:
        beam_input_dir = get_beam_input_dir()
        required.append(
            {
                "key": "beam_mutable_data_dir",
                "path": beam_input_dir,
                "reason": (
                    "BEAM mutable data root required for restart metadata and "
                    "resumed traffic assignment"
                ),
            }
        )
        required.append(
            {
                "key": "beam_region_input_dir",
                "path": os.path.join(beam_input_dir, region),
                "reason": (
                    "BEAM mutable input tree required for resumed traffic assignment"
                ),
            }
        )
        beam_cfg = getattr(settings, "beam", None)
        beam_config_name = getattr(beam_cfg, "config", None)
        if beam_config_name:
            required.append(
                {
                    "key": "beam_primary_config_file",
                    "path": os.path.join(beam_input_dir, region, beam_config_name),
                    "reason": (
                        "BEAM primary config required for resumed traffic assignment"
                    ),
                }
            )

    requires_atlas_locals = (
        getattr(model_cfg, "vehicle_ownership", None) == "atlas"
        and current_stage == workflow_stage.vehicle_ownership_model
    )
    get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if requires_atlas_locals and callable(get_atlas_input_dir):
        atlas_input_dir = get_atlas_input_dir()
        for relpath in atlas_static_input_relpaths_fn(settings):
            required.append(
                {
                    "key": f"atlas_static::{relpath}",
                    "path": os.path.join(atlas_input_dir, relpath),
                    "reason": "ATLAS static input required during vehicle ownership restart",
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
        if not os.path.exists(path):
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
    step_filter = (
        set(_ATLAS_MANIFEST_STEPS) if require_complete_steps else None
    )
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
        current_stage in {workflow_stage.vehicle_ownership_model, workflow_stage.supply_demand_loop}
        and vehicle_enabled is not False
    ):
        atlas_targets, atlas_issues = _atlas_manifest_targets(
            archive_run_dir=archive_run_dir,
            year=year,
            forecast_year=forecast_year,
            require_all_subyears=(
                current_stage == workflow_stage.supply_demand_loop
                and vehicle_enabled is True
            ),
            require_complete_steps=current_stage == workflow_stage.supply_demand_loop,
        )
        targets.extend(atlas_targets)
        issues.extend(atlas_issues)

    if current_stage == workflow_stage.supply_demand_loop:
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
) -> Dict[str, Any]:
    year = _coerce_int(getattr(state, "current_year", None))
    if year is None:
        return {"run_ids": [], "issues": [("state.current_year", "missing year")], "manifest_paths": []}

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

    return {
        "run_ids": run_ids,
        "source_root": source_root,
        "target_root": target_root,
        "manifest_paths": list(discovery["manifest_paths"]),
        "materialization_result": aggregate,
    }


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
