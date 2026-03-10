from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


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
) -> List[Dict[str, str]]:
    """
    Build a pragmatic set of local artifacts that must exist to safely skip bootstrap.
    """
    required: List[Dict[str, str]] = []

    usim_data_dir = workspace.get_usim_mutable_data_dir()
    usim_base_fname = get_usim_datastore_fname_fn(settings, io="input")
    required.append(
        {
            "key": "usim_datastore_base_h5",
            "path": os.path.join(usim_data_dir, usim_base_fname),
            "reason": "UrbanSim base datastore required for downstream restart inputs",
        }
    )

    current_stage = getattr(state, "current_major_stage", None)
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

    model_cfg = getattr(getattr(settings, "run", None), "models", None)
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
        asim_data_dir = workspace.get_asim_mutable_data_dir()
        for filename in ("households.csv", "persons.csv", "land_use.csv"):
            required.append(
                {
                    "key": f"activitysim_input_{filename}",
                    "path": os.path.join(asim_data_dir, filename),
                    "reason": "ActivitySim mutable input required on restart",
                }
            )
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
    raw = getattr(run_cfg, "restart_rehydrate_mode", "bundle")
    mode = str(raw).strip().lower() if raw is not None else "bundle"
    if mode in {"bundle", "full", "off"}:
        return mode
    logger.warning(
        "Unknown run.restart_rehydrate_mode=%r; defaulting to 'bundle'.",
        raw,
    )
    return "bundle"


def is_restart_strict(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "restart_strict", False))


def read_archive_run_state_year(
    state_path: str,
    *,
    read_current_stage_fn: Callable[[str], tuple[Any, ...]],
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


def map_local_path_to_archive(
    *,
    local_path: str,
    local_run_dir: str,
    archive_run_dir: str,
) -> Optional[str]:
    local_abs = os.path.realpath(local_path)
    local_root = os.path.realpath(local_run_dir)
    archive_root = os.path.realpath(archive_run_dir)
    try:
        if os.path.commonpath([local_abs, local_root]) != local_root:
            return None
    except ValueError:
        return None
    rel = os.path.relpath(local_abs, local_root)
    return os.path.join(archive_root, rel)


def copy_archive_entry_preserve_existing(
    *,
    archive_path: str,
    local_path: str,
) -> Tuple[int, int]:
    copied = 0
    skipped_existing = 0

    if os.path.isdir(archive_path):
        for root, _, files in os.walk(archive_path):
            rel_root = os.path.relpath(root, archive_path)
            dest_root = local_path if rel_root == "." else os.path.join(local_path, rel_root)
            os.makedirs(dest_root, exist_ok=True)
            for filename in files:
                src = os.path.join(root, filename)
                dest = os.path.join(dest_root, filename)
                if os.path.exists(dest):
                    skipped_existing += 1
                    continue
                shutil.copy2(src, dest)
                copied += 1
        return copied, skipped_existing

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return 0, 1
    shutil.copy2(archive_path, local_path)
    return 1, 0


def rehydrate_missing_local_artifacts_from_archive(
    *,
    missing_artifacts: List[Dict[str, str]],
    local_run_dir: str,
    archive_run_dir: str,
    map_local_path_to_archive_fn: Callable[..., Optional[str]] = map_local_path_to_archive,
    copy_archive_entry_fn: Callable[..., Tuple[int, int]] = copy_archive_entry_preserve_existing,
) -> Dict[str, int]:
    summary = {
        "copied": 0,
        "skipped_existing": 0,
        "skipped_missing_archive": 0,
        "skipped_unmapped": 0,
        "copy_errors": 0,
    }
    for artifact in missing_artifacts:
        local_path = os.path.realpath(artifact["path"])
        key = artifact.get("key", "unknown")
        kind = artifact.get("kind", "file")

        if os.path.exists(local_path) and not (kind == "dir" and os.path.isdir(local_path)):
            summary["skipped_existing"] += 1
            logger.info("[RestartRehydrate] Skip existing local artifact key=%s path=%s", key, local_path)
            continue

        archive_path = map_local_path_to_archive_fn(
            local_path=local_path,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )
        if archive_path is None:
            summary["skipped_unmapped"] += 1
            logger.warning("[RestartRehydrate] Cannot map local path to archive key=%s path=%s", key, local_path)
            continue
        archive_path = os.path.realpath(archive_path)
        if not os.path.exists(archive_path):
            summary["skipped_missing_archive"] += 1
            logger.warning("[RestartRehydrate] Archive source missing key=%s archive_path=%s", key, archive_path)
            continue

        try:
            copied, skipped_existing = copy_archive_entry_fn(
                archive_path=archive_path,
                local_path=local_path,
            )
            summary["copied"] += copied
            summary["skipped_existing"] += skipped_existing
            logger.info(
                "[RestartRehydrate] key=%s copied=%s skipped_existing=%s source=%s dest=%s",
                key,
                copied,
                skipped_existing,
                archive_path,
                local_path,
            )
        except Exception as exc:
            summary["copy_errors"] += 1
            logger.warning(
                "[RestartRehydrate] Failed copy key=%s source=%s dest=%s error=%s",
                key,
                archive_path,
                local_path,
                exc,
            )

    logger.info(
        "[RestartRehydrate] Summary copied=%s skipped_existing=%s skipped_missing_archive=%s skipped_unmapped=%s copy_errors=%s",
        summary["copied"],
        summary["skipped_existing"],
        summary["skipped_missing_archive"],
        summary["skipped_unmapped"],
        summary["copy_errors"],
    )
    return summary


def rehydrate_full_local_run_from_archive(
    *,
    local_run_dir: str,
    archive_run_dir: str,
    copy_archive_entry_fn: Callable[..., Tuple[int, int]] = copy_archive_entry_preserve_existing,
) -> Dict[str, int]:
    summary = {
        "copied": 0,
        "skipped_existing": 0,
        "skipped_missing_archive": 0,
        "skipped_unmapped": 0,
        "copy_errors": 0,
    }
    archive_root = os.path.realpath(archive_run_dir)
    if not os.path.exists(archive_root):
        summary["skipped_missing_archive"] = 1
        logger.warning("[RestartRehydrate] Full mode archive root missing: %s", archive_root)
        return summary

    try:
        copied, skipped_existing = copy_archive_entry_fn(
            archive_path=archive_root,
            local_path=os.path.realpath(local_run_dir),
        )
        summary["copied"] = copied
        summary["skipped_existing"] = skipped_existing
    except Exception as exc:
        summary["copy_errors"] = 1
        logger.warning(
            "[RestartRehydrate] Full mode copy failed source=%s dest=%s error=%s",
            archive_root,
            os.path.realpath(local_run_dir),
            exc,
        )
    return summary


def rehydrate_bundle_local_artifacts_from_archive(
    *,
    bundle_manifest: Optional[Dict[str, Any]],
    local_run_dir: str,
    archive_run_dir: str,
    manifest_entries_to_local_artifacts_fn: Callable[..., List[Dict[str, str]]],
    rehydrate_missing_local_artifacts_fn: Callable[..., Dict[str, int]] = rehydrate_missing_local_artifacts_from_archive,
) -> Dict[str, int]:
    bundle_artifacts = manifest_entries_to_local_artifacts_fn(
        manifest=bundle_manifest,
        local_run_dir=local_run_dir,
    )
    if not bundle_artifacts:
        logger.warning("[RestartRehydrate] Bundle mode found no manifest artifacts to hydrate.")
        return {
            "copied": 0,
            "skipped_existing": 0,
            "skipped_missing_archive": 0,
            "skipped_unmapped": 0,
            "copy_errors": 0,
        }
    return rehydrate_missing_local_artifacts_fn(
        missing_artifacts=bundle_artifacts,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )


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
    restart_missing_artifacts_after_rehydrate: List[Dict[str, str]],
    snapshot_latest_dir_fn: Callable[[str], Path],
    build_manifest_path_fn: Callable[..., Path],
    map_local_path_to_archive_fn: Callable[..., Optional[str]] = map_local_path_to_archive,
    format_missing_artifact_summary_fn: Callable[[List[Dict[str, str]]], str] = format_missing_artifact_summary,
    log_resume_doctor_check_fn: Callable[..., None] = log_resume_doctor_check,
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
        missing_summary = format_missing_artifact_summary_fn(restart_missing_artifacts_after_rehydrate)
        record(
            "required_restart_local_artifacts",
            not restart_missing_artifacts_after_rehydrate,
            "data_initialized=true "
            f"initial_missing={len(restart_missing_artifacts_initial)} "
            f"remaining_missing={len(restart_missing_artifacts_after_rehydrate)} "
            f"missing={missing_summary}",
        )
    else:
        record(
            "required_restart_local_artifacts",
            True,
            "data_initialized=false reason=bootstrap_required",
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
        record("supply_demand_manifest_archive_mapped", False, f"year={year} iteration={iteration} error=local_manifest_path_unavailable")

    if local_manifest_path is not None:
        local_manifest_real = os.path.realpath(str(local_manifest_path))
        record(
            "supply_demand_manifest_local",
            os.path.exists(local_manifest_real),
            f"year={year} iteration={iteration} path={local_manifest_real}",
        )
        archive_manifest_path = map_local_path_to_archive_fn(
            local_path=local_manifest_real,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )
        if archive_manifest_path is None:
            record(
                "supply_demand_manifest_archive_mapped",
                False,
                f"year={year} iteration={iteration} local_path={local_manifest_real} archive_path=unmapped",
            )
        else:
            archive_manifest_real = os.path.realpath(archive_manifest_path)
            record(
                "supply_demand_manifest_archive_mapped",
                os.path.exists(archive_manifest_real),
                f"year={year} iteration={iteration} local_path={local_manifest_real} archive_path={archive_manifest_real}",
            )

    if degraded_checks:
        logger.warning(
            "[ResumeDoctor] summary status=degraded reason=missing_checks:%s",
            ",".join(degraded_checks),
        )
    else:
        logger.info("[ResumeDoctor] summary status=ready reason=all_checks_ok")
