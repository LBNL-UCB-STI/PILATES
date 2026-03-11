"""
Restart bundle manifest helpers.

This module defines a manifest under ``.restart/bundle_manifest.yaml`` in each
archive run directory. The manifest enumerates a targeted set of artifacts that
can be used to rehydrate ephemeral node-local workspaces for restart/resume.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from pilates.atlas.inputs import atlas_static_input_relpaths
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.consist_db_snapshot import snapshot_latest_dir

logger = logging.getLogger(__name__)

RESTART_BUNDLE_MANIFEST_REL_PATH = os.path.join(".restart", "bundle_manifest.yaml")


def restart_bundle_manifest_path(archive_run_dir: str) -> str:
    return os.path.join(os.path.realpath(archive_run_dir), RESTART_BUNDLE_MANIFEST_REL_PATH)


def _safe_relpath(path: str, root: str) -> Optional[str]:
    abs_path = os.path.realpath(path)
    abs_root = os.path.realpath(root)
    try:
        if os.path.commonpath([abs_path, abs_root]) != abs_root:
            return None
    except ValueError:
        return None
    rel = os.path.relpath(abs_path, abs_root)
    if rel.startswith(".."):
        return None
    return rel


def _append_artifact(
    artifacts: List[Dict[str, Any]],
    *,
    key: str,
    rel_path: Optional[str],
    reason: str,
    archive_run_dir: str,
) -> None:
    if not rel_path:
        return
    archive_path = os.path.join(archive_run_dir, rel_path)
    kind = "dir" if os.path.isdir(archive_path) else "file"
    artifacts.append(
        {
            "key": key,
            "rel_path": rel_path,
            "reason": reason,
            "kind": kind,
            "exists_in_archive": os.path.exists(archive_path),
        }
    )


def _append_local_candidate(
    artifacts: List[Dict[str, Any]],
    *,
    key: str,
    local_path: str,
    reason: str,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    rel = _safe_relpath(local_path, local_run_dir)
    _append_artifact(
        artifacts,
        key=key,
        rel_path=rel,
        reason=reason,
        archive_run_dir=archive_run_dir,
    )


def _add_activitysim_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "activity_demand", None) != "activitysim":
        return

    get_data_dir = getattr(workspace, "get_asim_mutable_data_dir", None)
    get_cfg_dir = getattr(workspace, "get_asim_mutable_configs_dir", None)
    if not callable(get_data_dir) or not callable(get_cfg_dir):
        return

    main_configs_dir = (
        getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
        or "configs"
    )
    for dirname in required_asim_config_dirs(main_configs_dir):
        _append_local_candidate(
            artifacts,
            key=f"activitysim_config_dir_{dirname}",
            local_path=os.path.join(get_cfg_dir(), dirname),
            reason=f"ActivitySim restart config directory ({dirname})",
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )

    get_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if callable(get_output_dir):
        _append_local_candidate(
            artifacts,
            key="zarr_skims",
            local_path=os.path.join(get_output_dir(), "cache", "skims.zarr"),
            reason="ActivitySim compiled skims cache for restart",
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )

    _append_local_candidate(
        artifacts,
        key=ASIM_SHARROW_CACHE_DIR,
        local_path=os.path.join(local_run_dir, "shared_cache", "numba"),
        reason="ActivitySim persisted numba/sharrow cache directory for faster restart",
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )


def _add_urbansim_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace: Any,
    state: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    activity_demand_model = getattr(model_cfg, "activity_demand", None)
    land_use_model = getattr(model_cfg, "land_use", None)
    if land_use_model != "urbansim" and activity_demand_model != "activitysim":
        return

    get_usim_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    if not callable(get_usim_dir):
        return

    usim_dir = get_usim_dir()
    if not usim_dir:
        return
    try:
        base_fname = get_usim_datastore_fname(settings, io="input")
    except Exception:
        base_fname = None
    if base_fname:
        _append_local_candidate(
            artifacts,
            key="usim_datastore_base_h5",
            local_path=os.path.join(usim_dir, base_fname),
            reason="UrbanSim restart base datastore",
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )

    if land_use_model == "urbansim":
        current_year = getattr(state, "current_year", None)
        try:
            current_fname = get_usim_datastore_fname(
                settings, io="output", year=current_year
            )
        except Exception:
            current_fname = None
        if current_fname:
            _append_local_candidate(
                artifacts,
                key="usim_datastore_current_h5",
                local_path=os.path.join(usim_dir, current_fname),
                reason="UrbanSim current-year datastore (if present)",
                local_run_dir=local_run_dir,
                archive_run_dir=archive_run_dir,
            )


def _add_beam_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "traffic_assignment", None) != "beam":
        return

    get_beam_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
    region = getattr(getattr(settings, "run", None), "region", None)
    if not callable(get_beam_dir) or not region:
        return

    _append_local_candidate(
        artifacts,
        key="beam_region_input_dir",
        local_path=os.path.join(get_beam_dir(), region),
        reason=f"BEAM mutable input directory for region {region}",
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )


def _add_atlas_year_dir_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return
    for getter_name, prefix in (
        ("get_atlas_mutable_input_dir", "atlas_input"),
        ("get_atlas_output_dir", "atlas_output"),
    ):
        getter = getattr(workspace, getter_name, None)
        if not callable(getter):
            continue
        local_base = getter()
        if not local_base:
            continue
        rel = _safe_relpath(local_base, local_run_dir)
        if not rel:
            continue
        archive_base = os.path.join(archive_run_dir, rel)
        if not os.path.isdir(archive_base):
            continue
        for entry in sorted(os.listdir(archive_base)):
            if not entry.startswith("year"):
                continue
            archive_year_dir = os.path.join(archive_base, entry)
            if not os.path.isdir(archive_year_dir):
                continue
            _append_artifact(
                artifacts,
                key=f"{prefix}_{entry}",
                rel_path=_safe_relpath(archive_year_dir, archive_run_dir),
                reason="ATLAS year directory",
                archive_run_dir=archive_run_dir,
            )


def _add_atlas_static_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> None:
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return
    getter = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if not callable(getter):
        return
    atlas_input_dir = getter()
    for relpath in atlas_static_input_relpaths(settings):
        _append_local_candidate(
            artifacts,
            key=f"atlas_static::{relpath}",
            local_path=os.path.join(atlas_input_dir, relpath),
            reason="ATLAS static input required for restart",
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )


def _add_workflow_manifest_candidates(
    artifacts: List[Dict[str, Any]],
    *,
    archive_run_dir: str,
) -> None:
    workflow_dir = os.path.join(archive_run_dir, ".workflow")
    if os.path.isdir(workflow_dir):
        _append_artifact(
            artifacts,
            key="workflow_manifests",
            rel_path=".workflow",
            reason="Workflow step manifests for restart/resume",
            archive_run_dir=archive_run_dir,
        )


def _dedupe_artifacts(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for artifact in artifacts:
        rel_path = artifact.get("rel_path")
        key = artifact.get("key")
        dedupe_key = (str(key), str(rel_path))
        if not rel_path or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(artifact)
    return deduped


def build_restart_bundle_manifest(
    *,
    archive_run_dir: str,
    local_run_dir: str,
    settings: Any,
    workspace: Any,
    state: Any,
    local_consist_db_path: Optional[str],
) -> Dict[str, Any]:
    archive_root = os.path.realpath(archive_run_dir)
    artifacts: List[Dict[str, Any]] = []

    _append_artifact(
        artifacts,
        key="run_state",
        rel_path="run_state.yaml",
        reason="Workflow restart state",
        archive_run_dir=archive_root,
    )

    latest_snapshot = snapshot_latest_dir(archive_root)
    if latest_snapshot.exists():
        _append_artifact(
            artifacts,
            key="latest_consist_snapshot",
            rel_path=_safe_relpath(str(latest_snapshot), archive_root),
            reason="Latest Consist DB snapshot",
            archive_run_dir=archive_root,
        )

    if local_consist_db_path:
        _append_local_candidate(
            artifacts,
            key="local_consist_db",
            local_path=local_consist_db_path,
            reason="Primary local Consist DB path",
            local_run_dir=local_run_dir,
            archive_run_dir=archive_root,
        )

    _add_activitysim_candidates(
        artifacts,
        settings=settings,
        workspace=workspace,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_root,
    )
    _add_beam_candidates(
        artifacts,
        settings=settings,
        workspace=workspace,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_root,
    )
    _add_urbansim_candidates(
        artifacts,
        settings=settings,
        workspace=workspace,
        state=state,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_root,
    )
    _add_atlas_year_dir_candidates(
        artifacts,
        settings=settings,
        workspace=workspace,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_root,
    )
    _add_atlas_static_candidates(
        artifacts,
        settings=settings,
        workspace=workspace,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_root,
    )
    _add_workflow_manifest_candidates(artifacts, archive_run_dir=archive_root)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "artifacts": _dedupe_artifacts(artifacts),
    }


def write_restart_bundle_manifest(*, archive_run_dir: str, manifest: Dict[str, Any]) -> str:
    manifest_path = restart_bundle_manifest_path(archive_run_dir)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return manifest_path


def load_restart_bundle_manifest(archive_run_dir: str) -> Optional[Dict[str, Any]]:
    manifest_path = restart_bundle_manifest_path(archive_run_dir)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(
            "Failed reading restart bundle manifest at %s: %s",
            manifest_path,
            exc,
        )
    return None


def manifest_entries_to_local_artifacts(
    *,
    manifest: Optional[Dict[str, Any]],
    local_run_dir: str,
) -> List[Dict[str, str]]:
    if not isinstance(manifest, dict):
        return []

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []

    local_root = os.path.realpath(local_run_dir)
    mapped: List[Dict[str, str]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("rel_path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        local_path = os.path.realpath(os.path.join(local_root, rel_path))
        try:
            if os.path.commonpath([local_path, local_root]) != local_root:
                continue
        except ValueError:
            continue
        mapped.append(
            {
                "key": str(item.get("key") or rel_path),
                "path": local_path,
                "reason": str(item.get("reason") or "bundle manifest entry"),
                "kind": str(item.get("kind") or "file"),
            }
        )
    return mapped
