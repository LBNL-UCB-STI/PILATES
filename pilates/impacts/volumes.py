from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from pilates.config.models import ImpactsConfig
from pilates.utils.path_utils import find_project_root
from pilates.utils.settings_helper import get as get_setting
from pilates.workspace import Workspace


def _normalize_container_path(path: str) -> str:
    normalized = path.rstrip("/") or "/"
    if not normalized.startswith("/"):
        raise ValueError(f"Container path must be absolute: {path!r}")
    return normalized


def _resolve_isrm_host_path(path: str, *, workspace: Workspace) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate.resolve()

    workspace_candidate = Path(workspace.full_path) / candidate
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    project_root = find_project_root(start_path=workspace.full_path)
    if project_root:
        project_candidate = Path(project_root) / candidate
        if project_candidate.exists():
            return project_candidate.resolve()

    return workspace_candidate.resolve()


def _require_existing_path(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Impacts mount {label} does not exist: {path}")


def volumes_manifest(volumes: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """Serialize Docker volume dict for run manifests."""
    manifest: List[Dict[str, str]] = []
    for host_path in sorted(volumes):
        spec = volumes[host_path]
        manifest.append(
            {
                "host": host_path,
                "container": spec["bind"],
                "mode": spec.get("mode", "rw"),
            }
        )
    return manifest


def build_impacts_docker_volumes(
    *,
    workspace: Workspace,
    cfg: ImpactsConfig,
) -> Dict[str, Dict[str, str]]:
    """
    Build Docker volume mounts for impacts container execution.

    Host paths come from the active PILATES workspace (same resolution as BEAM
    runner/preprocessor): mutable BEAM input/output copied into the workspace,
    impacts staged input/output, and the configured ISRM zarr store.
    """
    region = get_setting(workspace.settings, "run.region")
    beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_output_dir = Path(workspace.get_beam_output_dir())
    impacts_input_dir = Path(workspace.get_impacts_input_dir())
    impacts_output_dir = Path(workspace.get_impacts_output_dir())

    mount_specs: List[Tuple[Path, str, str]] = [
        (impacts_input_dir, cfg.container_input_folder, "rw"),
        (impacts_output_dir, cfg.container_output_folder, "rw"),
        (beam_input_dir, cfg.container_beam_input_folder, "rw"),
        (beam_output_dir, cfg.container_beam_output_folder, "rw"),
        (
            _resolve_isrm_host_path(cfg.isrm_source_directory, workspace=workspace),
            cfg.container_isrm_path,
            "ro",
        ),
    ]

    _require_existing_path(impacts_input_dir, label="impacts input")
    _require_existing_path(impacts_output_dir, label="impacts output")
    _require_existing_path(beam_input_dir, label="beam input")
    _require_existing_path(beam_output_dir, label="beam output")
    _require_existing_path(mount_specs[-1][0], label="isrm zarr")

    if region:
        beam_region_input = beam_input_dir / region
        _require_existing_path(
            beam_region_input,
            label=f"beam input region ({region})",
        )

    volumes: Dict[str, Dict[str, str]] = {}
    for host_path, container_path, mode in mount_specs:
        resolved_host = str(host_path.resolve())
        normalized_container = _normalize_container_path(container_path)
        if resolved_host in volumes:
            continue
        volumes[resolved_host] = {"bind": normalized_container, "mode": mode}

    return volumes
