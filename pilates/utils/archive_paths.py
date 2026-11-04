from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
    runtime_checkable,
)

from pilates.utils import consist_runtime as cr

if TYPE_CHECKING:
    from pilates.workspace import Workspace

PathValue = TypeVar("PathValue", str, Path)
PathLike = Union[str, os.PathLike[str]]


@runtime_checkable
class ArchiveStateLike(Protocol):
    @property
    def run_info_path(self) -> Optional[PathLike]: ...


@runtime_checkable
class WorkspaceRootLike(Protocol):
    @property
    def full_path(self) -> PathLike: ...


@runtime_checkable
class ArchiveRunDirLike(Protocol):
    @property
    def archive_run_dir(self) -> Optional[PathLike]: ...


@runtime_checkable
class _TrackerMountsLike(Protocol):
    @property
    def mounts(self) -> Mapping[str, Any]: ...


@runtime_checkable
class _TrackerRunDirLike(Protocol):
    @property
    def run_dir(self) -> Optional[PathLike]: ...


def _non_empty_path(val: object) -> Optional[Path]:
    if isinstance(val, (str, os.PathLike)) and str(val).strip():
        return Path(os.fspath(val)).expanduser()
    return None


def resolve_archive_run_dir(
    *,
    workspace: object = None,
    state: object = None,
) -> Optional[Path]:
    """Resolve the canonical archive run directory from explicit state or workspace."""
    if isinstance(state, ArchiveRunDirLike):
        if (state_archive := _non_empty_path(state.archive_run_dir)) is not None:
            return state_archive.resolve()
    if isinstance(workspace, ArchiveRunDirLike):
        if (ws_archive := _non_empty_path(workspace.archive_run_dir)) is not None:
            return ws_archive.resolve()
    if isinstance(state, ArchiveStateLike):
        if (run_info_path := _non_empty_path(state.run_info_path)) is not None:
            try:
                return run_info_path.resolve().parent
            except Exception:
                return None
    return None


def workspace_storage_roots(
    *,
    workspace: object = None,
    state: object = None,
) -> Optional[tuple[str, str]]:
    """Return ``(local_root, archive_root)`` resolved from workspace, state, or active tracker."""
    local_root: Optional[str] = None
    if isinstance(workspace, WorkspaceRootLike):
        if (full_path := _non_empty_path(workspace.full_path)) is not None:
            local_root = os.path.abspath(full_path)

    archive_dir = resolve_archive_run_dir(workspace=workspace, state=state)
    archive_root: Optional[str] = str(archive_dir) if archive_dir is not None else None

    if (not local_root or not archive_root) and workspace is None:
        try:
            tracker = cr.current_tracker()
        except Exception:
            tracker = None
        if tracker is not None:
            if not local_root and isinstance(tracker, _TrackerMountsLike):
                mounts = tracker.mounts
                if isinstance(mounts, Mapping):
                    if (
                        ws_mount := _non_empty_path(mounts.get("workspace"))
                    ) is not None:
                        local_root = os.path.abspath(ws_mount)
            if not archive_root and isinstance(tracker, _TrackerRunDirLike):
                if (run_dir_val := _non_empty_path(tracker.run_dir)) is not None:
                    archive_root = os.path.abspath(run_dir_val)

    if not local_root or not archive_root:
        return None
    return local_root, archive_root


def resolve_workspace_uri_path(
    path: str,
    workspace: Optional["Workspace"] = None,
) -> Optional[str]:
    if not isinstance(path, str):
        return None
    prefix = "workspace://"
    if not path.startswith(prefix):
        return path
    rel_path = path[len(prefix) :].lstrip("/")
    if isinstance(workspace, WorkspaceRootLike):
        if _non_empty_path(workspace.full_path) is not None:
            return os.path.join(os.fspath(workspace.full_path), rel_path)
    if workspace is None:
        try:
            tracker = cr.current_tracker()
        except Exception:
            tracker = None
        if isinstance(tracker, _TrackerMountsLike) and isinstance(
            tracker.mounts, Mapping
        ):
            if (
                ws_mount := _non_empty_path(tracker.mounts.get("workspace"))
            ) is not None:
                return os.path.join(os.path.abspath(ws_mount), rel_path)
    return None


def archive_fallback_path(
    *,
    state: object,
    workspace: object,
    local_path: PathLike,
) -> Optional[Path]:
    archive_run_dir = resolve_archive_run_dir(workspace=workspace, state=state)
    full_path = (
        _non_empty_path(workspace.full_path)
        if isinstance(workspace, WorkspaceRootLike)
        else None
    )
    if archive_run_dir is None or full_path is None:
        return None
    local_root = full_path.resolve()
    try:
        rel = Path(local_path).expanduser().resolve().relative_to(local_root)
    except Exception:
        return None
    return archive_run_dir / rel


def first_existing_path(*paths: Optional[PathValue]) -> Optional[PathValue]:
    for path in paths:
        if path is None:
            continue
        if isinstance(path, Path):
            if path.exists():
                return path
            continue
        if os.path.exists(path):
            return path
    return None
