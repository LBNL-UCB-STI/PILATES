from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    from pilates.workspace import Workspace

ARCHIVE_LOCAL_ENV = "PILATES_LOCAL_RUN_DIR"
ARCHIVE_ROOT_ENV = "PILATES_ARCHIVE_RUN_DIR"

PathValue = TypeVar("PathValue", str, Path)
PathLike = Union[str, os.PathLike[str]]


class ArchiveStateLike(Protocol):
    @property
    def run_info_path(self) -> Optional[PathLike]: ...


class WorkspaceRootLike(Protocol):
    @property
    def full_path(self) -> PathLike: ...


def archive_roots() -> Optional[tuple[str, str]]:
    local_root = os.environ.get(ARCHIVE_LOCAL_ENV)
    archive_root = os.environ.get(ARCHIVE_ROOT_ENV)
    if not local_root or not archive_root:
        return None
    return os.path.abspath(local_root), os.path.abspath(archive_root)


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
    if workspace is not None and getattr(workspace, "full_path", None):
        return os.path.join(str(workspace.full_path), rel_path)
    roots = archive_roots()
    if roots is None:
        return None
    local_root, _archive_root = roots
    return os.path.join(local_root, rel_path)


def archive_fallback_path(
    *,
    state: ArchiveStateLike,
    workspace: WorkspaceRootLike,
    local_path: PathLike,
) -> Optional[Path]:
    run_info_path = getattr(state, "run_info_path", None)
    full_path = getattr(workspace, "full_path", None)
    if not run_info_path or full_path is None:
        return None
    archive_run_dir = Path(run_info_path).expanduser().resolve().parent
    local_root = Path(full_path).expanduser().resolve()
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
