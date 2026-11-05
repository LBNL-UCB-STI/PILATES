import os
import logging
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)

from pilates.utils import consist_runtime as cr
from pilates.utils.archive_paths import (
    WorkspaceRootLike,
    resolve_workspace_uri_path as _resolve_workspace_uri_path,
    workspace_storage_roots as _workspace_storage_roots,
)
from pilates.utils.consist_types import ArtifactLike

logger = logging.getLogger(__name__)


def _resolve_artifact_source_workspace_path(value: Any) -> Optional[str]:
    container_uri = getattr(value, "container_uri", None) or getattr(value, "uri", None)
    if not isinstance(container_uri, str) or not container_uri.startswith(
        "workspace://"
    ):
        return None
    meta = getattr(value, "meta", None)
    if not isinstance(meta, Mapping):
        return None
    mount_root = meta.get("mount_root")
    if not mount_root:
        return None
    rel_path = container_uri[len("workspace://") :].lstrip("/")
    return os.path.abspath(os.path.join(str(mount_root), rel_path))


def _materialization_roots(
    workspace: Optional["Workspace"],
) -> tuple[Optional[str], Optional[str]]:
    if (roots := _workspace_storage_roots(workspace=workspace)) is not None:
        return roots
    if isinstance(workspace, WorkspaceRootLike) and workspace.full_path:
        return os.path.abspath(str(workspace.full_path)), None
    return None, None


from consist.models.artifact import get_tracker_ref


@runtime_checkable
class _MaterializableTrackerLike(Protocol):
    def materialize_artifact(
        self,
        artifact: Any,
        *,
        target_root: str,
        source_root: Optional[str] = None,
        preserve_existing: bool = True,
        on_missing: str = "warn",
        validate_content_hash: str = "if-present",
    ) -> Any: ...


@runtime_checkable
class _MaterializationResultLike(Protocol):
    path: Any


def _materialize_artifact_to_existing_path(
    value: Any,
    *,
    workspace: Optional["Workspace"],
) -> Optional[str]:
    if not isinstance(value, ArtifactLike):
        return None
    target_root, source_root = _materialization_roots(workspace)
    if target_root is None:
        return None
    try:
        tracker: Any = cr.current_tracker()
    except Exception:
        tracker = None
    if not isinstance(tracker, _MaterializableTrackerLike):
        tracker_ref = get_tracker_ref(value)
        if tracker_ref is not None:
            candidate_tracker = tracker_ref() if callable(tracker_ref) else tracker_ref
            if isinstance(candidate_tracker, _MaterializableTrackerLike):
                tracker = candidate_tracker
    if not isinstance(tracker, _MaterializableTrackerLike):
        return None
    try:
        result = tracker.materialize_artifact(
            value,
            target_root=target_root,
            source_root=source_root,
            preserve_existing=True,
            on_missing="warn",
            validate_content_hash="if-present",
        )
    except Exception:
        logger.debug(
            "[Archive] Consist artifact materialization failed for %s",
            value.key,
            exc_info=True,
        )
        return None
    if not isinstance(result, _MaterializationResultLike) or result.path is None:
        return None
    resolved_path = os.path.abspath(os.fspath(result.path))
    if os.path.exists(resolved_path):
        return resolved_path
    return None


if TYPE_CHECKING:
    from pilates.workspace import Workspace


def artifact_to_path(
    value: Any, workspace: Optional["Workspace"] = None
) -> Optional[str]:
    """
    Resolve an artifact-like object or path into a concrete filesystem path.

    Parameters
    ----------
    value : object
        Artifact-like object or path value to resolve.
    workspace : Workspace, optional
        Workspace used to resolve relative paths.

    Returns
    -------
    str or None
        Resolved filesystem path, or None if value is None.
    """
    if value is None:
        return None
    container_uri = getattr(value, "container_uri", None) or getattr(value, "uri", None)
    if isinstance(container_uri, str) and container_uri.startswith("workspace://"):
        rel_path = container_uri[len("workspace://") :].lstrip("/")
        if workspace is not None and getattr(workspace, "full_path", None):
            current_workspace_path = os.path.join(str(workspace.full_path), rel_path)
            if os.path.exists(current_workspace_path):
                return current_workspace_path
        source_workspace_path = _resolve_artifact_source_workspace_path(value)
        if source_workspace_path and os.path.exists(source_workspace_path):
            return source_workspace_path
        if workspace is not None and getattr(workspace, "full_path", None):
            return os.path.join(str(workspace.full_path), rel_path)
    path = getattr(value, "path", None) or container_uri or value
    if isinstance(path, Path):
        path = os.fspath(path)
    elif isinstance(path, os.PathLike):
        path = os.fspath(path)
    if isinstance(path, str) and workspace is not None and not os.path.isabs(path):
        if "://" not in path:
            return os.path.join(workspace.full_path, path)
    return path


def resolve_existing_path(
    path: Optional[str],
    *,
    workspace: Optional["Workspace"] = None,
) -> Optional[str]:
    """
    Resolve a path or workspace URI to a local path that currently exists.
    """
    if not path:
        return None
    resolved = _resolve_workspace_uri_path(path, workspace=workspace)
    if not resolved:
        return None
    if "://" in resolved:
        return None
    abs_path = os.path.abspath(resolved)
    if os.path.exists(abs_path):
        logger.debug("[Archive] Using local path: %s", abs_path)
        return abs_path
    logger.debug("[Archive] Unable to resolve existing path for: %s", abs_path)
    return None


def artifact_to_existing_path(
    value: Any,
    workspace: Optional["Workspace"] = None,
) -> Optional[str]:
    """
    Resolve an artifact-like value to an existing local path.
    """
    path = artifact_to_path(value, workspace=workspace)
    if path is None and isinstance(value, str):
        path = value
    resolved = resolve_existing_path(path, workspace=workspace)
    if resolved is not None:
        return resolved
    source_workspace_path = _resolve_artifact_source_workspace_path(value)
    if source_workspace_path and os.path.exists(source_workspace_path):
        return source_workspace_path
    materialized_path = _materialize_artifact_to_existing_path(
        value, workspace=workspace
    )
    if materialized_path:
        return materialized_path
    return None
