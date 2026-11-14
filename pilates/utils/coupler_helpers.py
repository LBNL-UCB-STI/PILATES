import os
import logging
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    TYPE_CHECKING,
)

from consist import (
    Artifact,
    ArtifactLike as ConsistArtifactLike,
    MaterializedArtifact,
    Tracker,
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
    if not isinstance(value, ArtifactLike):
        return None
    uri = value.uri
    if not isinstance(uri, str) or not uri.startswith("workspace://"):
        return None
    if not isinstance(value.meta, Mapping):
        return None
    mount_root = value.meta.get("mount_root")
    if not mount_root:
        return None
    rel_path = uri[len("workspace://") :].lstrip("/")
    return os.path.abspath(os.path.join(str(mount_root), rel_path))


def _materialization_roots(
    workspace: Optional["Workspace"],
) -> tuple[Optional[str], Optional[str]]:
    if (roots := _workspace_storage_roots(workspace=workspace)) is not None:
        return roots
    if isinstance(workspace, WorkspaceRootLike) and workspace.full_path:
        return os.path.abspath(str(workspace.full_path)), None
    return None, None


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
    if tracker is None and isinstance(value, Artifact):
        tracker = value.resolver
    active_tracker = tracker if isinstance(tracker, Tracker) else None
    if isinstance(value, Artifact):
        try:
            resolved = value.ensure_local_path(
                tracker=active_tracker,
                target_root=target_root,
                source_root=source_root,
                on_missing="warn",
            )
            if resolved is not None:
                return os.path.abspath(os.fspath(resolved))
        except Exception:
            logger.debug(
                "[Archive] Consist artifact materialization failed for %s",
                value.key,
                exc_info=True,
            )
            return None
    if active_tracker is None:
        return None
    try:
        result = active_tracker.materialize_artifact(
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
    if not isinstance(result, MaterializedArtifact) or result.path is None:
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
    uri = None
    if isinstance(value, ConsistArtifactLike):
        uri = value.uri
    elif isinstance(value, str) and "://" in value:
        uri = value

    ws_full_path = (
        str(workspace.full_path)
        if isinstance(workspace, WorkspaceRootLike) and workspace.full_path
        else None
    )

    if isinstance(uri, str) and uri.startswith("workspace://"):
        rel_path = uri[len("workspace://") :].lstrip("/")
        if ws_full_path is not None:
            current_workspace_path = os.path.join(ws_full_path, rel_path)
            if os.path.exists(current_workspace_path):
                return current_workspace_path
        source_workspace_path = _resolve_artifact_source_workspace_path(value)
        if source_workspace_path and os.path.exists(source_workspace_path):
            return source_workspace_path
        if ws_full_path is not None:
            return os.path.join(ws_full_path, rel_path)

    if isinstance(value, ConsistArtifactLike):
        path = value.path
    elif uri:
        path = uri
    else:
        path = value

    if isinstance(path, (Path, os.PathLike)):
        path = os.fspath(path)
    if isinstance(path, str) and not os.path.isabs(path) and ws_full_path is not None:
        if "://" not in path:
            return os.path.join(ws_full_path, path)
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
