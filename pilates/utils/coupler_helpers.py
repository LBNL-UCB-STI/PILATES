import os
import logging
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    TYPE_CHECKING,
)

from pilates.utils import consist_runtime as cr
from pilates.utils.archive_paths import (
    archive_roots as _archive_roots,
    resolve_workspace_uri_path as _resolve_workspace_uri_path,
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
    roots = _archive_roots()
    source_root = os.path.abspath(roots[1]) if roots is not None else None
    if workspace is not None and getattr(workspace, "full_path", None):
        return os.path.abspath(str(workspace.full_path)), source_root
    if roots is None:
        return None, None
    local_root, _archive_root = roots
    return os.path.abspath(local_root), source_root


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
        tracker = cr.current_tracker()
    except Exception:
        tracker = None
    materialize_artifact = getattr(tracker, "materialize_artifact", None)
    if not callable(materialize_artifact):
        return None
    try:
        result = materialize_artifact(
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
            getattr(value, "key", value),
            exc_info=True,
        )
        return None
    materialized_path = getattr(result, "path", None)
    if materialized_path is None:
        return None
    resolved_path = os.path.abspath(os.fspath(materialized_path))
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


def resolve_artifact_from_value(
    value: Any,
    *,
    key: Optional[str] = None,
    workspace: Optional["Workspace"] = None,
) -> Any:
    """
    Resolve a coupler value to a Consist Artifact when possible.

    If the value is already an Artifact, it is returned unchanged. If it is a
    path-like value, we attempt to locate a previously-logged Artifact by container URI
    (without re-hashing). Falls back to the original value when unavailable.
    """
    if value is None:
        return None
    if isinstance(value, ArtifactLike):
        if key and getattr(value, "key", None) != key:
            value.key = key
        return value

    path = artifact_to_path(value, workspace)
    if not path:
        return value

    tracker = cr.current_tracker()
    if tracker is None or getattr(tracker, "db", None) is None:
        return value

    try:
        if "://" in str(path):
            container_uri = str(path)
        else:
            abs_path = os.path.abspath(path)
            container_uri = tracker.fs.virtualize_path(abs_path)
        artifact = tracker.db.find_latest_artifact_at_uri(
            container_uri,
            include_inputs=True,
        )
        if artifact is None:
            return value
        if key and getattr(artifact, "key", None) != key:
            artifact.key = key
        return artifact
    except Exception:
        return value


def log_coupler_value(
    *,
    key: str,
    value: Any,
    workspace: Optional["Workspace"] = None,
    context: str = "coupler",
) -> None:
    """
    Debug helper to show whether a coupler value is an Artifact or raw path.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    value_type = type(value).__name__
    has_container_uri = isinstance(value, ArtifactLike)
    has_key = hasattr(value, "key")
    path = artifact_to_path(value, workspace)

    tracker = cr.current_tracker()
    container_uri = None
    db_hit = None
    if has_container_uri:
        container_uri = getattr(value, "container_uri", None)
    elif path:
        if tracker is not None:
            if "://" in str(path):
                container_uri = str(path)
            else:
                container_uri = tracker.fs.virtualize_path(os.path.abspath(path))
        else:
            container_uri = str(path)

    if (
        tracker is not None
        and getattr(tracker, "db", None) is not None
        and container_uri
    ):
        try:
            db_hit = (
                tracker.db.find_latest_artifact_at_uri(
                    container_uri,
                    include_inputs=True,
                )
                is not None
            )
        except Exception:
            db_hit = None

    logger.debug(
        "[CouplerDebug] %s key=%s type=%s has_container_uri=%s has_key=%s path=%s container_uri=%s db_hit=%s",
        context,
        key,
        value_type,
        has_container_uri,
        has_key,
        path,
        container_uri,
        db_hit,
    )
