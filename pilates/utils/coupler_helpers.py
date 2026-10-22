import os
import logging
import atexit
import fnmatch
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    TYPE_CHECKING,
    Union,
)

from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import ArtifactLike, CouplerProtocol
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.workflows.coupler_namespace import (
    ResolvedCouplerValue,
    canonical_artifact_key_from_raw_key,
    namespaced_view_target,
    resolve_coupler_value,
)
from pilates.runtime.archive_paths import (
    archive_roots as _archive_roots,
    resolve_archive_path as _resolve_archive_path,
    resolve_workspace_uri_path as _resolve_workspace_uri_path,
)
from pilates.runtime import recovery_root_adoption

logger = logging.getLogger(__name__)

_ARCHIVE_ENABLE_ENV = "PILATES_ENABLE_ARCHIVE_COPY"
_ARCHIVE_ALLOWED_DIR_PATTERNS = (
    "urbansim_bootstrap_data_root",
    "beam_mutable_data_dir",
    "beam_output_dir",
    "beam_region_input_dir",
    "beam_input_config_references_archived",
    "zarr_skims",
    "zarr_skims_*",
    "raw_od_skims_zarr_*",
    "asim_output_dir",
    "asim_mutable_data_dir",
    "activitysim_bootstrap_data_root",
    "activitysim_bootstrap_configs_root",
    "atlas_input_year_dir",
    "atlas_input_year_dir_*",
)
_archive_queue: Optional["queue.Queue[Optional[str]]"] = None
_archive_thread: Optional[threading.Thread] = None
_archive_lock = threading.Lock()
_archive_pending_tasks: Dict[
    str, tuple[str, str, str, bool, Optional[tuple[int, int, bool]]]
] = {}
_archive_queued_destinations: set[str] = set()
_archive_inflight_signature: Dict[str, tuple[int, int, bool]] = {}
_archive_last_copied_signature: Dict[str, tuple[int, int, bool]] = {}


def _is_noop_artifact(candidate: Any) -> bool:
    if candidate is None:
        return False
    candidate_type = type(candidate)
    candidate_name = getattr(candidate_type, "__name__", "").lower()
    candidate_module = getattr(candidate_type, "__module__", "").lower()
    return "noopartifact" in candidate_name or ".noop" in candidate_module


def _archive_enabled() -> bool:
    value = os.environ.get(_ARCHIVE_ENABLE_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


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


def _archive_dir_allowed(key: str) -> bool:
    return any(
        fnmatch.fnmatch(key, pattern) for pattern in _ARCHIVE_ALLOWED_DIR_PATTERNS
    )


def _resolve_archive_copy_target(
    *,
    key: str,
    path: Optional[Union[str, os.PathLike]],
    workspace: Optional["Workspace"] = None,
) -> Optional[tuple[str, str, bool, Optional[tuple[int, int, bool]]]]:
    """
    Resolve and validate an archive copy request.

    Both async enqueueing and synchronous copies use the same target selection
    and guardrails so path guessing stays centralized.
    """
    if path is None:
        return None

    resolved = _resolve_workspace_uri_path(os.fspath(path), workspace=workspace)
    if not resolved or "://" in resolved:
        return None
    if not os.path.exists(resolved):
        logger.warning(
            "[Archive] Output path does not exist: %s (key=%s)", resolved, key
        )
        return None

    is_dir = os.path.isdir(resolved)
    if is_dir and not _archive_dir_allowed(key):
        logger.warning(
            "[Archive] Skipping directory output (not allowlisted): %s (key=%s)",
            resolved,
            key,
        )
        return None

    roots = _archive_roots()
    if roots is None:
        return None
    local_root, archive_root = roots
    dest = _resolve_archive_path(resolved, local_root, archive_root)
    if dest is None or dest == resolved:
        return None
    signature = _archive_path_signature(resolved, is_dir)
    return resolved, dest, is_dir, signature


def _archive_path_signature(path: str, is_dir: bool) -> Optional[tuple[int, int, bool]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (int(stat.st_size), int(stat.st_mtime_ns), bool(is_dir))


def _archive_log_method(key: str) -> Callable[..., None]:
    del key
    return logger.info


def _emit_artifact_lifecycle_event(*_args: Any, **_kwargs: Any) -> None:
    """Compatibility hook retained after lifecycle audit emission removal."""
    return None


def _register_phase2_recovery_root_copies() -> int:
    return recovery_root_adoption.adopt_pending_recovery_root_copies(
        find_artifact=lambda key, path: _find_current_run_output_artifact(
            key=key,
            path=path,
        ),
        emit_lifecycle_event=_emit_artifact_lifecycle_event,
    )


def _copy_archive_payload(
    *,
    key: str,
    src: str,
    dest: str,
    is_dir: bool,
    signature: Optional[tuple[int, int, bool]],
) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if is_dir:
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    if signature is not None:
        with _archive_lock:
            if _archive_inflight_signature.get(dest) == signature:
                _archive_inflight_signature.pop(dest, None)
            _archive_last_copied_signature[dest] = signature
    recovery_root_adoption.record_archive_copy(
        key=key,
        src=src,
        dest=dest,
        is_dir=is_dir,
        signature=signature,
    )
    _archive_log_method(key)("[Archive] Copied %s -> %s (key=%s)", src, dest, key)


def _archive_worker() -> None:
    while True:
        task = _archive_queue.get() if _archive_queue is not None else None
        if task is None:
            if _archive_queue is not None:
                _archive_queue.task_done()
            break
        dest = task
        with _archive_lock:
            _archive_queued_destinations.discard(dest)
            payload = _archive_pending_tasks.pop(dest, None)
            if payload is not None:
                signature = payload[-1]
                if signature is not None:
                    _archive_inflight_signature[dest] = signature
        if payload is None:
            if _archive_queue is not None:
                _archive_queue.task_done()
            continue
        key, src, dest, is_dir, signature = payload
        try:
            _copy_archive_payload(
                key=key,
                src=src,
                dest=dest,
                is_dir=is_dir,
                signature=signature,
            )
        except Exception as exc:
            if signature is not None:
                with _archive_lock:
                    if _archive_inflight_signature.get(dest) == signature:
                        _archive_inflight_signature.pop(dest, None)
            logger.warning(
                "[Archive] Failed to copy %s -> %s (key=%s): %s",
                src,
                dest,
                key,
                exc,
            )
        finally:
            if _archive_queue is not None:
                _archive_queue.task_done()


def _ensure_archive_worker() -> None:
    if not _archive_enabled():
        return
    with _archive_lock:
        global _archive_queue, _archive_thread
        if _archive_queue is None:
            _archive_queue = queue.Queue()
        if _archive_thread is None or not _archive_thread.is_alive():
            _archive_thread = threading.Thread(
                target=_archive_worker, name="pilates-archiver", daemon=True
            )
            _archive_thread.start()


def _enqueue_archive_copy(
    key: str,
    path: Optional[Union[str, os.PathLike]],
    workspace: Optional["Workspace"] = None,
) -> None:
    if not _archive_enabled():
        return
    prepared = _resolve_archive_copy_target(key=key, path=path, workspace=workspace)
    if prepared is None:
        return
    resolved, dest, is_dir, signature = prepared
    should_queue = False
    if signature is not None:
        with _archive_lock:
            pending_task = _archive_pending_tasks.get(dest)
            pending_signature = pending_task[-1] if pending_task is not None else None
            if pending_signature == signature:
                logger.debug(
                    "[Archive] Skipping duplicate enqueue (pending): %s (key=%s)",
                    resolved,
                    key,
                )
                return
            if _archive_inflight_signature.get(dest) == signature:
                logger.debug(
                    "[Archive] Skipping duplicate enqueue (in-flight): %s (key=%s)",
                    resolved,
                    key,
                )
                return
            if _archive_last_copied_signature.get(dest) == signature:
                logger.debug(
                    "[Archive] Skipping duplicate enqueue (already copied): %s (key=%s)",
                    resolved,
                    key,
                )
                return
            _archive_pending_tasks[dest] = (key, resolved, dest, is_dir, signature)
            if dest not in _archive_queued_destinations:
                _archive_queued_destinations.add(dest)
                should_queue = True
    else:
        with _archive_lock:
            _archive_pending_tasks[dest] = (key, resolved, dest, is_dir, signature)
            if dest not in _archive_queued_destinations:
                _archive_queued_destinations.add(dest)
                should_queue = True
    _ensure_archive_worker()
    if should_queue and _archive_queue is not None:
        _archive_log_method(key)(
            "[Archive] Enqueued %s -> %s (key=%s)", resolved, dest, key
        )
        _archive_queue.put(dest)
    else:
        logger.debug(
            "[Archive] Coalesced pending enqueue %s -> %s (key=%s)",
            resolved,
            dest,
            key,
        )


def enqueue_archive_copy(
    *,
    key: str,
    path: Optional[Union[str, os.PathLike]],
    workspace: Optional["Workspace"] = None,
) -> None:
    """
    Public wrapper to enqueue archive copy work for a file or allowlisted directory.

    Parameters
    ----------
    key : str
        Artifact key associated with ``path``.
    path : str or PathLike, optional
        Local or workspace URI path to enqueue.
    workspace : Workspace, optional
        Workspace used to resolve ``workspace://`` paths.
    """
    _enqueue_archive_copy(key, path, workspace=workspace)


def flush_archive_queue(
    timeout: Optional[float] = None,
    *,
    fail_on_timeout: bool = False,
) -> bool:
    if _archive_queue is None:
        _register_phase2_recovery_root_copies()
        return True
    pending = _archive_queue.unfinished_tasks
    logger.info("[Archive] Flushing queue (pending=%s)", pending)
    if timeout is None:
        _archive_queue.join()
        logger.info("[Archive] Flush complete")
        _register_phase2_recovery_root_copies()
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _archive_queue.unfinished_tasks == 0:
            logger.info("[Archive] Flush complete")
            _register_phase2_recovery_root_copies()
            return True
        time.sleep(0.1)
    message = f"[Archive] Flush timed out (pending={_archive_queue.unfinished_tasks})"
    if fail_on_timeout:
        raise TimeoutError(message)
    logger.warning(message)
    return False


def stop_archive_worker(timeout: Optional[float] = None) -> None:
    if _archive_queue is None:
        return
    if _archive_thread is None:
        return
    logger.info("[Archive] Stopping worker")
    _archive_queue.put(None)
    if timeout is None:
        _archive_thread.join()
        logger.info("[Archive] Worker stopped")
        return
    _archive_thread.join(timeout=timeout)
    if _archive_thread.is_alive():
        logger.warning("[Archive] Worker stop timed out")
    else:
        logger.info("[Archive] Worker stopped")


atexit.register(stop_archive_worker)

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


def _normalize_artifact_path_for_match(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = _resolve_workspace_uri_path(path)
    if not resolved:
        return None
    if "://" in resolved:
        return resolved
    return os.path.abspath(resolved)


def _find_current_run_output_artifact(*, key: str, path: str) -> Optional[Any]:
    tracker = cr.current_tracker()
    current_consist = getattr(tracker, "current_consist", None) if tracker else None
    if current_consist is None:
        return None

    target_path = _normalize_artifact_path_for_match(path)
    if target_path is None:
        return None

    for artifact in list(getattr(current_consist, "outputs", []) or []):
        if getattr(artifact, "key", None) != key:
            continue
        artifact_path = _normalize_artifact_path_for_match(artifact_to_path(artifact))
        if artifact_path == target_path:
            return artifact
    return None


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


def resolve_input_precedence(
    *,
    key: str,
    coupler: Optional[CouplerProtocol],
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
) -> ResolvedCouplerValue:
    """
    Resolve one input key using canonical precedence.

    Precedence is: explicit input -> coupler value -> fallback input.
    """
    if explicit_inputs is not None and key in explicit_inputs:
        value = explicit_inputs.get(key)
        if value is not None:
            return ResolvedCouplerValue(
                requested_key=key,
                canonical_key=canonical_artifact_key_from_raw_key(key),
                storage_key=None,
                value=value,
                source="explicit",
            )

    resolved = resolve_coupler_value(coupler, key)
    if resolved.value is not None:
        return resolved

    if fallback_inputs is not None and key in fallback_inputs:
        value = fallback_inputs.get(key)
        if value is not None:
            return ResolvedCouplerValue(
                requested_key=key,
                canonical_key=canonical_artifact_key_from_raw_key(key),
                storage_key=None,
                value=value,
                source="fallback",
            )

    return ResolvedCouplerValue(
        requested_key=key,
        canonical_key=canonical_artifact_key_from_raw_key(key),
        storage_key=None,
        value=None,
        source="missing",
    )


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


def set_coupler_from_artifact(
    coupler: CouplerProtocol,
    key: str,
    artifact: Optional[Any],
    fallback: Optional[str] = None,
) -> None:
    """
    Set a coupler key from an artifact, falling back to a raw path if needed.

    Parameters
    ----------
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    key : str
        Coupler key to set.
    artifact : object
        Artifact-like object returned by logging helpers.
    fallback : str, optional
        Path to set if artifact is None.
    """
    if artifact is None and fallback is None:
        return
    canonical_key = resolve_artifact_key(key)
    value = artifact
    if _is_noop_artifact(value) and fallback is not None:
        value = fallback
    elif value is None:
        value = fallback

    def _preserve_artifact_identity(candidate: Any) -> bool:
        if candidate is None or isinstance(candidate, (str, os.PathLike)):
            return False
        if _is_noop_artifact(candidate):
            return False
        return any(
            getattr(candidate, attr_name, None) is not None
            for attr_name in ("id", "uri", "container_uri", "hash")
        )

    def _set_value(target: Any, target_key: str) -> None:
        set_from_artifact = getattr(target, "set_from_artifact", None)
        if callable(set_from_artifact):
            set_from_artifact(target_key, value)
            return
        set_value = getattr(target, "set", None)
        if callable(set_value):
            set_value(target_key, value)

    # Preferred path: if available, also publish through model namespace view.
    target = namespaced_view_target(canonical_key)
    view_fn = getattr(coupler, "view", None)
    if (
        target is not None
        and callable(view_fn)
        and not _preserve_artifact_identity(value)
    ):
        namespace, local_key = target
        try:
            view = view_fn(namespace)
            _set_value(view, local_key)
        except Exception:
            logger.debug(
                "Failed to publish key %s via coupler view namespace %s",
                canonical_key,
                namespace,
                exc_info=True,
            )

    # Transitional compatibility: keep unscoped key writes so existing consumers
    # and historical runs remain valid while namespaced lookups are adopted.
    _set_value(coupler, canonical_key)
