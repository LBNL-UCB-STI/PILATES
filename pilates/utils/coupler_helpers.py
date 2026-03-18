import os
import logging
import re
import atexit
import fnmatch
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TYPE_CHECKING, Union

from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.workflows.coupler_namespace import (
    ResolvedCouplerValue,
    canonical_artifact_key_from_raw_key,
    namespaced_view_target,
    resolve_coupler_value,
)
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
)

logger = logging.getLogger(__name__)
_STEP_OUTPUT_WARNING_SIGNATURES: set[tuple[Any, ...]] = set()

_ARCHIVE_ENABLE_ENV = "PILATES_ENABLE_ARCHIVE_COPY"
_ARCHIVE_LOCAL_ENV = "PILATES_LOCAL_RUN_DIR"
_ARCHIVE_ROOT_ENV = "PILATES_ARCHIVE_RUN_DIR"
_ARCHIVE_ALLOWED_DIR_PATTERNS = (
    "urbansim_bootstrap_data_root",
    "beam_mutable_data_dir",
    "beam_region_input_dir",
    "zarr_skims",
    "zarr_skims_*",
    "asim_input_skims_zarr_archived",
    ASIM_SHARROW_CACHE_DIR,
    "activitysim_bootstrap_data_root",
    "activitysim_bootstrap_configs_root",
    "atlas_input_year_dir",
    "atlas_input_year_dir_*",
)
_archive_queue: Optional[
    "queue.Queue[Optional[tuple[str, str, str, bool, Optional[tuple[int, int, bool]]]]]"
] = None
_archive_thread: Optional[threading.Thread] = None
_archive_lock = threading.Lock()
_archive_inflight_signature: Dict[str, tuple[int, int, bool]] = {}
_archive_last_copied_signature: Dict[str, tuple[int, int, bool]] = {}


def _archive_enabled() -> bool:
    value = os.environ.get(_ARCHIVE_ENABLE_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _archive_roots() -> Optional[tuple[str, str]]:
    local_root = os.environ.get(_ARCHIVE_LOCAL_ENV)
    archive_root = os.environ.get(_ARCHIVE_ROOT_ENV)
    if not local_root or not archive_root:
        return None
    return os.path.abspath(local_root), os.path.abspath(archive_root)


def _path_under_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _resolve_archive_path(path: str, local_root: str, archive_root: str) -> Optional[str]:
    abs_path = os.path.abspath(path)
    if _path_under_root(abs_path, archive_root):
        return None
    if not _path_under_root(abs_path, local_root):
        return None
    rel_path = os.path.relpath(abs_path, local_root)
    return os.path.join(archive_root, rel_path)


def _resolve_local_path(path: str, local_root: str, archive_root: str) -> Optional[str]:
    abs_path = os.path.abspath(path)
    if _path_under_root(abs_path, local_root):
        return abs_path
    if not _path_under_root(abs_path, archive_root):
        return None
    rel_path = os.path.relpath(abs_path, archive_root)
    return os.path.join(local_root, rel_path)


def _resolve_workspace_uri_path(
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
    roots = _archive_roots()
    if roots is None:
        return None
    local_root, _archive_root = roots
    return os.path.join(local_root, rel_path)


def _copy_archive_to_local(
    *,
    local_path: str,
    archive_path: str,
) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.isdir(archive_path):
            shutil.copytree(archive_path, local_path, dirs_exist_ok=True)
        else:
            shutil.copy2(archive_path, local_path)
        return local_path
    except Exception as exc:
        logger.warning(
            "[Archive] Failed to materialize %s from archive %s: %s",
            local_path,
            archive_path,
            exc,
        )
        return None


def _archive_dir_allowed(key: str) -> bool:
    return any(fnmatch.fnmatch(key, pattern) for pattern in _ARCHIVE_ALLOWED_DIR_PATTERNS)


def _archive_path_signature(path: str, is_dir: bool) -> Optional[tuple[int, int, bool]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (int(stat.st_size), int(stat.st_mtime_ns), bool(is_dir))


def _archive_worker() -> None:
    while True:
        task = _archive_queue.get() if _archive_queue is not None else None
        if task is None:
            if _archive_queue is not None:
                _archive_queue.task_done()
            break
        key, src, dest, is_dir, signature = task
        try:
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
            logger.info("[Archive] Copied %s -> %s (key=%s)", src, dest, key)
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


def _enqueue_archive_copy(key: str, path: str) -> None:
    if not _archive_enabled():
        return
    roots = _archive_roots()
    if roots is None:
        return
    if not path or (isinstance(path, str) and "://" in path):
        return
    if not os.path.exists(path):
        logger.warning("[Archive] Output path does not exist: %s (key=%s)", path, key)
        return
    is_dir = os.path.isdir(path)
    if is_dir and not _archive_dir_allowed(key):
        logger.warning(
            "[Archive] Skipping directory output (not allowlisted): %s (key=%s)",
            path,
            key,
        )
        return
    local_root, archive_root = roots
    dest = _resolve_archive_path(path, local_root, archive_root)
    if dest is None or dest == path:
        return
    signature = _archive_path_signature(path, is_dir)
    if signature is not None:
        with _archive_lock:
            if _archive_inflight_signature.get(dest) == signature:
                logger.debug(
                    "[Archive] Skipping duplicate enqueue (in-flight): %s (key=%s)",
                    path,
                    key,
                )
                return
            if _archive_last_copied_signature.get(dest) == signature:
                logger.debug(
                    "[Archive] Skipping duplicate enqueue (already copied): %s (key=%s)",
                    path,
                    key,
                )
                return
            _archive_inflight_signature[dest] = signature
    _ensure_archive_worker()
    logger.info("[Archive] Enqueued %s -> %s (key=%s)", path, dest, key)
    if _archive_queue is not None:
        _archive_queue.put((key, path, dest, is_dir, signature))


def _warn_once(signature: tuple[Any, ...], message: str, *args: Any) -> None:
    if signature in _STEP_OUTPUT_WARNING_SIGNATURES:
        return
    _STEP_OUTPUT_WARNING_SIGNATURES.add(signature)
    logger.warning(message, *args)


def _warn_for_undeclared_step_output(
    *,
    step_name: Optional[str],
    key: str,
) -> None:
    if not step_name:
        return
    from pilates.workflows.catalog import (
        workflow_step_key_match,
        workflow_step_spec_for_step_name,
    )

    match = workflow_step_key_match(step_name, key, direction="output")
    if match.declared:
        return
    spec = workflow_step_spec_for_step_name(step_name)
    dynamic_families = tuple(spec.dynamic_output_families) if spec is not None else ()
    if dynamic_families:
        message = (
            "[CONTRACT-ENFORCEMENT][%s] Step '%s' published undeclared output "
            "key '%s'%s; it matches no declared output key and no dynamic "
            "output family %s."
        )
        args = (
            step_name,
            step_name,
            key,
            match.alias_note,
            dynamic_families,
        )
    else:
        message = (
            "[CONTRACT-ENFORCEMENT][%s] Step '%s' published undeclared output "
            "key '%s'%s; the step declares no dynamic output families."
        )
        args = (
            step_name,
            step_name,
            key,
            match.alias_note,
        )
    _warn_once(
        ("undeclared_step_output", step_name, key),
        message,
        *args,
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
    if path is None:
        return
    resolved = _resolve_workspace_uri_path(os.fspath(path), workspace=workspace)
    if not resolved:
        return
    _enqueue_archive_copy(key, resolved)


def flush_archive_queue(
    timeout: Optional[float] = None,
    *,
    fail_on_timeout: bool = False,
) -> bool:
    if _archive_queue is None:
        return True
    pending = _archive_queue.unfinished_tasks
    logger.info("[Archive] Flushing queue (pending=%s)", pending)
    if timeout is None:
        _archive_queue.join()
        logger.info("[Archive] Flush complete")
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _archive_queue.unfinished_tasks == 0:
            logger.info("[Archive] Flush complete")
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
    path = (
        getattr(value, "path", None)
        or getattr(value, "container_uri", None)
        or getattr(value, "uri", None)
        or value
    )
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
    materialize_from_archive: bool = False,
) -> Optional[str]:
    """
    Resolve a path to one that currently exists, optionally using archive fallback.
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

    roots = _archive_roots()
    if roots is None:
        logger.debug(
            "[Archive] No archive roots configured while resolving missing path: %s",
            abs_path,
        )
        return None
    local_root, archive_root = roots

    archive_path = _resolve_archive_path(abs_path, local_root, archive_root)
    if archive_path and os.path.exists(archive_path):
        if materialize_from_archive:
            local_path = _resolve_local_path(archive_path, local_root, archive_root)
            if local_path:
                logger.info(
                    "[Archive] Local path missing; materializing from archive %s -> %s",
                    archive_path,
                    local_path,
                )
                materialized = _copy_archive_to_local(
                    local_path=local_path, archive_path=archive_path
                )
                if materialized and os.path.exists(materialized):
                    logger.info(
                        "[Archive] Materialized local path from archive: %s",
                        materialized,
                    )
                    return materialized
                logger.warning(
                    "[Archive] Materialization attempt did not produce local path: %s",
                    local_path,
                )
        logger.info(
            "[Archive] Local path missing; using archive path directly: %s",
            archive_path,
        )
        return archive_path

    local_path = _resolve_local_path(abs_path, local_root, archive_root)
    if local_path and os.path.exists(local_path):
        logger.debug("[Archive] Resolved path already available locally: %s", local_path)
        return local_path
    logger.debug("[Archive] Unable to resolve existing path for: %s", abs_path)
    return None


def artifact_to_existing_path(
    value: Any,
    workspace: Optional["Workspace"] = None,
    *,
    materialize_from_archive: bool = False,
) -> Optional[str]:
    """
    Resolve an artifact-like value to an existing path with optional archive fallback.
    """
    path = artifact_to_path(value, workspace=workspace)
    if path is None and isinstance(value, str):
        path = value
    return resolve_existing_path(
        path,
        workspace=workspace,
        materialize_from_archive=materialize_from_archive,
    )


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
    if hasattr(value, "container_uri") and getattr(value, "key", None):
        return value
    if hasattr(value, "uri") and getattr(value, "key", None):
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
    has_container_uri = hasattr(value, "container_uri")
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


def _select_beam_output_record(
    current: Optional[Any],
    candidate: Any,
    base_key: str,
) -> Optional[Any]:
    """
    Choose the most recent BEAM output record for a base key.

    Parameters
    ----------
    current : object, optional
        Currently selected record.
    candidate : object
        Candidate record to compare.
    base_key : str
        Base short_name prefix (e.g., ``linkstats``).

    Returns
    -------
    object or None
        Selected record with the highest (year, iteration) rank.
    """
    candidate_rank = _beam_record_rank(candidate, base_key)
    if candidate_rank is None:
        return current
    if current is None:
        return candidate
    current_rank = _beam_record_rank(current, base_key)
    if current_rank is None or candidate_rank > current_rank:
        return candidate
    return current


def _beam_record_rank(record: Any, base_key: str) -> Optional[tuple]:
    """
    Return a sortable (year, iteration) rank for a BEAM record short_name.

    Parameters
    ----------
    record : object
        Record with a ``short_name`` attribute.
    base_key : str
        Base short_name prefix.

    Returns
    -------
    tuple or None
        (year, iteration) rank, or None when the record cannot be ranked.
    """
    short_name = getattr(record, "short_name", None)
    if not short_name:
        return None
    if short_name == base_key:
        return (0, 0)
    if not short_name.startswith(f"{base_key}_"):
        return None
    suffix = short_name[len(base_key) + 1 :]
    parts = suffix.split("_")
    if len(parts) < 2:
        return None
    if parts[-1].startswith("sub"):
        return None
    try:
        year = int(parts[-2])
        iteration = int(parts[-1])
    except ValueError:
        return None
    return (year, iteration)


def _is_linkstats_unmodified_phys_sim_key(short_name: Optional[str]) -> bool:
    if not short_name:
        return False
    return short_name.startswith(
        "linkstats_unmodified_phys_sim_iter_parquet_"
    ) or short_name.startswith("linkstats_unmodified_parquet__")


def _is_sub_iteration_key(short_name: Optional[str]) -> bool:
    if not short_name:
        return False
    return "_sub" in short_name or "__beam_sub_iter" in short_name


def _beam_linkstats_facet_meta(
    short_name: Optional[str],
    *,
    family: str,
) -> Dict[str, Any]:
    if not short_name:
        return {}
    parsed = _parse_linkstats_iteration_key(short_name)
    if not parsed:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **parsed,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _parse_linkstats_iteration_key(short_name: str) -> Optional[Dict[str, Any]]:
    for prefix in ("linkstats_parquet_", "linkstats_"):
        if not short_name.startswith(prefix):
            continue
        tail = short_name[len(prefix) :]
        parts = tail.split("_")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            iteration = int(parts[1])
        except ValueError:
            continue
        payload: Dict[str, Any] = {"year": year, "iteration": iteration}
        if len(parts) > 2 and parts[2].startswith("sub"):
            try:
                payload["beam_sub_iteration"] = int(parts[2][3:])
            except ValueError:
                continue
        return payload
    return None


def _parse_linkstats_unmodified_phys_sim_facets(
    short_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Parse structured facets from phys-sim unmodified linkstats artifact keys.

    Expected key format:
      linkstats_unmodified_phys_sim_iter_parquet_<phys_sim_iter>_<year>_<iteration>[_sub<beam_sub_iteration>]
    """
    if not short_name:
        return None

    year = None
    iteration = None
    phys_sim_iteration = None
    sub_iteration = None

    modern_match = re.fullmatch(
        r"linkstats_unmodified_parquet__y(?P<year>\d+)__i(?P<iteration>\d+)"
        r"__phys_sim_iter(?P<phys>\d+)(?:__beam_sub_iter(?P<sub>\d+))?",
        short_name,
    )
    if modern_match:
        year = int(modern_match.group("year"))
        iteration = int(modern_match.group("iteration"))
        phys_sim_iteration = int(modern_match.group("phys"))
        if modern_match.group("sub") is not None:
            sub_iteration = int(modern_match.group("sub"))
    else:
        legacy_prefix = "linkstats_unmodified_phys_sim_iter_parquet_"
        if not short_name.startswith(legacy_prefix):
            return None

        tail = short_name[len(legacy_prefix) :]
        parts = tail.split("_")
        if len(parts) < 3:
            return None

        if parts[-1].startswith("sub"):
            try:
                sub_iteration = int(parts[-1][3:])
            except ValueError:
                return None
            parts = parts[:-1]

        if len(parts) != 3:
            return None

        try:
            phys_sim_iteration = int(parts[0])
            year = int(parts[1])
            iteration = int(parts[2])
        except ValueError:
            return None

    facets: Dict[str, Any] = {
        "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
        "year": year,
        "iteration": iteration,
        "phys_sim_iteration": phys_sim_iteration,
    }
    if sub_iteration is not None:
        facets["beam_sub_iteration"] = sub_iteration
    return facets


def _log_and_set_beam_record(
    record: Optional[Any],
    *,
    key: str,
    description: str,
    coupler: CouplerProtocol,
    workspace: "Workspace",
    **meta: Any,
) -> None:
    """
    Log a BEAM output record and set it on the coupler.

    Parameters
    ----------
    record : object, optional
        Record describing the output artifact.
    key : str
        Coupler key to set.
    description : str
        Description used in provenance logging.
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    workspace : Workspace
        Workspace used to resolve output paths.
    """
    if record is None:
        return
    output_path = artifact_to_path(record.file_path, workspace)
    if not output_path or not os.path.exists(output_path):
        return
    log_and_set_output(
        key=key,
        path=output_path,
        description=description,
        coupler=coupler,
        **meta,
    )


def _log_beam_record_only(
    record: Optional[Any],
    *,
    key: str,
    description: str,
    workspace: "Workspace",
    **meta: Any,
) -> None:
    if record is None:
        return
    output_path = artifact_to_path(record.file_path, workspace)
    if not output_path or not os.path.exists(output_path):
        return
    log_output_only(
        key=key,
        path=output_path,
        description=description,
        **meta,
    )


def clean_expected_outputs(outputs: dict) -> dict:
    """
    Filter out expected outputs that resolved to None.

    Parameters
    ----------
    outputs : dict
        Mapping of expected outputs to filter.

    Returns
    -------
    dict
        Outputs with None-valued entries removed.
    """
    return {key: value for key, value in outputs.items() if value is not None}


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
    value = artifact or fallback

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
    if target is not None and callable(view_fn):
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


def _log_with_optional_h5_container(
    *,
    direction: str,
    key: str,
    path: str,
    description: str,
    meta: Dict[str, Any],
) -> Optional[Any]:
    """
    Log either an HDF5 container or a standard artifact based on meta flags.
    """
    def _is_internal_h5_table(table_name: str) -> bool:
        leaf = table_name.rsplit("/", 1)[-1]
        normalized_leaf = leaf.lower()
        # Pandas HDF internals can appear either as leaf names (e.g. axis1,
        # block0_items, level0) or flattened into larger table names
        # (e.g. travel_data_axis1_level0).
        if re.fullmatch(r"(axis|block|level|label)\d+(?:_.*)?", normalized_leaf):
            return True
        return bool(
            re.search(r"_(axis|block|level|label)\d*(?:_|$)", normalized_leaf)
        )

    def _table_filter_to_callable(
        table_filter: Optional[Union[Callable[[str], bool], Sequence[str]]]
    ) -> Optional[Callable[[str], bool]]:
        if table_filter is None:
            return None
        if callable(table_filter):
            return table_filter
        normalized = {
            name if str(name).startswith("/") else f"/{name}"
            for name in table_filter
            if name
        }
        return lambda table_name: table_name in normalized

    def _h5_table_filter_from_list(tables_used):
        normalized = {
            name if name.startswith("/") else f"/{name}"
            for name in tables_used
            if name
        }

        def _filter(table_name: str) -> bool:
            if _is_internal_h5_table(table_name):
                return False
            return table_name in normalized

        return _filter

    tables_used = meta.pop("h5_tables_used", None)
    requested_filter = _table_filter_to_callable(meta.pop("table_filter", None))
    h5_container = bool(meta.pop("h5_container", False)) or bool(tables_used)
    if h5_container:
        if tables_used:
            normalized_paths = sorted(
                {
                    name if str(name).startswith("/") else f"/{name}"
                    for name in tables_used
                    if name
                }
            )
            meta.setdefault("h5_table_paths", normalized_paths)
            meta.setdefault("h5_table_count", len(normalized_paths))
        base_filter = (
            _h5_table_filter_from_list(tables_used)
            if tables_used
            else (lambda table_name: not _is_internal_h5_table(table_name))
        )
        if requested_filter is None:
            table_filter = base_filter
        else:
            def table_filter(table_name: str) -> bool:
                return base_filter(table_name) and requested_filter(table_name)
        return cr.log_h5_container(
            path,
            key=key,
            direction=direction,
            description=description,
            table_filter=table_filter,
            **meta,
        )
    if direction == "output":
        return cr.log_output(path, key=key, description=description, **meta)
    return cr.log_input(path, key=key, description=description, **meta)


def _log_and_maybe_publish_artifact(
    *,
    direction: str,
    key: str,
    path: str,
    description: str,
    meta: Dict[str, Any],
    coupler: Optional[CouplerProtocol] = None,
    publish_to_coupler: bool,
    enqueue_archive_copy: bool,
    skip_logging_without_active_run: bool,
) -> None:
    """
    Shared primitive for the coupler helper logging functions.

    This centralizes active-run vs no-active-run behavior so the public wrappers
    cannot diverge.
    """
    has_active_run = cr.current_run() is not None
    artifact: Optional[Any] = None

    if has_active_run or not skip_logging_without_active_run:
        artifact = _log_with_optional_h5_container(
            direction=direction,
            key=key,
            path=path,
            description=description,
            meta=dict(meta),
        )

    if enqueue_archive_copy:
        _enqueue_archive_copy(key, path)

    if publish_to_coupler:
        if coupler is None:
            raise TypeError("coupler must be provided when publish_to_coupler=True")
        set_coupler_from_artifact(coupler, key, artifact, fallback=path)


def log_and_set_output(
    *,
    key: str,
    path: str,
    description: str,
    coupler: CouplerProtocol,
    step_name: Optional[str] = None,
    **meta: Any,
) -> None:
    """
    Log an output path and set it on the coupler.

    Parameters
    ----------
    key : str
        Coupler key to set.
    path : str
        Output path to log.
    description : str
        Description used in provenance logging.
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    step_name : str, optional
        Canonical workflow step name used for semantic-contract warnings.
    """
    _warn_for_undeclared_step_output(step_name=step_name, key=key)
    _log_and_maybe_publish_artifact(
        direction="output",
        key=key,
        path=path,
        description=description,
        meta=meta,
        coupler=coupler,
        publish_to_coupler=True,
        enqueue_archive_copy=True,
        skip_logging_without_active_run=True,
    )


def log_output_only(
    *,
    key: str,
    path: str,
    description: str,
    step_name: Optional[str] = None,
    **meta: Any,
) -> None:
    """
    Log an output path without writing to the coupler.

    Parameters
    ----------
    key : str
        Artifact key used for provenance logging.
    path : str
        Output path to log.
    description : str
        Description used in provenance logging.
    step_name : str, optional
        Canonical workflow step name used for semantic-contract warnings.
    """
    _warn_for_undeclared_step_output(step_name=step_name, key=key)
    _log_and_maybe_publish_artifact(
        direction="output",
        key=key,
        path=path,
        description=description,
        meta=meta,
        publish_to_coupler=False,
        enqueue_archive_copy=True,
        skip_logging_without_active_run=False,
    )


def log_and_set_input(
    *,
    key: str,
    path: str,
    description: str,
    coupler: CouplerProtocol,
    **meta: Any,
) -> None:
    """
    Log an input path and set it on the coupler.

    Parameters
    ----------
    key : str
        Coupler key to set.
    path : str
        Input path to log.
    description : str
        Description used in provenance logging.
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    """
    _log_and_maybe_publish_artifact(
        direction="input",
        key=key,
        path=path,
        description=description,
        meta=meta,
        coupler=coupler,
        publish_to_coupler=True,
        enqueue_archive_copy=False,
        skip_logging_without_active_run=True,
    )


def log_input_only(
    *,
    key: str,
    path: str,
    description: str,
    **meta: Any,
) -> None:
    """
    Log an input path without writing to the coupler.

    Parameters
    ----------
    key : str
        Artifact key used for provenance logging.
    path : str
        Input path to log.
    description : str
        Description used in provenance logging.
    """
    _log_and_maybe_publish_artifact(
        direction="input",
        key=key,
        path=path,
        description=description,
        meta=meta,
        publish_to_coupler=False,
        enqueue_archive_copy=False,
        skip_logging_without_active_run=False,
    )
