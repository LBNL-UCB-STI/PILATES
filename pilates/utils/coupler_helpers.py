import os
import logging
import atexit
import fnmatch
import queue
import shutil
import threading
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Type

from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol
from pilates.workflows.artifact_constants import (
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)

logger = logging.getLogger(__name__)

_ARCHIVE_ENABLE_ENV = "PILATES_ENABLE_ARCHIVE_COPY"
_ARCHIVE_LOCAL_ENV = "PILATES_LOCAL_RUN_DIR"
_ARCHIVE_ROOT_ENV = "PILATES_ARCHIVE_RUN_DIR"
_ARCHIVE_ALLOWED_DIR_PATTERNS = (
    "zarr_skims",
    "zarr_skims_*",
    "asim_input_skims_zarr_archived",
)
_archive_queue: Optional["queue.Queue[Optional[tuple[str, str, str, bool]]]"] = None
_archive_thread: Optional[threading.Thread] = None
_archive_lock = threading.Lock()


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


def _archive_dir_allowed(key: str) -> bool:
    return any(fnmatch.fnmatch(key, pattern) for pattern in _ARCHIVE_ALLOWED_DIR_PATTERNS)


def _archive_worker() -> None:
    while True:
        task = _archive_queue.get() if _archive_queue is not None else None
        if task is None:
            if _archive_queue is not None:
                _archive_queue.task_done()
            break
        key, src, dest, is_dir = task
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if is_dir:
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)
            logger.info("[Archive] Copied %s -> %s (key=%s)", src, dest, key)
        except Exception as exc:
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
    _ensure_archive_worker()
    logger.info("[Archive] Enqueued %s -> %s (key=%s)", path, dest, key)
    if _archive_queue is not None:
        _archive_queue.put((key, path, dest, is_dir))


def flush_archive_queue(timeout: Optional[float] = None) -> None:
    if _archive_queue is None:
        return
    pending = _archive_queue.unfinished_tasks
    logger.info("[Archive] Flushing queue (pending=%s)", pending)
    if timeout is None:
        _archive_queue.join()
        logger.info("[Archive] Flush complete")
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _archive_queue.unfinished_tasks == 0:
            logger.info("[Archive] Flush complete")
            return
        time.sleep(0.1)
    logger.warning(
        "[Archive] Flush timed out (pending=%s)", _archive_queue.unfinished_tasks
    )


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
    from pilates.generic.records import RecordStore
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


def update_coupler_from_beam_outputs(
    output_store: Optional["RecordStore"],
    coupler: CouplerProtocol,
    workspace: "Workspace",
) -> None:
    """
    Log and propagate BEAM output artifacts into the workflow coupler.

    Parameters
    ----------
    output_store : RecordStore
        Output store from BEAM run + postprocess outputs.
    coupler : CouplerProtocol
        Consist coupler or compatible interface.
    workspace : Workspace
        Workspace used to resolve output paths.
    """
    if not output_store:
        return
    linkstats_record = None
    beam_plans_record = None
    linkstats_parquet_records = []
    for record in output_store.all_records():
        if record.short_name == ZARR_SKIMS:
            zarr_path = artifact_to_path(record.file_path, workspace)
            if zarr_path and os.path.exists(zarr_path):
                log_and_set_output(
                    key=ZARR_SKIMS,
                    path=zarr_path,
                    description="Zarr skims updated with BEAM outputs",
                    coupler=coupler,
                )
        elif record.short_name == FINAL_SKIMS_OMX:
            omx_path = artifact_to_path(record.file_path, workspace)
            if omx_path and os.path.exists(omx_path):
                log_and_set_output(
                    key=FINAL_SKIMS_OMX,
                    path=omx_path,
                    description="Final skims OMX for downstream models",
                    coupler=coupler,
                )
        elif record.short_name and record.short_name.startswith("linkstats_parquet_"):
            if "_sub" not in record.short_name:
                linkstats_parquet_records.append(record)
        elif (
            record.short_name
            and record.short_name.startswith(LINKSTATS)
            and not record.short_name.startswith("linkstats_unmodified")
        ):
            linkstats_record = _select_beam_output_record(
                linkstats_record, record, LINKSTATS
            )
        elif record.short_name and record.short_name.startswith(BEAM_PLANS_OUT):
            beam_plans_record = _select_beam_output_record(
                beam_plans_record, record, BEAM_PLANS_OUT
            )

    _log_and_set_beam_record(
        linkstats_record,
        key=LINKSTATS,
        description="BEAM linkstats output for downstream runs",
        coupler=coupler,
        workspace=workspace,
    )
    _log_and_set_beam_record(
        linkstats_record,
        key=LINKSTATS_WARMSTART,
        description="BEAM warm-start linkstats for downstream runs",
        coupler=coupler,
        workspace=workspace,
        profile_file_schema=True,
    )
    for record in linkstats_parquet_records:
        _log_and_set_beam_record(
            record,
            key=record.short_name,
            description="BEAM linkstats parquet output for downstream runs",
            coupler=coupler,
            workspace=workspace,
        )
    _log_and_set_beam_record(
        beam_plans_record,
        key=BEAM_PLANS_OUT,
        description="BEAM plans output for downstream runs",
        coupler=coupler,
        workspace=workspace,
        profile_file_schema=True,
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
    set_from_artifact = getattr(coupler, "set_from_artifact", None)
    if callable(set_from_artifact):
        if artifact is None:
            set_from_artifact(key, fallback)
        else:
            set_from_artifact(key, artifact)
        return
    coupler.set(key, artifact or fallback)


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
    def _h5_table_filter_from_list(tables_used):
        normalized = {
            name if name.startswith("/") else f"/{name}"
            for name in tables_used
            if name
        }

        def _filter(table_name: str) -> bool:
            if any(tok in table_name for tok in ("_axis", "_block", "_level", "_label")):
                return False
            return table_name in normalized

        return _filter

    tables_used = meta.pop("h5_tables_used", None)
    h5_container = bool(meta.pop("h5_container", False)) or bool(tables_used)
    if h5_container:
        return cr.log_h5_container(
            path,
            key=key,
            direction=direction,
            description=description,
            table_filter=_h5_table_filter_from_list(tables_used)
            if tables_used
            else None,
            **meta,
        )
    if direction == "output":
        return cr.log_output(path, key=key, description=description, **meta)
    return cr.log_input(path, key=key, description=description, **meta)


def log_and_set_output(
    *,
    key: str,
    path: str,
    description: str,
    coupler: CouplerProtocol,
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
    """
    artifact = _log_with_optional_h5_container(
        direction="output",
        key=key,
        path=path,
        description=description,
        meta=meta,
    )
    _enqueue_archive_copy(key, path)
    if cr.current_run() is None:
        set_coupler_from_artifact(coupler, key, artifact, fallback=path)


def log_output_only(
    *,
    key: str,
    path: str,
    description: str,
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
    """
    _log_with_optional_h5_container(
        direction="output",
        key=key,
        path=path,
        description=description,
        meta=meta,
    )
    _enqueue_archive_copy(key, path)


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
    artifact = _log_with_optional_h5_container(
        direction="input",
        key=key,
        path=path,
        description=description,
        meta=meta,
    )
    if cr.current_run() is None:
        set_coupler_from_artifact(coupler, key, artifact, fallback=path)


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
    _log_with_optional_h5_container(
        direction="input",
        key=key,
        path=path,
        description=description,
        meta=meta,
    )


def record_store_to_outputs(
    record_store: "RecordStore",
    output_class: Type[Any],
    workspace: "Workspace",
) -> Any:
    """
    Convert a RecordStore into a typed StepOutputs dataclass.

    Parameters
    ----------
    record_store : RecordStore
        RecordStore returned by a component execution.
    output_class : type
        Dataclass type to instantiate.
    workspace : Workspace
        Workspace used to resolve relative paths.
    """
    if hasattr(output_class, "from_record_store"):
        return output_class.from_record_store(record_store, workspace)
    if not is_dataclass(output_class):
        raise TypeError(
            "output_class must be a dataclass or implement from_record_store"
        )

    mapping = record_store.to_mapping() if record_store is not None else {}
    record_keys = getattr(output_class, "record_keys", {}) or {}
    values: Dict[str, Any] = {}

    for field in fields(output_class):
        key = record_keys.get(field.name, field.name)
        if key not in mapping:
            continue
        path = artifact_to_path(mapping[key], workspace)
        if path is None:
            continue
        values[field.name] = Path(path)

    return output_class(**values)
