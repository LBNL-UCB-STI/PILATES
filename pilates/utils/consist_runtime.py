"""
pilates/utils/consist_runtime.py

Consist runtime wrappers.

Consist is a mandatory dependency for PILATES. This module keeps a thin,
test-friendly layer around the public Consist API while preserving local
helpers (path normalization and schema hints).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, cast

from pilates.utils.consist_types import (
    ArtifactLike,
    ScenarioWithCoupler,
    TrackerLike,
)

logger = logging.getLogger(__name__)

import consist

_enabled_override: Optional[bool] = None


def _is_enabled(enabled: Optional[bool] = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    if _enabled_override is not None:
        return _enabled_override
    return True


def set_enabled(enabled: Optional[bool]) -> None:
    global _enabled_override
    _enabled_override = bool(enabled) if enabled is not None else None


def create_tracker(
    *,
    settings: Optional[Any] = None,
    enabled: Optional[bool] = None,
    tracker_factory: Optional[Callable[[], TrackerLike]] = None,
    **tracker_kwargs: Any,
) -> Optional[TrackerLike]:
    resolved_enabled = _is_enabled(enabled)
    set_enabled(resolved_enabled)
    if "hashing_strategy" not in tracker_kwargs:
        # Avoid expensive full-directory hashing by default.
        tracker_kwargs["hashing_strategy"] = "fast"

    if tracker_factory is None:
        tracker_factory = lambda: consist.Tracker(**tracker_kwargs)

    tracker = consist.create_tracker(
        enabled=resolved_enabled,
        tracker_factory=tracker_factory,
    )
    if resolved_enabled and _is_noop_tracker(tracker):
        repaired = _repair_tracker_db_schema_compatibility(tracker_kwargs)
        if repaired:
            logger.warning(
                "Applied compatibility migration for Consist artifact URI columns; "
                "retrying tracker initialization."
            )
            tracker = consist.create_tracker(
                enabled=resolved_enabled,
                tracker_factory=tracker_factory,
            )
    if resolved_enabled and _is_noop_tracker(tracker):
        logger.error(
            "Consist tracker initialization returned %s while enabled=True. "
            "PILATES requires an active tracker; check preceding Consist tracker "
            "initialization errors for API/version mismatches.",
            type(tracker).__name__,
        )
        return None
    return tracker


@contextmanager
def use_tracker(tracker: Optional[TrackerLike]) -> Iterator[Optional[TrackerLike]]:
    with consist.use_tracker(tracker) as tr:
        yield tr


@contextmanager
def scenario(
    name: str,
    tracker: Optional[TrackerLike] = None,
    *,
    enabled: Optional[bool] = None,
    **kwargs: Any,
) -> Iterator[ScenarioWithCoupler]:
    resolved_enabled = _is_enabled(enabled)
    with consist.scenario(
        name,
        tracker=tracker,
        enabled=resolved_enabled,
        **kwargs,
    ) as sc:
        # Narrow for callers: PILATES requires a coupler-capable scenario.
        yield cast(ScenarioWithCoupler, sc)


def log_input(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    if not resolved_enabled:
        return consist.log_input(path, key=key, enabled=False, **meta)
    try:
        return consist.log_input(path, key=key, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping input artifact logging outside active Consist run.")
        return consist.log_input(path, key=key, enabled=False, **meta)


def log_output(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    if not resolved_enabled:
        resolved_path = _resolve_artifact_path(path)
        if resolved_path and not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Output path does not exist: {resolved_path}")
        return consist.log_output(path, key=key, enabled=False, **meta)
    try:
        return consist.log_output(path, key=key, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping output artifact logging outside active Consist run.")
        return consist.log_output(path, key=key, enabled=False, **meta)


def log_h5_container(
    path: Any,
    key: Optional[str] = None,
    *,
    direction: str = "input",
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log an HDF5 container using the Tracker API when available.

    Falls back to log_input/log_output if the Tracker method is unavailable.
    """
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        return (
            log_output(path, key=key, enabled=False, **meta)
            if direction == "output"
            else log_input(path, key=key, enabled=False, **meta)
        )
    tracker = current_tracker()
    if tracker is None or not hasattr(tracker, "log_h5_container"):
        if direction == "output":
            return log_output(path, key=key, enabled=resolved_enabled, **meta)
        return log_input(path, key=key, enabled=resolved_enabled, **meta)
    return tracker.log_h5_container(path, key=key, direction=direction, **meta)


def log_h5_table(
    path: Any,
    key: Optional[str] = None,
    *,
    table_path: str,
    direction: str = "input",
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log a single HDF5 dataset using the Tracker API when available.

    Requires table_path for schema profiling. Falls back to log_input/log_output
    if the Tracker method is unavailable.
    """
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        return (
            log_output(path, key=key, enabled=False, **meta)
            if direction == "output"
            else log_input(path, key=key, enabled=False, **meta)
        )
    tracker = current_tracker()
    if tracker is None or not hasattr(tracker, "log_h5_table"):
        if direction == "output":
            return log_output(path, key=key, enabled=resolved_enabled, **meta)
        return log_input(path, key=key, enabled=resolved_enabled, **meta)
    artifact = tracker.log_h5_table(
        path, key=key, table_path=table_path, direction=direction, **meta
    )
    return _ensure_legacy_table_path_meta(artifact, table_path=table_path)


def log_artifacts(
    mapping: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[Dict[str, ArtifactLike]]:
    resolved_enabled = _is_enabled(enabled)
    normalized = {key: _normalize_path(value) for key, value in mapping.items()}
    if not resolved_enabled:
        return _log_artifacts_with_enabled(normalized, enabled=False, **meta)
    try:
        return _log_artifacts_with_enabled(normalized, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping artifact logging outside active Consist run.")
        return _log_artifacts_with_enabled(normalized, enabled=False, **meta)


def _log_artifacts_with_enabled(
    outputs: Mapping[str, Any], *, enabled: bool, **meta: Any
) -> Optional[Dict[str, ArtifactLike]]:
    try:
        return consist.log_artifacts(outputs=outputs, enabled=enabled, **meta)
    except TypeError:
        # Legacy compatibility for older Consist call forms.
        return consist.log_artifacts(outputs, enabled=enabled, **meta)


def _schema_for_key(key: str) -> Optional[Any]:
    try:
        from pilates.database.schema.registry import get_schema_for_key
    except Exception:
        return None
    return get_schema_for_key(key)


def log_meta(**meta: Any) -> None:
    if not _is_enabled():
        return None
    try:
        return consist.log_meta(**meta)
    except RuntimeError:
        logger.debug("Skipping metadata logging outside active Consist run.")
        return None


def current_run() -> Optional[Any]:
    if not _is_enabled():
        return None
    try:
        return consist.current_run()
    except RuntimeError:
        return None


def set_tracker(tracker: Optional[TrackerLike]) -> None:
    """Set a global tracker for Consist API entrypoints."""
    if hasattr(consist, "set_current_tracker"):
        consist.set_current_tracker(tracker)
    else:
        consist.use_tracker(tracker).__enter__()


def current_tracker() -> Optional[TrackerLike]:
    """Get the current global tracker (or None if not set)."""
    try:
        return consist.current_tracker()
    except Exception:
        return None


def _is_noop_tracker(tracker: Optional[Any]) -> bool:
    if tracker is None:
        return False
    tracker_type = type(tracker)
    name = getattr(tracker_type, "__name__", "").lower()
    module = getattr(tracker_type, "__module__", "").lower()
    return "nooptracker" in name or ".noop" in module


def _repair_tracker_db_schema_compatibility(tracker_kwargs: Mapping[str, Any]) -> bool:
    db_path = tracker_kwargs.get("db_path")
    if not db_path:
        return False
    try:
        db_path_str = os.fspath(db_path)
    except TypeError:
        return False
    if not db_path_str:
        return False
    if not os.path.exists(db_path_str):
        return False

    try:
        import duckdb
    except Exception:
        logger.exception(
            "DuckDB not available to apply Consist schema compatibility migration."
        )
        return False

    conn = None
    try:
        conn = duckdb.connect(db_path_str)
        columns = conn.execute("PRAGMA table_info('artifact')").fetchall()
        if not columns:
            return False
        names = {str(row[1]) for row in columns if len(row) > 1}
        migrated = False
        if "uri" in names and "container_uri" not in names:
            conn.execute("ALTER TABLE artifact ADD COLUMN container_uri VARCHAR")
            conn.execute(
                "UPDATE artifact SET container_uri = uri WHERE container_uri IS NULL"
            )
            migrated = True
        if "container_uri" in names and "uri" not in names:
            conn.execute("ALTER TABLE artifact ADD COLUMN uri VARCHAR")
            conn.execute("UPDATE artifact SET uri = container_uri WHERE uri IS NULL")
            migrated = True
        return migrated
    except Exception:
        logger.exception(
            "Failed applying Consist artifact URI schema compatibility migration for %s.",
            db_path_str,
        )
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def require_runtime_kwargs(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return consist.require_runtime_kwargs(*names)


def _normalize_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        path = os.fspath(value)
    elif isinstance(value, str):
        path = value
    else:
        return value
    if "://" in path:
        return value
    if os.path.isabs(path):
        return os.path.realpath(path)
    return value


def _maybe_fast_hash_h5(path: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    if "hashing_strategy" in meta:
        return meta
    resolved = _resolve_artifact_path(path)
    if not resolved:
        return meta
    lowered = resolved.lower()
    if lowered.endswith((".h5", ".hdf5")):
        return {**meta, "hashing_strategy": "fast"}
    return meta


def _resolve_artifact_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        return os.fspath(_normalize_path(value))
    if isinstance(value, str):
        return _normalize_path(value)
    if hasattr(value, "path"):
        return os.fspath(getattr(value, "path"))
    if hasattr(value, "container_uri"):
        return str(getattr(value, "container_uri"))
    if hasattr(value, "uri"):
        return str(getattr(value, "uri"))
    return None


def _ensure_legacy_table_path_meta(
    artifact: Optional[ArtifactLike], *, table_path: str
) -> Optional[ArtifactLike]:
    """
    Backward-compatibility shim for callers/tests that read table_path from meta.

    Newer Consist builds may keep table_path as a first-class field without
    duplicating it in artifact.meta.
    """
    if artifact is None:
        return None
    artifact_meta = getattr(artifact, "meta", None)
    if isinstance(artifact_meta, dict):
        artifact_meta.setdefault("table_path", table_path)
        return artifact
    if hasattr(artifact, "model_copy"):
        try:
            normalized_meta = {}
            if isinstance(artifact_meta, Mapping):
                normalized_meta = dict(artifact_meta)
            normalized_meta.setdefault("table_path", table_path)
            return artifact.model_copy(update={"meta": normalized_meta})
        except Exception:
            return artifact
    try:
        setattr(artifact, "meta", {"table_path": table_path})
    except Exception:
        pass
    return artifact
