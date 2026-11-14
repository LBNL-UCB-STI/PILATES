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
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    cast,
)

from pilates.utils.consist_types import (
    ArtifactLike,
    ScenarioWithCoupler,
    TrackerLike,
)

logger = logging.getLogger(__name__)

import consist


def create_tracker(
    *,
    settings: Optional[Any] = None,
    tracker_factory: Optional[Callable[[], TrackerLike]] = None,
    **tracker_kwargs: Any,
) -> Optional[TrackerLike]:
    tracker_kwargs.pop("enabled", None)
    if "hashing_strategy" not in tracker_kwargs:
        try:
            configured_strategy = settings.run.consist_hashing_strategy
        except AttributeError:
            configured_strategy = "fast"
        if configured_strategy not in {"fast", "full"}:
            configured_strategy = "fast"
        tracker_kwargs["hashing_strategy"] = configured_strategy

    if tracker_factory is not None:
        return tracker_factory()

    return consist.Tracker(**tracker_kwargs)


@contextmanager
def use_tracker(tracker: Optional[TrackerLike]) -> Iterator[Optional[TrackerLike]]:
    with consist.use_tracker(tracker) as tr:
        yield tr


@contextmanager
def scenario(
    name: str,
    tracker: Optional[TrackerLike] = None,
    **kwargs: Any,
) -> Iterator[ScenarioWithCoupler]:
    kwargs.pop("enabled", None)
    with consist.scenario(
        name,
        tracker=tracker,
        **kwargs,
    ) as sc:
        # Narrow for callers: PILATES requires a coupler-capable scenario.
        yield cast(ScenarioWithCoupler, sc)


def log_input(
    path: Any,
    key: Optional[str] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    meta.pop("enabled", None)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    path = _normalize_path(path)
    try:
        return consist.log_input(path, key=key, **meta)
    except Exception as exc:
        return _retry_without_schema_meta(
            log_fn=consist.log_input,
            path=path,
            key=key,
            meta=meta,
            exc=exc,
            direction="input",
        )


def artifact_fingerprint(value: Any) -> Optional[str]:
    """
    Return Consist's canonical public artifact fingerprint when available.

    Downstream workflow code should treat ``Artifact.digest`` as the only stable
    fingerprint surface. Missing fingerprints remain ``None``; callers should
    not probe legacy or wrapper-specific fields.
    """
    if isinstance(value, ArtifactLike):
        digest = value.digest
        if digest is not None:
            return str(digest)
    return None


def log_output(
    path: Any,
    key: Optional[str] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    meta.pop("enabled", None)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    path = _normalize_path(path)
    try:
        return consist.log_output(path, key=key, **meta)
    except Exception as exc:
        return _retry_without_schema_meta(
            log_fn=consist.log_output,
            path=path,
            key=key,
            meta=meta,
            exc=exc,
            direction="output",
        )


def log_h5_container(
    path: Any,
    key: Optional[str] = None,
    *,
    direction: str = "input",
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log an HDF5 container using the Tracker API when available.

    Falls back to log_input/log_output if no active Tracker is set.
    """
    meta.pop("enabled", None)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    tracker = current_tracker()
    if tracker is None:
        if direction == "output":
            return log_output(path, key=key, **meta)
        return log_input(path, key=key, **meta)
    with tracker.persistence.batch_artifact_writes():
        return tracker.log_h5_container(path, key=key, direction=direction, **meta)


def log_h5_table(
    path: Any,
    key: Optional[str] = None,
    *,
    table_path: str,
    direction: str = "input",
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log a single HDF5 dataset using the Tracker API when available.

    Requires table_path for schema profiling. Falls back to log_input/log_output
    if no active Tracker is set.
    """
    meta.pop("enabled", None)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    tracker = current_tracker()
    if tracker is None:
        if direction == "output":
            return log_output(path, key=key, **meta)
        return log_input(path, key=key, **meta)
    return tracker.log_h5_table(
        path, key=key, table_path=table_path, direction=direction, **meta
    )


def log_artifacts(
    mapping: Mapping[str, Any],
    **meta: Any,
) -> Optional[Dict[str, ArtifactLike]]:
    meta.pop("enabled", None)
    normalized = {key: _normalize_path(value) for key, value in mapping.items()}
    return consist.log_artifacts(outputs=normalized, **meta)


def _schema_for_key(key: str) -> Optional[Any]:
    try:
        from pilates.database.schema.registry import get_schema_for_key
    except Exception:
        return None
    return get_schema_for_key(key)


_schema_warning_signatures: set[tuple[Any, ...]] = set()


def _schema_warn_only_enabled() -> bool:
    value = os.getenv("PILATES_SCHEMA_WARN_ONLY")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _warn_once(signature: tuple[Any, ...], msg: str, *args: Any) -> None:
    if signature in _schema_warning_signatures:
        return
    _schema_warning_signatures.add(signature)
    logger.warning(msg, *args)


def _retry_without_schema_meta(
    *,
    log_fn: Callable[..., Optional[ArtifactLike]],
    path: Any,
    key: Optional[str],
    meta: Mapping[str, Any],
    exc: Exception,
    direction: str,
) -> Optional[ArtifactLike]:
    if "schema" not in meta or not _schema_warn_only_enabled():
        raise exc
    schema = meta.get("schema")
    try:
        schema_name = str(schema.__name__)
    except AttributeError:
        schema_name = str(schema)
    reduced_meta = {
        k: v
        for k, v in meta.items()
        if k not in {"schema", "declared_schema_class", "declared_schema_table"}
    }
    signature = ("schema_log_retry", direction, key, schema_name, type(exc).__name__)
    _warn_once(
        signature,
        (
            "[SCHEMA WARNING] %s logging failed for key=%r using schema=%s; "
            "retrying without schema metadata. Error: %s"
        ),
        direction,
        key,
        schema_name,
        exc,
    )
    try:
        return log_fn(path, key=key, enabled=True, **reduced_meta)
    except RuntimeError:
        raise


def _with_declared_schema_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    schema = meta.get("schema")
    if schema is None:
        return dict(meta)
    try:
        schema_name = str(schema.__name__)
    except AttributeError:
        schema_name = str(schema)
    try:
        raw_table_name = schema.__tablename__
    except AttributeError:
        raw_table_name = None
    updated = dict(meta)
    updated.setdefault("declared_schema_class", schema_name)
    updated.setdefault(
        "declared_schema_table",
        str(raw_table_name) if raw_table_name is not None else None,
    )
    return updated


def log_meta(**meta: Any) -> None:
    return consist.log_meta(**meta)


def set_output_meta(
    key: str,
    meta: Mapping[str, Any],
    *,
    merge: bool = True,
) -> None:
    if current_run() is None:
        return None
    consist.set_output_meta(key, meta, merge=merge)


def current_run() -> Optional[Any]:
    try:
        return consist.current_run()
    except (AttributeError, RuntimeError):
        return None


def current_run_id() -> Optional[str]:
    run = current_run()
    if run is None:
        return None
    try:
        raw_id = run.id
    except AttributeError:
        return None
    if raw_id is None:
        return None
    run_id_text = str(raw_id).strip()
    return run_id_text or None


def set_tracker(tracker: Optional[TrackerLike]) -> None:
    """Set a global tracker for Consist API entrypoints."""
    consist.set_current_tracker(tracker)


def current_tracker() -> Optional[TrackerLike]:
    """Get the current global tracker (or None if not set)."""
    try:
        return consist.current_tracker()
    except (AttributeError, RuntimeError):
        return None


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


def dispose_tracker(tracker: Optional[TrackerLike]) -> None:
    """Safely dispose the underlying tracker / database handles."""
    if tracker is None:
        return
    try:
        tracker.close()
    except (AttributeError, TypeError):
        pass
    except Exception:
        logger.warning("Failed to close tracker database", exc_info=True)
