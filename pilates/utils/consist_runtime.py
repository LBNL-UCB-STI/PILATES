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
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, cast

from pilates.utils.consist_types import (
    ArtifactLike,
    ScenarioWithCoupler,
    TrackerLike,
)

logger = logging.getLogger(__name__)

import consist

_enabled_override: Optional[bool] = None
_schema_warning_signatures: set[tuple[Any, ...]] = set()
_tracker_hashing_strategy: str = "fast"


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
    global _tracker_hashing_strategy
    resolved_enabled = _is_enabled(enabled)
    set_enabled(resolved_enabled)
    if "hashing_strategy" not in tracker_kwargs:
        run_cfg = getattr(settings, "run", None)
        configured_strategy = getattr(run_cfg, "consist_hashing_strategy", "fast")
        if configured_strategy not in {"fast", "full"}:
            configured_strategy = "fast"
        tracker_kwargs["hashing_strategy"] = configured_strategy
    strategy = str(tracker_kwargs.get("hashing_strategy", "fast")).lower()
    _tracker_hashing_strategy = strategy if strategy in {"fast", "full"} else "fast"

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
    if consist is None:
        return None
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    _warn_schema_compatibility(path, key=key, meta=meta, direction="input")
    if not resolved_enabled:
        return consist.log_input(path, key=key, enabled=False, **meta)
    try:
        return consist.log_input(path, key=key, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping input artifact logging outside active Consist run.")
        return consist.log_input(path, key=key, enabled=False, **meta)
    except Exception as exc:
        return _retry_without_schema_meta(
            log_fn=consist.log_input,
            path=path,
            key=key,
            meta=meta,
            exc=exc,
            direction="input",
        )


def log_output(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    if consist is None:
        return None
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    _warn_schema_compatibility(path, key=key, meta=meta, direction="output")
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
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log an HDF5 container using the Tracker API when available.

    Falls back to log_input/log_output if the Tracker method is unavailable.
    """
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
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
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    meta = _with_declared_schema_meta(meta)
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
    schema_name = getattr(schema, "__name__", str(schema))
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
        logger.debug(
            "Skipping %s artifact logging outside active Consist run.", direction
        )
        return log_fn(path, key=key, enabled=False, **reduced_meta)


def _warn_schema_compatibility(
    path: Any,
    *,
    key: Optional[str],
    meta: Mapping[str, Any],
    direction: str,
) -> None:
    if not _schema_warn_only_enabled():
        return
    schema = meta.get("schema")
    if schema is None:
        return
    _warn_schema_fk_targets(schema, key=key, direction=direction)
    resolved_path = _resolve_artifact_path(path)
    if not resolved_path or "://" in str(resolved_path) or not os.path.exists(
        resolved_path
    ):
        return
    if not _is_schema_check_file(resolved_path):
        return
    try:
        expected_columns = _schema_expected_columns(schema)
        if not expected_columns:
            return
        observed_columns, observed_families = _observed_columns_and_families(
            resolved_path
        )
        if not observed_columns:
            return
        missing = sorted(set(expected_columns) - set(observed_columns))
        unexpected = sorted(set(observed_columns) - set(expected_columns))
        mismatches = _type_mismatch_columns(
            schema=schema,
            observed_families=observed_families,
            path=resolved_path,
        )
        if not missing and not unexpected and not mismatches:
            return
        schema_name = getattr(schema, "__name__", str(schema))
        signature = (
            "schema_compatibility",
            direction,
            key,
            schema_name,
            tuple(missing[:20]),
            tuple(unexpected[:20]),
            tuple(mismatches[:20]),
        )
        _warn_once(
            signature,
            (
                "[SCHEMA WARNING] %s key=%r path=%s schema=%s "
                "missing_columns=%s unexpected_columns=%s type_mismatches=%s"
            ),
            direction,
            key,
            resolved_path,
            schema_name,
            _render_list(missing),
            _render_list(unexpected),
            _render_list(mismatches),
        )
    except Exception as exc:
        schema_name = getattr(schema, "__name__", str(schema))
        signature = ("schema_compatibility_error", direction, key, schema_name)
        _warn_once(
            signature,
            (
                "[SCHEMA WARNING] %s key=%r schema=%s could not run schema "
                "compatibility check for %s: %s"
            ),
            direction,
            key,
            schema_name,
            resolved_path,
            exc,
        )


def _warn_schema_fk_targets(schema: Any, *, key: Optional[str], direction: str) -> None:
    try:
        registry_tables = _registered_schema_table_columns()
        model_fields = getattr(schema, "model_fields", None)
        if not isinstance(model_fields, Mapping):
            return
        schema_name = getattr(schema, "__name__", str(schema))
        for field in model_fields.values():
            sa_column = getattr(field, "sa_column", None)
            if sa_column is None:
                continue
            foreign_keys = getattr(sa_column, "foreign_keys", None) or set()
            for fk in foreign_keys:
                target = getattr(fk, "target_fullname", "")
                if not target or "." not in target:
                    continue
                table_name, column_name = target.split(".", 1)
                known_columns = registry_tables.get(table_name)
                if known_columns is None:
                    signature = (
                        "schema_fk_missing_table",
                        direction,
                        key,
                        schema_name,
                        table_name,
                        column_name,
                    )
                    _warn_once(
                        signature,
                        (
                            "[SCHEMA WARNING] %s key=%r schema=%s has FK %s -> %s.%s, "
                            "but target table is not registered in schema registry."
                        ),
                        direction,
                        key,
                        schema_name,
                        sa_column.name,
                        table_name,
                        column_name,
                    )
                    continue
                if column_name not in known_columns:
                    signature = (
                        "schema_fk_missing_column",
                        direction,
                        key,
                        schema_name,
                        table_name,
                        column_name,
                    )
                    _warn_once(
                        signature,
                        (
                            "[SCHEMA WARNING] %s key=%r schema=%s has FK %s -> %s.%s, "
                            "but target column is not present in registered schema."
                        ),
                        direction,
                        key,
                        schema_name,
                        sa_column.name,
                        table_name,
                        column_name,
                    )
    except Exception as exc:
        schema_name = getattr(schema, "__name__", str(schema))
        signature = ("schema_fk_validation_error", direction, key, schema_name)
        _warn_once(
            signature,
            (
                "[SCHEMA WARNING] %s key=%r schema=%s could not validate foreign keys: %s"
            ),
            direction,
            key,
            schema_name,
            exc,
        )


def _registered_schema_table_columns() -> dict[str, set[str]]:
    try:
        from pilates.database.schema.registry import get_consist_schemas
    except Exception:
        return {}
    table_map: dict[str, set[str]] = {}
    for schema in get_consist_schemas():
        table_name = getattr(schema, "__tablename__", None)
        if not table_name:
            continue
        table_map[str(table_name)] = set(_schema_expected_columns(schema))
    return table_map


def _warn_once(signature: tuple[Any, ...], msg: str, *args: Any) -> None:
    if signature in _schema_warning_signatures:
        return
    _schema_warning_signatures.add(signature)
    logger.warning(msg, *args)


def _with_declared_schema_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    schema = meta.get("schema")
    if schema is None:
        return dict(meta)
    updated = dict(meta)
    updated.setdefault("declared_schema_class", getattr(schema, "__name__", str(schema)))
    updated.setdefault("declared_schema_table", getattr(schema, "__tablename__", None))
    return updated


def _schema_warn_only_enabled() -> bool:
    value = os.getenv("PILATES_SCHEMA_WARN_ONLY")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _schema_expected_columns(schema: Any) -> list[str]:
    model_fields = getattr(schema, "model_fields", None)
    if not isinstance(model_fields, Mapping):
        return []
    names: list[str] = []
    for field_name, field in model_fields.items():
        sa_column = getattr(field, "sa_column", None)
        if sa_column is not None and getattr(sa_column, "name", None):
            names.append(str(sa_column.name))
        else:
            names.append(str(field_name))
    return names


def _observed_columns_and_families(path: str) -> tuple[list[str], dict[str, str]]:
    lowered = path.lower()
    if lowered.endswith((".csv", ".csv.gz")):
        import pandas as pd

        df = pd.read_csv(path, nrows=0)
        return [str(c) for c in df.columns], {}
    if lowered.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema.to_arrow_schema()
        names = [str(name) for name in schema.names]
        families = {
            str(field.name): _arrow_type_family(field.type, pa_module=pa)
            for field in schema
        }
        return names, families
    return [], {}


def _type_mismatch_columns(
    *,
    schema: Any,
    observed_families: Mapping[str, str],
    path: str,
) -> list[str]:
    if not observed_families:
        return []
    if not path.lower().endswith(".parquet"):
        return []
    model_fields = getattr(schema, "model_fields", None)
    if not isinstance(model_fields, Mapping):
        return []
    mismatches: list[str] = []
    for field in model_fields.values():
        sa_column = getattr(field, "sa_column", None)
        if sa_column is None or not getattr(sa_column, "name", None):
            continue
        col_name = str(sa_column.name)
        observed = observed_families.get(col_name)
        if not observed:
            continue
        expected = _sqlalchemy_type_family(getattr(sa_column, "type", None))
        if expected == "unknown":
            continue
        if not _type_families_compatible(expected, observed):
            mismatches.append(f"{col_name}:{expected}->{observed}")
    return sorted(mismatches)


def _arrow_type_family(arrow_type: Any, *, pa_module: Any) -> str:
    if pa_module.types.is_integer(arrow_type):
        return "integer"
    if pa_module.types.is_floating(arrow_type):
        return "float"
    if pa_module.types.is_boolean(arrow_type):
        return "boolean"
    if pa_module.types.is_string(arrow_type) or pa_module.types.is_large_string(
        arrow_type
    ):
        return "string"
    if pa_module.types.is_timestamp(arrow_type) or pa_module.types.is_date(arrow_type):
        return "datetime"
    if pa_module.types.is_decimal(arrow_type):
        return "numeric"
    return "unknown"


def _sqlalchemy_type_family(sa_type: Any) -> str:
    if sa_type is None:
        return "unknown"
    type_name = sa_type.__class__.__name__.lower()
    if "bigint" in type_name or "integer" in type_name:
        return "integer"
    if "float" in type_name or "double" in type_name or "real" in type_name:
        return "float"
    if "numeric" in type_name or "decimal" in type_name:
        return "numeric"
    if "bool" in type_name:
        return "boolean"
    if "char" in type_name or "string" in type_name or "text" in type_name:
        return "string"
    if "date" in type_name or "time" in type_name:
        return "datetime"
    return "unknown"


def _type_families_compatible(expected: str, observed: str) -> bool:
    compatibility = {
        "integer": {"integer"},
        "float": {"float", "integer"},
        "numeric": {"numeric", "float", "integer"},
        "boolean": {"boolean", "integer"},
        "string": {"string"},
        "datetime": {"datetime", "string"},
    }
    return observed in compatibility.get(expected, {expected})


def _render_list(values: Sequence[str], limit: int = 8) -> str:
    if not values:
        return "[]"
    head = list(values[:limit])
    remainder = len(values) - len(head)
    if remainder > 0:
        return f"{head} (+{remainder} more)"
    return str(head)


def _is_schema_check_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".csv", ".csv.gz", ".parquet"))


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
    if _tracker_hashing_strategy != "fast":
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
