from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sqlmodel import Session, col, select, text

_PHYS_SIM_LINKSTATS_KEY = re.compile(
    r"^linkstats_unmodified_parquet__y(?P<year>\d+)__i(?P<iteration>\d+)"
    r"__phys_sim_iter(?P<phys_sim_iteration>\d+)"
    r"(?:__beam_sub_iter(?P<beam_sub_iteration>\d+))?$"
)

_ITERATION_LINKSTATS_KEY = re.compile(
    r"^(?P<prefix>linkstats|linkstats_parquet)_(?P<year>\d+)_(?P<iteration>\d+)"
    r"(?:_sub(?P<beam_sub_iteration>\d+))?$"
)

_LINKSTATS_FACET_KEYS = (
    "artifact_family",
    "year",
    "iteration",
    "phys_sim_iteration",
    "beam_sub_iteration",
)
_DEFAULT_LINKSTATS_SCHEMA_ID = (
    "a0490d6beb290b489cf08c7fd6b93177095d4a9d7d6d4782d613dcbc94e4199b"
)
_METERS_PER_MILE = 1609.344
_SECONDS_PER_HOUR = 3600.0
_MPS_TO_MPH = 2.2369362920544


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and pd.isna(value):
            return "NULL"
        return str(value)
    if pd.isna(value):
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _apply_duckdb_runtime_settings(
    session: Session,
    *,
    duckdb_threads: Optional[int] = None,
    duckdb_memory_limit: Optional[str] = None,
    duckdb_temp_directory: Optional[str | Path] = None,
    duckdb_preserve_insertion_order: Optional[bool] = None,
) -> None:
    """
    Apply per-connection DuckDB runtime settings for heavy analysis queries.
    """
    if duckdb_threads is not None:
        threads = max(1, int(duckdb_threads))
        session.exec(text(f"SET threads = {threads}"))
    if duckdb_memory_limit:
        session.exec(text(f"SET memory_limit = {_sql_literal(str(duckdb_memory_limit))}"))
    if duckdb_temp_directory:
        temp_dir = str(Path(duckdb_temp_directory).expanduser().resolve())
        session.exec(text(f"SET temp_directory = {_sql_literal(temp_dir)}"))
    if duckdb_preserve_insertion_order is not None:
        value = "true" if duckdb_preserve_insertion_order else "false"
        session.exec(text(f"SET preserve_insertion_order = {value}"))


def _stable_linkstats_grouped_view_name(
    *,
    namespace: str,
    artifact_family: Optional[str],
    year: Optional[int],
    iteration: Optional[int],
) -> str:
    payload = "|".join(
        [
            namespace,
            artifact_family or "",
            str(year) if year is not None else "",
            str(iteration) if iteration is not None else "",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"v_linkstats_grouped_{digest}"


def build_archive_mounts(
    *,
    archive_run_dir: str | Path,
    project_root: str | Path,
    output_root: Optional[str | Path] = None,
    extra_mounts: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, str]:
    """
    Build Consist mounts for post-run analysis against archived outputs.

    The key behavior is mapping ``workspace://`` to the archive run directory
    (shared storage), not the node-local execution directory.
    """
    archive_run_dir_path = Path(archive_run_dir).expanduser().resolve()
    project_root_path = Path(project_root).expanduser().resolve()
    mounts: Dict[str, str] = {
        "inputs": str(project_root_path),
        "workspace": str(archive_run_dir_path),
    }
    if output_root is not None:
        mounts["scratch"] = str(Path(output_root).expanduser().resolve())
    if extra_mounts:
        for key, value in extra_mounts.items():
            mounts[str(key)] = str(Path(value).expanduser().resolve())
    return mounts


def create_analysis_tracker(
    *,
    db_path: str | Path,
    archive_run_dir: str | Path,
    project_root: str | Path,
    output_root: Optional[str | Path] = None,
    extra_mounts: Optional[Mapping[str, str | Path]] = None,
    access_mode: str = "analysis",
    hashing_strategy: str = "fast",
):
    """
    Create a Consist tracker configured for archived-run analysis.
    """
    import consist

    db_path_obj = Path(db_path).expanduser().resolve()
    archive_run_dir_path = Path(archive_run_dir).expanduser().resolve()
    if not db_path_obj.exists():
        raise FileNotFoundError(f"Consist DB not found: {db_path_obj}")
    if not archive_run_dir_path.exists():
        raise FileNotFoundError(f"Archive run directory not found: {archive_run_dir_path}")

    mounts = build_archive_mounts(
        archive_run_dir=archive_run_dir_path,
        project_root=project_root,
        output_root=output_root,
        extra_mounts=extra_mounts,
    )
    return consist.Tracker(
        run_dir=str(archive_run_dir_path),
        db_path=str(db_path_obj),
        mounts=mounts,
        access_mode=access_mode,
        hashing_strategy=hashing_strategy,
    )


def get_duckdb_health(
    *,
    db_path: str | Path,
    probe_open: bool = True,
) -> Dict[str, Any]:
    """
    Collect lightweight health diagnostics for a DuckDB file.
    """
    db_path_obj = Path(db_path).expanduser().resolve()
    wal_path = Path(f"{db_path_obj}.wal")

    info: Dict[str, Any] = {
        "db_path": str(db_path_obj),
        "db_exists": db_path_obj.exists(),
        "db_size_bytes": None,
        "db_size_gb": None,
        "wal_exists": wal_path.exists(),
        "wal_size_bytes": None,
        "wal_size_gb": None,
        "duckdb_open_seconds": None,
        "duckdb_open_error": None,
    }

    if info["db_exists"]:
        db_size_bytes = db_path_obj.stat().st_size
        info["db_size_bytes"] = int(db_size_bytes)
        info["db_size_gb"] = float(db_size_bytes) / (1024**3)
    if info["wal_exists"]:
        wal_size_bytes = wal_path.stat().st_size
        info["wal_size_bytes"] = int(wal_size_bytes)
        info["wal_size_gb"] = float(wal_size_bytes) / (1024**3)

    if not probe_open or not info["db_exists"]:
        return info

    try:
        import duckdb

        start = time.perf_counter()
        conn = duckdb.connect(str(db_path_obj), read_only=True)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        info["duckdb_open_seconds"] = float(time.perf_counter() - start)
    except Exception as exc:
        info["duckdb_open_error"] = f"{type(exc).__name__}: {exc}"

    return info


def print_duckdb_health(
    *,
    db_path: str | Path,
    probe_open: bool = True,
) -> Dict[str, Any]:
    """
    Print DuckDB health diagnostics and return the raw metric dictionary.
    """
    info = get_duckdb_health(db_path=db_path, probe_open=probe_open)
    print("DuckDB health:")
    print(f"  DB path: {info['db_path']}")
    print(f"  DB exists: {info['db_exists']}")
    if info["db_size_bytes"] is not None:
        print(
            "  DB size: "
            f"{int(info['db_size_bytes']):,} bytes "
            f"({float(info['db_size_gb']):.3f} GiB)"
        )
    print(f"  WAL exists: {info['wal_exists']}")
    if info["wal_size_bytes"] is not None:
        print(
            "  WAL size: "
            f"{int(info['wal_size_bytes']):,} bytes "
            f"({float(info['wal_size_gb']):.3f} GiB)"
        )
    if probe_open:
        if info["duckdb_open_error"]:
            print(f"  Open probe error: {info['duckdb_open_error']}")
        else:
            print(
                "  Open probe: "
                f"{float(info['duckdb_open_seconds'] or 0.0):.3f} seconds"
            )
    return info


def parse_linkstats_facets_from_key(key: str) -> Dict[str, Any]:
    """
    Parse facet-like fields from BEAM linkstats keys.
    """
    phys = _PHYS_SIM_LINKSTATS_KEY.match(key)
    if phys:
        payload = {
            "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
            "year": int(phys.group("year")),
            "iteration": int(phys.group("iteration")),
            "phys_sim_iteration": int(phys.group("phys_sim_iteration")),
        }
        beam_sub = phys.group("beam_sub_iteration")
        if beam_sub is not None:
            payload["beam_sub_iteration"] = int(beam_sub)
        return payload

    parsed = _ITERATION_LINKSTATS_KEY.match(key)
    if parsed:
        payload = {
            "artifact_family": parsed.group("prefix"),
            "year": int(parsed.group("year")),
            "iteration": int(parsed.group("iteration")),
        }
        beam_sub = parsed.group("beam_sub_iteration")
        if beam_sub is not None:
            payload["beam_sub_iteration"] = int(beam_sub)
        return payload

    return {}


def _decode_artifact_kv_value(row: Any) -> Any:
    value_type = getattr(row, "value_type", None)
    if value_type == "bool":
        return getattr(row, "value_bool", None)
    if value_type == "int":
        value_num = getattr(row, "value_num", None)
        return int(value_num) if value_num is not None else None
    if value_type == "float":
        return getattr(row, "value_num", None)
    if value_type == "str":
        return getattr(row, "value_str", None)
    return None


def _normalize_artifact_id(artifact_id: Any) -> Optional[str]:
    if artifact_id is None:
        return None
    value = str(artifact_id).strip()
    return value or None


def _chunked(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _facet_map_from_artifact_kv(
    tracker: Any,
    artifact_ids: Iterable[Any],
    *,
    namespace: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Load linkstats facet fields from artifact_kv with batched artifact_id queries.
    """
    db = getattr(tracker, "db", None)
    if db is None or not hasattr(db, "session_scope"):
        return {}

    try:
        from consist.models.artifact_kv import ArtifactKV
    except Exception:
        return {}

    unique_ids: list[Any] = []
    seen: set[str] = set()
    for artifact_id in artifact_ids:
        normalized = _normalize_artifact_id(artifact_id)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        unique_ids.append(artifact_id)

    if not unique_ids:
        return {}

    facet_map: Dict[str, Dict[str, Any]] = {}

    def _consume_rows(rows: list[Any]) -> set[str]:
        resolved_ids: set[str] = set()
        for row in rows:
            normalized = _normalize_artifact_id(getattr(row, "artifact_id", None))
            if normalized is None:
                continue
            key_path = str(getattr(row, "key_path", "") or "")
            if key_path not in _LINKSTATS_FACET_KEYS:
                continue
            resolved_ids.add(normalized)
            facet_map.setdefault(normalized, {})[key_path] = _decode_artifact_kv_value(row)
        return resolved_ids

    def _query_rows(artifact_ids_subset: list[Any], ns: Optional[str]) -> list[Any]:
        if not artifact_ids_subset:
            return []
        rows: list[Any] = []
        for chunk in _chunked(artifact_ids_subset, 1000):
            with db.session_scope() as session:
                statement = select(ArtifactKV).where(col(ArtifactKV.artifact_id).in_(chunk))
                if ns is not None:
                    statement = statement.where(ArtifactKV.namespace == ns)
                statement = statement.where(col(ArtifactKV.key_path).in_(_LINKSTATS_FACET_KEYS))
                rows.extend(session.exec(statement).all())
        return rows

    if namespace is None:
        _consume_rows(_query_rows(unique_ids, None))
        return facet_map

    rows_for_namespace = _query_rows(unique_ids, namespace)
    resolved_in_namespace = _consume_rows(rows_for_namespace)
    unresolved_ids = [
        artifact_id
        for artifact_id in unique_ids
        if (_normalize_artifact_id(artifact_id) or "") not in resolved_in_namespace
    ]
    if unresolved_ids:
        _consume_rows(_query_rows(unresolved_ids, None))
    return facet_map


def _parent_run_map(
    tracker: Any,
    run_ids: Iterable[Any],
) -> Dict[str, Optional[str]]:
    """
    Resolve run_id -> parent_run_id from Consist run table when available.
    """
    db = getattr(tracker, "db", None)
    if db is None or not hasattr(db, "session_scope"):
        return {}
    try:
        from consist.models.run import Run
    except Exception:
        return {}

    unique_ids: list[str] = []
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id is None:
            continue
        normalized = str(run_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_ids.append(normalized)
    if not unique_ids:
        return {}

    mapping: Dict[str, Optional[str]] = {}
    for chunk in _chunked(unique_ids, 1000):
        with db.session_scope() as session:
            statement = select(Run).where(col(Run.id).in_(chunk))
            rows = session.exec(statement).all()
        for row in rows:
            row_id = str(getattr(row, "id", "") or "").strip()
            if not row_id:
                continue
            parent = getattr(row, "parent_run_id", None)
            parent_normalized = str(parent).strip() if parent is not None else None
            mapping[row_id] = parent_normalized or None
    return mapping


def _filter_linkstats_facet(facet: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: facet.get(key)
        for key in _LINKSTATS_FACET_KEYS
        if key in facet and facet.get(key) is not None
    }


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_artifacts_by_key_prefix_sqlmodel(
    tracker: Any,
    *,
    key_prefix: str,
    limit: int,
) -> list[Any]:
    """
    Fallback artifact discovery using explicit SQLModel query syntax.
    """
    db = getattr(tracker, "db", None)
    if db is None or not hasattr(db, "session_scope"):
        return []
    try:
        from consist.models.artifact import Artifact
    except Exception:
        return []

    with db.session_scope() as session:
        statement = (
            select(Artifact)
            .where(col(Artifact.key).like(f"{key_prefix}%"))
            .order_by(col(Artifact.created_at).desc())
            .limit(limit)
        )
        return list(session.exec(statement).all())


def find_linkstats_artifacts(
    tracker: Any,
    *,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    artifact_family: str = "linkstats_unmodified_phys_sim_iter_parquet",
    namespace: str = "beam",
    limit: int = 10000,
) -> pd.DataFrame:
    """
    Query linkstats artifacts by indexed facet values and return a DataFrame.
    """
    params = [f"{namespace}.artifact_family={artifact_family}"]
    if year is not None:
        params.append(f"{namespace}.year={year}")
    if iteration is not None:
        params.append(f"{namespace}.iteration={iteration}")

    artifacts = tracker.find_artifacts_by_params(
        params=params,
        namespace=namespace,
        limit=limit,
    )
    if not artifacts:
        # Fallback for runs where artifact facet indexing was absent/incomplete.
        artifacts = _find_artifacts_by_key_prefix_sqlmodel(
            tracker, key_prefix="linkstats", limit=limit
        )

    if not artifacts:
        return pd.DataFrame(
            columns=[
                "key",
                "artifact_id",
                "run_id",
                "parent_run_id",
                "header_run_id",
                "container_uri",
                "facet_source",
                "artifact_family",
                "year",
                "iteration",
                "phys_sim_iteration",
                "beam_sub_iteration",
            ]
        )

    artifact_ids = [getattr(artifact, "id", None) for artifact in artifacts]
    run_ids = [getattr(artifact, "run_id", None) for artifact in artifacts]
    facet_map_from_kv = _facet_map_from_artifact_kv(
        tracker,
        artifact_ids,
        namespace=namespace,
    )
    parent_map = _parent_run_map(tracker, run_ids)

    output_rows = []
    for artifact in artifacts:
        key = str(getattr(artifact, "key", "") or "")
        meta = dict(getattr(artifact, "meta", None) or {})
        artifact_id = getattr(artifact, "id", None)
        normalized_artifact_id = _normalize_artifact_id(artifact_id)

        facet = {}
        facet_source = "none"

        facet_from_kv = {}
        if normalized_artifact_id is not None:
            facet_from_kv = _filter_linkstats_facet(
                facet_map_from_kv.get(normalized_artifact_id, {})
            )
        if facet_from_kv:
            facet.update(facet_from_kv)
            facet_source = "artifact_kv"

        meta_facet = meta.get("facet") if isinstance(meta.get("facet"), dict) else {}
        if meta_facet:
            facet_from_meta = _filter_linkstats_facet(meta_facet)
            if facet_from_meta:
                facet = {**facet_from_meta, **facet}
                if facet_source == "none":
                    facet_source = "artifact_meta"

        if not facet:
            facet = parse_linkstats_facets_from_key(key)
            if facet:
                facet_source = "key_parse"

        family = facet.get("artifact_family")
        row_year = _to_int_or_none(facet.get("year"))
        row_iteration = _to_int_or_none(facet.get("iteration"))
        if artifact_family and family != artifact_family:
            continue
        if year is not None and row_year is not None and row_year != int(year):
            continue
        if iteration is not None and row_iteration is not None and row_iteration != int(iteration):
            continue

        output_rows.append(
            {
                "key": key,
                "artifact_id": str(artifact_id or ""),
                "run_id": getattr(artifact, "run_id", None),
                "parent_run_id": parent_map.get(str(getattr(artifact, "run_id", "") or "").strip()),
                "header_run_id": parent_map.get(str(getattr(artifact, "run_id", "") or "").strip()),
                "container_uri": getattr(artifact, "container_uri", None),
                "facet_source": facet_source,
                "artifact_family": facet.get("artifact_family"),
                "year": facet.get("year"),
                "iteration": facet.get("iteration"),
                "phys_sim_iteration": facet.get("phys_sim_iteration"),
                "beam_sub_iteration": facet.get("beam_sub_iteration"),
            }
        )

    frame = pd.DataFrame(output_rows)
    for column in ("year", "iteration", "phys_sim_iteration", "beam_sub_iteration"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    sort_cols = [
        col
        for col in ("year", "iteration", "beam_sub_iteration", "phys_sim_iteration", "key")
        if col in frame.columns
    ]
    return frame.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def assign_effective_beam_sub_iteration(
    artifacts_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("run_id", "year", "iteration", "phys_sim_iteration"),
    source_col: str = "beam_sub_iteration",
    output_col: str = "beam_sub_iteration_effective",
    ordinal_output_col: Optional[str] = "beam_sub_iteration_ordinal",
) -> pd.DataFrame:
    """
    Derive an effective sub-iteration index for promoted final BEAM sub-iterations.

    For each group, null sub-iteration values are mapped to `max(non-null) + 1`.
    This avoids hard-coding the number of BEAM sub-iterations.
    """
    frame = artifacts_df.copy()
    if frame.empty:
        if output_col not in frame.columns:
            frame[output_col] = pd.Series(dtype="float64")
        if ordinal_output_col and ordinal_output_col not in frame.columns:
            frame[ordinal_output_col] = pd.Series(dtype="float64")
        return frame

    missing = [col for col in (*group_cols, source_col) if col not in frame.columns]
    if missing:
        raise ValueError(
            "assign_effective_beam_sub_iteration missing required columns: "
            f"{missing}"
        )

    frame[source_col] = pd.to_numeric(frame[source_col], errors="coerce")
    max_sub = frame.groupby(list(group_cols), dropna=False)[source_col].transform("max")

    effective = frame[source_col].copy()
    fill_mask = effective.isna() & max_sub.notna()
    effective.loc[fill_mask] = max_sub.loc[fill_mask] + 1
    frame[output_col] = effective

    if ordinal_output_col:
        ordinal = frame[output_col] + 1
        frame[ordinal_output_col] = ordinal.where(frame[output_col].notna())

    return frame


def _first_non_null_string(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    for value in series:
        if pd.isna(value):
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _single_string_value(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    values: set[str] = set()
    for value in series:
        if pd.isna(value):
            continue
        normalized = str(value).strip()
        if normalized:
            values.add(normalized)
    if len(values) == 1:
        return next(iter(values))
    return None


def _single_numeric_value(series: Optional[pd.Series]) -> Optional[int]:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    unique = sorted({int(value) for value in numeric.tolist()})
    if len(unique) == 1:
        return unique[0]
    return None


def _resolve_linkstats_schema_model() -> Optional[type]:
    try:
        from pilates.database.schema.beam_schema import BeamLinkstats
    except Exception:
        return None
    return BeamLinkstats


def _create_linkstats_grouped_view(
    *,
    tracker: Any,
    artifacts_df: pd.DataFrame,
    namespace: str = "beam",
    drivers: Optional[list[str]] = None,
    mode: str = "hybrid",
    missing_files: str = "warn",
    view_name: Optional[str] = None,
    schema_id: str = _DEFAULT_LINKSTATS_SCHEMA_ID,
    use_facet_filters: bool = True,
) -> str:
    artifact_family = _first_non_null_string(artifacts_df.get("artifact_family"))
    year = _single_numeric_value(artifacts_df.get("year"))
    iteration = _single_numeric_value(artifacts_df.get("iteration"))
    run_id = _single_string_value(artifacts_df.get("run_id"))
    facet_source_values = (
        artifacts_df.get("facet_source").dropna().astype(str)
        if "facet_source" in artifacts_df.columns
        else pd.Series(dtype=str)
    )
    has_artifact_kv_facets = bool((facet_source_values == "artifact_kv").any())

    params: list[str] = []
    if use_facet_filters and has_artifact_kv_facets and artifact_family:
        params.append(f"{namespace}.artifact_family={artifact_family}")

    grouped_view_name = view_name or _stable_linkstats_grouped_view_name(
        namespace=namespace,
        artifact_family=artifact_family,
        year=year,
        iteration=iteration,
    )

    selector: Dict[str, Any] = {}
    if schema_id:
        selector["schema_id"] = schema_id
    else:
        schema_model = _resolve_linkstats_schema_model()
        if schema_model is None:
            raise ValueError(
                "Unable to resolve BeamLinkstats schema model and no schema_id provided."
            )
        selector["schema"] = schema_model

    tracker.create_grouped_view(
        view_name=grouped_view_name,
        namespace=namespace,
        params=params or None,
        drivers=drivers or ["parquet"],
        attach_facets=list(_LINKSTATS_FACET_KEYS),
        include_system_columns=True,
        mode=mode,
        if_exists="replace",
        missing_files=missing_files,
        run_id=run_id,
        year=year,
        iteration=iteration,
        **selector,
    )
    return grouped_view_name


def _summarize_linkstats_grouped_view(
    *,
    tracker: Any,
    view_name: str,
    artifact_ids: Optional[list[str]] = None,
    traveltime_weighting: Literal["unweighted", "volume_weighted"] = "unweighted",
) -> pd.DataFrame:
    volume_sum_expr = "SUM(CAST(src.volume AS DOUBLE))"
    vmt_miles_expr = (
        f"SUM(CAST(src.volume AS DOUBLE) * CAST(src.length AS DOUBLE)) / {_METERS_PER_MILE}"
    )
    freeflow_time_expr = (
        "CASE "
        "WHEN CAST(src.freespeed AS DOUBLE) > 0 "
        "THEN CAST(src.length AS DOUBLE) / CAST(src.freespeed AS DOUBLE) "
        "ELSE NULL END"
    )
    delay_seconds_per_vehicle_expr = (
        "GREATEST(CAST(src.traveltime AS DOUBLE) - COALESCE("
        + freeflow_time_expr
        + ", CAST(src.traveltime AS DOUBLE)), 0.0)"
    )
    total_delay_hours_expr = (
        "SUM(CAST(src.volume AS DOUBLE) * "
        + delay_seconds_per_vehicle_expr
        + f") / {_SECONDS_PER_HOUR}"
    )
    vht_hours_expr = (
        "SUM(CAST(src.volume AS DOUBLE) * CAST(src.traveltime AS DOUBLE)) "
        f"/ {_SECONDS_PER_HOUR}"
    )
    vmt_per_vht_expr = (
        "CASE "
        f"WHEN ({vht_hours_expr}) > 0 THEN ({vmt_miles_expr}) / ({vht_hours_expr}) "
        "ELSE NULL END"
    )
    vht_per_vmt_expr = (
        "CASE "
        f"WHEN ({vmt_miles_expr}) > 0 THEN ({vht_hours_expr}) / ({vmt_miles_expr}) "
        "ELSE NULL END"
    )

    if traveltime_weighting == "volume_weighted":
        traveltime_mean_expr = (
            "CASE "
            f"WHEN {volume_sum_expr} > 0 "
            "THEN SUM(CAST(src.volume AS DOUBLE) * CAST(src.traveltime AS DOUBLE)) "
            f"/ {volume_sum_expr} "
            "ELSE NULL END"
        )
    else:
        traveltime_mean_expr = "AVG(src.traveltime)"

    quoted_view = _quote_ident(view_name)
    query = f"""
        SELECT
            src.consist_artifact_id AS artifact_id,
            COUNT(*) AS row_count,
            COUNT(DISTINCT src.link) AS distinct_links,
            {volume_sum_expr} AS volume_sum,
            {vmt_miles_expr} AS vmt_miles,
            {vht_hours_expr} AS vht_hours,
            {total_delay_hours_expr} AS total_delay_hours,
            {vmt_per_vht_expr} AS vmt_per_vht_mph,
            {vht_per_vmt_expr} AS vht_per_vmt_hours_per_mile,
            {traveltime_mean_expr} AS traveltime_mean,
            quantile_cont(src.traveltime, 0.95) AS traveltime_p95
        FROM {quoted_view} src
    """
    if artifact_ids:
        unique_ids = sorted({str(v) for v in artifact_ids if str(v).strip()})
        if unique_ids:
            values_sql = ", ".join(f"({_sql_literal(v)})" for v in unique_ids)
            query = f"""
                WITH target_artifacts(artifact_id) AS (
                    VALUES {values_sql}
                )
                SELECT
                    src.consist_artifact_id AS artifact_id,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT src.link) AS distinct_links,
                    {volume_sum_expr} AS volume_sum,
                    {vmt_miles_expr} AS vmt_miles,
                    {vht_hours_expr} AS vht_hours,
                    {total_delay_hours_expr} AS total_delay_hours,
                    {vmt_per_vht_expr} AS vmt_per_vht_mph,
                    {vht_per_vmt_expr} AS vht_per_vmt_hours_per_mile,
                    {traveltime_mean_expr} AS traveltime_mean,
                    quantile_cont(src.traveltime, 0.95) AS traveltime_p95
                FROM {quoted_view} src
                JOIN target_artifacts ta
                  ON src.consist_artifact_id = ta.artifact_id
            """
    query += "\nGROUP BY 1"

    with Session(tracker.engine) as session:
        rows = session.exec(text(query)).all()

    frame = pd.DataFrame(
        rows,
        columns=[
            "artifact_id",
            "row_count",
            "distinct_links",
            "volume_sum",
            "vmt_miles",
            "vht_hours",
            "total_delay_hours",
            "vmt_per_vht_mph",
            "vht_per_vmt_hours_per_mile",
            "traveltime_mean",
            "traveltime_p95",
        ],
    )
    if frame.empty:
        return frame
    frame["artifact_id"] = frame["artifact_id"].astype(str)
    for column in (
        "row_count",
        "distinct_links",
        "volume_sum",
        "vmt_miles",
        "vht_hours",
        "total_delay_hours",
        "vmt_per_vht_mph",
        "vht_per_vmt_hours_per_mile",
        "traveltime_mean",
        "traveltime_p95",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["row_count"] = frame["row_count"].astype(int)
    frame["distinct_links"] = frame["distinct_links"].astype(int)
    return frame


def _summarize_linkstats_metric_deltas_from_summary(
    *,
    tracker: Any,
    view_name: str,
    summary_df: pd.DataFrame,
    metric_column: Literal["traveltime", "volume"],
) -> pd.DataFrame:
    """
    Compute consecutive phys-sim deltas for one metric fully in DuckDB.
    """
    if summary_df.empty:
        return pd.DataFrame()

    metric_prefix = "traveltime" if metric_column == "traveltime" else "volume"
    metric_alias_prev = f"{metric_prefix}_previous"
    metric_alias_curr = f"{metric_prefix}_current"

    rows_sql: list[str] = []
    for row in summary_df.itertuples(index=False):
        rows_sql.append(
            "("
            + ", ".join(
                [
                    _sql_literal(getattr(row, "year", None)),
                    _sql_literal(getattr(row, "iteration", None)),
                    _sql_literal(getattr(row, "beam_sub_iteration", None)),
                    _sql_literal(getattr(row, "phys_sim_iteration", None)),
                    _sql_literal(str(getattr(row, "artifact_id", "") or "")),
                    _sql_literal(getattr(row, "key", None)),
                ]
            )
            + ")"
        )

    if not rows_sql:
        return pd.DataFrame()

    quoted_view = _quote_ident(view_name)
    query = text(
        f"""
        WITH summary_input(
            year,
            iteration,
            beam_sub_iteration,
            phys_sim_iteration,
            artifact_id,
            key
        ) AS (
            VALUES
                {", ".join(rows_sql)}
        ),
        ordered_summary AS (
            SELECT
                CAST(year AS BIGINT) AS year,
                CAST(iteration AS BIGINT) AS iteration,
                CAST(beam_sub_iteration AS BIGINT) AS beam_sub_iteration,
                CAST(phys_sim_iteration AS BIGINT) AS phys_sim_iteration,
                CAST(artifact_id AS VARCHAR) AS artifact_id,
                key,
                LAG(CAST(phys_sim_iteration AS BIGINT)) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS phys_sim_iteration_prev,
                LAG(CAST(artifact_id AS VARCHAR)) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS artifact_id_prev,
                LAG(key) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS key_prev
            FROM summary_input
            WHERE artifact_id IS NOT NULL
              AND CAST(artifact_id AS VARCHAR) != ''
              AND phys_sim_iteration IS NOT NULL
        ),
        pairs AS (
            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY
                        year,
                        iteration,
                        beam_sub_iteration,
                        phys_sim_iteration,
                        artifact_id
                ) AS pair_id,
                year,
                iteration,
                beam_sub_iteration,
                phys_sim_iteration_prev,
                phys_sim_iteration AS phys_sim_iteration_curr,
                artifact_id_prev,
                artifact_id AS artifact_id_curr,
                key_prev,
                key AS key_curr
            FROM ordered_summary
            WHERE artifact_id_prev IS NOT NULL
              AND phys_sim_iteration_prev IS NOT NULL
        ),
        prev_rows AS (
            SELECT
                p.pair_id,
                p.year,
                p.iteration,
                p.beam_sub_iteration,
                p.phys_sim_iteration_prev,
                p.phys_sim_iteration_curr,
                p.artifact_id_prev,
                p.artifact_id_curr,
                p.key_prev,
                p.key_curr,
                link,
                hour,
                CAST(a.{metric_column} AS DOUBLE) AS metric_prev
            FROM pairs p
            JOIN {quoted_view} a ON a.consist_artifact_id = p.artifact_id_prev
        ),
        curr_rows AS (
            SELECT
                p.pair_id,
                p.year,
                p.iteration,
                p.beam_sub_iteration,
                p.phys_sim_iteration_prev,
                p.phys_sim_iteration_curr,
                p.artifact_id_prev,
                p.artifact_id_curr,
                p.key_prev,
                p.key_curr,
                link,
                hour,
                CAST(a.{metric_column} AS DOUBLE) AS metric_curr
            FROM pairs p
            JOIN {quoted_view} a ON a.consist_artifact_id = p.artifact_id_curr
        ),
        joined AS (
            SELECT
                COALESCE(c.year, p.year) AS year,
                COALESCE(c.iteration, p.iteration) AS iteration,
                COALESCE(c.beam_sub_iteration, p.beam_sub_iteration) AS beam_sub_iteration,
                COALESCE(c.phys_sim_iteration_prev, p.phys_sim_iteration_prev) AS phys_sim_iteration_prev,
                COALESCE(c.phys_sim_iteration_curr, p.phys_sim_iteration_curr) AS phys_sim_iteration_curr,
                COALESCE(c.artifact_id_prev, p.artifact_id_prev) AS artifact_id_prev,
                COALESCE(c.artifact_id_curr, p.artifact_id_curr) AS artifact_id_curr,
                COALESCE(c.key_prev, p.key_prev) AS key_prev,
                COALESCE(c.key_curr, p.key_curr) AS key_curr,
                COALESCE(c.link, p.link) AS link,
                COALESCE(c.hour, p.hour) AS hour,
                COALESCE(p.metric_prev, 0.0) AS {metric_alias_prev},
                COALESCE(c.metric_curr, 0.0) AS {metric_alias_curr}
            FROM curr_rows c
            FULL OUTER JOIN prev_rows p
                ON c.pair_id = p.pair_id
               AND c.link = p.link
               AND c.hour = p.hour
        )
        SELECT
            year,
            iteration,
            beam_sub_iteration,
            phys_sim_iteration_prev,
            phys_sim_iteration_curr,
            artifact_id_prev,
            artifact_id_curr,
            key_prev,
            key_curr,
            COUNT(*) AS group_count,
            AVG({metric_alias_curr} - {metric_alias_prev}) AS {metric_prefix}_delta_mean,
            AVG(ABS({metric_alias_curr} - {metric_alias_prev})) AS {metric_prefix}_delta_abs_mean
        FROM joined
        GROUP BY 1,2,3,4,5,6,7,8,9
        ORDER BY year, iteration, beam_sub_iteration, phys_sim_iteration_curr
        """
    )
    with Session(tracker.engine) as session:
        result_rows = session.exec(query).all()

    frame = pd.DataFrame(
        result_rows,
        columns=[
            "year",
            "iteration",
            "beam_sub_iteration",
            "phys_sim_iteration_prev",
            "phys_sim_iteration_curr",
            "artifact_id_prev",
            "artifact_id_curr",
            "key_prev",
            "key_curr",
            "group_count",
            f"{metric_prefix}_delta_mean",
            f"{metric_prefix}_delta_abs_mean",
        ],
    )
    if frame.empty:
        return frame
    for column in (
        "year",
        "iteration",
        "beam_sub_iteration",
        "phys_sim_iteration_prev",
        "phys_sim_iteration_curr",
        "group_count",
        f"{metric_prefix}_delta_mean",
        f"{metric_prefix}_delta_abs_mean",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["group_count"] = frame["group_count"].fillna(0).astype(int)
    return frame


def _resolve_delta_grouped_view_name(summary_df: pd.DataFrame) -> str:
    required_cols = {"year", "iteration", "phys_sim_iteration", "artifact_id", "view_name"}
    missing = sorted(required_cols - set(summary_df.columns))
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")
    non_null_views = summary_df["view_name"].dropna().astype(str).unique().tolist()
    if len(non_null_views) != 1:
        raise ValueError("summary_df must reference exactly one grouped view in 'view_name'")
    return non_null_views[0]


def summarize_linkstats_traveltime_deltas(
    summary_df: pd.DataFrame,
    *,
    tracker: Any,
) -> pd.DataFrame:
    """
    Compute consecutive phys-sim travel time deltas for one grouped view.
    """
    if summary_df.empty:
        return pd.DataFrame()
    grouped_view_name = _resolve_delta_grouped_view_name(summary_df)
    delta_df = _summarize_linkstats_metric_deltas_from_summary(
        tracker=tracker,
        view_name=grouped_view_name,
        summary_df=summary_df,
        metric_column="traveltime",
    )
    if delta_df.empty:
        return delta_df
    delta_df["view_prev"] = grouped_view_name
    delta_df["view_curr"] = grouped_view_name
    return delta_df


def _summarize_linkstats_traveltime_deltas_hourly_weighted_from_summary(
    *,
    tracker: Any,
    view_name: str,
    summary_df: pd.DataFrame,
    exclude_zero_volume: bool = True,
) -> pd.DataFrame:
    """
    Compute travel-time deltas using hourly, volume-weighted aggregates.
    """
    if summary_df.empty:
        return pd.DataFrame()

    rows_sql: list[str] = []
    for row in summary_df.itertuples(index=False):
        rows_sql.append(
            "("
            + ", ".join(
                [
                    _sql_literal(getattr(row, "year", None)),
                    _sql_literal(getattr(row, "iteration", None)),
                    _sql_literal(getattr(row, "beam_sub_iteration", None)),
                    _sql_literal(getattr(row, "phys_sim_iteration", None)),
                    _sql_literal(str(getattr(row, "artifact_id", "") or "")),
                    _sql_literal(getattr(row, "key", None)),
                ]
            )
            + ")"
        )
    if not rows_sql:
        return pd.DataFrame()

    volume_filter = "AND volume > 0" if exclude_zero_volume else ""
    quoted_view = _quote_ident(view_name)
    query = text(
        f"""
        WITH summary_input(
            year,
            iteration,
            beam_sub_iteration,
            phys_sim_iteration,
            artifact_id,
            key
        ) AS (
            VALUES
                {", ".join(rows_sql)}
        ),
        ordered_summary AS (
            SELECT
                CAST(year AS BIGINT) AS year,
                CAST(iteration AS BIGINT) AS iteration,
                CAST(beam_sub_iteration AS BIGINT) AS beam_sub_iteration,
                CAST(phys_sim_iteration AS BIGINT) AS phys_sim_iteration,
                CAST(artifact_id AS VARCHAR) AS artifact_id,
                key,
                LAG(CAST(phys_sim_iteration AS BIGINT)) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS phys_sim_iteration_prev,
                LAG(CAST(artifact_id AS VARCHAR)) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS artifact_id_prev,
                LAG(key) OVER (
                    PARTITION BY
                        CAST(year AS BIGINT),
                        CAST(iteration AS BIGINT),
                        CAST(beam_sub_iteration AS BIGINT)
                    ORDER BY CAST(phys_sim_iteration AS BIGINT), CAST(artifact_id AS VARCHAR)
                ) AS key_prev
            FROM summary_input
            WHERE artifact_id IS NOT NULL
              AND CAST(artifact_id AS VARCHAR) != ''
              AND phys_sim_iteration IS NOT NULL
        ),
        pairs AS (
            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY
                        year,
                        iteration,
                        beam_sub_iteration,
                        phys_sim_iteration,
                        artifact_id
                ) AS pair_id,
                year,
                iteration,
                beam_sub_iteration,
                phys_sim_iteration_prev,
                phys_sim_iteration AS phys_sim_iteration_curr,
                artifact_id_prev,
                artifact_id AS artifact_id_curr,
                key_prev,
                key AS key_curr
            FROM ordered_summary
            WHERE artifact_id_prev IS NOT NULL
              AND phys_sim_iteration_prev IS NOT NULL
        ),
        relevant_artifacts(artifact_id) AS (
            SELECT artifact_id_prev FROM pairs
            UNION
            SELECT artifact_id_curr FROM pairs
        ),
        hourly AS (
            SELECT
                consist_artifact_id AS artifact_id,
                hour,
                SUM(CAST(volume AS DOUBLE)) AS hour_volume,
                CASE
                    WHEN SUM(CAST(volume AS DOUBLE)) > 0
                    THEN SUM(CAST(volume AS DOUBLE) * CAST(traveltime AS DOUBLE))
                         / SUM(CAST(volume AS DOUBLE))
                    ELSE NULL
                END AS hour_traveltime_weighted
            FROM {quoted_view}
            WHERE consist_artifact_id IN (
                SELECT artifact_id FROM relevant_artifacts
            )
            {volume_filter}
            GROUP BY 1, 2
        ),
        prev_hours AS (
            SELECT
                p.pair_id,
                p.year,
                p.iteration,
                p.beam_sub_iteration,
                p.phys_sim_iteration_prev,
                p.phys_sim_iteration_curr,
                p.artifact_id_prev,
                p.artifact_id_curr,
                p.key_prev,
                p.key_curr,
                h.hour,
                h.hour_volume AS prev_volume,
                h.hour_traveltime_weighted AS prev_tt
            FROM pairs p
            JOIN hourly h ON h.artifact_id = p.artifact_id_prev
        ),
        curr_hours AS (
            SELECT
                p.pair_id,
                p.year,
                p.iteration,
                p.beam_sub_iteration,
                p.phys_sim_iteration_prev,
                p.phys_sim_iteration_curr,
                p.artifact_id_prev,
                p.artifact_id_curr,
                p.key_prev,
                p.key_curr,
                h.hour,
                h.hour_volume AS curr_volume,
                h.hour_traveltime_weighted AS curr_tt
            FROM pairs p
            JOIN hourly h ON h.artifact_id = p.artifact_id_curr
        ),
        joined AS (
            SELECT
                COALESCE(c.year, p.year) AS year,
                COALESCE(c.iteration, p.iteration) AS iteration,
                COALESCE(c.beam_sub_iteration, p.beam_sub_iteration) AS beam_sub_iteration,
                COALESCE(c.phys_sim_iteration_prev, p.phys_sim_iteration_prev) AS phys_sim_iteration_prev,
                COALESCE(c.phys_sim_iteration_curr, p.phys_sim_iteration_curr) AS phys_sim_iteration_curr,
                COALESCE(c.artifact_id_prev, p.artifact_id_prev) AS artifact_id_prev,
                COALESCE(c.artifact_id_curr, p.artifact_id_curr) AS artifact_id_curr,
                COALESCE(c.key_prev, p.key_prev) AS key_prev,
                COALESCE(c.key_curr, p.key_curr) AS key_curr,
                COALESCE(c.hour, p.hour) AS hour,
                COALESCE(p.prev_volume, 0.0) AS prev_volume,
                COALESCE(c.curr_volume, 0.0) AS curr_volume,
                COALESCE(p.prev_tt, 0.0) AS prev_tt,
                COALESCE(c.curr_tt, 0.0) AS curr_tt
            FROM curr_hours c
            FULL OUTER JOIN prev_hours p
                ON c.pair_id = p.pair_id
               AND c.hour = p.hour
        )
        SELECT
            year,
            iteration,
            beam_sub_iteration,
            phys_sim_iteration_prev,
            phys_sim_iteration_curr,
            artifact_id_prev,
            artifact_id_curr,
            key_prev,
            key_curr,
            COUNT(*) AS hour_count,
            SUM(prev_volume) AS prev_volume_total,
            SUM(curr_volume) AS curr_volume_total,
            AVG(curr_tt - prev_tt) AS traveltime_delta_mean,
            AVG(ABS(curr_tt - prev_tt)) AS traveltime_delta_abs_mean
        FROM joined
        GROUP BY 1,2,3,4,5,6,7,8,9
        ORDER BY year, iteration, beam_sub_iteration, phys_sim_iteration_curr
        """
    )
    with Session(tracker.engine) as session:
        result_rows = session.exec(query).all()

    frame = pd.DataFrame(
        result_rows,
        columns=[
            "year",
            "iteration",
            "beam_sub_iteration",
            "phys_sim_iteration_prev",
            "phys_sim_iteration_curr",
            "artifact_id_prev",
            "artifact_id_curr",
            "key_prev",
            "key_curr",
            "hour_count",
            "prev_volume_total",
            "curr_volume_total",
            "traveltime_delta_mean",
            "traveltime_delta_abs_mean",
        ],
    )
    if frame.empty:
        return frame
    for column in (
        "year",
        "iteration",
        "beam_sub_iteration",
        "phys_sim_iteration_prev",
        "phys_sim_iteration_curr",
        "hour_count",
        "prev_volume_total",
        "curr_volume_total",
        "traveltime_delta_mean",
        "traveltime_delta_abs_mean",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hour_count"] = frame["hour_count"].fillna(0).astype(int)
    return frame


def summarize_linkstats_traveltime_deltas_hourly_weighted(
    summary_df: pd.DataFrame,
    *,
    tracker: Any,
    exclude_zero_volume: bool = True,
) -> pd.DataFrame:
    """
    Compute consecutive phys-sim travel-time deltas using hourly volume weighting.
    """
    if summary_df.empty:
        return pd.DataFrame()
    grouped_view_name = _resolve_delta_grouped_view_name(summary_df)
    delta_df = _summarize_linkstats_traveltime_deltas_hourly_weighted_from_summary(
        tracker=tracker,
        view_name=grouped_view_name,
        summary_df=summary_df,
        exclude_zero_volume=exclude_zero_volume,
    )
    if delta_df.empty:
        return delta_df
    delta_df["view_prev"] = grouped_view_name
    delta_df["view_curr"] = grouped_view_name
    return delta_df


def summarize_linkstats_volume_deltas(
    summary_df: pd.DataFrame,
    *,
    tracker: Any,
) -> pd.DataFrame:
    """
    Compute consecutive phys-sim volume deltas for one grouped view.
    """
    if summary_df.empty:
        return pd.DataFrame()
    grouped_view_name = _resolve_delta_grouped_view_name(summary_df)
    delta_df = _summarize_linkstats_metric_deltas_from_summary(
        tracker=tracker,
        view_name=grouped_view_name,
        summary_df=summary_df,
        metric_column="volume",
    )
    if delta_df.empty:
        return delta_df
    delta_df["view_prev"] = grouped_view_name
    delta_df["view_curr"] = grouped_view_name
    return delta_df


def summarize_linkstats_artifacts(
    artifacts_df: pd.DataFrame,
    *,
    tracker: Any,
    namespace: str = "beam",
    grouped_mode: str = "hybrid",
    grouped_missing_files: str = "warn",
    grouped_view_name: Optional[str] = None,
    grouped_schema_id: str = _DEFAULT_LINKSTATS_SCHEMA_ID,
    grouped_drivers: Optional[list[str]] = None,
    traveltime_weighting: Literal["unweighted", "volume_weighted"] = "unweighted",
) -> pd.DataFrame:
    """
    Compute per-artifact summary metrics using Consist hybrid views.
    """
    if artifacts_df.empty:
        return artifacts_df.copy()
    if "artifact_id" not in artifacts_df.columns:
        raise ValueError("artifacts_df must contain an 'artifact_id' column")

    grouped_view = _create_linkstats_grouped_view(
        tracker=tracker,
        artifacts_df=artifacts_df,
        namespace=namespace,
        drivers=grouped_drivers,
        mode=grouped_mode,
        missing_files=grouped_missing_files,
        view_name=grouped_view_name,
        schema_id=grouped_schema_id,
        use_facet_filters=True,
    )

    stats_df = _summarize_linkstats_grouped_view(
        tracker=tracker,
        view_name=grouped_view,
        artifact_ids=artifacts_df["artifact_id"].astype(str).tolist(),
        traveltime_weighting=traveltime_weighting,
    )
    if stats_df.empty and not artifacts_df.empty:
        grouped_view = _create_linkstats_grouped_view(
            tracker=tracker,
            artifacts_df=artifacts_df,
            namespace=namespace,
            drivers=grouped_drivers,
            mode=grouped_mode,
            missing_files=grouped_missing_files,
            view_name=grouped_view_name,
            schema_id=grouped_schema_id,
            use_facet_filters=False,
        )
        stats_df = _summarize_linkstats_grouped_view(
            tracker=tracker,
            view_name=grouped_view,
            artifact_ids=artifacts_df["artifact_id"].astype(str).tolist(),
            traveltime_weighting=traveltime_weighting,
        )
    if stats_df.empty:
        return pd.DataFrame()

    metadata_df = artifacts_df.copy()
    metadata_df["artifact_id"] = metadata_df["artifact_id"].astype(str)
    metadata_df = metadata_df.drop_duplicates(subset=["artifact_id"], keep="first")

    summary_df = metadata_df.merge(stats_df, on="artifact_id", how="inner")
    summary_df["view_name"] = grouped_view
    return summary_df


def summarize_linkstats_deltas(
    summary_df: pd.DataFrame,
    *,
    tracker: Any,
) -> pd.DataFrame:
    """
    Compute consecutive phys-sim travel time and volume deltas.
    """
    if summary_df.empty:
        return pd.DataFrame()

    travel_df = summarize_linkstats_traveltime_deltas(summary_df, tracker=tracker)
    volume_df = summarize_linkstats_volume_deltas(summary_df, tracker=tracker)
    if travel_df.empty:
        return volume_df
    if volume_df.empty:
        return travel_df

    key_cols = [
        "year",
        "iteration",
        "beam_sub_iteration",
        "phys_sim_iteration_prev",
        "phys_sim_iteration_curr",
        "artifact_id_prev",
        "artifact_id_curr",
        "key_prev",
        "key_curr",
        "view_prev",
        "view_curr",
    ]
    volume_cols = key_cols + ["group_count", "volume_delta_mean", "volume_delta_abs_mean"]
    merged = travel_df.merge(volume_df[volume_cols], on=key_cols, how="outer", suffixes=("", "_volume"))
    if "group_count_volume" in merged.columns:
        merged["group_count"] = merged["group_count"].fillna(merged["group_count_volume"])
        merged = merged.drop(columns=["group_count_volume"])
    return merged


def _prepare_linkstats_pca_artifact_index(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a stable artifact ordering for matrix rows.
    """
    required_cols = {"artifact_id"}
    missing = sorted(required_cols - set(summary_df.columns))
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    frame = summary_df.copy()
    frame["artifact_id"] = frame["artifact_id"].astype(str).str.strip()
    frame = frame[frame["artifact_id"] != ""]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "observation_index",
                "artifact_id",
                "run_id",
                "parent_run_id",
                "header_run_id",
                "year",
                "iteration",
                "beam_sub_iteration",
                "phys_sim_iteration",
                "key",
            ]
        )

    for column in ("year", "iteration", "beam_sub_iteration", "phys_sim_iteration"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="float64")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "key" not in frame.columns:
        frame["key"] = pd.Series(dtype="object")
    for column in ("run_id", "parent_run_id", "header_run_id"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")

    frame = frame.drop_duplicates(subset=["artifact_id"], keep="first")
    frame = frame.sort_values(
        [
            "parent_run_id",
            "run_id",
            "year",
            "iteration",
            "beam_sub_iteration",
            "phys_sim_iteration",
            "artifact_id",
        ],
        na_position="last",
    ).reset_index(drop=True)
    frame["observation_index"] = np.arange(len(frame), dtype=int)
    return frame[
        [
            "observation_index",
            "artifact_id",
            "run_id",
            "parent_run_id",
            "header_run_id",
            "year",
            "iteration",
            "beam_sub_iteration",
            "phys_sim_iteration",
            "key",
        ]
    ]


def _resolve_link_filter_reference_artifact_ids(
    artifact_index_df: pd.DataFrame,
    *,
    strategy: Literal["first", "last", "all"],
) -> list[str]:
    """
    Resolve reference artifacts used to compute link-volume coverage.
    """
    artifact_ids = artifact_index_df.get("artifact_id", pd.Series(dtype=str)).astype(str).tolist()
    artifact_ids = [artifact_id.strip() for artifact_id in artifact_ids if artifact_id.strip()]
    if not artifact_ids:
        return []
    if strategy == "first":
        return [artifact_ids[0]]
    if strategy == "last":
        return [artifact_ids[-1]]
    if strategy == "all":
        return artifact_ids
    raise ValueError(f"Unsupported link_filter_strategy: {strategy}")


def _linkstats_metric_row_expr(metric: Literal["speed_mph", "traveltime_sec", "delay_ratio"]) -> str:
    if metric == "speed_mph":
        return (
            "CASE "
            "WHEN CAST(src.traveltime AS DOUBLE) > 0 "
            f"THEN (CAST(src.length AS DOUBLE) / CAST(src.traveltime AS DOUBLE)) * {_MPS_TO_MPH} "
            "ELSE NULL END"
        )
    if metric == "traveltime_sec":
        return "CAST(src.traveltime AS DOUBLE)"
    if metric == "delay_ratio":
        return (
            "CASE "
            "WHEN CAST(src.freespeed AS DOUBLE) > 0 AND CAST(src.length AS DOUBLE) > 0 "
            "THEN CAST(src.traveltime AS DOUBLE) / "
            "(CAST(src.length AS DOUBLE) / CAST(src.freespeed AS DOUBLE)) "
            "ELSE NULL END"
        )
    raise ValueError(f"Unsupported metric: {metric}")


def build_linkstats_hourly_pca_matrix(
    summary_df: pd.DataFrame,
    *,
    tracker: Any,
    link_filter_strategy: Literal["first", "last", "all"] = "all",
    volume_coverage: float = 0.98,
    hour_group_size: int = 1,
    hours_per_day: int = 24,
    metric: Literal["speed_mph", "traveltime_sec", "delay_ratio"] = "speed_mph",
    exclude_zero_volume: bool = True,
    fill_value: float = 0.0,
    dtype: str = "float32",
    duckdb_threads: Optional[int] = None,
    duckdb_memory_limit: Optional[str] = None,
    duckdb_temp_directory: Optional[str | Path] = None,
    duckdb_preserve_insertion_order: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build a PCA-ready matrix with rows=artifacts and columns=(selected links x hour bins).

    Link selection is by cumulative daily volume coverage using one of:
      - ``first``: first artifact in the ordered summary
      - ``last``: last artifact in the ordered summary
      - ``all``: summed volume across all artifacts in the ordered summary

    Set ``exclude_zero_volume=True`` to skip zero-volume rows during both
    link-selection and hourly aggregation scans (often a major speedup).
    """
    if not (0.0 < float(volume_coverage) <= 1.0):
        raise ValueError("volume_coverage must be in (0, 1].")
    if hour_group_size <= 0:
        raise ValueError("hour_group_size must be >= 1.")
    if hours_per_day <= 0:
        raise ValueError("hours_per_day must be >= 1.")

    hour_bins = int((hours_per_day + hour_group_size - 1) // hour_group_size)
    artifact_index_df = _prepare_linkstats_pca_artifact_index(summary_df)
    if artifact_index_df.empty:
        empty = np.zeros((0, 0), dtype=np.dtype(dtype))
        return {
            "matrix": empty,
            "artifact_index": artifact_index_df,
            "feature_index": pd.DataFrame(
                columns=[
                    "feature_index",
                    "link",
                    "hour_bin",
                    "hour_start",
                    "hour_end_exclusive",
                    "link_index",
                    "daily_volume",
                    "cumulative_share",
                ]
            ),
            "link_index": pd.DataFrame(
                columns=["link_index", "link", "daily_volume", "cumulative_share"]
            ),
            "column_weights": np.array([], dtype="float64"),
            "column_weights_normalized": np.array([], dtype="float64"),
            "metric": metric,
            "hour_bins": hour_bins,
            "hour_group_size": hour_group_size,
            "link_filter_strategy": link_filter_strategy,
            "volume_coverage": float(volume_coverage),
        }

    view_name = _resolve_delta_grouped_view_name(summary_df)
    quoted_view = _quote_ident(view_name)
    metric_row_expr = _linkstats_metric_row_expr(metric)
    metric_agg_expr = (
        "CASE "
        "WHEN SUM(CAST(src.volume AS DOUBLE)) > 0 "
        f"THEN SUM(CAST(src.volume AS DOUBLE) * ({metric_row_expr})) "
        "/ SUM(CAST(src.volume AS DOUBLE)) "
        f"ELSE AVG({metric_row_expr}) END"
    )
    volume_filter_sql = "AND CAST(src.volume AS DOUBLE) > 0" if exclude_zero_volume else ""

    artifact_values_sql = ", ".join(
        f"({_sql_literal(row.artifact_id)}, {int(row.observation_index)})"
        for row in artifact_index_df.itertuples(index=False)
    )
    reference_artifact_ids = _resolve_link_filter_reference_artifact_ids(
        artifact_index_df,
        strategy=link_filter_strategy,
    )
    reference_values_sql = ", ".join(
        f"({_sql_literal(artifact_id)})" for artifact_id in reference_artifact_ids
    )

    common_cte = f"""
        WITH target_artifacts(artifact_id, observation_index) AS (
            VALUES {artifact_values_sql}
        ),
        reference_artifacts(artifact_id) AS (
            VALUES {reference_values_sql}
        ),
        link_daily_volume AS (
            SELECT
                src.link AS link,
                SUM(CAST(src.volume AS DOUBLE)) AS daily_volume
            FROM {quoted_view} src
            JOIN reference_artifacts ref
              ON src.consist_artifact_id = ref.artifact_id
            WHERE 1=1
              {volume_filter_sql}
            GROUP BY 1
        ),
        ranked_links AS (
            SELECT
                link,
                daily_volume,
                SUM(daily_volume) OVER () AS total_volume,
                SUM(daily_volume) OVER (
                    ORDER BY daily_volume DESC, link
                ) AS cumulative_volume
            FROM link_daily_volume
        ),
        coverage_cutoff AS (
            SELECT
                MIN(cumulative_volume) AS cutoff_volume
            FROM ranked_links
            WHERE cumulative_volume >= :volume_coverage * total_volume
        ),
        selected_links AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY rl.daily_volume DESC, rl.link) - 1 AS link_index,
                rl.link,
                rl.daily_volume,
                CASE
                    WHEN rl.total_volume > 0 THEN rl.cumulative_volume / rl.total_volume
                    ELSE 0.0
                END AS cumulative_share
            FROM ranked_links rl
            CROSS JOIN coverage_cutoff cc
            WHERE rl.total_volume <= 0
               OR cc.cutoff_volume IS NULL
               OR rl.cumulative_volume <= cc.cutoff_volume
        )
    """

    params = {"volume_coverage": float(volume_coverage)}
    with Session(tracker.engine) as session:
        _apply_duckdb_runtime_settings(
            session,
            duckdb_threads=duckdb_threads,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_temp_directory=duckdb_temp_directory,
            duckdb_preserve_insertion_order=duckdb_preserve_insertion_order,
        )
        link_rows = session.exec(
            text(
                common_cte
                + """
                SELECT
                    link_index,
                    link,
                    daily_volume,
                    cumulative_share
                FROM selected_links
                ORDER BY link_index
                """
            ),
            params=params,
        ).all()

    link_index_df = pd.DataFrame(
        link_rows,
        columns=["link_index", "link", "daily_volume", "cumulative_share"],
    )
    if link_index_df.empty:
        empty_matrix = np.full(
            (len(artifact_index_df), 0),
            float(fill_value),
            dtype=np.dtype(dtype),
        )
        return {
            "matrix": empty_matrix,
            "artifact_index": artifact_index_df,
            "feature_index": pd.DataFrame(
                columns=[
                    "feature_index",
                    "link",
                    "hour_bin",
                    "hour_start",
                    "hour_end_exclusive",
                    "link_index",
                    "daily_volume",
                    "cumulative_share",
                ]
            ),
            "link_index": link_index_df,
            "column_weights": np.array([], dtype="float64"),
            "column_weights_normalized": np.array([], dtype="float64"),
            "metric": metric,
            "hour_bins": hour_bins,
            "hour_group_size": hour_group_size,
            "link_filter_strategy": link_filter_strategy,
            "volume_coverage": float(volume_coverage),
            "view_name": view_name,
        }

    hour_bin_expr = f"CAST(FLOOR(CAST(src.hour AS DOUBLE) / {int(hour_group_size)}) AS BIGINT)"
    matrix_query = text(
        common_cte
        + f"""
        , hourly_values AS (
            SELECT
                ta.observation_index,
                sl.link_index,
                {hour_bin_expr} AS hour_bin,
                {metric_agg_expr} AS metric_value
            FROM {quoted_view} src
            JOIN target_artifacts ta
              ON src.consist_artifact_id = ta.artifact_id
            JOIN selected_links sl
              ON src.link = sl.link
            WHERE src.hour IS NOT NULL
              {volume_filter_sql}
              AND {hour_bin_expr} >= 0
              AND {hour_bin_expr} < {hour_bins}
            GROUP BY 1, 2, 3
        )
        SELECT
            observation_index,
            link_index,
            hour_bin,
            metric_value
        FROM hourly_values
        """
    )
    with Session(tracker.engine) as session:
        _apply_duckdb_runtime_settings(
            session,
            duckdb_threads=duckdb_threads,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_temp_directory=duckdb_temp_directory,
            duckdb_preserve_insertion_order=duckdb_preserve_insertion_order,
        )
        matrix_rows = session.exec(matrix_query, params=params).all()

    matrix_long_df = pd.DataFrame(
        matrix_rows,
        columns=["observation_index", "link_index", "hour_bin", "metric_value"],
    )
    n_observations = int(len(artifact_index_df))
    n_links = int(len(link_index_df))
    n_features = int(n_links * hour_bins)
    matrix = np.full(
        (n_observations, n_features),
        float(fill_value),
        dtype=np.dtype(dtype),
    )
    if not matrix_long_df.empty:
        obs_idx = pd.to_numeric(
            matrix_long_df["observation_index"],
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        link_idx = pd.to_numeric(
            matrix_long_df["link_index"],
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        hour_idx = pd.to_numeric(
            matrix_long_df["hour_bin"],
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        feature_idx = link_idx * hour_bins + hour_idx
        values = pd.to_numeric(
            matrix_long_df["metric_value"],
            errors="coerce",
        ).fillna(float(fill_value)).to_numpy(dtype=np.dtype(dtype))

        valid = (
            (obs_idx >= 0)
            & (obs_idx < n_observations)
            & (feature_idx >= 0)
            & (feature_idx < n_features)
        )
        matrix[obs_idx[valid], feature_idx[valid]] = values[valid]

    link_index_df["link_index"] = pd.to_numeric(
        link_index_df["link_index"],
        errors="coerce",
    ).fillna(-1).astype(int)
    link_index_df["daily_volume"] = pd.to_numeric(
        link_index_df["daily_volume"],
        errors="coerce",
    ).fillna(0.0)
    link_index_df["cumulative_share"] = pd.to_numeric(
        link_index_df["cumulative_share"],
        errors="coerce",
    ).fillna(0.0)

    feature_index_df = link_index_df.loc[
        link_index_df.index.repeat(hour_bins),
        ["link_index", "link", "daily_volume", "cumulative_share"],
    ].reset_index(drop=True)
    feature_index_df["hour_bin"] = np.tile(np.arange(hour_bins, dtype=int), n_links)
    feature_index_df["hour_start"] = feature_index_df["hour_bin"] * int(hour_group_size)
    feature_index_df["hour_end_exclusive"] = (
        feature_index_df["hour_start"] + int(hour_group_size)
    ).clip(upper=int(hours_per_day))
    feature_index_df["feature_index"] = (
        feature_index_df["link_index"] * hour_bins + feature_index_df["hour_bin"]
    ).astype(int)
    feature_index_df = feature_index_df[
        [
            "feature_index",
            "link",
            "hour_bin",
            "hour_start",
            "hour_end_exclusive",
            "link_index",
            "daily_volume",
            "cumulative_share",
        ]
    ]

    column_weights = np.repeat(
        link_index_df["daily_volume"].to_numpy(dtype="float64"),
        hour_bins,
    )
    weight_sum = float(column_weights.sum())
    if weight_sum > 0:
        column_weights_normalized = column_weights / weight_sum
    elif len(column_weights) > 0:
        column_weights_normalized = np.full(
            shape=len(column_weights),
            fill_value=1.0 / len(column_weights),
            dtype="float64",
        )
    else:
        column_weights_normalized = np.array([], dtype="float64")

    return {
        "matrix": matrix,
        "artifact_index": artifact_index_df,
        "feature_index": feature_index_df,
        "link_index": link_index_df,
        "column_weights": column_weights,
        "column_weights_normalized": column_weights_normalized,
        "metric": metric,
        "hour_bins": hour_bins,
        "hour_group_size": int(hour_group_size),
        "link_filter_strategy": link_filter_strategy,
        "volume_coverage": float(volume_coverage),
        "view_name": view_name,
    }
