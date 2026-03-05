from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sqlmodel import Session, col, select, text

from .keys import CANONICAL_KEY_COLUMNS, ensure_canonical_key_columns
from .manifest import DatasetManifest

_TRIPS_FACET_KEYS = ("artifact_family", "year", "iteration")


@dataclass
class ActivitySimTripsDataset:
    artifacts: pd.DataFrame
    mode_counts: pd.DataFrame
    purpose_mode_counts: pd.DataFrame
    depart_hour_counts: pd.DataFrame
    iteration_summary: pd.DataFrame
    mode_deltas: pd.DataFrame
    equilibrium_pairs: pd.DataFrame


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


def _facet_map_from_artifact_kv(
    tracker: Any,
    artifact_ids: Iterable[Any],
    *,
    namespace: Optional[str],
) -> Dict[str, Dict[str, Any]]:
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

    def _consume(rows: list[Any]) -> set[str]:
        resolved: set[str] = set()
        for row in rows:
            normalized = _normalize_artifact_id(getattr(row, "artifact_id", None))
            if normalized is None:
                continue
            key_path = str(getattr(row, "key_path", "") or "")
            if key_path not in _TRIPS_FACET_KEYS:
                continue
            resolved.add(normalized)
            facet_map.setdefault(normalized, {})[key_path] = _decode_artifact_kv_value(row)
        return resolved

    def _query(artifact_ids_subset: list[Any], ns: Optional[str]) -> list[Any]:
        rows: list[Any] = []
        for chunk in _chunked(artifact_ids_subset, 1000):
            with db.session_scope() as session:
                statement = select(ArtifactKV).where(col(ArtifactKV.artifact_id).in_(chunk))
                if ns is not None:
                    statement = statement.where(ArtifactKV.namespace == ns)
                statement = statement.where(col(ArtifactKV.key_path).in_(_TRIPS_FACET_KEYS))
                rows.extend(session.exec(statement).all())
        return rows

    if namespace is None:
        _consume(_query(unique_ids, None))
        return facet_map

    rows_for_ns = _query(unique_ids, namespace)
    resolved_in_ns = _consume(rows_for_ns)
    unresolved = [
        artifact_id
        for artifact_id in unique_ids
        if (_normalize_artifact_id(artifact_id) or "") not in resolved_in_ns
    ]
    if unresolved:
        _consume(_query(unresolved, None))
    return facet_map


def _parent_run_map(tracker: Any, run_ids: Iterable[Any]) -> Dict[str, Optional[str]]:
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
            rows = session.exec(select(Run).where(col(Run.id).in_(chunk))).all()
        for row in rows:
            row_id = str(getattr(row, "id", "") or "").strip()
            if not row_id:
                continue
            parent = getattr(row, "parent_run_id", None)
            parent_normalized = str(parent).strip() if parent is not None else None
            mapping[row_id] = parent_normalized or None
    return mapping


def _find_artifacts_by_key_prefix_sqlmodel(
    tracker: Any,
    *,
    key_prefix: str,
    limit: int,
) -> list[Any]:
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


def find_trip_artifacts(
    tracker: Any,
    *,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    artifact_family: str = "trips",
    namespace: str = "activitysim",
    limit: int = 10000,
) -> pd.DataFrame:
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
        artifacts = _find_artifacts_by_key_prefix_sqlmodel(
            tracker,
            key_prefix="trips_asim_out",
            limit=limit,
        )
    if not artifacts:
        return pd.DataFrame(
            columns=[
                "key",
                "artifact_id",
                "run_id",
                "parent_run_id",
                "comparison_group",
                "container_uri",
                "created_at",
                "facet_source",
                "artifact_family",
                "year",
                "iteration",
            ]
        )

    artifact_ids = [getattr(artifact, "id", None) for artifact in artifacts]
    run_ids = [getattr(artifact, "run_id", None) for artifact in artifacts]
    facet_map = _facet_map_from_artifact_kv(tracker, artifact_ids, namespace=namespace)
    parent_map = _parent_run_map(tracker, run_ids)

    rows: list[Dict[str, Any]] = []
    for artifact in artifacts:
        key = str(getattr(artifact, "key", "") or "")
        artifact_id = str(getattr(artifact, "id", "") or "")
        run_id = str(getattr(artifact, "run_id", "") or "")
        created_at = getattr(artifact, "created_at", None)
        meta = dict(getattr(artifact, "meta", None) or {})

        facet: Dict[str, Any] = {}
        facet_source = "none"
        facet_from_kv = facet_map.get(_normalize_artifact_id(artifact_id) or "", {})
        if facet_from_kv:
            facet.update(
                {
                    k: facet_from_kv.get(k)
                    for k in _TRIPS_FACET_KEYS
                    if facet_from_kv.get(k) is not None
                }
            )
            facet_source = "artifact_kv"

        meta_facet = meta.get("facet") if isinstance(meta.get("facet"), dict) else {}
        if meta_facet:
            meta_filtered = {
                k: meta_facet.get(k)
                for k in _TRIPS_FACET_KEYS
                if meta_facet.get(k) is not None
            }
            if meta_filtered:
                facet = {**meta_filtered, **facet}
                if facet_source == "none":
                    facet_source = "artifact_meta"

        family = facet.get("artifact_family")
        row_year = facet.get("year")
        row_iteration = facet.get("iteration")
        if artifact_family and family not in {artifact_family, "trips_asim_out"}:
            continue
        if year is not None and row_year is not None and int(row_year) != int(year):
            continue
        if (
            iteration is not None
            and row_iteration is not None
            and int(row_iteration) != int(iteration)
        ):
            continue

        parent_run_id = parent_map.get(run_id)
        rows.append(
            {
                "key": key,
                "artifact_id": artifact_id,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "comparison_group": parent_run_id or run_id,
                "container_uri": getattr(artifact, "container_uri", None),
                "created_at": created_at,
                "facet_source": facet_source,
                "artifact_family": family or artifact_family,
                "year": row_year,
                "iteration": row_iteration,
                "model": "activitysim",
                "seed": None,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["iteration"] = pd.to_numeric(frame["iteration"], errors="coerce")
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    return frame.sort_values(
        ["year", "iteration", "created_at", "artifact_id"],
        na_position="last",
    ).reset_index(drop=True)


def _stable_grouped_view_name(
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
    return f"v_asim_trips_grouped_{digest}"


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


def _resolve_trips_schema_model() -> Optional[type]:
    try:
        from pilates.database.schema.activitysim_schema import TripsAsimOut
        return TripsAsimOut
    except Exception:
        pass
    try:
        from pilates.database.schema.registry import get_schema_for_key

        return get_schema_for_key("trips_asim_out")
    except Exception:
        return None


def _parse_artifact_param_value(raw: str) -> tuple[str, Any]:
    value = raw.strip()
    lowered = value.lower()
    if lowered == "true":
        return "bool", True
    if lowered == "false":
        return "bool", False
    if lowered == "null":
        return "null", None
    try:
        if "." not in value and "e" not in lowered:
            return "num", int(value)
        return "num", float(value)
    except ValueError:
        return "str", value


def _parse_artifact_param_expression(expression: str) -> Dict[str, Any]:
    raw = expression.strip()
    operator = None
    lhs = ""
    rhs = ""
    for candidate in (">=", "<=", "="):
        idx = raw.find(candidate)
        if idx > 0:
            operator = candidate
            lhs = raw[:idx].strip()
            rhs = raw[idx + len(candidate) :].strip()
            break
    if operator is None:
        raise ValueError(
            f"Invalid artifact facet predicate {expression!r}. "
            "Expected <key>=<value>, <key>>=<value>, or <key><=<value>."
        )
    if not lhs:
        raise ValueError(f"Artifact facet predicate is missing a key: {expression!r}")
    if rhs == "":
        raise ValueError(f"Artifact facet predicate is missing a value: {expression!r}")

    predicate_namespace = None
    key_path = lhs
    if "." in lhs:
        maybe_namespace, remainder = lhs.split(".", 1)
        if maybe_namespace and remainder:
            predicate_namespace = maybe_namespace
            key_path = remainder

    value_kind, value = _parse_artifact_param_value(rhs)
    if operator in {">=", "<="} and value_kind != "num":
        raise ValueError(
            f"Artifact facet predicate {expression!r} uses {operator} with a "
            "non-numeric value."
        )

    return {
        "namespace": predicate_namespace,
        "key_path": key_path,
        "op": operator,
        "kind": value_kind,
        "value": value,
    }


def _build_artifact_predicates(tracker: Any, params: list[str]) -> list[Dict[str, Any]]:
    if not params:
        return []
    parser = getattr(tracker, "_parse_artifact_param_expression", None)
    if callable(parser):
        return [parser(param) for param in params]
    return [_parse_artifact_param_expression(param) for param in params]


def _resolve_grouped_schema_selector(
    *,
    tracker: Any,
    schema_id: Optional[str],
) -> Dict[str, Any]:
    if schema_id:
        return {"schema_id": schema_id}

    schema_model = _resolve_trips_schema_model()
    if schema_model is None:
        return {}

    db = getattr(tracker, "db", None)
    if db is not None and hasattr(db, "find_schema_ids_for_model"):
        schema_ids = db.find_schema_ids_for_model(schema_model=schema_model, compatible=False)
        if not schema_ids:
            schema_ids = db.find_schema_ids_for_model(schema_model=schema_model, compatible=True)
        if schema_ids:
            return {"schema_ids": list(schema_ids), "schema": schema_model}
    return {"schema": schema_model}


def _call_with_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    accepts_var_kw = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kw:
        return func(**kwargs)
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return func(**filtered_kwargs)


def _resolve_grouped_hybrid_creator(tracker: Any) -> Optional[Any]:
    view_factory = getattr(tracker, "view_factory", None)
    creator = getattr(view_factory, "create_grouped_hybrid_view", None)
    if callable(creator):
        return creator
    try:
        from consist.core.views import ViewFactory
    except Exception:
        return None
    try:
        creator = getattr(ViewFactory(tracker), "create_grouped_hybrid_view", None)
    except Exception:
        return None
    return creator if callable(creator) else None


def _create_grouped_view(
    *,
    tracker: Any,
    artifacts_df: pd.DataFrame,
    namespace: str = "activitysim",
    drivers: Optional[list[str]] = None,
    mode: str = "hybrid",
    missing_files: str = "warn",
    view_name: Optional[str] = None,
    schema_id: Optional[str] = None,
    use_facet_filters: bool = True,
) -> str:
    artifact_family = _single_string_value(artifacts_df.get("artifact_family"))
    year = _single_numeric_value(artifacts_df.get("year"))
    iteration = _single_numeric_value(artifacts_df.get("iteration"))
    run_id = _single_string_value(artifacts_df.get("run_id"))

    facet_source_values = (
        artifacts_df.get("facet_source").dropna().astype(str)
        if "facet_source" in artifacts_df.columns
        else pd.Series(dtype=str)
    )
    has_kv_facets = bool((facet_source_values == "artifact_kv").any())

    params: list[str] = []
    if use_facet_filters and has_kv_facets and artifact_family:
        params.append(f"{namespace}.artifact_family={artifact_family}")
    if use_facet_filters and has_kv_facets and year is not None:
        params.append(f"{namespace}.year={year}")
    if use_facet_filters and has_kv_facets and iteration is not None:
        params.append(f"{namespace}.iteration={iteration}")

    grouped_view_name = view_name or _stable_grouped_view_name(
        namespace=namespace,
        artifact_family=artifact_family,
        year=year,
        iteration=iteration,
    )
    selector = _resolve_grouped_schema_selector(tracker=tracker, schema_id=schema_id)
    predicates = _build_artifact_predicates(tracker, params)
    facets = list(_TRIPS_FACET_KEYS)
    create_grouped_hybrid = _resolve_grouped_hybrid_creator(tracker)
    can_use_hybrid = (
        callable(create_grouped_hybrid)
        and ("schema_id" in selector or "schema_ids" in selector)
    )

    if can_use_hybrid:
        grouped_kwargs: Dict[str, Any] = {
            "view_name": grouped_view_name,
            "namespace": namespace,
            "predicates": predicates or None,
            "drivers": drivers or ["parquet", "csv"],
            "attach_facets": facets,
            "facets": facets,
            "include_system_columns": True,
            "mode": mode,
            "if_exists": "replace",
            "missing_files": missing_files,
            "run_id": run_id,
            "year": year,
            "iteration": iteration,
            "schema_id": selector.get("schema_id"),
            "schema_ids": selector.get("schema_ids"),
        }
        _call_with_supported_kwargs(create_grouped_hybrid, grouped_kwargs)
        return grouped_view_name

    legacy_selector: Dict[str, Any] = {}
    if selector.get("schema_id"):
        legacy_selector["schema_id"] = selector["schema_id"]
    elif selector.get("schema") is not None:
        legacy_selector["schema"] = selector["schema"]
    elif selector.get("schema_ids"):
        schema_ids = selector["schema_ids"]
        if schema_ids:
            legacy_selector["schema_id"] = schema_ids[0]

    tracker.create_grouped_view(
        view_name=grouped_view_name,
        namespace=namespace,
        params=params or None,
        drivers=drivers or ["parquet", "csv"],
        attach_facets=facets,
        include_system_columns=True,
        mode=mode,
        if_exists="replace",
        missing_files=missing_files,
        run_id=run_id,
        year=year,
        iteration=iteration,
        **legacy_selector,
    )
    return grouped_view_name


def _query_grouped_view(
    *,
    tracker: Any,
    view_name: str,
    select_sql: str,
    columns: list[str],
    artifact_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    quoted_view = _quote_ident(view_name)
    query = f"""
        SELECT {select_sql}
        FROM {quoted_view} src
    """
    if artifact_ids:
        unique_ids = sorted({str(v).strip() for v in artifact_ids if str(v).strip()})
        if unique_ids:
            values_sql = ", ".join(f"({_sql_literal(v)})" for v in unique_ids)
            query = f"""
                WITH target_artifacts(artifact_id) AS (
                    VALUES {values_sql}
                )
                SELECT {select_sql}
                FROM {quoted_view} src
                JOIN target_artifacts ta
                  ON src.consist_artifact_id = ta.artifact_id
            """
    with Session(tracker.engine) as session:
        rows = session.exec(text(query)).all()
    return pd.DataFrame(rows, columns=columns)


def _summarize_mode_counts(
    tracker: Any,
    *,
    view_name: str,
    artifact_ids: list[str],
) -> pd.DataFrame:
    frame = _query_grouped_view(
        tracker=tracker,
        view_name=view_name,
        artifact_ids=artifact_ids,
        select_sql="""
            src.consist_artifact_id AS artifact_id,
            COALESCE(NULLIF(TRIM(CAST(src.trip_mode AS VARCHAR)), ''), '__MISSING__') AS trip_mode,
            COUNT(*) AS trip_count,
            COUNT(DISTINCT src.person_id) AS distinct_persons,
            AVG(CAST(src.mode_choice_logsum AS DOUBLE)) AS mode_choice_logsum_mean
            GROUP BY 1, 2
        """,
        columns=[
            "artifact_id",
            "trip_mode",
            "trip_count",
            "distinct_persons",
            "mode_choice_logsum_mean",
        ],
    )
    if frame.empty:
        return frame
    frame["artifact_id"] = frame["artifact_id"].astype(str)
    frame["trip_mode"] = frame["trip_mode"].astype(str)
    frame["trip_count"] = pd.to_numeric(frame["trip_count"], errors="coerce").fillna(0).astype(int)
    frame["distinct_persons"] = (
        pd.to_numeric(frame["distinct_persons"], errors="coerce").fillna(0).astype(int)
    )
    frame["mode_choice_logsum_mean"] = pd.to_numeric(
        frame["mode_choice_logsum_mean"], errors="coerce"
    )
    return frame


def _summarize_purpose_mode_counts(
    tracker: Any,
    *,
    view_name: str,
    artifact_ids: list[str],
) -> pd.DataFrame:
    frame = _query_grouped_view(
        tracker=tracker,
        view_name=view_name,
        artifact_ids=artifact_ids,
        select_sql="""
            src.consist_artifact_id AS artifact_id,
            COALESCE(NULLIF(TRIM(CAST(src.primary_purpose AS VARCHAR)), ''), '__MISSING__') AS primary_purpose,
            COALESCE(NULLIF(TRIM(CAST(src.trip_mode AS VARCHAR)), ''), '__MISSING__') AS trip_mode,
            COUNT(*) AS trip_count
            GROUP BY 1, 2, 3
        """,
        columns=["artifact_id", "primary_purpose", "trip_mode", "trip_count"],
    )
    if frame.empty:
        return frame
    frame["artifact_id"] = frame["artifact_id"].astype(str)
    frame["primary_purpose"] = frame["primary_purpose"].astype(str)
    frame["trip_mode"] = frame["trip_mode"].astype(str)
    frame["trip_count"] = pd.to_numeric(frame["trip_count"], errors="coerce").fillna(0).astype(int)
    return frame


def _summarize_depart_hour_counts(
    tracker: Any,
    *,
    view_name: str,
    artifact_ids: list[str],
) -> pd.DataFrame:
    frame = _query_grouped_view(
        tracker=tracker,
        view_name=view_name,
        artifact_ids=artifact_ids,
        select_sql="""
            src.consist_artifact_id AS artifact_id,
            CAST(FLOOR(CAST(src.depart AS DOUBLE)) AS BIGINT) AS depart_hour,
            COUNT(*) AS trip_count
            GROUP BY 1, 2
        """,
        columns=["artifact_id", "depart_hour", "trip_count"],
    )
    if frame.empty:
        return frame
    frame["artifact_id"] = frame["artifact_id"].astype(str)
    frame["depart_hour"] = pd.to_numeric(frame["depart_hour"], errors="coerce")
    frame["trip_count"] = pd.to_numeric(frame["trip_count"], errors="coerce").fillna(0).astype(int)
    frame["depart_hour_in_day"] = (frame["depart_hour"] % 24).astype("Int64")
    return frame


def _summarize_iteration(
    tracker: Any,
    *,
    view_name: str,
    artifact_ids: list[str],
) -> pd.DataFrame:
    frame = _query_grouped_view(
        tracker=tracker,
        view_name=view_name,
        artifact_ids=artifact_ids,
        select_sql="""
            src.consist_artifact_id AS artifact_id,
            COUNT(*) AS total_trips,
            COUNT(DISTINCT src.person_id) AS distinct_persons,
            COUNT(DISTINCT COALESCE(NULLIF(TRIM(CAST(src.trip_mode AS VARCHAR)), ''), '__MISSING__')) AS distinct_modes,
            AVG(CAST(src.mode_choice_logsum AS DOUBLE)) AS mode_choice_logsum_mean,
            SUM(CASE WHEN src.outbound THEN 1 ELSE 0 END) AS outbound_trip_count
            GROUP BY 1
        """,
        columns=[
            "artifact_id",
            "total_trips",
            "distinct_persons",
            "distinct_modes",
            "mode_choice_logsum_mean",
            "outbound_trip_count",
        ],
    )
    if frame.empty:
        return frame
    frame["artifact_id"] = frame["artifact_id"].astype(str)
    for column in ("total_trips", "distinct_persons", "distinct_modes", "outbound_trip_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["mode_choice_logsum_mean"] = pd.to_numeric(
        frame["mode_choice_logsum_mean"], errors="coerce"
    )
    return frame


def _select_latest_artifact_ids(iteration_summary: pd.DataFrame) -> set[str]:
    if iteration_summary.empty:
        return set()
    frame = iteration_summary.copy()
    if "created_at" not in frame.columns:
        return set(frame["artifact_id"].astype(str).tolist())
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    frame = frame.sort_values(
        ["comparison_group", "year", "iteration", "created_at", "artifact_id"],
        na_position="last",
    )
    latest = frame.groupby(
        ["comparison_group", "year", "iteration"], dropna=False, as_index=False
    ).tail(1)
    return set(latest["artifact_id"].astype(str).tolist())


def _add_mode_entropy(iteration_mode_summary: pd.DataFrame) -> pd.DataFrame:
    if iteration_mode_summary.empty:
        return pd.DataFrame(
            columns=["comparison_group", "year", "iteration", "mode_entropy_nats"]
        )
    frame = iteration_mode_summary.copy()
    frame = frame[frame["mode_share"] > 0]
    frame["entropy_component"] = -(frame["mode_share"] * np.log(frame["mode_share"]))
    entropy = (
        frame.groupby(["comparison_group", "year", "iteration"], dropna=False)[
            "entropy_component"
        ]
        .sum()
        .reset_index(name="mode_entropy_nats")
    )
    return entropy


def _build_mode_deltas(iteration_mode_summary: pd.DataFrame) -> pd.DataFrame:
    if iteration_mode_summary.empty:
        return pd.DataFrame()
    frame = iteration_mode_summary.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["iteration"] = pd.to_numeric(frame["iteration"], errors="coerce")
    frame = frame.sort_values(
        ["comparison_group", "year", "trip_mode", "iteration"],
        na_position="last",
    )
    frame["iteration_prev"] = frame.groupby(["comparison_group", "year", "trip_mode"])[
        "iteration"
    ].shift(1)
    frame["trip_count_prev"] = frame.groupby(["comparison_group", "year", "trip_mode"])[
        "trip_count"
    ].shift(1)
    frame["mode_share_prev"] = frame.groupby(["comparison_group", "year", "trip_mode"])[
        "mode_share"
    ].shift(1)
    deltas = frame[frame["iteration_prev"].notna()].copy()
    if deltas.empty:
        return pd.DataFrame()
    deltas["trip_count_delta"] = deltas["trip_count"] - deltas["trip_count_prev"]
    deltas["trip_count_delta_abs"] = deltas["trip_count_delta"].abs()
    deltas["mode_share_delta"] = deltas["mode_share"] - deltas["mode_share_prev"]
    deltas["mode_share_delta_abs"] = deltas["mode_share_delta"].abs()
    return deltas[
        [
            "comparison_group",
            "year",
            "trip_mode",
            "iteration_prev",
            "iteration",
            "trip_count_prev",
            "trip_count",
            "trip_count_delta",
            "trip_count_delta_abs",
            "mode_share_prev",
            "mode_share",
            "mode_share_delta",
            "mode_share_delta_abs",
        ]
    ].reset_index(drop=True)


def _build_equilibrium_pairs(
    iteration_mode_summary: pd.DataFrame,
    iteration_summary: pd.DataFrame,
) -> pd.DataFrame:
    if iteration_mode_summary.empty or iteration_summary.empty:
        return pd.DataFrame()

    share_pivot = iteration_mode_summary.pivot_table(
        index=["comparison_group", "year", "iteration"],
        columns="trip_mode",
        values="mode_share",
        fill_value=0.0,
        aggfunc="sum",
    ).sort_index()

    totals = iteration_summary.set_index(
        ["comparison_group", "year", "iteration"]
    )["total_trips"]

    rows: list[Dict[str, Any]] = []
    grouped_indices = share_pivot.index.to_frame(index=False)
    for (comparison_group, year), group in grouped_indices.groupby(
        ["comparison_group", "year"], dropna=False
    ):
        iterations = sorted(
            [int(v) for v in group["iteration"].dropna().astype(int).tolist()]
        )
        if len(iterations) < 2:
            continue
        for idx in range(1, len(iterations)):
            prev_iter = iterations[idx - 1]
            curr_iter = iterations[idx]
            prev_key = (comparison_group, year, prev_iter)
            curr_key = (comparison_group, year, curr_iter)
            prev_vec = share_pivot.loc[prev_key]
            curr_vec = share_pivot.loc[curr_key]
            tvd = float(0.5 * np.abs(curr_vec.values - prev_vec.values).sum())

            total_prev = float(totals.get(prev_key, np.nan))
            total_curr = float(totals.get(curr_key, np.nan))
            total_delta = total_curr - total_prev
            rows.append(
                {
                    "comparison_group": comparison_group,
                    "year": year,
                    "iteration_prev": prev_iter,
                    "iteration": curr_iter,
                    "mode_share_total_variation_distance": tvd,
                    "total_trips_prev": total_prev,
                    "total_trips": total_curr,
                    "total_trips_delta": total_delta,
                    "total_trips_delta_abs": abs(total_delta),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["comparison_group", "year", "iteration_prev", "iteration"],
        na_position="last",
    )


def build_activitysim_trips_dataset(
    tracker: Any,
    *,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    artifact_family: str = "trips",
    namespace: str = "activitysim",
    grouped_mode: str = "hybrid",
    grouped_missing_files: str = "warn",
    grouped_schema_id: Optional[str] = None,
    grouped_drivers: Optional[list[str]] = None,
    latest_per_iteration: bool = True,
    limit: int = 10000,
) -> ActivitySimTripsDataset:
    artifacts_df = find_trip_artifacts(
        tracker,
        year=year,
        iteration=iteration,
        artifact_family=artifact_family,
        namespace=namespace,
        limit=limit,
    )
    if artifacts_df.empty:
        empty = pd.DataFrame()
        return ActivitySimTripsDataset(
            artifacts=artifacts_df,
            mode_counts=empty,
            purpose_mode_counts=empty,
            depart_hour_counts=empty,
            iteration_summary=empty,
            mode_deltas=empty,
            equilibrium_pairs=empty,
        )

    grouped_view = _create_grouped_view(
        tracker=tracker,
        artifacts_df=artifacts_df,
        namespace=namespace,
        drivers=grouped_drivers,
        mode=grouped_mode,
        missing_files=grouped_missing_files,
        schema_id=grouped_schema_id,
        use_facet_filters=True,
    )
    artifact_ids = artifacts_df["artifact_id"].astype(str).tolist()

    mode_counts = _summarize_mode_counts(
        tracker,
        view_name=grouped_view,
        artifact_ids=artifact_ids,
    )
    purpose_mode_counts = _summarize_purpose_mode_counts(
        tracker,
        view_name=grouped_view,
        artifact_ids=artifact_ids,
    )
    depart_hour_counts = _summarize_depart_hour_counts(
        tracker,
        view_name=grouped_view,
        artifact_ids=artifact_ids,
    )
    iteration_summary_by_artifact = _summarize_iteration(
        tracker,
        view_name=grouped_view,
        artifact_ids=artifact_ids,
    )
    if iteration_summary_by_artifact.empty:
        empty = pd.DataFrame()
        return ActivitySimTripsDataset(
            artifacts=artifacts_df,
            mode_counts=empty,
            purpose_mode_counts=purpose_mode_counts,
            depart_hour_counts=depart_hour_counts,
            iteration_summary=empty,
            mode_deltas=empty,
            equilibrium_pairs=empty,
        )

    metadata = artifacts_df.copy()
    metadata["artifact_id"] = metadata["artifact_id"].astype(str)
    metadata = metadata.drop_duplicates(subset=["artifact_id"], keep="first")
    iteration_summary_by_artifact = metadata.merge(
        iteration_summary_by_artifact, on="artifact_id", how="inner"
    )

    selected_ids = set(iteration_summary_by_artifact["artifact_id"].astype(str).tolist())
    if latest_per_iteration:
        selected_ids = _select_latest_artifact_ids(iteration_summary_by_artifact)
    selected = iteration_summary_by_artifact[
        iteration_summary_by_artifact["artifact_id"].astype(str).isin(selected_ids)
    ].copy()

    mode_counts = mode_counts[mode_counts["artifact_id"].astype(str).isin(selected_ids)].copy()
    purpose_mode_counts = purpose_mode_counts[
        purpose_mode_counts["artifact_id"].astype(str).isin(selected_ids)
    ].copy()
    depart_hour_counts = depart_hour_counts[
        depart_hour_counts["artifact_id"].astype(str).isin(selected_ids)
    ].copy()

    iteration_summary = (
        selected.groupby(
            ["comparison_group", "year", "iteration", "model", "seed"], dropna=False
        )
        .agg(
            artifact_count=("artifact_id", "nunique"),
            total_trips=("total_trips", "sum"),
            distinct_persons=("distinct_persons", "sum"),
            distinct_modes=("distinct_modes", "max"),
            mode_choice_logsum_mean=("mode_choice_logsum_mean", "mean"),
            outbound_trip_count=("outbound_trip_count", "sum"),
        )
        .reset_index()
    )
    iteration_summary["scenario_id"] = iteration_summary["comparison_group"]
    iteration_summary["run_id"] = iteration_summary["comparison_group"]
    iteration_summary["phys_sim_iteration"] = None
    iteration_summary["beam_sub_iteration"] = None

    mode_counts_with_meta = metadata[
        ["artifact_id", "comparison_group", "year", "iteration", "model", "seed"]
    ].merge(mode_counts, on="artifact_id", how="inner")
    mode_iteration_summary = (
        mode_counts_with_meta.groupby(
            ["comparison_group", "year", "iteration", "trip_mode", "model", "seed"],
            dropna=False,
        )
        .agg(
            trip_count=("trip_count", "sum"),
            distinct_persons=("distinct_persons", "sum"),
            mode_choice_logsum_mean=("mode_choice_logsum_mean", "mean"),
        )
        .reset_index()
    )
    total_trips = iteration_summary[
        ["comparison_group", "year", "iteration", "total_trips"]
    ].rename(columns={"total_trips": "total_trips_iteration"})
    mode_iteration_summary = mode_iteration_summary.merge(
        total_trips,
        on=["comparison_group", "year", "iteration"],
        how="left",
    )
    mode_iteration_summary["mode_share"] = np.where(
        mode_iteration_summary["total_trips_iteration"] > 0,
        mode_iteration_summary["trip_count"]
        / mode_iteration_summary["total_trips_iteration"],
        np.nan,
    )
    mode_iteration_summary["scenario_id"] = mode_iteration_summary["comparison_group"]
    mode_iteration_summary["run_id"] = mode_iteration_summary["comparison_group"]
    mode_iteration_summary["phys_sim_iteration"] = None
    mode_iteration_summary["beam_sub_iteration"] = None

    entropy = _add_mode_entropy(mode_iteration_summary)
    if not entropy.empty:
        iteration_summary = iteration_summary.merge(
            entropy,
            on=["comparison_group", "year", "iteration"],
            how="left",
        )

    mode_deltas = _build_mode_deltas(mode_iteration_summary)
    if not mode_deltas.empty:
        mode_deltas["scenario_id"] = mode_deltas["comparison_group"]
        mode_deltas["run_id"] = mode_deltas["comparison_group"]
        mode_deltas["model"] = "activitysim"
        mode_deltas["seed"] = None
        mode_deltas["phys_sim_iteration"] = None
        mode_deltas["beam_sub_iteration"] = None

    equilibrium_pairs = _build_equilibrium_pairs(mode_iteration_summary, iteration_summary)
    if not equilibrium_pairs.empty:
        equilibrium_pairs["scenario_id"] = equilibrium_pairs["comparison_group"]
        equilibrium_pairs["run_id"] = equilibrium_pairs["comparison_group"]
        equilibrium_pairs["model"] = "activitysim"
        equilibrium_pairs["seed"] = None
        equilibrium_pairs["phys_sim_iteration"] = None
        equilibrium_pairs["beam_sub_iteration"] = None

    return ActivitySimTripsDataset(
        artifacts=artifacts_df,
        mode_counts=mode_iteration_summary,
        purpose_mode_counts=purpose_mode_counts.merge(
            metadata[
                ["artifact_id", "comparison_group", "year", "iteration", "model", "seed"]
            ],
            on="artifact_id",
            how="left",
        ),
        depart_hour_counts=depart_hour_counts.merge(
            metadata[
                ["artifact_id", "comparison_group", "year", "iteration", "model", "seed"]
            ],
            on="artifact_id",
            how="left",
        ),
        iteration_summary=iteration_summary,
        mode_deltas=mode_deltas,
        equilibrium_pairs=equilibrium_pairs,
    )


def write_activitysim_trips_dataset(
    dataset: ActivitySimTripsDataset,
    *,
    output_dir: str | Path,
    archive_run_dir: str,
    db_path: str,
    query: Dict[str, Any],
) -> DatasetManifest:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    artifacts_path = out / "asim_trips_artifacts.csv"
    mode_counts_path = out / "asim_trips_mode_counts.csv"
    purpose_mode_counts_path = out / "asim_trips_purpose_mode_counts.csv"
    depart_hour_counts_path = out / "asim_trips_depart_hour_counts.csv"
    iteration_summary_path = out / "asim_trips_iteration_summary.csv"
    mode_deltas_path = out / "asim_trips_mode_deltas.csv"
    equilibrium_pairs_path = out / "asim_trips_equilibrium_pairs.csv"

    dataset.artifacts.to_csv(artifacts_path, index=False)
    ensure_canonical_key_columns(dataset.mode_counts).to_csv(mode_counts_path, index=False)
    ensure_canonical_key_columns(dataset.purpose_mode_counts).to_csv(
        purpose_mode_counts_path, index=False
    )
    ensure_canonical_key_columns(dataset.depart_hour_counts).to_csv(
        depart_hour_counts_path, index=False
    )
    ensure_canonical_key_columns(dataset.iteration_summary).to_csv(
        iteration_summary_path, index=False
    )
    ensure_canonical_key_columns(dataset.mode_deltas).to_csv(mode_deltas_path, index=False)
    ensure_canonical_key_columns(dataset.equilibrium_pairs).to_csv(
        equilibrium_pairs_path, index=False
    )

    manifest = DatasetManifest(
        dataset_name="activitysim_trips_dataset",
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
        files={
            "artifacts": str(artifacts_path),
            "mode_counts": str(mode_counts_path),
            "purpose_mode_counts": str(purpose_mode_counts_path),
            "depart_hour_counts": str(depart_hour_counts_path),
            "iteration_summary": str(iteration_summary_path),
            "mode_deltas": str(mode_deltas_path),
            "equilibrium_pairs": str(equilibrium_pairs_path),
        },
        row_counts={
            "artifacts": int(len(dataset.artifacts)),
            "mode_counts": int(len(dataset.mode_counts)),
            "purpose_mode_counts": int(len(dataset.purpose_mode_counts)),
            "depart_hour_counts": int(len(dataset.depart_hour_counts)),
            "iteration_summary": int(len(dataset.iteration_summary)),
            "mode_deltas": int(len(dataset.mode_deltas)),
            "equilibrium_pairs": int(len(dataset.equilibrium_pairs)),
        },
        key_columns=list(CANONICAL_KEY_COLUMNS),
    )
    manifest.write_json(out / "dataset_manifest.json")
    return manifest
