from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

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
    facet_map_from_kv = _facet_map_from_artifact_kv(
        tracker,
        artifact_ids,
        namespace=namespace,
    )

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
    run_id = _first_non_null_string(artifacts_df.get("run_id"))
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
) -> pd.DataFrame:
    quoted_view = _quote_ident(view_name)
    query = f"""
        SELECT
            CAST(src.consist_artifact_id AS VARCHAR) AS artifact_id,
            COUNT(*) AS row_count,
            COUNT(DISTINCT src.link) AS distinct_links,
            SUM(src.volume) AS volume_sum,
            AVG(src.traveltime) AS traveltime_mean,
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
                    CAST(src.consist_artifact_id AS VARCHAR) AS artifact_id,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT src.link) AS distinct_links,
                    SUM(src.volume) AS volume_sum,
                    AVG(src.traveltime) AS traveltime_mean,
                    quantile_cont(src.traveltime, 0.95) AS traveltime_p95
                FROM {quoted_view} src
                JOIN target_artifacts ta
                  ON CAST(src.consist_artifact_id AS VARCHAR) = ta.artifact_id
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
        "traveltime_mean",
        "traveltime_p95",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["row_count"] = frame["row_count"].astype(int)
    frame["distinct_links"] = frame["distinct_links"].astype(int)
    return frame


def _summarize_linkstats_grouped_view_deltas_bulk(
    *,
    tracker: Any,
    view_name: str,
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()

    required = {
        "year",
        "iteration",
        "beam_sub_iteration",
        "phys_sim_iteration_prev",
        "phys_sim_iteration_curr",
        "artifact_id_prev",
        "artifact_id_curr",
        "key_prev",
        "key_curr",
    }
    missing = sorted(required - set(pairs_df.columns))
    if missing:
        raise ValueError(f"pairs_df missing required columns: {missing}")

    rows_sql: list[str] = []
    for pair_id, row in enumerate(pairs_df.itertuples(index=False)):
        rows_sql.append(
            "("
            + ", ".join(
                [
                    str(pair_id),
                    _sql_literal(getattr(row, "year")),
                    _sql_literal(getattr(row, "iteration")),
                    _sql_literal(getattr(row, "beam_sub_iteration")),
                    _sql_literal(getattr(row, "phys_sim_iteration_prev")),
                    _sql_literal(getattr(row, "phys_sim_iteration_curr")),
                    _sql_literal(str(getattr(row, "artifact_id_prev"))),
                    _sql_literal(str(getattr(row, "artifact_id_curr"))),
                    _sql_literal(getattr(row, "key_prev")),
                    _sql_literal(getattr(row, "key_curr")),
                ]
            )
            + ")"
        )

    quoted_view = _quote_ident(view_name)
    query = text(
        f"""
        WITH pairs(
            pair_id,
            year,
            iteration,
            beam_sub_iteration,
            phys_sim_iteration_prev,
            phys_sim_iteration_curr,
            artifact_id_prev,
            artifact_id_curr,
            key_prev,
            key_curr
        ) AS (
            VALUES
                {", ".join(rows_sql)}
        ),
        relevant_artifacts(artifact_id) AS (
            SELECT artifact_id_prev FROM pairs
            UNION
            SELECT artifact_id_curr FROM pairs
        ),
        agg AS (
            SELECT
                CAST(consist_artifact_id AS VARCHAR) AS artifact_id,
                link,
                hour,
                SUM(volume) AS volume,
                AVG(traveltime) AS traveltime
            FROM {quoted_view}
            WHERE CAST(consist_artifact_id AS VARCHAR) IN (
                SELECT artifact_id FROM relevant_artifacts
            )
            GROUP BY 1, 2, 3
        ),
        pair_rows AS (
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
                a.link,
                a.hour,
                a.volume AS volume_prev,
                0.0 AS volume_curr,
                a.traveltime AS traveltime_prev,
                0.0 AS traveltime_curr
            FROM pairs p
            JOIN agg a ON a.artifact_id = p.artifact_id_prev
            UNION ALL
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
                a.link,
                a.hour,
                0.0 AS volume_prev,
                a.volume AS volume_curr,
                0.0 AS traveltime_prev,
                a.traveltime AS traveltime_curr
            FROM pairs p
            JOIN agg a ON a.artifact_id = p.artifact_id_curr
        ),
        joined AS (
            SELECT
                pair_id,
                year,
                iteration,
                beam_sub_iteration,
                phys_sim_iteration_prev,
                phys_sim_iteration_curr,
                artifact_id_prev,
                artifact_id_curr,
                key_prev,
                key_curr,
                link,
                hour,
                SUM(volume_prev) AS volume_previous,
                SUM(volume_curr) AS volume_current,
                SUM(traveltime_prev) AS traveltime_previous,
                SUM(traveltime_curr) AS traveltime_current
            FROM pair_rows
            GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
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
            AVG(volume_current - volume_previous) AS volume_delta_mean,
            AVG(ABS(volume_current - volume_previous)) AS volume_delta_abs_mean,
            AVG(traveltime_current - traveltime_previous) AS traveltime_delta_mean,
            AVG(ABS(traveltime_current - traveltime_previous)) AS traveltime_delta_abs_mean
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
            "volume_delta_mean",
            "volume_delta_abs_mean",
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
        "group_count",
        "volume_delta_mean",
        "volume_delta_abs_mean",
        "traveltime_delta_mean",
        "traveltime_delta_abs_mean",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["group_count"] = frame["group_count"].fillna(0).astype(int)
    return frame


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
    Compute consecutive phys-sim deltas within each (year, iteration, sub-iter) series.
    """
    if summary_df.empty:
        return pd.DataFrame()

    required_cols = {"year", "iteration", "phys_sim_iteration", "artifact_id", "view_name"}
    missing = sorted(required_cols - set(summary_df.columns))
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    non_null_views = summary_df["view_name"].dropna().astype(str).unique().tolist()
    if len(non_null_views) != 1:
        raise ValueError("summary_df must reference exactly one grouped view in 'view_name'")
    grouped_view_name = non_null_views[0]

    pair_rows: list[Dict[str, Any]] = []
    group_cols = ["year", "iteration", "beam_sub_iteration"]
    for _, group in summary_df.groupby(group_cols, dropna=False):
        ordered = group.sort_values("phys_sim_iteration")
        if len(ordered) < 2:
            continue
        for prev_row, curr_row in zip(
            ordered.iloc[:-1].itertuples(index=False),
            ordered.iloc[1:].itertuples(index=False),
        ):
            prev_artifact_id = getattr(prev_row, "artifact_id", None)
            curr_artifact_id = getattr(curr_row, "artifact_id", None)
            if prev_artifact_id is None or curr_artifact_id is None:
                continue
            pair_rows.append(
                {
                    "year": getattr(curr_row, "year"),
                    "iteration": getattr(curr_row, "iteration"),
                    "beam_sub_iteration": getattr(curr_row, "beam_sub_iteration"),
                    "phys_sim_iteration_prev": getattr(prev_row, "phys_sim_iteration"),
                    "phys_sim_iteration_curr": getattr(curr_row, "phys_sim_iteration"),
                    "artifact_id_prev": str(prev_artifact_id),
                    "artifact_id_curr": str(curr_artifact_id),
                    "key_prev": getattr(prev_row, "key", None),
                    "key_curr": getattr(curr_row, "key", None),
                }
            )

    if not pair_rows:
        return pd.DataFrame()

    pairs_df = pd.DataFrame(pair_rows)
    delta_df = _summarize_linkstats_grouped_view_deltas_bulk(
        tracker=tracker,
        view_name=grouped_view_name,
        pairs_df=pairs_df,
    )
    if delta_df.empty:
        return delta_df
    delta_df["view_prev"] = grouped_view_name
    delta_df["view_curr"] = grouped_view_name
    return delta_df
