from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sqlmodel import col, select

from .keys import CANONICAL_KEY_COLUMNS, ensure_canonical_key_columns
from .manifest import DatasetManifest


@dataclass
class SkimConvergenceDataset:
    artifacts: pd.DataFrame
    matrices: pd.DataFrame
    summary: pd.DataFrame
    deltas: pd.DataFrame


def _chunked(values: Sequence[str], size: int = 1000) -> Iterable[list[str]]:
    chunk: list[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _empty_artifacts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "concept_key",
            "run_id",
            "parent_run_id",
            "comparison_group",
            "model",
            "year",
            "iteration",
            "matrix_count",
            "row_count_min",
            "row_count_max",
            "row_count_mean",
            "col_count_min",
            "col_count_max",
            "col_count_mean",
            "matrix_row_count_stable",
            "matrix_col_count_stable",
            "scenario_id",
            "phys_sim_iteration",
            "beam_sub_iteration",
            "seed",
        ]
    )


def _empty_matrices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "concept_key",
            "matrix_name",
            "shape",
            "dtype",
            "n_rows",
            "n_cols",
            "attributes",
            "run_id",
            "parent_run_id",
            "comparison_group",
            "model",
            "year",
            "iteration",
            "scenario_id",
            "phys_sim_iteration",
            "beam_sub_iteration",
            "seed",
        ]
    )


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "concept_key",
            "comparison_group",
            "run_count",
            "model",
            "year",
            "iteration",
            "matrix_count_mean",
            "matrix_count_min",
            "matrix_count_max",
            "matrix_count_stable",
            "row_count_mean",
            "row_count_min",
            "row_count_max",
            "col_count_mean",
            "col_count_min",
            "col_count_max",
            "row_count_uniform_share",
            "col_count_uniform_share",
            "scenario_id",
            "run_id",
            "phys_sim_iteration",
            "beam_sub_iteration",
            "seed",
        ]
    )


def _empty_deltas_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "concept_key",
            "comparison_group",
            "model",
            "year",
            "iteration_prev",
            "iteration",
            "iteration_gap",
            "run_count_prev",
            "run_count",
            "matrix_count_mean_prev",
            "matrix_count_mean",
            "matrix_count_mean_delta",
            "matrix_count_mean_delta_abs",
            "matrix_count_stable_prev",
            "matrix_count_stable",
            "matrix_count_stability_pair",
            "row_count_mean_prev",
            "row_count_mean",
            "row_count_mean_delta",
            "row_count_mean_delta_abs",
            "row_count_uniform_share_prev",
            "row_count_uniform_share",
            "row_count_uniform_share_delta",
            "col_count_uniform_share_prev",
            "col_count_uniform_share",
            "col_count_uniform_share_delta",
            "scenario_id",
            "run_id",
            "phys_sim_iteration",
            "beam_sub_iteration",
            "seed",
        ]
    )


def _run_metadata_map(
    tracker: Any, run_ids: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    db = getattr(tracker, "db", None)
    if db is None or not hasattr(db, "session_scope"):
        return {}
    try:
        from consist.models.run import Run
    except Exception:
        return {}

    unique_ids = sorted({str(value).strip() for value in run_ids if str(value).strip()})
    if not unique_ids:
        return {}

    mapping: Dict[str, Dict[str, Any]] = {}
    for chunk in _chunked(unique_ids, size=1000):
        with db.session_scope() as session:
            rows = session.exec(select(Run).where(col(Run.id).in_(chunk))).all()
        for row in rows:
            run_id = str(getattr(row, "id", "") or "").strip()
            if not run_id:
                continue
            parent_run_id = getattr(row, "parent_run_id", None)
            mapping[run_id] = {
                "parent_run_id": str(parent_run_id).strip() if parent_run_id else None,
                "model_name": getattr(row, "model_name", None),
                "year": getattr(row, "year", None),
                "iteration": getattr(row, "iteration", None),
            }
    return mapping


def _discover_openmatrix_concept_keys(
    tracker: Any,
    *,
    key_contains: str = "skim",
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    limit: int = 10000,
) -> list[str]:
    db = getattr(tracker, "db", None)
    if db is None or not hasattr(db, "session_scope"):
        return []
    try:
        from consist.models.artifact import Artifact
        from consist.models.run import Run
    except Exception:
        return []

    key_contains_pattern = str(key_contains or "").strip()
    explicit_run_ids = sorted(
        {str(v).strip() for v in (run_ids or []) if str(v).strip()}
    )

    with db.session_scope() as session:
        statement = (
            select(Artifact.key)
            .join(Run, col(Artifact.run_id) == col(Run.id))
            .where(Artifact.driver == "openmatrix")
            .order_by(col(Artifact.created_at).desc())
            .limit(int(limit))
        )
        if key_contains_pattern:
            statement = statement.where(
                col(Artifact.key).ilike(f"%{key_contains_pattern}%")
            )
        if explicit_run_ids:
            statement = statement.where(col(Artifact.run_id).in_(explicit_run_ids))
        if year is not None:
            statement = statement.where(Run.year == int(year))
        if iteration is not None:
            statement = statement.where(Run.iteration == int(iteration))
        rows = session.exec(statement).all()

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _load_openmatrix_matrices(
    tracker: Any,
    *,
    concept_key: str,
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
) -> pd.DataFrame:
    if not hasattr(tracker, "openmatrix_metadata"):
        raise RuntimeError("Tracker does not expose openmatrix metadata views.")

    run_ids_list = sorted({str(v).strip() for v in (run_ids or []) if str(v).strip()})
    metadata_view = tracker.openmatrix_metadata(concept_key)
    matrices = metadata_view.get_matrices(
        concept_key,
        run_ids=run_ids_list or None,
        year=int(year) if year is not None else None,
    )
    if matrices.empty:
        return _empty_matrices_frame()

    frame = matrices.copy()
    frame["concept_key"] = concept_key
    frame["run_id"] = frame["run_id"].astype(str)
    frame["year"] = pd.to_numeric(frame.get("year"), errors="coerce")
    frame["iteration"] = pd.to_numeric(frame.get("iteration"), errors="coerce")
    frame["n_rows"] = pd.to_numeric(frame.get("n_rows"), errors="coerce")
    frame["n_cols"] = pd.to_numeric(frame.get("n_cols"), errors="coerce")
    if iteration is not None:
        frame = frame[frame["iteration"] == int(iteration)].copy()
    if frame.empty:
        return _empty_matrices_frame()
    return frame


def _annotate_run_fields(tracker: Any, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    run_ids = frame["run_id"].astype(str).tolist()
    run_meta = _run_metadata_map(tracker, run_ids)
    output = frame.copy()
    output["parent_run_id"] = output["run_id"].map(
        lambda rid: run_meta.get(str(rid), {}).get("parent_run_id")
    )
    output["comparison_group"] = np.where(
        output["parent_run_id"].notna() & output["parent_run_id"].astype(str).ne(""),
        output["parent_run_id"],
        output["run_id"].astype(str),
    )
    output["model"] = output["run_id"].map(
        lambda rid: run_meta.get(str(rid), {}).get("model_name")
    )
    missing_year = output["year"].isna()
    if missing_year.any():
        output.loc[missing_year, "year"] = output.loc[missing_year, "run_id"].map(
            lambda rid: run_meta.get(str(rid), {}).get("year")
        )
    missing_iteration = output["iteration"].isna()
    if missing_iteration.any():
        output.loc[missing_iteration, "iteration"] = output.loc[
            missing_iteration, "run_id"
        ].map(lambda rid: run_meta.get(str(rid), {}).get("iteration"))
    output["year"] = pd.to_numeric(output["year"], errors="coerce")
    output["iteration"] = pd.to_numeric(output["iteration"], errors="coerce")
    output["scenario_id"] = output["comparison_group"]
    output["phys_sim_iteration"] = None
    output["beam_sub_iteration"] = None
    output["seed"] = None
    return output


def _summarize_artifacts_from_matrices(matrices_df: pd.DataFrame) -> pd.DataFrame:
    if matrices_df.empty:
        return _empty_artifacts_frame()

    grouped = matrices_df.groupby(
        [
            "concept_key",
            "run_id",
            "parent_run_id",
            "comparison_group",
            "model",
            "year",
            "iteration",
        ],
        dropna=False,
    )
    artifacts = grouped.agg(
        matrix_count=("matrix_name", "nunique"),
        row_count_min=("n_rows", "min"),
        row_count_max=("n_rows", "max"),
        row_count_mean=("n_rows", "mean"),
        col_count_min=("n_cols", "min"),
        col_count_max=("n_cols", "max"),
        col_count_mean=("n_cols", "mean"),
        matrix_row_count_stable=("n_rows", lambda s: bool(s.dropna().nunique() <= 1)),
        matrix_col_count_stable=("n_cols", lambda s: bool(s.dropna().nunique() <= 1)),
    ).reset_index()
    artifacts["scenario_id"] = artifacts["comparison_group"]
    artifacts["phys_sim_iteration"] = None
    artifacts["beam_sub_iteration"] = None
    artifacts["seed"] = None
    artifacts = artifacts.sort_values(
        ["concept_key", "comparison_group", "year", "iteration", "run_id"],
        na_position="last",
    ).reset_index(drop=True)
    return ensure_canonical_key_columns(artifacts)


def _summarize_iterations_from_matrices(matrices_df: pd.DataFrame) -> pd.DataFrame:
    if matrices_df.empty:
        return _empty_summary_frame()

    run_level = (
        matrices_df.groupby(
            ["concept_key", "comparison_group", "run_id", "model", "year", "iteration"],
            dropna=False,
        )
        .agg(
            matrix_count=("matrix_name", "nunique"),
            row_count_mean=("n_rows", "mean"),
            row_count_min=("n_rows", "min"),
            row_count_max=("n_rows", "max"),
            col_count_mean=("n_cols", "mean"),
            col_count_min=("n_cols", "min"),
            col_count_max=("n_cols", "max"),
            row_count_uniform=("n_rows", lambda s: bool(s.dropna().nunique() <= 1)),
            col_count_uniform=("n_cols", lambda s: bool(s.dropna().nunique() <= 1)),
        )
        .reset_index()
    )
    if run_level.empty:
        return _empty_summary_frame()

    summary = (
        run_level.groupby(
            ["concept_key", "comparison_group", "model", "year", "iteration"],
            dropna=False,
        )
        .agg(
            run_count=("run_id", "nunique"),
            matrix_count_mean=("matrix_count", "mean"),
            matrix_count_min=("matrix_count", "min"),
            matrix_count_max=("matrix_count", "max"),
            row_count_mean=("row_count_mean", "mean"),
            row_count_min=("row_count_min", "min"),
            row_count_max=("row_count_max", "max"),
            col_count_mean=("col_count_mean", "mean"),
            col_count_min=("col_count_min", "min"),
            col_count_max=("col_count_max", "max"),
            row_count_uniform_share=("row_count_uniform", "mean"),
            col_count_uniform_share=("col_count_uniform", "mean"),
        )
        .reset_index()
    )
    summary["matrix_count_stable"] = (
        summary["matrix_count_min"] == summary["matrix_count_max"]
    )
    summary["scenario_id"] = summary["comparison_group"]
    summary["run_id"] = summary["comparison_group"]
    summary["phys_sim_iteration"] = None
    summary["beam_sub_iteration"] = None
    summary["seed"] = None
    summary = summary.sort_values(
        ["concept_key", "comparison_group", "year", "iteration"],
        na_position="last",
    ).reset_index(drop=True)
    return ensure_canonical_key_columns(summary)


def discover_skim_artifacts(
    tracker: Any,
    *,
    concept_keys: Optional[Sequence[str]] = None,
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    key_contains: str = "skim",
    limit: int = 10000,
) -> pd.DataFrame:
    keys = [str(key).strip() for key in (concept_keys or []) if str(key).strip()]
    if not keys:
        keys = _discover_openmatrix_concept_keys(
            tracker,
            key_contains=key_contains,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
            limit=limit,
        )
    if not keys:
        return _empty_artifacts_frame()

    matrices_frames: list[pd.DataFrame] = []
    for concept_key in keys:
        matrix_frame = _load_openmatrix_matrices(
            tracker,
            concept_key=concept_key,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
        )
        if matrix_frame.empty:
            continue
        matrices_frames.append(matrix_frame)

    if not matrices_frames:
        return _empty_artifacts_frame()

    matrices_df = pd.concat(matrices_frames, ignore_index=True)
    matrices_df = _annotate_run_fields(tracker, matrices_df)
    return _summarize_artifacts_from_matrices(matrices_df)


def summarize_skim_matrices(
    tracker: Any,
    *,
    concept_keys: Optional[Sequence[str]] = None,
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    key_contains: str = "skim",
    limit: int = 10000,
) -> pd.DataFrame:
    keys = [str(key).strip() for key in (concept_keys or []) if str(key).strip()]
    if not keys:
        keys = _discover_openmatrix_concept_keys(
            tracker,
            key_contains=key_contains,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
            limit=limit,
        )
    if not keys:
        return _empty_summary_frame()

    matrices_frames: list[pd.DataFrame] = []
    for concept_key in keys:
        frame = _load_openmatrix_matrices(
            tracker,
            concept_key=concept_key,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
        )
        if frame.empty:
            continue
        matrices_frames.append(frame)
    if not matrices_frames:
        return _empty_summary_frame()

    matrices_df = pd.concat(matrices_frames, ignore_index=True)
    matrices_df = _annotate_run_fields(tracker, matrices_df)
    if matrices_df.empty:
        return _empty_summary_frame()

    return _summarize_iterations_from_matrices(matrices_df)


def summarize_skim_iteration_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return _empty_deltas_frame()

    required_columns = {
        "concept_key",
        "comparison_group",
        "model",
        "year",
        "iteration",
        "run_count",
        "matrix_count_mean",
        "matrix_count_stable",
        "row_count_mean",
        "row_count_uniform_share",
        "col_count_uniform_share",
    }
    missing = sorted(required_columns.difference(summary_df.columns))
    if missing:
        raise ValueError(f"summary_df missing required columns: {', '.join(missing)}")

    frame = summary_df.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["iteration"] = pd.to_numeric(frame["iteration"], errors="coerce")
    frame["run_count"] = pd.to_numeric(frame["run_count"], errors="coerce")
    frame["matrix_count_mean"] = pd.to_numeric(
        frame["matrix_count_mean"], errors="coerce"
    )
    frame["row_count_mean"] = pd.to_numeric(frame["row_count_mean"], errors="coerce")
    frame["row_count_uniform_share"] = pd.to_numeric(
        frame["row_count_uniform_share"], errors="coerce"
    )
    frame["col_count_uniform_share"] = pd.to_numeric(
        frame["col_count_uniform_share"], errors="coerce"
    )
    frame["matrix_count_stable"] = (
        frame["matrix_count_stable"].fillna(False).astype(bool)
    )

    frame = frame.sort_values(
        ["concept_key", "comparison_group", "year", "iteration"],
        na_position="last",
    ).reset_index(drop=True)
    group_keys = ["concept_key", "comparison_group", "year"]
    frame["iteration_prev"] = frame.groupby(group_keys, dropna=False)[
        "iteration"
    ].shift(1)
    frame["run_count_prev"] = frame.groupby(group_keys, dropna=False)[
        "run_count"
    ].shift(1)
    frame["matrix_count_mean_prev"] = frame.groupby(group_keys, dropna=False)[
        "matrix_count_mean"
    ].shift(1)
    frame["matrix_count_stable_prev"] = frame.groupby(group_keys, dropna=False)[
        "matrix_count_stable"
    ].shift(1)
    frame["row_count_mean_prev"] = frame.groupby(group_keys, dropna=False)[
        "row_count_mean"
    ].shift(1)
    frame["row_count_uniform_share_prev"] = frame.groupby(group_keys, dropna=False)[
        "row_count_uniform_share"
    ].shift(1)
    frame["col_count_uniform_share_prev"] = frame.groupby(group_keys, dropna=False)[
        "col_count_uniform_share"
    ].shift(1)

    deltas = frame[frame["iteration_prev"].notna()].copy()
    if deltas.empty:
        return _empty_deltas_frame()

    deltas["iteration_gap"] = deltas["iteration"] - deltas["iteration_prev"]
    deltas["matrix_count_mean_delta"] = (
        deltas["matrix_count_mean"] - deltas["matrix_count_mean_prev"]
    )
    deltas["matrix_count_mean_delta_abs"] = deltas["matrix_count_mean_delta"].abs()
    deltas["row_count_mean_delta"] = (
        deltas["row_count_mean"] - deltas["row_count_mean_prev"]
    )
    deltas["row_count_mean_delta_abs"] = deltas["row_count_mean_delta"].abs()
    deltas["row_count_uniform_share_delta"] = (
        deltas["row_count_uniform_share"] - deltas["row_count_uniform_share_prev"]
    )
    deltas["col_count_uniform_share_delta"] = (
        deltas["col_count_uniform_share"] - deltas["col_count_uniform_share_prev"]
    )
    stable_curr = deltas["matrix_count_stable"].fillna(False).astype(bool)
    stable_prev = deltas["matrix_count_stable_prev"].fillna(False).astype(bool)
    deltas["matrix_count_stability_pair"] = (
        stable_curr & stable_prev & (deltas["matrix_count_mean_delta"] == 0)
    )
    deltas["scenario_id"] = deltas["comparison_group"]
    deltas["run_id"] = deltas["comparison_group"]
    deltas["phys_sim_iteration"] = None
    deltas["beam_sub_iteration"] = None
    deltas["seed"] = None

    ordered = deltas[
        [
            "concept_key",
            "comparison_group",
            "model",
            "year",
            "iteration_prev",
            "iteration",
            "iteration_gap",
            "run_count_prev",
            "run_count",
            "matrix_count_mean_prev",
            "matrix_count_mean",
            "matrix_count_mean_delta",
            "matrix_count_mean_delta_abs",
            "matrix_count_stable_prev",
            "matrix_count_stable",
            "matrix_count_stability_pair",
            "row_count_mean_prev",
            "row_count_mean",
            "row_count_mean_delta",
            "row_count_mean_delta_abs",
            "row_count_uniform_share_prev",
            "row_count_uniform_share",
            "row_count_uniform_share_delta",
            "col_count_uniform_share_prev",
            "col_count_uniform_share",
            "col_count_uniform_share_delta",
            "scenario_id",
            "run_id",
            "phys_sim_iteration",
            "beam_sub_iteration",
            "seed",
        ]
    ].copy()
    ordered = ordered.sort_values(
        ["concept_key", "comparison_group", "year", "iteration_prev", "iteration"],
        na_position="last",
    ).reset_index(drop=True)
    return ensure_canonical_key_columns(ordered)


def _build_matrices_table(
    tracker: Any,
    *,
    concept_keys: Sequence[str],
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for concept_key in concept_keys:
        frame = _load_openmatrix_matrices(
            tracker,
            concept_key=concept_key,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
        )
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return _empty_matrices_frame()
    matrices = pd.concat(frames, ignore_index=True)
    matrices = _annotate_run_fields(tracker, matrices)
    matrices = matrices.sort_values(
        [
            "concept_key",
            "comparison_group",
            "year",
            "iteration",
            "run_id",
            "matrix_name",
        ],
        na_position="last",
    ).reset_index(drop=True)
    return ensure_canonical_key_columns(matrices)


def build_skim_convergence_dataset(
    tracker: Any,
    *,
    concept_keys: Optional[Sequence[str]] = None,
    run_ids: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    key_contains: str = "skim",
    limit: int = 10000,
) -> SkimConvergenceDataset:
    keys = [str(key).strip() for key in (concept_keys or []) if str(key).strip()]
    if not keys:
        keys = _discover_openmatrix_concept_keys(
            tracker,
            key_contains=key_contains,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
            limit=limit,
        )
    if not keys:
        empty_artifacts = _empty_artifacts_frame()
        empty_matrices = _empty_matrices_frame()
        empty_summary = _empty_summary_frame()
        empty_deltas = _empty_deltas_frame()
        return SkimConvergenceDataset(
            artifacts=empty_artifacts,
            matrices=empty_matrices,
            summary=empty_summary,
            deltas=empty_deltas,
        )

    matrices = _build_matrices_table(
        tracker,
        concept_keys=keys,
        run_ids=run_ids,
        year=year,
        iteration=iteration,
    )
    artifacts = _summarize_artifacts_from_matrices(matrices)
    summary = _summarize_iterations_from_matrices(matrices)
    deltas = summarize_skim_iteration_deltas(summary)
    return SkimConvergenceDataset(
        artifacts=artifacts,
        matrices=matrices,
        summary=summary,
        deltas=deltas,
    )


def write_skim_convergence_dataset(
    dataset: SkimConvergenceDataset,
    *,
    output_dir: str | Path,
    archive_run_dir: str,
    db_path: str,
    query: Mapping[str, Any],
) -> DatasetManifest:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    artifacts_path = out / "skim_artifacts.csv"
    matrices_path = out / "skim_matrices.csv"
    summary_path = out / "skim_summary.csv"
    deltas_path = out / "skim_deltas.csv"

    ensure_canonical_key_columns(dataset.artifacts).to_csv(artifacts_path, index=False)
    ensure_canonical_key_columns(dataset.matrices).to_csv(matrices_path, index=False)
    ensure_canonical_key_columns(dataset.summary).to_csv(summary_path, index=False)
    ensure_canonical_key_columns(dataset.deltas).to_csv(deltas_path, index=False)

    manifest = DatasetManifest(
        dataset_name="skim_convergence_dataset",
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=dict(query),
        files={
            "artifacts": str(artifacts_path),
            "matrices": str(matrices_path),
            "summary": str(summary_path),
            "deltas": str(deltas_path),
        },
        row_counts={
            "artifacts": int(len(dataset.artifacts)),
            "matrices": int(len(dataset.matrices)),
            "summary": int(len(dataset.summary)),
            "deltas": int(len(dataset.deltas)),
        },
        key_columns=list(CANONICAL_KEY_COLUMNS),
    )
    manifest.write_json(out / "dataset_manifest.json")
    return manifest
