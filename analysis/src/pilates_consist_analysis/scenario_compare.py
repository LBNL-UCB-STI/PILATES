from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .activitysim_trips import build_activitysim_trips_dataset
from .datasets import build_linkstats_dataset
from .manifest import DatasetManifest
from .runset import RunSet, runset_from_runs, runset_label, runset_run_ids
from .skim_analysis import build_skim_convergence_dataset

SUPPORTED_DATASETS = ("linkstats", "asim_trips", "skims")

_RUN_FIELD_ALIASES = {
    "year": ("year", "simulation_year", "facet.year"),
    "iteration": (
        "iteration",
        "outer_iteration",
        "simulation_iteration",
        "facet.iteration",
    ),
    "scenario_id": (
        "scenario_id",
        "scenario",
        "scenario.id",
        "facet.scenario_id",
    ),
    "model": ("model_name", "model", "facet.model"),
}


@dataclass
class ScenarioComparison:
    left_name: str
    right_name: str
    left_run_ids: list[str]
    right_run_ids: list[str]
    aligned_on: str
    aligned_keys: list[Any]
    config_diff: pd.DataFrame
    dataset_summaries: pd.DataFrame
    dataset_frames: Dict[str, pd.DataFrame]


def _normalize_dataset_list(datasets: Optional[Sequence[str]]) -> list[str]:
    if not datasets:
        return list(SUPPORTED_DATASETS)
    output: list[str] = []
    for raw in datasets:
        value = str(raw).strip().lower()
        if not value:
            continue
        if value not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{value}'. Expected one of: {SUPPORTED_DATASETS}."
            )
        if value not in output:
            output.append(value)
    if not output:
        return list(SUPPORTED_DATASETS)
    return output


def _normalize_group_by(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    if values is None:
        return None
    grouping = [str(value).strip() for value in values if str(value).strip()]
    return grouping or None


def _to_string_set(values: Iterable[Any]) -> set[str]:
    return {str(v).strip() for v in values if str(v).strip()}


def _run_parent_ids(runset: RunSet) -> set[str]:
    return _to_string_set(getattr(run, "parent_run_id", None) for run in runset)


def _restrict_to_allowed_ids(frame: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not allowed:
        return frame.iloc[0:0].copy()

    candidate_cols = ("run_id", "parent_run_id", "comparison_group")
    mask = pd.Series(False, index=frame.index)
    has_candidate = False
    for column in candidate_cols:
        if column not in frame.columns:
            continue
        has_candidate = True
        mask = mask | frame[column].astype(str).isin(allowed)
    if not has_candidate:
        return frame.copy()
    return frame.loc[mask].copy().reset_index(drop=True)


def _restrict_to_runset(frame: pd.DataFrame, runset: RunSet) -> pd.DataFrame:
    allowed = set(runset_run_ids(runset)) | _run_parent_ids(runset)
    return _restrict_to_allowed_ids(frame, allowed)


def _numeric_columns(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def _metric_agg(metric: str) -> str:
    metric_lower = metric.lower()
    sum_tokens = (
        "count",
        "sum",
        "total",
        "trips",
        "volume",
        "vmt",
        "vht",
        "rows",
        "links",
        "persons",
        "artifacts",
    )
    if any(token in metric_lower for token in sum_tokens):
        return "sum"
    return "mean"


def _metadata_sources(run: Any) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for name in ("metadata", "meta"):
        value = getattr(run, name, None)
        if isinstance(value, Mapping):
            output.append(value)
    return output


def _lookup_mapping(mapping: Mapping[str, Any], key_path: str) -> Any:
    if key_path in mapping:
        return mapping.get(key_path)
    current: Any = mapping
    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _run_field(run: Any, field: str) -> Any:
    aliases = _RUN_FIELD_ALIASES.get(field, (field,))
    for key in aliases:
        if "." not in key and hasattr(run, key):
            value = getattr(run, key)
            if value is not None and str(value).strip() != "":
                return value

    for source in _metadata_sources(run):
        for key in aliases:
            value = _lookup_mapping(source, key)
            if value is not None and str(value).strip() != "":
                return value

    if field == "model":
        value = getattr(run, "description", None)
        if value is not None and str(value).strip():
            return value
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_status(run: Any) -> str:
    return str(getattr(run, "status", "") or "").strip().lower()


def _normalized_align_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    output = str(value).strip()
    return output or None


def _align_key_for_run(run: Any, *, align_on: str) -> Optional[str]:
    return _normalized_align_key(_run_field(run, align_on))


def _is_complete_epoch_candidate(run: Any) -> bool:
    return (
        _run_status(run) == "completed"
        and _as_int(_run_field(run, "iteration")) is not None
    )


def _runset_keys(
    runset: RunSet,
    *,
    align_on: str,
    completed_only: bool,
) -> set[str]:
    output: set[str] = set()
    for run in runset:
        if completed_only and not _is_complete_epoch_candidate(run):
            continue
        key = _align_key_for_run(run, align_on=align_on)
        if key is not None:
            output.add(key)
    return output


def _format_key_list(values: Sequence[str], *, limit: int = 8) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return "<none>"
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    hidden = len(cleaned) - limit
    return f"{', '.join(cleaned[:limit])} (+{hidden} more)"


def _validate_converged_alignment_candidates(
    *,
    run_set_left: RunSet,
    run_set_right: RunSet,
    left_converged: RunSet,
    right_converged: RunSet,
    align_on: str,
    converged_group_by: Optional[Sequence[str]],
) -> None:
    overlapping_keys = sorted(
        _runset_keys(run_set_left, align_on=align_on, completed_only=False)
        & _runset_keys(run_set_right, align_on=align_on, completed_only=False)
    )
    if not overlapping_keys:
        return

    left_keys = _runset_keys(left_converged, align_on=align_on, completed_only=True)
    right_keys = _runset_keys(right_converged, align_on=align_on, completed_only=True)
    missing_left = sorted(key for key in overlapping_keys if key not in left_keys)
    missing_right = sorted(key for key in overlapping_keys if key not in right_keys)
    if not missing_left and not missing_right:
        return

    resolved_grouping = (
        list(converged_group_by) if converged_group_by else ["year", "scenario_id"]
    )
    raise ValueError(
        "Converged scenario compare requires complete epoch candidates on both sides "
        f"for aligned keys (align_on='{align_on}'). "
        f"Overlapping keys: {_format_key_list(overlapping_keys)}. "
        f"Missing on left: {_format_key_list(missing_left)}. "
        f"Missing on right: {_format_key_list(missing_right)}. "
        "Ensure each side has completed runs with iteration values for these keys, "
        f"adjust --converged-group-by (current={resolved_grouping}), "
        "tighten side filters, or disable converged mode."
    )


def _aggregate_for_compare(
    frame: pd.DataFrame, key_cols: Sequence[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = [column for column in key_cols if column in frame.columns]
    if not keys:
        frame = frame.copy()
        frame["__all__"] = "all"
        keys = ["__all__"]
    exclude = set(keys) | {
        "run_id",
        "parent_run_id",
        "comparison_group",
        "scenario_id",
        "model",
        "seed",
    }
    metrics = _numeric_columns(frame, exclude=exclude)
    if not metrics:
        return frame[keys].drop_duplicates().reset_index(drop=True)
    aggregations = {metric: _metric_agg(metric) for metric in metrics}
    output = frame.groupby(keys, dropna=False).agg(aggregations).reset_index()
    return output


def _merge_comparison_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    left_keys = [column for column in key_cols if column in left.columns]
    right_keys = [column for column in key_cols if column in right.columns]
    keys = [
        column for column in key_cols if column in left_keys and column in right_keys
    ]
    if not keys:
        left = left.copy()
        right = right.copy()
        left["__all__"] = "all"
        right["__all__"] = "all"
        keys = ["__all__"]

    left_metrics = [column for column in left.columns if column not in keys]
    right_metrics = [column for column in right.columns if column not in keys]
    common_metrics = sorted(set(left_metrics) & set(right_metrics))

    left_payload = left[keys + common_metrics].rename(
        columns={metric: f"{metric}_{left_name}" for metric in common_metrics}
    )
    right_payload = right[keys + common_metrics].rename(
        columns={metric: f"{metric}_{right_name}" for metric in common_metrics}
    )
    merged = pd.merge(left_payload, right_payload, on=keys, how="outer", sort=True)

    for metric in common_metrics:
        left_col = f"{metric}_{left_name}"
        right_col = f"{metric}_{right_name}"
        delta_col = f"{metric}_delta"
        delta_abs_col = f"{metric}_delta_abs"
        rel_col = f"{metric}_delta_rel"
        merged[delta_col] = merged[right_col] - merged[left_col]
        merged[delta_abs_col] = merged[delta_col].abs()
        merged[rel_col] = np.where(
            merged[left_col].replace(0, np.nan).notna(),
            merged[delta_col] / merged[left_col].replace(0, np.nan),
            np.nan,
        )

    return merged


def _dataset_summary_row(
    dataset: str, merged: pd.DataFrame, *, left_name: str, right_name: str
) -> Dict[str, Any]:
    delta_cols = [column for column in merged.columns if column.endswith("_delta")]
    delta_abs_cols = [
        column for column in merged.columns if column.endswith("_delta_abs")
    ]
    rel_cols = [column for column in merged.columns if column.endswith("_delta_rel")]
    left_cols = [
        column for column in merged.columns if column.endswith(f"_{left_name}")
    ]
    right_cols = [
        column for column in merged.columns if column.endswith(f"_{right_name}")
    ]

    left_non_null = int(merged[left_cols].notna().any(axis=1).sum()) if left_cols else 0
    right_non_null = (
        int(merged[right_cols].notna().any(axis=1).sum()) if right_cols else 0
    )
    overlap = (
        int(
            (
                merged[left_cols].notna().any(axis=1)
                & merged[right_cols].notna().any(axis=1)
            ).sum()
        )
        if left_cols and right_cols
        else 0
    )

    def _safe_mean(columns: list[str]) -> float:
        if not columns:
            return float("nan")
        values = merged[columns].to_numpy(dtype="float64")
        if values.size == 0:
            return float("nan")
        return float(np.nanmean(values))

    return {
        "dataset": dataset,
        "row_count": int(len(merged)),
        "left_rows_with_data": left_non_null,
        "right_rows_with_data": right_non_null,
        "overlap_rows": overlap,
        "metric_count": int(len(delta_cols)),
        "mean_delta": _safe_mean(delta_cols),
        "mean_abs_delta": _safe_mean(delta_abs_cols),
        "mean_rel_delta": _safe_mean(rel_cols),
    }


def _build_aligned_pair(
    run_set_left: RunSet,
    run_set_right: RunSet,
    *,
    align_on: str,
    latest_group_by: Optional[Sequence[str]] = None,
    use_converged: bool = False,
    converged_group_by: Optional[Sequence[str]] = None,
):
    compare_left = run_set_left
    compare_right = run_set_right

    resolved_converged_grouping = _normalize_group_by(converged_group_by)
    if use_converged:
        compare_left = run_set_left.converged(group_by=resolved_converged_grouping)
        compare_right = run_set_right.converged(group_by=resolved_converged_grouping)
        _validate_converged_alignment_candidates(
            run_set_left=run_set_left,
            run_set_right=run_set_right,
            left_converged=compare_left,
            right_converged=compare_right,
            align_on=align_on,
            converged_group_by=resolved_converged_grouping,
        )

    grouping = _normalize_group_by(latest_group_by) or [align_on]
    if not grouping:
        grouping = [align_on]
    left_aligned = compare_left.latest(group_by=grouping)
    right_aligned = compare_right.latest(group_by=grouping)
    return left_aligned.align(right_aligned, on=align_on)


def _build_config_diff(
    aligned_pair: Any,
    *,
    namespace: Optional[str] = None,
    prefix: Optional[str] = None,
    include_equal: bool = False,
) -> pd.DataFrame:
    frame = aligned_pair.config_diffs(namespace=namespace, prefix=prefix)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "align_on",
                "on_value",
                "key",
                "namespace",
                "status",
                "left",
                "right",
                "run_id_left",
                "run_id_right",
                "left_status",
                "right_status",
            ]
        )
    if not include_equal:
        frame = frame.loc[frame["status"] != "equal"].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "align_on",
                "on_value",
                "key",
                "namespace",
                "status",
                "left",
                "right",
                "run_id_left",
                "run_id_right",
                "left_status",
                "right_status",
            ]
        )

    pair_frame = aligned_pair.to_frame().rename(columns={"key": "on_value"})
    pair_frame = pair_frame.rename(
        columns={
            "left_run_id": "run_id_left",
            "right_run_id": "run_id_right",
        }
    )
    merged = frame.merge(pair_frame, on="on_value", how="left")
    merged = merged.rename(columns={"left_value": "left", "right_value": "right"})
    merged["align_on"] = aligned_pair.on
    return (
        merged[
            [
                "align_on",
                "on_value",
                "key",
                "namespace",
                "status",
                "left",
                "right",
                "run_id_left",
                "run_id_right",
                "left_status",
                "right_status",
            ]
        ]
        .sort_values(["status", "key"])
        .reset_index(drop=True)
    )


def _build_dataset_frame(
    tracker: Any,
    *,
    dataset: str,
    runset: RunSet,
    year: Optional[int],
    iteration: Optional[int],
) -> pd.DataFrame:
    if dataset == "linkstats":
        built = build_linkstats_dataset(tracker, year=year, iteration=iteration)
        return _restrict_to_runset(built.summary, runset)
    if dataset == "asim_trips":
        built = build_activitysim_trips_dataset(tracker, year=year, iteration=iteration)
        return _restrict_to_runset(built.iteration_summary, runset)
    if dataset == "skims":
        built = build_skim_convergence_dataset(
            tracker,
            run_ids=runset_run_ids(runset),
            year=year,
            iteration=iteration,
        )
        return _restrict_to_runset(built.summary, runset)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _key_columns_for_dataset(dataset: str) -> list[str]:
    if dataset == "linkstats":
        return ["year", "iteration"]
    if dataset == "asim_trips":
        return ["year", "iteration"]
    if dataset == "skims":
        return ["concept_key", "year", "iteration"]
    return ["year", "iteration"]


def _dataset_pair_frame(
    left_run: Any,
    right_run: Any,
    _align_key: Any,
    *,
    dataset: str,
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    left_allowed = _to_string_set(
        [getattr(left_run, "id", None), getattr(left_run, "parent_run_id", None)]
    )
    right_allowed = _to_string_set(
        [getattr(right_run, "id", None), getattr(right_run, "parent_run_id", None)]
    )
    left_filtered = _restrict_to_allowed_ids(left_frame, left_allowed)
    right_filtered = _restrict_to_allowed_ids(right_frame, right_allowed)

    key_cols = _key_columns_for_dataset(dataset)
    left_agg = _aggregate_for_compare(left_filtered, key_cols=key_cols)
    right_agg = _aggregate_for_compare(right_filtered, key_cols=key_cols)
    merged = _merge_comparison_frames(
        left_agg,
        right_agg,
        key_cols=key_cols,
        left_name=left_name,
        right_name=right_name,
    )
    merged["dataset"] = dataset
    return merged


def compare_scenarios(
    tracker: Any,
    run_set_left: RunSet,
    run_set_right: RunSet,
    *,
    datasets: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    config_namespace: Optional[str] = None,
    config_prefix: Optional[str] = None,
    config_include_equal: bool = False,
    align_on: str = "year",
    latest_group_by: Optional[Sequence[str]] = None,
    use_converged: bool = False,
    converged_group_by: Optional[Sequence[str]] = None,
) -> ScenarioComparison:
    dataset_list = _normalize_dataset_list(datasets)
    left_name = runset_label(run_set_left, default="left")
    right_name = runset_label(run_set_right, default="right")
    aligned_pair = _build_aligned_pair(
        run_set_left,
        run_set_right,
        align_on=align_on,
        latest_group_by=latest_group_by,
        use_converged=use_converged,
        converged_group_by=converged_group_by,
    )

    output_frames: Dict[str, pd.DataFrame] = {}
    summary_rows: list[Dict[str, Any]] = []

    for dataset in dataset_list:
        left_dataset_frame = _build_dataset_frame(
            tracker,
            dataset=dataset,
            runset=run_set_left,
            year=year,
            iteration=iteration,
        )
        right_dataset_frame = _build_dataset_frame(
            tracker,
            dataset=dataset,
            runset=run_set_right,
            year=year,
            iteration=iteration,
        )
        merged = aligned_pair.apply(
            lambda left_run, right_run, align_key: _dataset_pair_frame(
                left_run,
                right_run,
                align_key,
                dataset=dataset,
                left_frame=left_dataset_frame,
                right_frame=right_dataset_frame,
                left_name=left_name,
                right_name=right_name,
            )
        )
        merged["align_on"] = align_on
        output_frames[dataset] = merged
        summary_rows.append(
            _dataset_summary_row(
                dataset,
                merged,
                left_name=left_name,
                right_name=right_name,
            )
        )

    config_diff = _build_config_diff(
        aligned_pair,
        namespace=config_namespace,
        prefix=config_prefix,
        include_equal=config_include_equal,
    )

    dataset_summaries = (
        pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    )
    return ScenarioComparison(
        left_name=left_name,
        right_name=right_name,
        left_run_ids=runset_run_ids(run_set_left),
        right_run_ids=runset_run_ids(run_set_right),
        aligned_on=align_on,
        aligned_keys=list(aligned_pair.keys),
        config_diff=config_diff,
        dataset_summaries=dataset_summaries,
        dataset_frames=output_frames,
    )


def runset_from_run_ids(
    tracker: Any,
    run_ids: Sequence[str],
    *,
    name: str,
) -> RunSet:
    resolved_runs: list[Any] = []
    for run_id in run_ids:
        value = str(run_id).strip()
        if not value:
            continue
        run = tracker.get_run(value) if hasattr(tracker, "get_run") else None
        if run is None and hasattr(tracker, "queries"):
            try:
                run = tracker.queries.find_run(id=value)
            except Exception:
                run = None
        if run is not None:
            resolved_runs.append(run)

    return runset_from_runs(resolved_runs, name=name, tracker=tracker)


def write_scenario_comparison(
    comparison: ScenarioComparison,
    *,
    output_dir: str | Path,
    archive_run_dir: str,
    db_path: str,
    query: Mapping[str, Any],
) -> DatasetManifest:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "scenario_comparison_summary.csv"
    config_diff_path = out / "scenario_comparison_config_diff.csv"
    comparison.dataset_summaries.to_csv(summary_path, index=False)
    comparison.config_diff.to_csv(config_diff_path, index=False)

    files: Dict[str, str] = {
        "summary": str(summary_path),
        "config_diff": str(config_diff_path),
    }
    row_counts: Dict[str, int] = {
        "summary": int(len(comparison.dataset_summaries)),
        "config_diff": int(len(comparison.config_diff)),
    }
    for dataset, frame in comparison.dataset_frames.items():
        path = out / f"scenario_{dataset}_comparison.csv"
        frame.to_csv(path, index=False)
        files[f"{dataset}_comparison"] = str(path)
        row_counts[f"{dataset}_comparison"] = int(len(frame))

    manifest = DatasetManifest(
        dataset_name="scenario_comparison",
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=dict(query),
        files=files,
        row_counts=row_counts,
        key_columns=["year", "iteration"],
        notes=[
            f"left_name={comparison.left_name}",
            f"right_name={comparison.right_name}",
            f"aligned_on={comparison.aligned_on}",
            f"aligned_pairs={len(comparison.aligned_keys)}",
        ],
    )
    manifest.write_json(out / "dataset_manifest.json")
    return manifest
