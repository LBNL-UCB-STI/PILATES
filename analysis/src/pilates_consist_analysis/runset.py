from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import warnings

import pandas as pd

from consist import RunSet as _ConsistRunSet

RUNSET_COLUMNS = [
    "run_id",
    "parent_run_id",
    "name",
    "model",
    "year",
    "iteration",
    "status",
    "created_at",
    "ended_at",
]

_FIELD_ALIASES = {
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


class RunSet(_ConsistRunSet):
    """PILATES-aware RunSet extension with epoch/convergence helpers."""

    def filter(self, **field_values: Any) -> "RunSet":
        base = super().filter(**field_values)
        return _coerce_runset(
            base,
            label=runset_label(base, default=runset_label(self)),
            tracker=getattr(self, "_tracker", None),
        )

    def latest(self, group_by: Optional[Sequence[str]] = None) -> "RunSet":
        base = super().latest(group_by=list(group_by) if group_by else None)
        return _coerce_runset(
            base,
            label=runset_label(base, default=runset_label(self)),
            tracker=getattr(self, "_tracker", None),
        )

    def split_by(self, field: str) -> Dict[Any, "RunSet"]:
        grouped = super().split_by(field)
        return {
            key: _coerce_runset(
                value,
                label=runset_label(value, default=runset_label(self)),
                tracker=getattr(self, "_tracker", None),
            )
            for key, value in grouped.items()
        }

    def align(self, other: _ConsistRunSet, on: str) -> Any:
        pair = super().align(other, on=on)
        pair.left = _coerce_runset(
            pair.left,
            label=runset_label(pair.left, default=runset_label(self)),
            tracker=getattr(self, "_tracker", None),
        )
        pair.right = _coerce_runset(
            pair.right,
            label=runset_label(
                pair.right, default=runset_label(other, default="right")
            ),
            tracker=getattr(other, "_tracker", None) or getattr(self, "_tracker", None),
        )
        return pair

    def converged(self, group_by: Optional[Sequence[str]] = None) -> "RunSet":
        grouping = [
            str(value).strip()
            for value in (group_by or ("year", "scenario_id"))
            if str(value).strip()
        ]
        if not grouping:
            grouping = ["year", "scenario_id"]

        completed_status = "completed"
        max_iteration_by_group: Dict[tuple[Any, ...], int] = {}
        missing_iteration_count = 0

        for run in self:
            if _run_status(run) != completed_status:
                continue
            iteration = _as_int(_run_field(run, "iteration"))
            if iteration is None:
                missing_iteration_count += 1
                continue
            key = tuple(_run_field(run, field) for field in grouping)
            previous = max_iteration_by_group.get(key)
            if previous is None or iteration > previous:
                max_iteration_by_group[key] = iteration

        if missing_iteration_count:
            warnings.warn(
                "RunSet.converged skipped completed runs missing iteration "
                f"({missing_iteration_count}).",
                RuntimeWarning,
                stacklevel=2,
            )

        selected: list[Any] = []
        for run in self:
            if _run_status(run) != completed_status:
                continue
            iteration = _as_int(_run_field(run, "iteration"))
            if iteration is None:
                continue
            key = tuple(_run_field(run, field) for field in grouping)
            max_iteration = max_iteration_by_group.get(key)
            if max_iteration is not None and iteration == max_iteration:
                selected.append(run)

        label = f"{runset_label(self)}.converged"
        return runset_from_runs(
            selected,
            name=label,
            tracker=getattr(self, "_tracker", None),
        )


def _as_run_list(runs: Any) -> list[Any]:
    if runs is None:
        return []
    if isinstance(runs, dict):
        return list(runs.values())
    return list(runs)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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
    aliases = _FIELD_ALIASES.get(field, (field,))
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


def runs_to_frame(runs: Iterable[Any]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "run_id": str(getattr(run, "id", "") or ""),
                "parent_run_id": getattr(run, "parent_run_id", None),
                "name": getattr(run, "name", None) or getattr(run, "description", None),
                "model": getattr(run, "model_name", None),
                "year": getattr(run, "year", None),
                "iteration": getattr(run, "iteration", None),
                "status": getattr(run, "status", None),
                "created_at": _iso(getattr(run, "created_at", None)),
                "ended_at": _iso(getattr(run, "ended_at", None)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=RUNSET_COLUMNS)
    return pd.DataFrame(rows, columns=RUNSET_COLUMNS)


def runset_label(runset: _ConsistRunSet, *, default: str = "runset") -> str:
    label = getattr(runset, "label", None)
    if label is None:
        return default
    value = str(label).strip()
    return value or default


def runset_run_ids(runset: _ConsistRunSet) -> list[str]:
    run_ids: list[str] = []
    for run in runset:
        run_id = str(getattr(run, "id", "") or "").strip()
        if run_id:
            run_ids.append(run_id)
    return run_ids


def _coerce_runset(
    runset: Any,
    *,
    label: str = "runset",
    tracker: Optional[Any] = None,
) -> RunSet:
    if isinstance(runset, RunSet):
        return runset

    if isinstance(runset, _ConsistRunSet):
        runs = list(runset)
        resolved_label = runset_label(runset, default=label)
        resolved_tracker = tracker or getattr(runset, "_tracker", None)
        try:
            return RunSet(runs=runs, label=resolved_label, _tracker=resolved_tracker)
        except TypeError:
            return RunSet.from_runs(runs, label=resolved_label)

    return runset_from_runs(runset, name=label, tracker=tracker)


def runset_from_query(
    tracker: Any,
    *,
    runset_name: str = "runset",
    tags: Optional[list[str]] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    parent_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    limit: int = 100,
    run_name: Optional[str] = None,
) -> RunSet:
    filters: Dict[str, Any] = {
        "tags": tags,
        "year": year,
        "iteration": iteration,
        "model": model,
        "status": status,
        "parent_id": parent_id,
        "metadata": dict(metadata) if metadata is not None else None,
        "limit": limit,
        "name": run_name,
    }
    filtered = {k: v for k, v in filters.items() if v is not None}
    if hasattr(tracker, "run_set"):
        runset = tracker.run_set(label=runset_name, **filtered)
        return _coerce_runset(runset, label=runset_name, tracker=tracker)
    if hasattr(_ConsistRunSet, "from_query"):
        runset = _ConsistRunSet.from_query(tracker, label=runset_name, **filtered)
        return _coerce_runset(runset, label=runset_name, tracker=tracker)
    raise RuntimeError(
        "Tracker does not expose run_set() and RunSet.from_query unavailable."
    )


def runset_from_runs(
    runs: Iterable[Any],
    *,
    name: str = "runset",
    tracker: Optional[Any] = None,
) -> RunSet:
    run_list = _as_run_list(runs)
    if tracker is not None:
        try:
            return RunSet(runs=run_list, label=name, _tracker=tracker)
        except TypeError:
            pass
    return RunSet.from_runs(run_list, label=name)
