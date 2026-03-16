from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import pandas as pd

from .activitysim_trips import build_activitysim_trips_dataset
from .runset import RunSet, runset_label
from .scenario_compare import (
    ScenarioComparison,
    _aggregate_for_compare,
    _build_aligned_pair,
    _merge_comparison_frames,
)

if TYPE_CHECKING:
    from .archive import Archive, ArchiveScenario


def _normalized_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _run_identifiers(runset: RunSet) -> set[str]:
    values: set[str] = set()
    for run in runset:
        run_id = _normalized_text(getattr(run, "id", None))
        parent_run_id = _normalized_text(getattr(run, "parent_run_id", None))
        if run_id:
            values.add(run_id)
        if parent_run_id:
            values.add(parent_run_id)
    return values


def _restrict_to_runset(frame: pd.DataFrame, runset: RunSet) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    allowed = _run_identifiers(runset)
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


def _compare_frames(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    left_agg = _aggregate_for_compare(left_frame, key_cols=key_cols)
    right_agg = _aggregate_for_compare(right_frame, key_cols=key_cols)
    return _merge_comparison_frames(
        left_agg,
        right_agg,
        key_cols=key_cols,
        left_name=left_name,
        right_name=right_name,
    )


def _selection_name(selection: Any, *, default: str) -> str:
    if hasattr(selection, "scenario_id"):
        value = _normalized_text(getattr(selection, "scenario_id", None))
        if value:
            return value
    if isinstance(selection, str):
        value = _normalized_text(selection)
        if value:
            return value
    if isinstance(selection, RunSet):
        return runset_label(selection, default=default)
    return default


@dataclass(frozen=True)
class Comparison:
    archive: "Archive"
    raw: ScenarioComparison
    left_runset: RunSet
    right_runset: RunSet
    selected_left_runset: RunSet
    selected_right_runset: RunSet
    year: Optional[int] = None
    iteration: Optional[int] = None
    use_converged: bool = False
    latest_group_by: Optional[tuple[str, ...]] = None
    converged_group_by: Optional[tuple[str, ...]] = None

    @property
    def left_name(self) -> str:
        return self.raw.left_name

    @property
    def right_name(self) -> str:
        return self.raw.right_name

    @property
    def align_on(self) -> str:
        return self.raw.aligned_on

    @property
    def aligned_keys(self) -> list[Any]:
        return list(self.raw.aligned_keys)

    def summary(self) -> pd.DataFrame:
        row = {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "align_on": self.align_on,
            "aligned_key_count": int(len(self.aligned_keys)),
            "aligned_keys": ",".join(str(value) for value in self.aligned_keys),
            "dataset_count": int(len(self.raw.dataset_frames)),
            "config_diff_count": int(len(self.raw.config_diff)),
            "left_run_count": int(len(list(self.left_runset))),
            "right_run_count": int(len(list(self.right_runset))),
            "use_converged": bool(self.use_converged),
            "year_filter": self.year,
            "iteration_filter": self.iteration,
        }
        return pd.DataFrame([row])

    def dataset_summaries(self) -> pd.DataFrame:
        return self.raw.dataset_summaries.copy()

    def config_diff(self) -> pd.DataFrame:
        return self.raw.config_diff.copy()

    def frame(self, dataset: str) -> pd.DataFrame:
        normalized = str(dataset).strip().lower()
        if normalized not in self.raw.dataset_frames:
            available = sorted(self.raw.dataset_frames.keys())
            raise KeyError(
                f"Dataset '{normalized}' not available on comparison. Available: {available}."
            )
        return self.raw.dataset_frames[normalized].copy()

    def linkstats_summary(self) -> pd.DataFrame:
        return self.frame("linkstats")

    def asim_iteration_summary(self) -> pd.DataFrame:
        return self.frame("asim_trips")

    def skims_summary(self) -> pd.DataFrame:
        return self.frame("skims")

    def mode_shares(self) -> pd.DataFrame:
        dataset = build_activitysim_trips_dataset(
            self.archive.tracker,
            year=self.year,
            iteration=self.iteration,
        )
        left_frame = _restrict_to_runset(dataset.mode_counts, self.selected_left_runset)
        right_frame = _restrict_to_runset(dataset.mode_counts, self.selected_right_runset)
        merged = _compare_frames(
            left_frame,
            right_frame,
            key_cols=["year", "iteration", "trip_mode"],
            left_name=self.left_name,
            right_name=self.right_name,
        )
        merged["dataset"] = "asim_mode_shares"
        merged["align_on"] = self.align_on
        return merged

    def trip_purposes(self) -> pd.DataFrame:
        dataset = build_activitysim_trips_dataset(
            self.archive.tracker,
            year=self.year,
            iteration=self.iteration,
        )
        left_frame = _restrict_to_runset(
            dataset.purpose_mode_counts,
            self.selected_left_runset,
        )
        right_frame = _restrict_to_runset(
            dataset.purpose_mode_counts,
            self.selected_right_runset,
        )
        merged = _compare_frames(
            left_frame,
            right_frame,
            key_cols=["year", "iteration", "primary_purpose", "trip_mode"],
            left_name=self.left_name,
            right_name=self.right_name,
        )
        merged["dataset"] = "asim_trip_purposes"
        merged["align_on"] = self.align_on
        return merged

    def __repr__(self) -> str:
        return (
            "Comparison("
            f"left_name={self.left_name!r}, "
            f"right_name={self.right_name!r}, "
            f"align_on={self.align_on!r}, "
            f"aligned_keys={self.aligned_keys}, "
            f"datasets={sorted(self.raw.dataset_frames.keys())})"
        )


def build_comparison(
    archive: "Archive",
    *,
    left: str | "ArchiveScenario" | RunSet | Iterable[str],
    right: str | "ArchiveScenario" | RunSet | Iterable[str],
    left_runset: RunSet,
    right_runset: RunSet,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    datasets: Optional[Sequence[str]] = None,
    config_namespace: Optional[str] = None,
    config_prefix: Optional[str] = None,
    config_include_equal: bool = False,
    align_on: str = "year",
    latest_group_by: Optional[Sequence[str]] = None,
    use_converged: bool = False,
    converged_group_by: Optional[Sequence[str]] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
) -> Comparison:
    aligned_pair = _build_aligned_pair(
        left_runset,
        right_runset,
        align_on=align_on,
        latest_group_by=list(latest_group_by) if latest_group_by is not None else None,
        use_converged=bool(use_converged),
        converged_group_by=list(converged_group_by)
        if converged_group_by is not None
        else None,
    )
    resolved_left_name = left_name or _selection_name(left, default="left")
    resolved_right_name = right_name or _selection_name(right, default="right")
    raw = archive.session.compare_scenarios(
        left_runset,
        right_runset,
        left_name=resolved_left_name,
        right_name=resolved_right_name,
        datasets=list(datasets) if datasets is not None else None,
        year=year,
        iteration=iteration,
        config_namespace=config_namespace,
        config_prefix=config_prefix,
        config_include_equal=config_include_equal,
        align_on=align_on,
        latest_group_by=list(latest_group_by) if latest_group_by is not None else None,
        use_converged=bool(use_converged),
        converged_group_by=list(converged_group_by)
        if converged_group_by is not None
        else None,
    )
    return Comparison(
        archive=archive,
        raw=raw,
        left_runset=left_runset,
        right_runset=right_runset,
        selected_left_runset=aligned_pair.left,
        selected_right_runset=aligned_pair.right,
        year=year,
        iteration=iteration,
        use_converged=bool(use_converged),
        latest_group_by=tuple(latest_group_by) if latest_group_by is not None else None,
        converged_group_by=tuple(converged_group_by)
        if converged_group_by is not None
        else None,
    )
