from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ANALYSIS_SRC = Path(__file__).resolve().parents[2] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis import open_archive

LOGICAL_KEYS = ["scenario_id", "year", "iteration", "model"]


def _normalize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    columns = LOGICAL_KEYS + ["run_id", "status", "created_at"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in frame.columns]
    normalized = frame.loc[:, available].copy()
    for column in LOGICAL_KEYS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized


def build_boundary_overlap_summary(
    baseline_runs: pd.DataFrame,
    rerun_runs: pd.DataFrame,
) -> pd.DataFrame:
    baseline = _normalize_runs(baseline_runs).rename(
        columns={
            "run_id": "baseline_run_id",
            "status": "baseline_status",
            "created_at": "baseline_created_at",
        }
    )
    rerun = _normalize_runs(rerun_runs).rename(
        columns={
            "run_id": "rerun_run_id",
            "status": "rerun_status",
            "created_at": "rerun_created_at",
        }
    )
    merged = baseline.merge(rerun, how="outer", on=LOGICAL_KEYS, indicator=True)
    merged["status"] = merged["_merge"].map(
        {"both": "both", "left_only": "baseline_only", "right_only": "rerun_only"}
    )
    for column in (
        "baseline_status",
        "rerun_status",
        "baseline_created_at",
        "rerun_created_at",
    ):
        if column not in merged.columns:
            merged[column] = None
    ordered = merged[
        LOGICAL_KEYS
        + [
            "status",
            "baseline_run_id",
            "rerun_run_id",
            "baseline_status",
            "rerun_status",
            "baseline_created_at",
            "rerun_created_at",
        ]
    ].sort_values(LOGICAL_KEYS, na_position="last")
    return ordered.reset_index(drop=True)


def _frame_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    return frame.to_string(index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare logical stage boundaries across a baseline archive and a rerun archive.",
    )
    parser.add_argument("baseline_archive_run_dir", help="Baseline archive run directory.")
    parser.add_argument("rerun_archive_run_dir", help="Rerun archive run directory.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root mounted as Consist inputs. Defaults to the current PILATES checkout.",
    )
    parser.add_argument("--scenario-id", help="Optional scenario filter.")
    parser.add_argument("--year", type=int, help="Optional year filter.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    baseline = open_archive(
        Path(args.baseline_archive_run_dir),
        project_root=Path(args.project_root),
    )
    rerun = open_archive(
        Path(args.rerun_archive_run_dir),
        project_root=Path(args.project_root),
    )

    baseline_runs = baseline.runs(
        scenario_id=args.scenario_id,
        year=args.year,
    )
    rerun_runs = rerun.runs(
        scenario_id=args.scenario_id,
        year=args.year,
    )
    overlap = build_boundary_overlap_summary(baseline_runs, rerun_runs)

    print("Baseline archive")
    print(_frame_text(baseline.summary()))
    print("\nRerun archive")
    print(_frame_text(rerun.summary()))
    print("\nLogical boundary overlap")
    print(_frame_text(overlap.head(args.limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
