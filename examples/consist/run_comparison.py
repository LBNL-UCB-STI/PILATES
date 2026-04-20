from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

ANALYSIS_SRC = Path(__file__).resolve().parents[2] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis import open_archive


def run_comparison(
    archive,
    *,
    left_scenario: str | None = None,
    right_scenario: str | None = None,
    left_run_ids: list[str] | None = None,
    right_run_ids: list[str] | None = None,
    year: int | None = None,
    align_on: str = "year",
    converged: bool = False,
):
    if left_scenario and right_scenario:
        left = left_scenario
        right = right_scenario
    elif left_run_ids and right_run_ids:
        left = left_run_ids
        right = right_run_ids
    else:
        raise ValueError(
            "Provide either both scenario selections or both run-id selections."
        )
    return archive.compare(
        left,
        right,
        year=year,
        align_on=align_on,
        converged=converged,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two scenarios or explicit run selections inside one archive.",
    )
    parser.add_argument("archive_run_dir", help="Archive run directory.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root mounted as Consist inputs. Defaults to the current PILATES checkout.",
    )
    parser.add_argument("--left-scenario", help="Left scenario id.")
    parser.add_argument("--right-scenario", help="Right scenario id.")
    parser.add_argument(
        "--left-run-id",
        action="append",
        dest="left_run_ids",
        help="Explicit left run id. Repeatable.",
    )
    parser.add_argument(
        "--right-run-id",
        action="append",
        dest="right_run_ids",
        help="Explicit right run id. Repeatable.",
    )
    parser.add_argument("--year", type=int, help="Optional year filter.")
    parser.add_argument("--align-on", default="year", help="RunSet alignment key.")
    parser.add_argument(
        "--converged",
        action="store_true",
        help="Select converged epochs before comparing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    archive = open_archive(
        Path(args.archive_run_dir),
        project_root=Path(args.project_root),
    )
    comparison = run_comparison(
        archive,
        left_scenario=args.left_scenario,
        right_scenario=args.right_scenario,
        left_run_ids=args.left_run_ids,
        right_run_ids=args.right_run_ids,
        year=args.year,
        align_on=args.align_on,
        converged=args.converged,
    )

    print("Comparison summary")
    print(comparison.summary().to_string(index=False))
    dataset_summaries = comparison.dataset_summaries()
    if not dataset_summaries.empty:
        print("\nDataset summaries")
        print(dataset_summaries.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
