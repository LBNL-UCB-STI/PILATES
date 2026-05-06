from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pandas as pd

ANALYSIS_SRC = Path(__file__).resolve().parents[2] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis import open_archive


def summarize_run_outputs(outputs: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for key, artifact in sorted(outputs.items()):
        recovery_roots = list(getattr(artifact, "recovery_roots", []) or [])
        path = getattr(artifact, "path", None)
        rows.append(
            {
                "key": key,
                "path": str(path) if path is not None else None,
                "hash": getattr(artifact, "hash", None),
                "recovery_root_count": len(recovery_roots),
                "recovery_roots": ",".join(str(root) for root in recovery_roots),
            }
        )
    return pd.DataFrame(rows)


def _latest_run_id(
    archive, *, scenario_id: str | None = None, model: str | None = None
):
    runs = archive.runs(
        scenario_id=scenario_id,
        model=model,
        completed_only=False,
    )
    if runs.empty:
        raise ValueError("No runs matched the requested filters.")
    if "created_at" in runs.columns:
        runs = runs.sort_values("created_at", na_position="last")
    return str(runs.iloc[-1]["run_id"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect replay/restart-relevant archived outputs for one run.",
    )
    parser.add_argument("archive_run_dir", help="Archive run directory.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root mounted as Consist inputs. Defaults to the current PILATES checkout.",
    )
    parser.add_argument("--run-id", help="Explicit run id to inspect.")
    parser.add_argument(
        "--scenario-id", help="Optional scenario filter when selecting the latest run."
    )
    parser.add_argument(
        "--model", help="Optional model filter when selecting the latest run."
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Limit output rows to these artifact keys. Repeatable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    archive = open_archive(
        Path(args.archive_run_dir),
        project_root=Path(args.project_root),
    )
    run_id = args.run_id or _latest_run_id(
        archive,
        scenario_id=args.scenario_id,
        model=args.model,
    )
    outputs = archive.tracker.get_run_outputs(run_id) or {}
    frame = summarize_run_outputs(outputs)
    if args.keys:
        frame = frame.loc[frame["key"].isin(set(args.keys))]

    print("Archive summary")
    print(archive.summary().to_string(index=False))
    print(f"\nSelected run_id: {run_id}")
    if frame.empty:
        print("\nNo outputs matched the requested filters.")
    else:
        print("\nArchived outputs")
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
