from __future__ import annotations

import argparse
from pathlib import Path
import sys

import consist

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pilates.utils.provenance_report import write_provenance_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a human-readable provenance report for a Consist run id."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id to render.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Consist run directory used when the run executed.",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the Consist DuckDB database.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path. Defaults to <run-dir>/provenance_report_<run-id>.md",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print report contents to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    db_path = Path(args.db_path).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = run_dir / f"provenance_report_{args.run_id}.md"

    tracker = consist.Tracker(
        run_dir=run_dir,
        db_path=str(db_path),
        hashing_strategy="fast",
    )
    run = tracker.get_run(args.run_id)
    mounts = {}
    if run is not None and isinstance(getattr(run, "meta", None), dict):
        run_mounts = run.meta.get("mounts")
        if isinstance(run_mounts, dict):
            mounts = {str(k): str(v) for k, v in run_mounts.items()}
    if mounts:
        tracker = consist.Tracker(
            run_dir=run_dir,
            db_path=str(db_path),
            mounts=mounts,
            hashing_strategy="fast",
        )

    report = write_provenance_report(
        tracker=tracker,
        run_id=args.run_id,
        output_path=output_path,
    )
    print(f"Wrote provenance report to: {output_path}")
    if args.print:
        print()
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
