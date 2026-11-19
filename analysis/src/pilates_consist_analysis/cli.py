from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


from .catalog import find_runs, runs_to_frame
from .epochs import build_epoch_panel
from .runtime import (
    create_analysis_tracker,
    db_health_to_frame,
    get_db_health,
    get_db_health_issues,
    get_run_tagging_issues,
    inspect_run_tagging,
    run_tagging_to_frame,
)


def _repo_root_default() -> Path:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) >= 5:
        return resolved.parents[4]
    return Path.cwd().resolve()


def _add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--archive-run-dir",
        required=True,
        help="Archived PILATES run directory (mapped to workspace:// for analysis).",
    )
    parser.add_argument(
        "--project-root",
        default=str(_repo_root_default()),
        help="PILATES repository root for inputs:// mount.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional explicit Consist DB path.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional scratch mount root.",
    )
    parser.add_argument(
        "--hashing-strategy",
        default="fast",
        choices=["fast", "full"],
        help="Tracker hashing strategy.",
    )
    parser.add_argument(
        "--access-mode",
        default="analysis",
        help="Consist tracker access mode.",
    )


def _build_tracker(args: argparse.Namespace) -> Any:
    return create_analysis_tracker(
        archive_run_dir=args.archive_run_dir,
        project_root=args.project_root,
        db_path=args.db_path,
        output_root=args.output_root,
        hashing_strategy=args.hashing_strategy,
        access_mode=args.access_mode,
    )


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_discover_runs(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    records = find_runs(
        tracker,
        model=args.model,
        status=args.status,
        year=args.year,
        iteration=args.iteration,
        name=args.name,
        limit=args.limit,
    )
    frame = runs_to_frame(records)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            frame.to_json(orient="records", indent=2), encoding="utf-8"
        )
        print(output_path)
        return 0

    if frame.empty:
        print("No runs found.")
    else:
        print(frame.to_string(index=False))
    return 0


def cmd_db_health(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    health = get_db_health(tracker, archive_run_dir=args.archive_run_dir)
    issues = get_db_health_issues(health, strict=args.strict)
    frame = db_health_to_frame(health)
    payload: Dict[str, Any] = {
        "healthy": bool(health.get("healthy", False)),
        "strict": bool(args.strict),
        "issues": issues,
        "summary": frame.to_dict(orient="records")[0] if not frame.empty else {},
    }
    _print_json(payload)
    if issues and args.fail_on_issues:
        return 2
    return 0


def cmd_run_tagging(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    tagging_report = inspect_run_tagging(tracker)
    issues = get_run_tagging_issues(tagging_report, strict=args.strict)
    frame = run_tagging_to_frame(tagging_report, strict=args.strict)

    if args.output_format == "table":
        if frame.empty:
            print("No tagging report rows.")
        else:
            print(frame.to_string(index=False))
        if args.include_warnings:
            warnings = list(tagging_report.get("warnings", []) or [])
            if warnings:
                print("")
                print("Warnings:")
                for warning in warnings:
                    print(f"- {warning}")
        if args.include_issues and issues:
            print("")
            print("Issues:")
            for issue in issues:
                print(f"- {issue}")
    else:
        payload: Dict[str, Any] = {
            "healthy": len(issues) == 0,
            "strict": bool(args.strict),
            "issues": issues,
            "summary": frame.to_dict(orient="records")[0] if not frame.empty else {},
            "report": tagging_report,
        }
        _print_json(payload)

    if issues and args.fail_on_issues:
        return 2
    return 0


def cmd_epoch_panel(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    panel = build_epoch_panel(
        tracker,
        scenario_id=args.scenario_id,
        models=args.model or None,
    )
    if args.converged_only:
        panel = panel.converged_epochs()
    frame = panel.to_frame()

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        print(output_csv)
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            frame.to_json(orient="records", indent=2), encoding="utf-8"
        )
        print(output_json)
    if not args.output_csv and not args.output_json:
        if frame.empty:
            print("No epochs found.")
        else:
            print(frame.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pilates-consist-analysis",
        description="Consist-enabled post-run analysis helpers for PILATES.",
    )
    parser.add_argument("--version", action="version", version="0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover-runs", help="List runs matching filters."
    )
    _add_tracker_args(discover)
    discover.add_argument("--model", default=None)
    discover.add_argument("--status", default=None)
    discover.add_argument("--year", type=int, default=None)
    discover.add_argument("--iteration", type=int, default=None)
    discover.add_argument("--name", default=None)
    discover.add_argument("--limit", type=int, default=100)
    discover.add_argument("--output-json", default=None)
    discover.set_defaults(func=cmd_discover_runs)

    health = subparsers.add_parser(
        "db-health",
        help="Run Consist DB inspect/doctor health checks for the archive DB.",
    )
    _add_tracker_args(health)
    health.add_argument("--strict", action="store_true", default=False)
    health.add_argument("--fail-on-issues", action="store_true", default=False)
    health.set_defaults(func=cmd_db_health)

    tagging = subparsers.add_parser(
        "run-tagging",
        help="Inspect run-tagging metadata quality and parent linkage consistency.",
    )
    _add_tracker_args(tagging)
    tagging.add_argument("--strict", action="store_true", default=False)
    tagging.add_argument("--fail-on-issues", action="store_true", default=False)
    tagging.add_argument(
        "--output-format",
        choices=["json", "table"],
        default="json",
        help="Output format for tagging report.",
    )
    tagging.add_argument(
        "--include-warnings",
        action="store_true",
        default=False,
        help="Print warning lines when output format is table.",
    )
    tagging.add_argument(
        "--include-issues",
        action="store_true",
        default=False,
        help="Print issue lines when output format is table.",
    )
    tagging.set_defaults(func=cmd_run_tagging)

    epoch_panel = subparsers.add_parser(
        "epoch-panel",
        help="Summarize runs grouped into simulation epochs.",
    )
    _add_tracker_args(epoch_panel)
    epoch_panel.add_argument(
        "--scenario-id",
        default=None,
        help="Optional scenario id filter for epoch grouping.",
    )
    epoch_panel.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional model filter; repeatable.",
    )
    epoch_panel.add_argument(
        "--converged-only",
        action="store_true",
        default=False,
        help="Show only converged epochs (max complete outer iteration per year/scenario).",
    )
    epoch_panel.add_argument("--output-csv", default=None)
    epoch_panel.add_argument("--output-json", default=None)
    epoch_panel.set_defaults(func=cmd_epoch_panel)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
