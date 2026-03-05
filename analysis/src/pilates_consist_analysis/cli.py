from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .activitysim_trips import (
    build_activitysim_trips_dataset,
    write_activitysim_trips_dataset,
)
from .catalog import find_runs, runs_to_frame
from .datasets import build_linkstats_dataset, write_linkstats_dataset
from .metrics_activitysim import (
    compute_activitysim_equilibrium_metrics,
    write_activitysim_equilibrium_metrics,
)
from .metrics_equilibrium import compute_equilibrium_metrics, write_equilibrium_metrics
from .packaging import export_bundle
from .runtime import create_analysis_tracker, resolve_archive_run_dir, resolve_db_path


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


def cmd_build_linkstats_dataset(args: argparse.Namespace) -> int:
    archive_run_dir = resolve_archive_run_dir(args.archive_run_dir)
    db_path = resolve_db_path(archive_run_dir, db_path=args.db_path)
    tracker = _build_tracker(args)
    dataset = build_linkstats_dataset(
        tracker,
        year=args.year,
        iteration=args.iteration,
        artifact_family=args.artifact_family,
        namespace=args.namespace,
        grouped_mode=args.grouped_mode,
        grouped_missing_files=args.grouped_missing_files,
        grouped_schema_id=args.grouped_schema_id,
        traveltime_weighting=args.traveltime_weighting,
        limit=args.limit,
    )
    query = {
        "year": args.year,
        "iteration": args.iteration,
        "artifact_family": args.artifact_family,
        "namespace": args.namespace,
        "grouped_mode": args.grouped_mode,
        "grouped_missing_files": args.grouped_missing_files,
        "grouped_schema_id": args.grouped_schema_id,
        "traveltime_weighting": args.traveltime_weighting,
        "limit": args.limit,
    }
    manifest = write_linkstats_dataset(
        dataset,
        output_dir=args.output_dir,
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
    )
    _print_json(manifest.to_dict())
    return 0


def cmd_build_asim_trips_dataset(args: argparse.Namespace) -> int:
    archive_run_dir = resolve_archive_run_dir(args.archive_run_dir)
    db_path = resolve_db_path(archive_run_dir, db_path=args.db_path)
    tracker = _build_tracker(args)
    dataset = build_activitysim_trips_dataset(
        tracker,
        year=args.year,
        iteration=args.iteration,
        artifact_family=args.artifact_family,
        namespace=args.namespace,
        grouped_mode=args.grouped_mode,
        grouped_missing_files=args.grouped_missing_files,
        grouped_schema_id=args.grouped_schema_id,
        latest_per_iteration=not args.no_latest_per_iteration,
        limit=args.limit,
    )
    query = {
        "year": args.year,
        "iteration": args.iteration,
        "artifact_family": args.artifact_family,
        "namespace": args.namespace,
        "grouped_mode": args.grouped_mode,
        "grouped_missing_files": args.grouped_missing_files,
        "grouped_schema_id": args.grouped_schema_id,
        "latest_per_iteration": not args.no_latest_per_iteration,
        "limit": args.limit,
    }
    manifest = write_activitysim_trips_dataset(
        dataset,
        output_dir=args.output_dir,
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
    )
    _print_json(manifest.to_dict())
    return 0


def cmd_equilibrium_metrics(args: argparse.Namespace) -> int:
    if args.deltas_csv:
        deltas_path = Path(args.deltas_csv).expanduser().resolve()
    else:
        deltas_path = Path(args.dataset_dir).expanduser().resolve() / "linkstats_deltas.csv"
    deltas_df = pd.read_csv(deltas_path)
    metrics = compute_equilibrium_metrics(deltas_df)
    output = write_equilibrium_metrics(metrics, args.output_json)
    _print_json(
        {
            "deltas_csv": str(deltas_path),
            "metrics_json": str(output),
            **metrics.to_dict(),
        }
    )
    return 0


def cmd_activitysim_equilibrium_metrics(args: argparse.Namespace) -> int:
    if args.equilibrium_pairs_csv:
        pairs_path = Path(args.equilibrium_pairs_csv).expanduser().resolve()
    else:
        pairs_path = (
            Path(args.dataset_dir).expanduser().resolve()
            / "asim_trips_equilibrium_pairs.csv"
        )
    mode_deltas_path = None
    if args.mode_deltas_csv:
        mode_deltas_path = Path(args.mode_deltas_csv).expanduser().resolve()
    elif args.dataset_dir:
        mode_deltas_path = (
            Path(args.dataset_dir).expanduser().resolve() / "asim_trips_mode_deltas.csv"
        )

    equilibrium_pairs_df = pd.read_csv(pairs_path)
    mode_deltas_df = (
        pd.read_csv(mode_deltas_path)
        if mode_deltas_path is not None and mode_deltas_path.exists()
        else None
    )
    metrics = compute_activitysim_equilibrium_metrics(
        equilibrium_pairs_df, mode_deltas_df=mode_deltas_df
    )
    output = write_activitysim_equilibrium_metrics(metrics, args.output_json)
    _print_json(
        {
            "equilibrium_pairs_csv": str(pairs_path),
            "mode_deltas_csv": str(mode_deltas_path) if mode_deltas_path else None,
            "metrics_json": str(output),
            **metrics.to_dict(),
        }
    )
    return 0


def cmd_export_bundle(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    payload = export_bundle(
        tracker,
        archive_run_dir=args.archive_run_dir,
        run_ids=args.run_id,
        out_path=args.out_path,
        include_data=args.include_data,
        include_snapshots=args.include_snapshots,
        include_children=not args.no_include_children,
        dry_run=args.dry_run,
    )
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pilates-consist-analysis",
        description="Consist-enabled post-run analysis helpers for PILATES.",
    )
    parser.add_argument("--version", action="version", version="0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover-runs", help="List runs matching filters.")
    _add_tracker_args(discover)
    discover.add_argument("--model", default=None)
    discover.add_argument("--status", default=None)
    discover.add_argument("--year", type=int, default=None)
    discover.add_argument("--iteration", type=int, default=None)
    discover.add_argument("--name", default=None)
    discover.add_argument("--limit", type=int, default=100)
    discover.add_argument("--output-json", default=None)
    discover.set_defaults(func=cmd_discover_runs)

    dataset = subparsers.add_parser(
        "build-linkstats-dataset",
        help="Build artifacts/summary/deltas CSVs for BEAM linkstats.",
    )
    _add_tracker_args(dataset)
    dataset.add_argument("--year", type=int, default=None)
    dataset.add_argument("--iteration", type=int, default=None)
    dataset.add_argument(
        "--artifact-family",
        default="linkstats_unmodified_phys_sim_iter_parquet",
    )
    dataset.add_argument("--namespace", default="beam")
    dataset.add_argument("--grouped-mode", default="hybrid", choices=["hybrid", "hot_only", "cold_only"])
    dataset.add_argument("--grouped-missing-files", default="warn", choices=["warn", "error", "ignore"])
    dataset.add_argument("--grouped-schema-id", default=None)
    dataset.add_argument(
        "--traveltime-weighting",
        default="unweighted",
        choices=["unweighted", "volume_weighted"],
    )
    dataset.add_argument("--limit", type=int, default=10000)
    dataset.add_argument("--output-dir", required=True)
    dataset.set_defaults(func=cmd_build_linkstats_dataset)

    asim_dataset = subparsers.add_parser(
        "build-asim-trips-dataset",
        help="Build ActivitySim trips mode/purpose/hour summaries + iteration deltas.",
    )
    _add_tracker_args(asim_dataset)
    asim_dataset.add_argument("--year", type=int, default=None)
    asim_dataset.add_argument("--iteration", type=int, default=None)
    asim_dataset.add_argument("--artifact-family", default="trips")
    asim_dataset.add_argument("--namespace", default="activitysim")
    asim_dataset.add_argument(
        "--grouped-mode", default="hybrid", choices=["hybrid", "hot_only", "cold_only"]
    )
    asim_dataset.add_argument(
        "--grouped-missing-files", default="warn", choices=["warn", "error", "ignore"]
    )
    asim_dataset.add_argument("--grouped-schema-id", default=None)
    asim_dataset.add_argument("--no-latest-per-iteration", action="store_true", default=False)
    asim_dataset.add_argument("--limit", type=int, default=10000)
    asim_dataset.add_argument("--output-dir", required=True)
    asim_dataset.set_defaults(func=cmd_build_asim_trips_dataset)

    eq = subparsers.add_parser(
        "equilibrium-metrics",
        help="Compute first-pass equilibrium diagnostics from linkstats deltas.",
    )
    eq.add_argument("--dataset-dir", default=None, help="Directory containing linkstats_deltas.csv.")
    eq.add_argument("--deltas-csv", default=None, help="Explicit path to linkstats_deltas.csv.")
    eq.add_argument("--output-json", required=True)
    eq.set_defaults(func=cmd_equilibrium_metrics)

    asim_eq = subparsers.add_parser(
        "activitysim-equilibrium-metrics",
        help="Compute equilibrium metrics from ActivitySim trips iteration-pair outputs.",
    )
    asim_eq.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing asim_trips_equilibrium_pairs.csv.",
    )
    asim_eq.add_argument(
        "--equilibrium-pairs-csv",
        default=None,
        help="Explicit path to asim_trips_equilibrium_pairs.csv.",
    )
    asim_eq.add_argument(
        "--mode-deltas-csv",
        default=None,
        help="Optional explicit path to asim_trips_mode_deltas.csv.",
    )
    asim_eq.add_argument("--output-json", required=True)
    asim_eq.set_defaults(func=cmd_activitysim_equilibrium_metrics)

    export = subparsers.add_parser(
        "export-bundle",
        help="Export portable Consist bundle for selected run ids.",
    )
    _add_tracker_args(export)
    export.add_argument("--run-id", action="append", required=True, help="Run id to include; repeatable.")
    export.add_argument("--out-path", required=True)
    export.add_argument("--include-data", action="store_true", default=False)
    export.add_argument("--include-snapshots", action="store_true", default=False)
    export.add_argument("--no-include-children", action="store_true", default=False)
    export.add_argument("--dry-run", action="store_true", default=False)
    export.set_defaults(func=cmd_export_bundle)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "equilibrium-metrics" and not args.dataset_dir and not args.deltas_csv:
        parser.error("equilibrium-metrics requires --dataset-dir or --deltas-csv.")
    if args.command == "activitysim-equilibrium-metrics":
        if not args.dataset_dir and not args.equilibrium_pairs_csv:
            parser.error(
                "activitysim-equilibrium-metrics requires --dataset-dir or --equilibrium-pairs-csv."
            )
    return int(args.func(args))
