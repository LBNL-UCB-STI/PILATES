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
from .epochs import build_epoch_panel
from .handoff import (
    export_activitysim_inputs,
    export_scenario_bundle,
    export_sql_query,
    ingest_artifacts,
    parse_artifact_arg,
    parse_columns_arg,
    parse_rename_args,
    TableTransformSpec,
)
from .metrics_activitysim import (
    compute_activitysim_equilibrium_metrics,
    write_activitysim_equilibrium_metrics,
)
from .metrics_equilibrium import compute_equilibrium_metrics, write_equilibrium_metrics
from .packaging import export_bundle
from .scenario_compare import (
    compare_scenarios,
    runset_from_run_ids,
    write_scenario_comparison,
)
from .runset import RunSet, runset_from_query, runset_run_ids
from .runtime import (
    create_analysis_tracker,
    db_health_to_frame,
    get_db_health,
    get_db_health_issues,
    get_run_tagging_issues,
    inspect_run_tagging,
    resolve_archive_run_dir,
    resolve_db_path,
    run_tagging_to_frame,
)
from .skim_analysis import build_skim_convergence_dataset, write_skim_convergence_dataset


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


def _parse_metadata_items(items: Optional[list[str]]) -> Optional[dict[str, Any]]:
    if not items:
        return None
    output: dict[str, Any] = {}
    for raw in items:
        value = str(raw).strip()
        if not value:
            continue
        if "=" not in value:
            raise ValueError(f"Invalid metadata filter '{value}'. Use key=value format.")
        key, raw_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid metadata filter '{value}': missing key.")
        parsed_value: Any
        parsed_value_text = raw_value.strip()
        if not parsed_value_text:
            parsed_value = ""
        else:
            try:
                parsed_value = json.loads(parsed_value_text)
            except json.JSONDecodeError:
                parsed_value = parsed_value_text
        output[key] = parsed_value
    return output or None


def _parse_tags(items: Optional[list[str]]) -> Optional[list[str]]:
    if not items:
        return None
    tags: list[str] = []
    for raw in items:
        for piece in str(raw).split(","):
            value = piece.strip()
            if value:
                tags.append(value)
    return tags or None


def _build_compare_runset(
    tracker: Any,
    args: argparse.Namespace,
    *,
    side: str,
) -> RunSet:
    name = getattr(args, f"{side}_name")
    run_ids = getattr(args, f"{side}_run_id")
    if run_ids:
        return runset_from_run_ids(tracker, run_ids, name=name)

    metadata = _parse_metadata_items(getattr(args, f"{side}_metadata"))
    tags = _parse_tags(getattr(args, f"{side}_tag"))
    return runset_from_query(
        tracker=tracker,
        runset_name=name,
        model=getattr(args, f"{side}_model"),
        year=getattr(args, f"{side}_year"),
        iteration=getattr(args, f"{side}_iteration"),
        status=getattr(args, f"{side}_status"),
        parent_id=getattr(args, f"{side}_parent_id"),
        run_name=getattr(args, f"{side}_run_name"),
        tags=tags,
        metadata=metadata,
        limit=getattr(args, f"{side}_limit"),
    )


def _has_side_selectors(args: argparse.Namespace, side: str) -> bool:
    selector_fields = (
        "run_id",
        "model",
        "year",
        "iteration",
        "status",
        "parent_id",
        "run_name",
        "tag",
        "metadata",
    )
    for field in selector_fields:
        value = getattr(args, f"{side}_{field}")
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        return True
    return False


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


def cmd_build_skim_dataset(args: argparse.Namespace) -> int:
    archive_run_dir = resolve_archive_run_dir(args.archive_run_dir)
    db_path = resolve_db_path(archive_run_dir, db_path=args.db_path)
    tracker = _build_tracker(args)
    concept_keys = args.concept_key or None
    run_ids = args.run_id or None
    dataset = build_skim_convergence_dataset(
        tracker,
        concept_keys=concept_keys,
        run_ids=run_ids,
        year=args.year,
        iteration=args.iteration,
        key_contains=args.key_contains,
        limit=args.limit,
    )
    query = {
        "concept_keys": concept_keys,
        "run_ids": run_ids,
        "year": args.year,
        "iteration": args.iteration,
        "key_contains": args.key_contains,
        "limit": args.limit,
    }
    manifest = write_skim_convergence_dataset(
        dataset,
        output_dir=args.output_dir,
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
    )
    _print_json(manifest.to_dict())
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


def cmd_ingest_artifacts(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    artifact_specs = [
        parse_artifact_arg(
            raw,
            direction=args.direction,
            driver=args.driver,
            artifact_family=args.artifact_family,
        )
        for raw in (args.artifact or [])
    ]
    payload = ingest_artifacts(
        tracker,
        artifact_specs,
        run_id=args.run_id,
        model=args.model,
        scenario_id=args.scenario_id,
        seed=args.seed,
        year=args.year,
        iteration=args.iteration,
        parent_run_id=args.parent_run_id,
        tags=_parse_tags(args.tag),
        run_config=_parse_metadata_items(args.run_config),
        ingest_data=not args.no_ingest,
        profile_schema=not args.no_profile_schema,
    )
    _print_json(payload)
    return 0


def cmd_export_scenario_db(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    payload = export_scenario_bundle(
        tracker,
        archive_run_dir=args.archive_run_dir,
        out_path=args.out_path,
        scenario_id=args.scenario_id,
        seed=args.seed,
        model=args.model,
        status=args.status,
        year=args.year,
        iteration=args.iteration,
        tags=_parse_tags(args.tag),
        metadata=_parse_metadata_items(args.metadata),
        limit=args.limit,
        use_converged=not args.no_converged,
        converged_group_by=args.converged_group_by,
        latest_group_by=args.latest_group_by,
        include_data=not args.no_include_data,
        include_snapshots=args.include_snapshots,
        include_children=not args.no_include_children,
        dry_run=args.dry_run,
    )
    _print_json(payload)
    return 0


def cmd_export_sql(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    sql_inline = args.sql.strip() if args.sql else ""
    sql_file = args.sql_file.strip() if args.sql_file else ""
    if bool(sql_inline) == bool(sql_file):
        raise ValueError("Provide exactly one of --sql or --sql-file.")
    sql_text = (
        sql_inline
        if sql_inline
        else Path(sql_file).expanduser().resolve().read_text(encoding="utf-8")
    )
    payload = export_sql_query(
        tracker,
        sql=sql_text,
        output_path=args.output_path,
        output_format=args.output_format,
        limit=args.limit,
    )
    _print_json(payload)
    return 0


def cmd_export_asim_inputs(args: argparse.Namespace) -> int:
    tracker = _build_tracker(args)
    trips_spec = TableTransformSpec(
        columns=parse_columns_arg(args.trips_columns),
        rename=parse_rename_args(args.trips_rename),
        where_sql=args.trips_where,
    )
    persons_spec = TableTransformSpec(
        columns=parse_columns_arg(args.persons_columns),
        rename=parse_rename_args(args.persons_rename),
        where_sql=args.persons_where,
    )
    payload = export_activitysim_inputs(
        tracker,
        output_dir=args.output_dir,
        scenario_id=args.scenario_id,
        year=args.year,
        iteration=args.iteration,
        use_converged=not args.no_converged,
        trips=trips_spec,
        persons=persons_spec,
        include_trips=not args.skip_trips,
        include_persons=not args.skip_persons,
        output_format=args.output_format,
    )
    _print_json(payload)
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
        output_json.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
        print(output_json)
    if not args.output_csv and not args.output_json:
        if frame.empty:
            print("No epochs found.")
        else:
            print(frame.to_string(index=False))
    return 0


def cmd_compare_scenarios(args: argparse.Namespace) -> int:
    archive_run_dir = resolve_archive_run_dir(args.archive_run_dir)
    db_path = resolve_db_path(archive_run_dir, db_path=args.db_path)
    tracker = _build_tracker(args)

    left = _build_compare_runset(tracker, args, side="left")
    right = _build_compare_runset(tracker, args, side="right")
    if len(left) == 0:
        raise ValueError("Left run set resolved to zero runs. Refine filters or specify --left-run-id.")
    if len(right) == 0:
        raise ValueError("Right run set resolved to zero runs. Refine filters or specify --right-run-id.")
    try:
        comparison = compare_scenarios(
            tracker,
            left,
            right,
            datasets=args.dataset,
            year=args.year,
            iteration=args.iteration,
            config_namespace=args.config_namespace,
            config_prefix=args.config_prefix,
            config_include_equal=args.config_include_equal,
            align_on=args.align_on,
            latest_group_by=args.latest_group_by,
            use_converged=args.use_converged,
            converged_group_by=args.converged_group_by,
        )
    except ValueError as exc:
        message = str(exc)
        if "Cannot align on" in message:
            raise ValueError(
                f"{message} Try --latest-group-by {args.align_on} or a more specific grouping."
            ) from exc
        raise

    left_filters = None
    if not args.left_run_id:
        left_filters = {
            "model": args.left_model,
            "year": args.left_year,
            "iteration": args.left_iteration,
            "status": args.left_status,
            "parent_id": args.left_parent_id,
            "name": args.left_run_name,
            "tags": _parse_tags(args.left_tag),
            "metadata": _parse_metadata_items(args.left_metadata),
            "limit": args.left_limit,
        }
    right_filters = None
    if not args.right_run_id:
        right_filters = {
            "model": args.right_model,
            "year": args.right_year,
            "iteration": args.right_iteration,
            "status": args.right_status,
            "parent_id": args.right_parent_id,
            "name": args.right_run_name,
            "tags": _parse_tags(args.right_tag),
            "metadata": _parse_metadata_items(args.right_metadata),
            "limit": args.right_limit,
        }

    query = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_run_ids": runset_run_ids(left),
        "right_run_ids": runset_run_ids(right),
        "left_filters": left_filters,
        "right_filters": right_filters,
        "align_on": args.align_on,
        "latest_group_by": args.latest_group_by,
        "use_converged": args.use_converged,
        "converged_group_by": args.converged_group_by,
        "datasets": args.dataset,
        "year": args.year,
        "iteration": args.iteration,
        "config_namespace": args.config_namespace,
        "config_prefix": args.config_prefix,
        "config_include_equal": args.config_include_equal,
    }
    manifest = write_scenario_comparison(
        comparison,
        output_dir=args.output_dir,
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
    )
    _print_json(manifest.to_dict())
    return 0


def _add_compare_side_runset_args(parser: argparse.ArgumentParser, side: str) -> None:
    parser.add_argument(
        f"--{side}-run-id",
        action="append",
        default=None,
        help=f"Explicit run id for {side} side; repeatable. Overrides {side} filters when provided.",
    )
    parser.add_argument(f"--{side}-model", default=None, help=f"Run model filter for {side} side.")
    parser.add_argument(f"--{side}-year", type=int, default=None, help=f"Run year filter for {side} side.")
    parser.add_argument(
        f"--{side}-iteration",
        type=int,
        default=None,
        help=f"Run iteration filter for {side} side.",
    )
    parser.add_argument(f"--{side}-status", default=None, help=f"Run status filter for {side} side.")
    parser.add_argument(
        f"--{side}-parent-id",
        default=None,
        help=f"Run parent id filter for {side} side.",
    )
    parser.add_argument(
        f"--{side}-run-name",
        default=None,
        help=f"Run name/model-name alias filter for {side} side.",
    )
    parser.add_argument(
        f"--{side}-tag",
        action="append",
        default=None,
        help=f"Tag filter for {side} side; repeatable or comma-separated.",
    )
    parser.add_argument(
        f"--{side}-metadata",
        action="append",
        default=None,
        help=f"Metadata predicate for {side} side in key=value format; repeatable.",
    )
    parser.add_argument(
        f"--{side}-limit",
        type=int,
        default=100,
        help=f"Run query limit for {side} side in filter mode.",
    )


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

    skim = subparsers.add_parser(
        "build-skim-dataset",
        help="Build OpenMatrix skim convergence dataset across runs/iterations.",
    )
    _add_tracker_args(skim)
    skim.add_argument("--concept-key", action="append", default=None, help="Explicit concept key; repeatable.")
    skim.add_argument("--run-id", action="append", default=None, help="Optional run id filter; repeatable.")
    skim.add_argument("--year", type=int, default=None)
    skim.add_argument("--iteration", type=int, default=None)
    skim.add_argument("--key-contains", default="skim")
    skim.add_argument("--limit", type=int, default=10000)
    skim.add_argument("--output-dir", required=True)
    skim.set_defaults(func=cmd_build_skim_dataset)

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

    ingest = subparsers.add_parser(
        "ingest-artifacts",
        help="Log and optionally ingest arbitrary files into the Consist DB.",
    )
    _add_tracker_args(ingest)
    ingest.set_defaults(access_mode="standard")
    ingest.add_argument("--run-id", default=None, help="Optional explicit run id.")
    ingest.add_argument("--model", default="analysis_ingest")
    ingest.add_argument("--scenario-id", default=None)
    ingest.add_argument("--seed", type=int, default=None)
    ingest.add_argument("--year", type=int, default=None)
    ingest.add_argument("--iteration", type=int, default=None)
    ingest.add_argument("--parent-run-id", default=None)
    ingest.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact spec PATH or key=PATH; repeatable.",
    )
    ingest.add_argument(
        "--artifact-family",
        default=None,
        help="Optional artifact_family metadata applied to every artifact in this command.",
    )
    ingest.add_argument(
        "--direction",
        choices=["input", "output"],
        default="output",
    )
    ingest.add_argument("--driver", default=None)
    ingest.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Optional run tags; repeatable or comma-separated.",
    )
    ingest.add_argument(
        "--run-config",
        action="append",
        default=None,
        help="Optional run config key=value; repeatable.",
    )
    ingest.add_argument("--no-ingest", action="store_true", default=False)
    ingest.add_argument("--no-profile-schema", action="store_true", default=False)
    ingest.set_defaults(func=cmd_ingest_artifacts)

    export_scenario = subparsers.add_parser(
        "export-scenario-db",
        help="Select scenario runs and export a standalone Consist DB shard.",
    )
    _add_tracker_args(export_scenario)
    export_scenario.add_argument("--out-path", required=True)
    export_scenario.add_argument("--scenario-id", default=None)
    export_scenario.add_argument("--seed", type=int, default=None)
    export_scenario.add_argument("--model", default=None)
    export_scenario.add_argument("--status", default="completed")
    export_scenario.add_argument("--year", type=int, default=None)
    export_scenario.add_argument("--iteration", type=int, default=None)
    export_scenario.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Run tags filter; repeatable or comma-separated.",
    )
    export_scenario.add_argument(
        "--metadata",
        action="append",
        default=None,
        help="Run metadata filter key=value; repeatable.",
    )
    export_scenario.add_argument("--limit", type=int, default=10000)
    export_scenario.add_argument(
        "--converged-group-by",
        action="append",
        default=None,
        help="Field/facet grouping keys for converged selection; repeatable.",
    )
    export_scenario.add_argument(
        "--latest-group-by",
        action="append",
        default=None,
        help="Field/facet grouping keys for latest selection; repeatable.",
    )
    export_scenario.add_argument("--no-converged", action="store_true", default=False)
    export_scenario.add_argument("--no-include-data", action="store_true", default=False)
    export_scenario.add_argument("--include-snapshots", action="store_true", default=False)
    export_scenario.add_argument("--no-include-children", action="store_true", default=False)
    export_scenario.add_argument("--dry-run", action="store_true", default=False)
    export_scenario.set_defaults(func=cmd_export_scenario_db)

    export_sql = subparsers.add_parser(
        "export-sql",
        help="Run SQL against the Consist DB and export result rows.",
    )
    _add_tracker_args(export_sql)
    export_sql.add_argument("--sql", default=None, help="Inline SQL query text.")
    export_sql.add_argument("--sql-file", default=None, help="Path to SQL file.")
    export_sql.add_argument("--output-path", required=True)
    export_sql.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    export_sql.add_argument("--limit", type=int, default=None)
    export_sql.set_defaults(func=cmd_export_sql)

    export_asim = subparsers.add_parser(
        "export-asim-inputs",
        help="Export ActivitySim trips/persons tables for one epoch with optional transforms.",
    )
    _add_tracker_args(export_asim)
    export_asim.add_argument("--output-dir", required=True)
    export_asim.add_argument("--scenario-id", default=None)
    export_asim.add_argument("--year", type=int, default=None)
    export_asim.add_argument("--iteration", type=int, default=None)
    export_asim.add_argument("--no-converged", action="store_true", default=False)
    export_asim.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    export_asim.add_argument("--skip-trips", action="store_true", default=False)
    export_asim.add_argument("--skip-persons", action="store_true", default=False)
    export_asim.add_argument(
        "--trips-columns",
        default=None,
        help="Comma-separated subset of trips columns to export.",
    )
    export_asim.add_argument(
        "--persons-columns",
        default=None,
        help="Comma-separated subset of persons columns to export.",
    )
    export_asim.add_argument(
        "--trips-rename",
        action="append",
        default=None,
        help="Trips column rename old:new; repeatable.",
    )
    export_asim.add_argument(
        "--persons-rename",
        action="append",
        default=None,
        help="Persons column rename old:new; repeatable.",
    )
    export_asim.add_argument("--trips-where", default=None, help="Optional SQL WHERE clause for trips.")
    export_asim.add_argument("--persons-where", default=None, help="Optional SQL WHERE clause for persons.")
    export_asim.set_defaults(func=cmd_export_asim_inputs)

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

    compare = subparsers.add_parser(
        "compare-scenarios",
        help="Compare two run sets across datasets and config diffs.",
    )
    _add_tracker_args(compare)
    compare.add_argument("--left-name", default="left")
    compare.add_argument("--right-name", default="right")
    _add_compare_side_runset_args(compare, "left")
    _add_compare_side_runset_args(compare, "right")
    compare.add_argument(
        "--align-on",
        default="year",
        help="Run field/facet key used for native RunSet alignment (for example year, iteration, model).",
    )
    compare.add_argument(
        "--latest-group-by",
        action="append",
        default=None,
        help="Field/facet key for RunSet.latest(...); repeatable. Defaults to align key.",
    )
    compare.add_argument(
        "--use-converged",
        action="store_true",
        default=False,
        help=(
            "Select RunSet.converged(...) before latest/alignment. "
            "Disabled by default for backwards compatibility."
        ),
    )
    compare.add_argument(
        "--converged-group-by",
        action="append",
        default=None,
        help=(
            "Field/facet key for RunSet.converged(...); repeatable. "
            "Defaults to year + scenario_id."
        ),
    )
    compare.add_argument(
        "--dataset",
        action="append",
        choices=["linkstats", "asim_trips", "skims"],
        default=None,
        help="Dataset to compare; repeatable. Defaults to all.",
    )
    compare.add_argument("--year", type=int, default=None)
    compare.add_argument("--iteration", type=int, default=None)
    compare.add_argument("--config-namespace", default=None)
    compare.add_argument("--config-prefix", default=None)
    compare.add_argument("--config-include-equal", action="store_true", default=False)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=cmd_compare_scenarios)

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
    if args.command == "compare-scenarios":
        if not _has_side_selectors(args, "left"):
            parser.error(
                "compare-scenarios requires left selectors (--left-run-id or --left-* filters)."
            )
        if not _has_side_selectors(args, "right"):
            parser.error(
                "compare-scenarios requires right selectors (--right-run-id or --right-* filters)."
            )
        if args.converged_group_by and not args.use_converged:
            parser.error("--converged-group-by requires --use-converged.")
    return int(args.func(args))
