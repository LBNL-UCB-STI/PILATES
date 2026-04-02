from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import duckdb
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pilates.atlas.preprocessor import _resolve_atlas_h5_table_key
from pilates.utils.consist_analysis import create_analysis_tracker
from pilates.utils.provenance_report import write_provenance_report
from pilates.workflows.artifact_keys import ATLAS_VEHICLES2_OUTPUT, USIM_DATASTORE_H5
from pilates.workflows.tracker_outputs import load_tracker_run_outputs

logger = logging.getLogger(__name__)

EXPORT_VERSION = "0.1"
EXPORT_TYPE = "post_atlas_population_extract"
README_TEXT = """# Post-ATLAS Population Extract

This package contains a source-aligned extract of canonical post-ATLAS
population state from a PILATES run.

Contents:
- `years/<year>/households.parquet`
- `years/<year>/persons.parquet`
- `years/<year>/vehicles.parquet`
- `years/<year>/table_manifest.json`
- `export_manifest.json`

The extract is intentionally source-aligned:
- no target-schema renames
- no cross-table semantic reshaping
- only minimal normalization for stable parquet output

Lineage comes from Consist step/run discovery and artifact resolution.
"""


def _add_common_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Archive run directory used to resolve workspace:// artifacts.",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Consist DuckDB path for the run.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        type=int,
        help="Forecast years to export.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the export package.",
    )
    parser.add_argument(
        "--scenario-run-id",
        default=None,
        help="Optional explicit scenario run id. If omitted, atlas_postprocess step runs are discovered by year.",
    )
    parser.add_argument(
        "--vehicles-source",
        choices=("auto", "vehicles2", "vehicles_raw"),
        default="auto",
        help="Preferred ATLAS vehicle source.",
    )
    parser.add_argument(
        "--lineage-mode",
        choices=("minimal", "full"),
        default="full",
        help="Whether to emit only JSON manifests or also a markdown provenance report.",
    )
    parser.add_argument(
        "--hash",
        choices=("none", "sha256"),
        default="sha256",
        help="Hash mode for exported files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical post-ATLAS population tables from an existing PILATES run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Compatibility alias for the HDF-backed extractor.",
    )
    _add_common_extract_args(extract)

    extract_hdf = subparsers.add_parser(
        "extract-hdf",
        help="Extract source-aligned post-ATLAS population tables from HDF + ATLAS vehicle outputs.",
    )
    _add_common_extract_args(extract_hdf)

    extract_sql = subparsers.add_parser(
        "extract-sql",
        help="Extract canonical post-ATLAS population tables from ATLAS CSVs using DuckDB.",
    )
    _add_common_extract_args(extract_sql)

    translate = subparsers.add_parser(
        "translate",
        help="Reserved for future target-schema translation from an intermediate extract.",
    )
    translate.add_argument("--source-dir", required=True)
    translate.add_argument("--output-dir", required=True)
    translate.add_argument("--schema-spec", required=True)

    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _file_digest(path: Path, mode: str) -> Optional[str]:
    if mode == "none":
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(value: Any, tracker: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    for attr in ("path", "file_path", "uri", "container_uri"):
        candidate = getattr(value, attr, None)
        if not candidate:
            continue
        text = str(candidate)
        if "://" in text:
            try:
                return str(tracker.resolve_uri(text))
            except Exception:
                continue
        return text
    if isinstance(value, Mapping):
        for attr in ("path", "file_path", "uri", "container_uri"):
            candidate = value.get(attr)
            if not candidate:
                continue
            text = str(candidate)
            if "://" in text:
                try:
                    return str(tracker.resolve_uri(text))
                except Exception:
                    continue
            return text
    return None


def _normalize_export_frame(
    frame: pd.DataFrame,
    *,
    expected_index_name: Optional[str] = None,
) -> pd.DataFrame:
    normalized = frame.copy()
    index_name = str(normalized.index.name) if normalized.index.name else None
    if expected_index_name and expected_index_name not in normalized.columns:
        if index_name == expected_index_name:
            normalized = normalized.reset_index()
    elif index_name and index_name not in normalized.columns:
        normalized = normalized.reset_index()

    for column in normalized.select_dtypes(include=["category"]).columns:
        normalized[column] = normalized[column].astype("string")
    return normalized


def _choose_latest_run(runs: Sequence[Any]) -> Any:
    def _sort_key(run: Any) -> tuple[int, str]:
        status = str(getattr(run, "status", "") or "").lower()
        created_at = getattr(run, "created_at", None)
        return (
            1 if status == "completed" else 0,
            created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
        )

    return sorted(runs, key=_sort_key, reverse=True)[0]


def _match_scenario_step_run_id(
    scenario_run: Any,
    *,
    year: int,
    step_name: str,
    phase: str,
    compat_model: str = "atlas",
) -> Optional[str]:
    meta = getattr(scenario_run, "meta", None) or {}
    steps = meta.get("steps")
    if not isinstance(steps, list):
        return None
    matches: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_name_value = str(step.get("name", "") or "")
        step_model = str(step.get("model", "") or "")
        step_phase = str(step.get("phase", "") or "")
        step_year = step.get("year")
        if step_name_value == step_name:
            pass
        elif step_model != compat_model or step_phase != phase:
            continue
        if step_year is None or int(step_year) != year:
            continue
        if step.get("id"):
            matches.append(step)
    if not matches:
        return None
    return str(matches[-1]["id"])


def _step_run_for_year(
    tracker: Any,
    *,
    year: int,
    scenario_run: Optional[Any],
    step_model: str,
    phase: str,
) -> Any:
    if scenario_run is not None:
        step_run_id = _match_scenario_step_run_id(
            scenario_run,
            year=year,
            step_name=step_model,
            phase=phase,
        )
        if step_run_id is not None:
            step_run = tracker.get_run(step_run_id)
            if step_run is not None:
                return step_run

    matches = tracker.find_runs(
        model=step_model,
        phase=phase,
        year=year,
        status="completed",
        limit=50,
    )
    if not matches:
        # Compatibility fallback for runs that may have recorded the model as
        # plain ``atlas`` with a postprocess phase facet.
        matches = tracker.find_runs(
            model="atlas",
            phase=phase,
            year=year,
            status="completed",
            limit=50,
        )
    if not matches:
        raise ValueError(f"No {step_model} step run found for year {year}.")
    if len(matches) > 1:
        logger.warning(
            "Multiple %s step runs matched year %s; using the latest completed run.",
            step_model,
            year,
        )
    return _choose_latest_run(matches)


def _scenario_root_for_run(tracker: Any, run: Any) -> Optional[Any]:
    current = run
    visited: set[str] = set()
    while current is not None:
        current_id = getattr(current, "id", None)
        if not current_id or current_id in visited:
            break
        visited.add(str(current_id))
        parent_id = getattr(current, "parent_run_id", None)
        if not parent_id:
            meta = getattr(current, "meta", None) or {}
            if isinstance(meta.get("steps"), list):
                return current
            return None
        current = tracker.get_run(str(parent_id))
    return None


def _discover_start_year(run: Any) -> Optional[int]:
    meta = getattr(run, "meta", None) or {}
    for key in ("facet", "run_facet"):
        payload = meta.get(key)
        if isinstance(payload, Mapping) and payload.get("start_year") is not None:
            return int(payload["start_year"])
    if meta.get("start_year") is not None:
        return int(meta["start_year"])
    return None


def _resolve_vehicle_source(
    tracker: Any,
    *,
    step_run: Any,
    year: int,
    vehicles_source: str,
) -> tuple[str, str]:
    outputs = load_tracker_run_outputs(getattr(step_run, "id", None), tracker=tracker)
    step_artifacts = tracker.get_artifacts_for_run(getattr(step_run, "id"))
    output_map = outputs or {}
    input_artifacts = getattr(step_artifacts, "inputs", None) or {}

    choices: list[tuple[str, Any]] = []
    if vehicles_source in {"auto", "vehicles2"}:
        choices.append((ATLAS_VEHICLES2_OUTPUT, output_map.get(ATLAS_VEHICLES2_OUTPUT)))
    raw_key = f"vehicles_{year}"
    if vehicles_source in {"auto", "vehicles_raw"}:
        choices.append((raw_key, input_artifacts.get(raw_key)))

    for artifact_key, artifact in choices:
        path = _artifact_path(artifact, tracker)
        if path and Path(path).exists():
            return artifact_key, path

    raise FileNotFoundError(
        f"Could not resolve vehicles artifact for year {year} from atlas_postprocess run {getattr(step_run, 'id', 'unknown')}."
    )


def _table_output_record(
    *,
    table_name: str,
    output_path: Path,
    frame: pd.DataFrame,
    hash_mode: str,
) -> dict[str, Any]:
    return {
        "path": output_path.name,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "sha256": _file_digest(output_path, hash_mode),
    }


def _extract_year(
    tracker: Any,
    *,
    step_run: Any,
    year: int,
    output_dir: Path,
    vehicles_source: str,
    hash_mode: str,
) -> dict[str, Any]:
    step_run_id = str(getattr(step_run, "id"))
    scenario_run = _scenario_root_for_run(tracker, step_run)
    scenario_run_id = getattr(scenario_run, "id", None) if scenario_run else None
    start_year = _discover_start_year(scenario_run) if scenario_run else None
    is_start_year = start_year is not None and int(start_year) == int(year)

    outputs = load_tracker_run_outputs(step_run_id, tracker=tracker)
    usim_artifact = outputs.get(USIM_DATASTORE_H5)
    usim_path_text = _artifact_path(usim_artifact, tracker)
    if not usim_path_text:
        raise FileNotFoundError(
            f"atlas_postprocess run {step_run_id} did not expose {USIM_DATASTORE_H5}."
        )
    usim_path = Path(usim_path_text)
    if not usim_path.exists():
        raise FileNotFoundError(f"Resolved UrbanSim datastore did not exist: {usim_path}")

    vehicle_artifact_key, vehicle_path_text = _resolve_vehicle_source(
        tracker,
        step_run=step_run,
        year=year,
        vehicles_source=vehicles_source,
    )
    vehicle_path = Path(vehicle_path_text)

    with pd.HDFStore(str(usim_path), mode="r") as store:
        households_table_path = _resolve_atlas_h5_table_key(
            store,
            year=year,
            table="households",
            is_start_year=is_start_year,
        )
        persons_table_path = _resolve_atlas_h5_table_key(
            store,
            year=year,
            table="persons",
            is_start_year=is_start_year,
        )
        households = _normalize_export_frame(
            store[households_table_path],
            expected_index_name="household_id",
        )
        persons = _normalize_export_frame(
            store[persons_table_path],
            expected_index_name="person_id",
        )

    vehicles = _normalize_export_frame(pd.read_csv(vehicle_path))

    year_dir = output_dir / "years" / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    households_path = year_dir / "households.parquet"
    persons_path = year_dir / "persons.parquet"
    vehicles_out_path = year_dir / "vehicles.parquet"

    households.to_parquet(households_path, index=False)
    persons.to_parquet(persons_path, index=False)
    vehicles.to_parquet(vehicles_out_path, index=False)

    manifest = {
        "year": int(year),
        "step": {
            "name": "atlas_postprocess",
            "run_id": step_run_id,
            "model": getattr(step_run, "model_name", None) or "atlas",
            "status": getattr(step_run, "status", None),
            "scenario_run_id": scenario_run_id,
        },
        "sources": {
            "usim_datastore_h5": {
                "path": str(usim_path),
                "artifact_key": USIM_DATASTORE_H5,
                "households_table_path": str(households_table_path),
                "persons_table_path": str(persons_table_path),
            },
            "vehicles": {
                "path": str(vehicle_path),
                "artifact_key": vehicle_artifact_key,
                "format": "csv",
            },
        },
        "outputs": {
            "households": _table_output_record(
                table_name="households",
                output_path=households_path,
                frame=households,
                hash_mode=hash_mode,
            ),
            "persons": _table_output_record(
                table_name="persons",
                output_path=persons_path,
                frame=persons,
                hash_mode=hash_mode,
            ),
            "vehicles": _table_output_record(
                table_name="vehicles",
                output_path=vehicles_out_path,
                frame=vehicles,
                hash_mode=hash_mode,
            ),
        },
    }
    _write_json(year_dir / "table_manifest.json", manifest)
    return manifest


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")


def _build_export_manifest(
    *,
    run_dir: Path,
    db_path: Path,
    years: Sequence[int],
    year_manifests: Sequence[Mapping[str, Any]],
    source_mode: str,
    skipped_years: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    scenario_ids = sorted(
        {
            str(manifest["step"]["scenario_run_id"])
            for manifest in year_manifests
            if manifest.get("step", {}).get("scenario_run_id")
        }
    )
    return {
        "export_version": EXPORT_VERSION,
        "export_type": EXPORT_TYPE,
        "source_mode": source_mode,
        "source": {
            "run_dir": str(run_dir),
            "db_path": str(db_path),
            "scenario_run_id": scenario_ids[0] if len(scenario_ids) == 1 else None,
            "scenario_run_ids": scenario_ids,
        },
        "requested_years": [int(year) for year in years],
        "years": [int(manifest["year"]) for manifest in year_manifests],
        "tables": ["households", "persons", "vehicles"],
        "year_manifests": {
            str(manifest["year"]): f"years/{manifest['year']}/table_manifest.json"
            for manifest in year_manifests
        },
        "skipped_years": list(skipped_years),
    }


def _atlas_input_year_dir(run_dir: Path, year: int) -> Path:
    return run_dir / "atlas" / "atlas_input" / f"year{year}"


def _require_existing_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file was not found: {path}")
    return path


def _resolve_householdv_source(tracker: Any, *, step_run: Any, year: int) -> tuple[str, str]:
    artifact_key = f"householdv_{year}"
    outputs = load_tracker_run_outputs(getattr(step_run, "id", None), tracker=tracker)
    artifact = outputs.get(artifact_key)
    path = _artifact_path(artifact, tracker)
    if path and Path(path).exists():
        return artifact_key, path
    raise FileNotFoundError(
        f"Could not resolve household vehicle update artifact for year {year} from atlas_run step {getattr(step_run, 'id', 'unknown')}."
    )


def _quoted_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _relation_columns(relation: Any) -> list[str]:
    columns = getattr(relation, "columns", None)
    if columns is not None:
        return [str(column) for column in columns]
    return [str(column) for column in relation.limit(0).df().columns]


def _create_households_base_stage(
    conn: duckdb.DuckDBPyConnection,
    *,
    base_view: str,
    stage_view: str,
) -> None:
    columns = _relation_columns(conn.table(base_view))
    select_parts = [_quoted_ident(column) for column in columns if column not in {"cars", "hh_cars"}]
    if "cars" in columns:
        select_parts.append("try_cast(cars as BIGINT) as cars")
    else:
        select_parts.append("NULL::BIGINT as cars")
    if "hh_cars" in columns:
        select_parts.append("cast(hh_cars as VARCHAR) as hh_cars")
    else:
        select_parts.append("NULL::VARCHAR as hh_cars")
    conn.execute(
        f"""
        create or replace temp view {stage_view} as
        select
          {", ".join(select_parts)}
        from {base_view}
        """
    )


def _resolve_sql_year_sources(
    tracker: Any,
    *,
    scenario_run: Optional[Any],
    preprocess_step_run: Any,
    run_step_run: Any,
    step_run: Any,
    year: int,
    vehicles_source: str,
) -> dict[str, dict[str, str]]:
    del scenario_run
    preprocess_outputs = load_tracker_run_outputs(
        getattr(preprocess_step_run, "id", None),
        tracker=tracker,
    )
    households_path = _artifact_path(preprocess_outputs.get("atlas_households_csv"), tracker)
    persons_path = _artifact_path(preprocess_outputs.get("atlas_persons_csv"), tracker)
    households_csv = _require_existing_path(
        Path(households_path) if households_path else Path(""),
        label=f"atlas households year {year}",
    )
    persons_csv = _require_existing_path(
        Path(persons_path) if persons_path else Path(""),
        label=f"atlas persons year {year}",
    )
    householdv_key, householdv_path = _resolve_householdv_source(
        tracker,
        step_run=run_step_run,
        year=year,
    )
    vehicles_key, vehicles_path = _resolve_vehicle_source(
        tracker,
        step_run=step_run,
        year=year,
        vehicles_source=vehicles_source,
    )
    return {
        "households_base": {
            "path": str(households_csv),
            "artifact_key": "atlas_households_csv",
            "format": "csv",
        },
        "persons": {
            "path": str(persons_csv),
            "artifact_key": "atlas_persons_csv",
            "format": "csv",
        },
        "household_vehicle_update": {
            "path": str(householdv_path),
            "artifact_key": householdv_key,
            "format": "csv",
        },
        "vehicles": {
            "path": str(vehicles_path),
            "artifact_key": vehicles_key,
            "format": "csv",
        },
    }


def _extract_year_sql(
    tracker: Any,
    *,
    step_run: Any,
    run_dir: Path,
    year: int,
    output_dir: Path,
    vehicles_source: str,
    hash_mode: str,
) -> dict[str, Any]:
    step_run_id = str(getattr(step_run, "id"))
    scenario_run = _scenario_root_for_run(tracker, step_run)
    scenario_run_id = getattr(scenario_run, "id", None) if scenario_run else None
    preprocess_step_run = _step_run_for_year(
        tracker,
        year=year,
        scenario_run=scenario_run,
        step_model="atlas_preprocess",
        phase="preprocess",
    )
    run_step_run = _step_run_for_year(
        tracker,
        year=year,
        scenario_run=scenario_run,
        step_model="atlas_run",
        phase="run",
    )
    sources = _resolve_sql_year_sources(
        tracker,
        scenario_run=scenario_run,
        preprocess_step_run=preprocess_step_run,
        run_step_run=run_step_run,
        step_run=step_run,
        year=year,
        vehicles_source=vehicles_source,
    )

    conn = duckdb.connect()
    try:
        conn.read_csv(sources["households_base"]["path"]).create_view("households_base")
        conn.read_csv(sources["persons"]["path"]).create_view("persons_post_atlas")
        conn.read_csv(sources["household_vehicle_update"]["path"]).create_view(
            "household_vehicle_update_raw"
        )
        conn.read_csv(sources["vehicles"]["path"]).create_view("vehicles_post_atlas")

        _create_households_base_stage(
            conn,
            base_view="households_base",
            stage_view="households_base_stage",
        )
        conn.execute(
            """
            create or replace temp view household_vehicle_update as
            select
              cast(household_id as BIGINT) as household_id,
              cast(nvehicles as BIGINT) as cars,
              case
                when cast(nvehicles as BIGINT) <= 0 then 'none'
                when cast(nvehicles as BIGINT) = 1 then 'one'
                else 'two or more'
              end as hh_cars
            from household_vehicle_update_raw
            """
        )
        conn.execute(
            """
            create or replace temp view households_post_atlas as
            select
              h.* exclude (cars, hh_cars),
              v.cars,
              v.hh_cars
            from households_base_stage h
            inner join household_vehicle_update v
              on try_cast(h.household_id as BIGINT) = v.household_id
            """
        )

        year_dir = output_dir / "years" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        households_path = year_dir / "households.parquet"
        persons_path = year_dir / "persons.parquet"
        vehicles_out_path = year_dir / "vehicles.parquet"

        conn.sql("select * from households_post_atlas").write_parquet(
            str(households_path),
            compression="zstd",
        )
        conn.sql("select * from persons_post_atlas").write_parquet(
            str(persons_path),
            compression="zstd",
        )
        conn.sql("select * from vehicles_post_atlas").write_parquet(
            str(vehicles_out_path),
            compression="zstd",
        )

        households_columns = _relation_columns(conn.table("households_post_atlas"))
        persons_columns = _relation_columns(conn.table("persons_post_atlas"))
        vehicles_columns = _relation_columns(conn.table("vehicles_post_atlas"))

        households_rows = int(
            conn.sql("select count(*) from households_post_atlas").fetchone()[0]
        )
        persons_rows = int(conn.sql("select count(*) from persons_post_atlas").fetchone()[0])
        vehicles_rows = int(conn.sql("select count(*) from vehicles_post_atlas").fetchone()[0])
    finally:
        conn.close()

    manifest = {
        "year": int(year),
        "source_mode": "atlas_csv_sql",
        "step": {
            "name": "atlas_postprocess",
            "run_id": step_run_id,
            "model": getattr(step_run, "model_name", None) or "atlas_postprocess",
            "status": getattr(step_run, "status", None),
            "scenario_run_id": scenario_run_id,
        },
        "sources": sources,
        "outputs": {
            "households": {
                "path": households_path.name,
                "rows": households_rows,
                "columns": households_columns,
                "sha256": _file_digest(households_path, hash_mode),
            },
            "persons": {
                "path": persons_path.name,
                "rows": persons_rows,
                "columns": persons_columns,
                "sha256": _file_digest(persons_path, hash_mode),
            },
            "vehicles": {
                "path": vehicles_out_path.name,
                "rows": vehicles_rows,
                "columns": vehicles_columns,
                "sha256": _file_digest(vehicles_out_path, hash_mode),
            },
        },
    }
    _write_json(year_dir / "table_manifest.json", manifest)
    return manifest


def _shared_extract_command(
    args: argparse.Namespace,
    *,
    extractor: Any,
    source_mode: str,
) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lineage").mkdir(parents=True, exist_ok=True)

    tracker = create_analysis_tracker(
        db_path=db_path,
        archive_run_dir=run_dir,
        project_root=Path(__file__).resolve().parents[2],
    )

    scenario_run = tracker.get_run(args.scenario_run_id) if args.scenario_run_id else None
    year_manifests: list[dict[str, Any]] = []
    skipped_years: list[dict[str, Any]] = []
    for year in args.years:
        year_int = int(year)
        try:
            step_run = _step_run_for_year(
                tracker,
                year=year_int,
                scenario_run=scenario_run,
                step_model="atlas_postprocess",
                phase="postprocess",
            )
            logger.info(
                "Exporting year %s from atlas_postprocess run %s",
                year_int,
                getattr(step_run, "id", "unknown"),
            )
            year_manifests.append(
                extractor(
                    tracker,
                    step_run=step_run,
                    year=year_int,
                    run_dir=run_dir,
                    output_dir=output_dir,
                    vehicles_source=args.vehicles_source,
                    hash_mode=args.hash,
                )
            )
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            logger.warning(
                "Skipping year %s due to unavailable source artifacts or metadata: %s",
                year_int,
                exc,
            )
            skipped_years.append(
                {
                    "year": year_int,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    _write_readme(output_dir)
    export_manifest = _build_export_manifest(
        run_dir=run_dir,
        db_path=db_path,
        years=args.years,
        year_manifests=year_manifests,
        source_mode=source_mode,
        skipped_years=skipped_years,
    )
    _write_json(output_dir / "export_manifest.json", export_manifest)

    scenario_ids = export_manifest["source"]["scenario_run_ids"]
    if args.lineage_mode == "full" and len(scenario_ids) == 1 and year_manifests:
        write_provenance_report(
            tracker=tracker,
            run_id=scenario_ids[0],
            output_path=output_dir / "lineage" / "provenance_report.md",
        )
    elif args.lineage_mode == "full":
        logger.warning(
            "Skipping provenance report because the export spans multiple scenario roots or none could be resolved."
        )

    if skipped_years:
        logger.warning(
            "Skipped %s year(s): %s",
            len(skipped_years),
            ", ".join(str(item["year"]) for item in skipped_years),
        )
    logger.info("Wrote export package to %s", output_dir)
    return 0


def _extract_year_hdf_compat(
    tracker: Any,
    *,
    step_run: Any,
    year: int,
    run_dir: Path,
    output_dir: Path,
    vehicles_source: str,
    hash_mode: str,
) -> dict[str, Any]:
    del run_dir
    manifest = _extract_year(
        tracker,
        step_run=step_run,
        year=year,
        output_dir=output_dir,
        vehicles_source=vehicles_source,
        hash_mode=hash_mode,
    )
    manifest["source_mode"] = "hdf"
    return manifest


def extract_hdf_command(args: argparse.Namespace) -> int:
    return _shared_extract_command(
        args,
        extractor=_extract_year_hdf_compat,
        source_mode="hdf",
    )


def extract_sql_command(args: argparse.Namespace) -> int:
    return _shared_extract_command(
        args,
        extractor=_extract_year_sql,
        source_mode="atlas_csv_sql",
    )


def translate_command(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "Target-schema translation is not implemented yet. Use the extract subcommand first."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "extract":
        return extract_hdf_command(args)
    if args.command == "extract-hdf":
        return extract_hdf_command(args)
    if args.command == "extract-sql":
        return extract_sql_command(args)
    if args.command == "translate":
        return translate_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
