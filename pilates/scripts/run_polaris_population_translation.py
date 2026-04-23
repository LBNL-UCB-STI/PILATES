from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

import duckdb

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


logger = logging.getLogger(__name__)

SQL_EXECUTION_ORDER = (
    "00_inputs.sql",
    "10_households.sql",
    "20_persons.sql",
    "30_vehicle_dimensions.sql",
    "40_vehicles.sql",
    "50_vehicle_class.sql",
    "60_vehicle_type.sql",
    "99_exports.sql",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first-pass POLARIS translation SQL against a post-ATLAS population extract."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Root directory of the post-ATLAS population export package.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Export year to translate.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("baseline", "ess_cons", "zev_mandate"),
        help="ATLAS scenario key used to choose adopt reference files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for POLARIS parquet outputs.",
    )
    parser.add_argument(
        "--translation-dir",
        default=None,
        help="Optional override for the SQL translation package directory.",
    )
    return parser.parse_args(argv)


def _translation_dir(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1] / "generated" / "polaris_population_translation"


def _source_year_dir(source_dir: Path, year: int) -> Path:
    return source_dir / "years" / str(year)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _atlas_lookup_paths(*, scenario: str, year: int) -> tuple[Path, Path]:
    atlas_input = _repo_root() / "pilates" / "atlas" / "atlas_input"
    new_vehicles = atlas_input / "adopt" / scenario / f"new_vehicles_biannual_values_{year}.csv"
    used_vehicles = atlas_input / "adopt" / scenario / f"used_vehicles_{year}.csv"
    return new_vehicles, used_vehicles


def _require_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} path was not found: {path}")
    return path.resolve()


def _quoted_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _parquet_columns(path: Path) -> list[str]:
    conn = duckdb.connect()
    try:
        relation = conn.read_parquet(str(path))
        return [str(column) for column in relation.columns]
    finally:
        conn.close()


def _resolve_person_mar_expr(persons_path: Path) -> str:
    columns = _parquet_columns(persons_path)
    for candidate in ("mar", "MAR"):
        if candidate in columns:
            return f"p.{_quoted_ident(candidate)}"
    logger.warning(
        "Persons parquet %s does not contain a marital-status column; defaulting marital_status to 0. Columns: %s",
        persons_path,
        columns,
    )
    return "NULL"


def _placeholder_map(
    *,
    source_dir: Path,
    year: int,
    scenario: str,
    output_dir: Path,
) -> dict[str, str]:
    year_dir = _source_year_dir(source_dir, year)
    new_vehicles, used_vehicles = _atlas_lookup_paths(scenario=scenario, year=year)
    households = _require_path(year_dir / "households.parquet", label=f"source households year {year}")
    persons = _require_path(year_dir / "persons.parquet", label=f"source persons year {year}")
    vehicles = _require_path(year_dir / "vehicles.parquet", label=f"source vehicles year {year}")
    new_ref = _require_path(new_vehicles, label=f"ATLAS new_vehicles reference year {year}")
    used_ref = _require_path(used_vehicles, label=f"ATLAS used_vehicles reference year {year}")
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "__HOUSEHOLDS_POST_ATLAS_PARQUET__": str(households),
        "__PERSONS_POST_ATLAS_PARQUET__": str(persons),
        "__VEHICLES_POST_ATLAS_PARQUET__": str(vehicles),
        "__ATLAS_NEW_VEHICLES_BIANNUAL_VALUES_CSV__": str(new_ref),
        "__ATLAS_USED_VEHICLES_CSV__": str(used_ref),
        "__OUT_DIR__": str(resolved_output_dir),
        "__EXPORT_YEAR__": str(year),
        "__PERSON_MAR_EXPR__": _resolve_person_mar_expr(persons),
    }


def _render_sql(text: str, replacements: Mapping[str, str]) -> str:
    rendered = text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value.replace("\\", "\\\\"))
    return rendered


def _copy_translation_package(*, translation_dir: Path, output_dir: Path) -> Path:
    target_dir = output_dir / "translation_sql"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(translation_dir, target_dir)
    return target_dir


def run_translation(
    *,
    source_dir: Path,
    year: int,
    scenario: str,
    output_dir: Path,
    translation_dir: Path,
) -> None:
    replacements = _placeholder_map(
        source_dir=source_dir,
        year=year,
        scenario=scenario,
        output_dir=output_dir,
    )
    copied_dir = _copy_translation_package(
        translation_dir=translation_dir,
        output_dir=output_dir.resolve(),
    )
    conn = duckdb.connect()
    try:
        for filename in SQL_EXECUTION_ORDER:
            sql_path = _require_path(translation_dir / filename, label=f"SQL step {filename}")
            logger.info("Running SQL step %s", sql_path.name)
            conn.execute(_render_sql(sql_path.read_text(encoding="utf-8"), replacements))
    finally:
        conn.close()
    logger.info("Copied translation SQL package to %s", copied_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s - %(levelname)s - %(message)s")
    run_translation(
        source_dir=Path(args.source_dir).resolve(),
        year=int(args.year),
        scenario=str(args.scenario),
        output_dir=Path(args.output_dir).resolve(),
        translation_dir=_translation_dir(args.translation_dir),
    )
    logger.info("Wrote POLARIS translation outputs to %s", Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
