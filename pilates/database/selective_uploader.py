import argparse
import json
import logging
import os
import re
from typing import Optional, Tuple

import pandas as pd
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from pilates.*
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.generic.records import (
    PilatesRunInfo,
    FileRecord,
    ModelRunInfo,
    H5TableRecord,
    H5FileRecord,
)
from pilates.database.schema_generator import (
    _normalize_table_name,
)  # Import the normalization function

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- CONSTANTS FOR SPECIAL LOGIC ---
TABLES_WITH_DATA_YEAR = {
    "householdv",
    "atlas_vehicles_input",
    "new_vehicle_annual_medians",
    "new_vehicle_representative_vehicle",
    "new_vehicles",
    "vehicles",
    "atlas_vehicles2_output",
}


def build_smart_select(
    conn,
    data_path: str,
    table_name: str,
    record: dict,
    run_info: dict,
    year: int,
    iteration: int,
) -> Tuple[str, list]:
    """
    Constructs a SQL SELECT clause that handles renames, casting, and metadata injection.
    Returns: (sql_query, list_of_sort_keys)
    """

    # 1. Peek at the file schema (Zero cost, just reads header)
    # Use read_parquet or read_csv_auto based on extension
    reader_func = "read_parquet" if data_path.endswith(".parquet") else "read_csv_auto"

    try:
        # DESCRIBE returns column_name, column_type, etc.
        file_schema = conn.execute(
            f"DESCRIBE SELECT * FROM {reader_func}('{data_path}')"
        ).fetchall()
        raw_columns = [row[0] for row in file_schema]  # List of column names
    except Exception as e:
        raise ValueError(f"Could not read file header for {data_path}: {e}")

    # 2. Build the Column Projection List
    projection_parts = []

    for col in raw_columns:
        col_lower = col.lower()

        # --- LOGIC A: Rename 'year' to 'data_year' ---
        if col_lower == "year" and table_name in TABLES_WITH_DATA_YEAR:
            # SQL equivalent of df.rename(columns={'year': 'data_year'})
            projection_parts.append(f'"{col}" AS data_year')

        # --- LOGIC B: Cast 'sector_id' to string ---
        elif col_lower == "sector_id" and table_name == "atlas_jobs_csv":
            # SQL equivalent of df['sector_id'].astype(str)
            projection_parts.append(f'CAST("{col}" AS VARCHAR) AS sector_id')

        # --- Standard Column (Sanitize name) ---
        else:
            # If name needs sanitizing (e.g. has dots), handle it here
            sanitized = sanitize_name(col)
            if sanitized != col:
                projection_parts.append(f'"{col}" AS {sanitized}')
            else:
                projection_parts.append(f'"{col}"')

    # 3. Inject Metadata Columns
    # We use string literals for IDs to ensure they are treated as such
    projection_parts.append(f"'{run_info['run_id']}' AS run_id")
    projection_parts.append(f"'{record['unique_id']}' AS file_record_id")
    projection_parts.append(f"{year} AS year")  # The simulation year
    projection_parts.append(f"{iteration} AS iteration")
    projection_parts.append("0 AS sub_iteration")

    # 4. Construct Query
    select_clause = f"""
        SELECT {', '.join(projection_parts)}
        FROM {reader_func}('{data_path}')
    """

    # 5. Determine Sort Keys (Heuristic)
    # We always want to sort by our metadata keys first
    sort_keys = ["run_id", "year", "iteration"]

    # Check sanitized schema for spatial keys
    # FIX: .strip('"') ensures we catch keys even if they are quoted strings like "zone_id"
    final_cols = [p.split(" AS ")[-1].strip().strip('"') for p in projection_parts]

    if "zone_id" in final_cols:
        sort_keys.append("zone_id")
    elif "origin" in final_cols:
        sort_keys.append("origin")

    return select_clause, sort_keys


def get_approved_tables(schema_dir: str) -> set:
    """
    Scans schema directory for valid table names.
    Captures MULTIPLE tables per file (safe for bundled SQL files).
    """
    approved_tables = set()
    # Only scan specific directories to avoid recursion issues
    dirs_to_scan = [schema_dir, os.path.join(schema_dir, "generated")]

    # Compile regex once for speed
    # Captures: CREATE TABLE [IF NOT EXISTS] table_name
    table_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.IGNORECASE
    )
    year_pattern = re.compile(r"(\d{4})")

    for current_dir in dirs_to_scan:
        if not os.path.isdir(current_dir):
            continue

        for filename in os.listdir(current_dir):
            if filename.endswith(".sql"):
                file_path = os.path.join(current_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Extract year from filename if present (e.g. 'households_2010.sql')
                    year_match = year_pattern.search(filename)
                    year = int(year_match.group(1)) if year_match else None

                    # Find ALL tables in this file (Fixes the single-table bug)
                    found_tables = table_pattern.findall(content)

                    for table_name in found_tables:
                        norm_name = _normalize_table_name(table_name, year).lower()
                        approved_tables.add(norm_name)

                except Exception as e:
                    logging.warning(f"Skipping unreadable schema file {filename}: {e}")
                    continue

    return approved_tables


def sanitize_name(name: str) -> str:
    """Sanitizes a name to be a valid SQL identifier."""
    name = name.strip().strip("/")
    name = re.sub(r"[\s./-]", "_", name)
    if re.match(r"^[0-9]", name):
        name = "_" + name
    return name.lower()


def resolve_file_path(run_dir: str, record: dict, run_info: dict) -> Optional[str]:
    """Smartly resolves file path given the messy reality of run directories."""
    path_from_record = record["file_path"]

    # Handle H5 Tables (they point to a container record)
    if record.get("h5_file_unique_id"):
        parent_id = record["h5_file_unique_id"]
        path_from_record = run_info["file_records"][parent_id]["file_path"]

    # 1. Check Absolute
    if os.path.isabs(path_from_record) and os.path.exists(path_from_record):
        return path_from_record

    # 2. Check relative to run_info.json
    candidate = os.path.join(run_dir, path_from_record)
    if os.path.exists(candidate):
        return candidate

    # 3. Check relative to parent (common in some directory structures)
    candidate_parent = os.path.join(os.path.dirname(run_dir), path_from_record)
    if os.path.exists(candidate_parent):
        return candidate_parent

    return None


def extract_iteration(short_name: str) -> Optional[int]:
    """Extracts iteration number from short_name (e.g., 'households_3')."""
    # Look for _iter3, _3, or similar patterns at the end
    match = re.search(r"[_\-](\d+)$", short_name)
    if match:
        return int(match.group(1))
    return None


def run_uploader(
    run_info_path: str,
    database_path: str,
    tables: list = None,
    no_upload_parquet: bool = False,
) -> bool:
    """
    The core logic of the uploader, refactored into a callable function.
    """
    if not os.path.exists(run_info_path):
        logging.error(f"Input file not found: {run_info_path}")
        return False

    with DuckDBManager(database_path) as db_manager:

        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        approved_schema_dir = os.path.join(script_dir, "schema")
        run_dir = os.path.dirname(run_info_path)

        # --- Logic ---
        approved_tables = get_approved_tables(approved_schema_dir)
        if not approved_tables:
            logging.warning("No approved schemas found. Nothing to upload.")
            print("No approved schemas found. Nothing to upload.")
            return False

        # If user specifies tables, filter the approved list
        target_tables = set(tables) if tables else approved_tables
        upload_list = approved_tables.intersection(target_tables)

        if not upload_list:
            logging.warning("No tables matched the criteria for upload.")
            print("No tables matched the criteria for upload.")
            return False

        logging.info(
            f"Attempting to upload data for tables: {sorted(list(upload_list))}"
        )

        # Load run_info.json
        try:
            with open(run_info_path, "r") as f:
                run_info = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to read or parse {run_info_path}: {e}")
            print(f"Failed to read or parse {run_info_path}: {e}")
            return False

        # Convert run_info dict to PilatesRunInfo object
        file_records = {}
        for uid, rec in run_info.get("file_records", {}).items():
            # Coalesce null iteration fields to 0 to satisfy NOT NULL constraint in DB
            if rec.get("iteration") is None:
                rec["iteration"] = 0
            if rec.get("sub_iteration") is None:
                rec["sub_iteration"] = 0
            if "h5_file_unique_id" in rec:
                file_records[uid] = H5TableRecord(**rec)
            elif "table_record_ids" in rec:
                file_records[uid] = H5FileRecord(**rec)
            else:
                file_records[uid] = FileRecord(**rec)
        model_runs = {
            uid: ModelRunInfo(**run)
            for uid, run in run_info.get("model_runs", {}).items()
        }

        run_info_obj = PilatesRunInfo(
            run_id=run_info.get("run_id"),
            created_at=run_info.get("created_at"),
            start_year=run_info.get("start_year"),
            end_year=run_info.get("end_year"),
            models_used=run_info.get("models_used", []),
            settings_hash=run_info.get("settings_hash"),
            code_version=run_info.get("code_version"),
            hostname=run_info.get("hostname"),
            file_records=file_records,
            repo_records=run_info.get("repo_records", {}),
            model_runs=model_runs,
            config_snapshot=run_info.get("config_snapshot"),
            openlineage_event_metadata=run_info.get("openlineage_event_metadata", []),
        )

        # Upload the entire run_info structure to ensure all records exist
        db_manager.upload_run_data(run_info_obj)
        logging.info(
            f"Successfully uploaded metadata for run {run_info.get('run_id')}."
        )

        for record in run_info.get("file_records", {}).values():
            try:
                # Validation checks
                if not record.get("short_name"):
                    continue

                # Determine Year, Table Name, Iteration, etc.
                year = record.get("year")
                if year is None:
                    match = re.search(r"(\d{4})", record["short_name"])
                    year = int(match.group(1)) if match else 0
                table_name = _normalize_table_name(record["short_name"], year)
                if table_name not in upload_list:
                    continue
                iteration = extract_iteration(record["short_name"]) or 0
                data_path = resolve_file_path(run_dir, record, run_info)
                if not data_path:
                    logging.error(f"Missing file for {record['short_name']}")
                    continue

                # Ensure the path is absolute for database storage
                data_path = str(Path(data_path).resolve())

                # Infer data format and logical table name
                data_format = "unknown"
                if data_path.endswith(".parquet"):
                    data_format = "parquet"
                elif data_path.endswith((".csv", ".csv.gz")):
                    data_format = "csv"
                elif data_path.endswith(".h5"):
                    data_format = "h5"
                logical_table_name = table_name

                # Decide the final storage location definitively.
                final_storage_location = "database"
                if data_format == "parquet" and no_upload_parquet:
                    final_storage_location = "external"

                logging.info(
                    f"Processing {logical_table_name} from {os.path.basename(data_path)} -> storage: {final_storage_location}"
                )

                # First, write the authoritative record for this file to the database.
                # This ensures the state is correct *before* any data is moved.
                db_manager.update_file_record_storage_info(
                    unique_id=record["unique_id"],
                    storage_location=final_storage_location,
                    logical_table_name=logical_table_name,
                    data_format=data_format,
                    file_path=data_path,
                )

                # Second, if the strategy is to upload, then upload the data.
                if final_storage_location == "database":
                    if data_format == "parquet" or data_format == "csv":
                        # === NATIVE UPLOAD (Parquet/CSV) ===
                        conn = db_manager._get_connection()
                        select_sql, sort_keys = build_smart_select(
                            conn,
                            data_path,
                            table_name,
                            record,
                            run_info,
                            year,
                            iteration,
                        )

                        # Get the list of columns from the select statement to use in the insert
                        desc_rel = conn.execute(f"DESCRIBE ({select_sql})")
                        insert_cols = [row[0] for row in desc_rel.fetchall()]
                        insert_cols_str = ", ".join([f'"{c}"' for c in insert_cols])

                        conn.execute(
                            f"INSERT INTO uploaded_{table_name} ({insert_cols_str}) SELECT * FROM ({select_sql}) ORDER BY {', '.join(sort_keys)}"
                        )
                        logging.info(f"Native upload complete for {logical_table_name}")
                    else:
                        # === PANDAS UPLOAD (H5 / Complex) ===
                        if record.get("h5_file_unique_id"):
                            df = pd.read_hdf(data_path, key=record["table_name"])
                        else:
                            logging.warning(
                                f"Unsupported format for direct upload: {data_path}"
                            )
                            continue

                        df.columns = [sanitize_name(c) for c in df.columns]
                        df["run_id"] = run_info["run_id"]
                        df["file_record_id"] = record["unique_id"]
                        df["year"] = year
                        df["iteration"] = iteration
                        df["sub_iteration"] = 0
                        db_manager.store_generic_table(f"uploaded_{table_name}", df)
                        logging.info(f"Pandas upload complete for {logical_table_name}")

            except Exception as e:
                logging.error(
                    f"Failed to process {logical_table_name}: {e}", exc_info=True
                )
                print(f"Failed to process {logical_table_name}: {e}")
                return False
    return True


def main():
    """Main function to drive the selective data upload process."""
    parser = argparse.ArgumentParser(
        description="Selectively upload table data from a PILATES run to the database."
    )
    parser.add_argument("run_info_path", help="Path to the run_info.json file.")
    parser.add_argument("database_path", help="Path to the DuckDB database file.")
    parser.add_argument(
        "--no-upload-parquet",
        action="store_true",
        help="If set, parquet files will not be uploaded to the database but linked externally.",
    )
    parser.add_argument(
        "--table",
        action="append",
        help="Specify a table to upload. Can be used multiple times. If not provided, all approved tables found in the run will be uploaded.",
    )
    args = parser.parse_args()

    run_uploader(
        run_info_path=args.run_info_path,
        database_path=args.database_path,
        tables=args.table,
        no_upload_parquet=args.no_upload_parquet,
    )


if __name__ == "__main__":
    main()
