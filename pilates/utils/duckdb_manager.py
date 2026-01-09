"""
DuckDB implementation of the PILATES database manager.

This module provides DuckDB-specific implementation for storing and retrieving
PILATES run data, optimized for analytical queries and local/cloud deployment.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
import pandas as pd
import numpy as np

import duckdb

from pilates.generic.records_legacy import (
    PilatesRunInfo,
    FileRecord,
    H5FileRecord,
    H5TableRecord,
)
from pilates.utils.database import (
    DatabaseManager,
    DatabaseUploadError,
    DatabaseQueryError,
)
from pilates.database.schema_generator import _normalize_table_name

if TYPE_CHECKING:
    from pilates.generic.records_legacy import ModelRun, OpenLineageEventMetadata

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _execute_sql_from_file(conn, sql_file_path: str):
    """Helper function to execute SQL commands from a file."""
    try:
        with open(sql_file_path, "r") as f:
            sql_commands = f.read()
            conn.execute(sql_commands)
        logger.info(f"Successfully executed SQL from {os.path.basename(sql_file_path)}")
    except Exception as e:
        logger.error(
            f"Failed to execute SQL from {os.path.basename(sql_file_path)}: {e}"
        )
        raise


def _infer_data_format(file_path: str) -> str:
    """Infers data format from file extension."""
    if not file_path:
        return "unknown"
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return "parquet"
    elif ext in [".csv", ".csv.gz"]:
        return "csv"
    elif ext == ".h5":
        return "h5"
    return "unknown"


def _get_logical_table_name(short_name: str, year: Optional[int]) -> str:
    """Wrapper for _normalize_table_name, ensuring it returns a string."""
    return _normalize_table_name(short_name, year) if short_name else "unknown"


class DuckDBManager(DatabaseManager):
    """
    DuckDB implementation of the database manager.

    Provides local analytical database capabilities with potential for
    cloud deployment via DuckDB's cloud features.
    """

    def __init__(self, database_path: str, **kwargs):
        """
        Initialize DuckDB database manager.

        Args:
            database_path: Path to DuckDB database file
            **kwargs: Additional DuckDB configuration options
        """
        if duckdb is None:
            raise ImportError(
                "DuckDB is required but not installed. Install with: pip install duckdb"
            )

        super().__init__(database_path, **kwargs)
        self.connection = None

    def __enter__(self):
        """Override for diagnostics to see when the context is entered."""
        # Call the parent's __enter__ if it does anything important in the future
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Override for diagnostics to see when the context is exited."""
        super().__exit__(exc_type, exc_val, exc_tb)  # This calls self.close()

    def _get_connection(self):
        """Get or create database connection."""
        if self.connection is None:
            # Create directory if it doesn't exist
            dir_name = os.path.dirname(self.database_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self.connection = duckdb.connect(self.database_path)
            logger.info(f"Connected to DuckDB database at {self.database_path}")
        return self.connection

    @staticmethod
    def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        try:
            res = conn.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = ?
                LIMIT 1
                """,
                [table_name],
            ).fetchone()
            return res is not None
        except Exception:
            # If information_schema is unavailable, fail closed.
            return False

    def _resolve_table_name(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        preferred: str,
        fallbacks: Tuple[str, ...] = (),
    ) -> str:
        """Return a safely-quoted table name, tolerating older schema variants.

        Some production DuckDBs were created with singular names (e.g. `run`)
        instead of newer pluralized names (e.g. `runs`). This helper detects the
        available table and returns a quoted identifier for SQL interpolation.
        """

        def quote_identifier(name: str) -> str:
            safe = name.replace('"', '""')
            return f'"{safe}"'

        for candidate in (preferred,) + fallbacks:
            if self._table_exists(conn, candidate):
                if candidate != preferred:
                    logger.warning(
                        f"Detected legacy DuckDB schema: using table '{candidate}' instead of '{preferred}'."
                    )
                return quote_identifier(candidate)

        logger.warning(
            f"Could not find DuckDB table '{preferred}' (fallbacks={fallbacks}); proceeding with '{preferred}'."
        )
        return quote_identifier(preferred)

    def execute_sql(self, sql_text: str, values_to_insert: Optional[list] = None):
        """
        Executes a SQL command with optional parameterization and commits the transaction.

        Args:
            sql_text: The SQL query string to execute.
            values_to_insert: Optional list of values to insert into the query,
                              used for parameterized queries to prevent SQL injection.

        Returns:
            The result of the SQL execution (e.g., a DuckDBPyRelation object for SELECT statements).
        """
        conn = self._get_connection()
        if values_to_insert:
            result = conn.execute(sql_text, values_to_insert)
        else:
            result = conn.execute(sql_text)
        conn.commit()
        return result

    def initialize_database(self, schema_dir: Optional[str] = None) -> bool:
        """
        Initialize DuckDB database with PILATES schema by executing SQL scripts.

        Args:
            schema_dir: Optional path to the schema directory. If not provided,
                        it's inferred from the location of this file.
        """
        try:
            conn = self._get_connection()

            # 1. Load Extensions first
            try:
                conn.execute("INSTALL spatial; LOAD spatial;")
            except Exception:
                logger.warning(
                    "Could not load DuckDB spatial extension. Geometry columns may fail."
                )

            if schema_dir is None:
                # Get schema and generated schema directories relative to this file
                utils_dir = os.path.dirname(os.path.abspath(__file__))
                pilates_dir = os.path.dirname(utils_dir)
                schema_dir = os.path.join(pilates_dir, "database", "schema")

            generated_schema_dir = os.path.join(schema_dir, "generated")

            # 2. Execute base schema files (excluding those handled by generation script)
            base_sql_files = sorted(
                [
                    f
                    for f in os.listdir(schema_dir)
                    if f.endswith(".sql") and f not in ["06_asim_outputs.sql"]
                ]
            )
            for sql_file in base_sql_files:
                full_path = os.path.join(schema_dir, sql_file)
                _execute_sql_from_file(conn, full_path)

            # 3. Execute generated schema files in order
            if os.path.isdir(generated_schema_dir):
                generated_sql_files = sorted(
                    [f for f in os.listdir(generated_schema_dir) if f.endswith(".sql")]
                )
                for sql_file in generated_sql_files:
                    full_path = os.path.join(generated_schema_dir, sql_file)
                    _execute_sql_from_file(conn, full_path)

            conn.commit()
            logger.info("DuckDB database initialized successfully from SQL files.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB database from SQL files: {e}")
            return False

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate data quality validation report.

        Returns:
            dict: Validation report with errors, warnings, and statistics
        """
        try:
            conn = self._get_connection()

            report = {
                "generated_at": datetime.now().isoformat(),
                "errors": [],
                "warnings": [],
                "statistics": {},
                "recommendations": [],
            }

            # Check 1: Orphaned file records (run_id doesn't exist in runs)
            orphaned = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM file_records fr
                LEFT JOIN runs r ON fr.run_id = r.run_id
                WHERE r.run_id IS NULL
            """
            ).fetchone()[0]

            if orphaned > 0:
                report["errors"].append(
                    f"{orphaned} file records reference non-existent runs"
                )

            # Check 2: Model runs without completion time
            incomplete = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM model_runs
                WHERE status = 'completed' AND completed_at IS NULL
            """
            ).fetchone()[0]

            if incomplete > 0:
                report["warnings"].append(
                    f"{incomplete} model runs marked complete but missing completion time"
                )

            # Check 3: Duplicate household IDs within a run
            dupes = conn.execute(
                """
                SELECT run_id, household_id, COUNT(*) as dup_count
                FROM activitysim_households
                GROUP BY run_id, household_id
                HAVING COUNT(*) > 1
            """
            ).fetchall()

            if dupes:
                report["errors"].append(
                    f"{len(dupes)} duplicate household_id values found"
                )

            # Check 4: Missing foreign key relationships
            missing_hh = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM activitysim_persons p
                LEFT JOIN activitysim_households h
                    ON p.household_id = h.household_id AND p.run_id = h.run_id
                WHERE h.household_id IS NULL
            """
            ).fetchone()[0]

            if missing_hh > 0:
                report["errors"].append(
                    f"{missing_hh} persons reference non-existent households"
                )

            # Statistics
            report["statistics"] = {
                "total_runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "total_file_records": conn.execute(
                    "SELECT COUNT(*) FROM file_records"
                ).fetchone()[0],
                "total_model_runs": conn.execute(
                    "SELECT COUNT(*) FROM model_runs"
                ).fetchone()[0],
                "failed_model_runs": conn.execute(
                    "SELECT COUNT(*) FROM model_runs WHERE status = 'failed'"
                ).fetchone()[0],
            }

            # Try to count data records (may not exist in all databases)
            try:
                report["statistics"]["total_households"] = conn.execute(
                    "SELECT COUNT(*) FROM activitysim_households"
                ).fetchone()[0]
                report["statistics"]["total_persons"] = conn.execute(
                    "SELECT COUNT(*) FROM activitysim_persons"
                ).fetchone()[0]
            except:
                pass

            # Recommendations
            if report["errors"]:
                report["recommendations"].append(
                    "Address data integrity errors before using database for analysis"
                )
            if report["statistics"].get("failed_model_runs", 0) > 0:
                report["recommendations"].append(
                    "Investigate failed model runs using recent_activity view"
                )

            return report

        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return {
                "generated_at": datetime.now().isoformat(),
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "statistics": {},
                "recommendations": [],
            }

    def check_dataset_exists_by_hash(self, unique_id: str) -> bool:
        """Check if a dataset exists by its unique_id (hash)."""
        try:
            conn = self._get_connection()
            result = conn.execute(
                "SELECT 1 FROM file_records WHERE unique_id = ? LIMIT 1",
                [unique_id],
            ).fetchone()

            return result is not None

        except Exception as e:
            logger.error(f"Failed to check dataset existence {unique_id}: {e}")
            return False

    # =========================================================================
    # NEW GRANULAR AND IDEMPOTENT UPLOAD METHODS
    # =========================================================================

    def upsert_config_snapshot(
        self, conn: duckdb.DuckDBPyConnection, config_snapshot: Dict[str, Any]
    ) -> Optional[str]:
        """
        Inserts a configuration snapshot if it doesn't already exist.
        Idempotent based on the snapshot_id.

        Args:
            conn: An active DuckDB connection.
            config_snapshot: A dictionary representing the config snapshot.

        Returns:
            The snapshot_id of the inserted or existing record.
        """
        if not config_snapshot:
            return None

        config_snapshot_id = config_snapshot.get("snapshot_id")
        if not config_snapshot_id:
            return None

        try:
            conn.execute(
                """
                INSERT INTO config_snapshots (
                    snapshot_id, created_timestamp, config_content_hash,
                    git_hashes, config_files, pilates_settings,
                    beam_config, asim_subdir, region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (snapshot_id) DO NOTHING
                """,
                [
                    config_snapshot_id,
                    config_snapshot.get("created_timestamp"),
                    config_snapshot.get("config_content_hash"),
                    json.dumps(config_snapshot.get("git_hashes", {})),
                    json.dumps(config_snapshot.get("config_files", {})),
                    json.dumps(config_snapshot.get("pilates_settings", {})),
                    config_snapshot.get("beam_config"),
                    config_snapshot.get("asim_subdir"),
                    config_snapshot.get("region"),
                ],
            )
            logger.debug(f"Upserted config snapshot {config_snapshot_id}")
            return config_snapshot_id
        except Exception as e:
            logger.error(f"Failed to upsert config snapshot {config_snapshot_id}: {e}")
            raise

    def upsert_pilates_run(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_info: PilatesRunInfo,
        config_snapshot_id: Optional[str],
    ):
        """
        Inserts the main run record in the 'runs' table if it does not exist.
        Idempotent based on the run_id.

        Args:
            conn: An active DuckDB connection.
            run_info: The main PilatesRunInfo object.
            config_snapshot_id: The foreign key to the config_snapshots table.
        """
        try:
            runs_table = self._resolve_table_name(conn, preferred="runs", fallbacks=("run",))
            conn.execute(
                f"""
                INSERT INTO {runs_table} (
                    run_id, created_at, start_year, end_year, models_used,
                    settings_hash, code_version, hostname, config_snapshot_id,
                    config_content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (run_id) DO NOTHING
                """,
                [
                    run_info.run_id,
                    run_info.created_at,
                    run_info.start_year,
                    run_info.end_year,
                    run_info.models_used,
                    run_info.settings_hash,
                    run_info.code_version,
                    run_info.hostname,
                    config_snapshot_id,
                    (
                        run_info.config_snapshot.get("config_content_hash")
                        if run_info.config_snapshot
                        else None
                    ),
                ],
            )
            logger.debug(f"Upserted pilates run {run_info.run_id}")
        except Exception as e:
            logger.error(f"Failed to upsert pilates run {run_info.run_id}: {e}")
            raise

    def upsert_file_record(
        self, conn: duckdb.DuckDBPyConnection, file_record: FileRecord, run_id: str
    ):
        """
        Inserts or updates a single file record, handling regular files,
        H5 containers, and H5 tables. Idempotent based on unique_id.

        Args:
            conn: An active DuckDB connection.
            file_record: The FileRecord, H5FileRecord, or H5TableRecord object.
            run_id: The ID of the parent Pilates run.
        """
        try:
            record_type = "file"
            if isinstance(file_record, H5TableRecord):
                record_type = "h5_table"
            elif isinstance(file_record, H5FileRecord):
                record_type = "h5_container"

            inferred_data_format = _infer_data_format(file_record.file_path)
            logical_table_name = (
                file_record.table_name
                if isinstance(file_record, H5TableRecord)
                else _get_logical_table_name(file_record.short_name, file_record.year)
            )
            if record_type == "h5_table":
                inferred_data_format = "h5"

            conn.execute(
                """
                INSERT INTO file_records (
                    unique_id, record_type, run_id, openlineage_id, file_path, created_at,
                    short_name, description, year, models, producing_run_id,
                    consuming_run_ids, source_file_paths, metadata, schema, exists,
                    storage_location, data_format, logical_table_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (unique_id) DO UPDATE SET
                    record_type = excluded.record_type,
                    run_id = excluded.run_id,
                    openlineage_id = excluded.openlineage_id,
                    file_path = excluded.file_path,
                    created_at = excluded.created_at,
                    short_name = excluded.short_name,
                    description = excluded.description,
                    year = excluded.year,
                    models = excluded.models,
                    producing_run_id = excluded.producing_run_id,
                    consuming_run_ids = excluded.consuming_run_ids,
                    source_file_paths = excluded.source_file_paths,
                    metadata = excluded.metadata,
                    schema = excluded.schema,
                    exists = excluded.exists,
                    storage_location = excluded.storage_location,
                    data_format = excluded.data_format,
                    logical_table_name = excluded.logical_table_name
                """,
                [
                    file_record.unique_id,
                    record_type,
                    run_id,
                    file_record.openlineage_id,
                    file_record.file_path,
                    file_record.created_at,
                    file_record.short_name,
                    file_record.description,
                    file_record.year,
                    file_record.models,
                    file_record.producing_run_id,
                    file_record.consuming_run_ids,
                    file_record.source_file_paths,
                    json.dumps(_convert_numpy_types(file_record.metadata)),
                    json.dumps(_convert_numpy_types(file_record.schema)),
                    file_record.exists,
                    "database",  # Default storage_location, can be updated later
                    inferred_data_format,
                    logical_table_name,
                ],
            )

            if isinstance(file_record, H5TableRecord):
                conn.execute(
                    """
                    INSERT INTO h5_table_records (
                        unique_id, h5_file_unique_id, table_name
                    ) VALUES (?, ?, ?)
                    ON CONFLICT (unique_id) DO NOTHING
                    """,
                    [
                        file_record.unique_id,
                        file_record.h5_file_unique_id,
                        file_record.table_name,
                    ],
                )
            logger.debug(f"Upserted file record {file_record.unique_id}")
        except Exception as e:
            logger.error(f"Failed to upsert file record {file_record.unique_id}: {e}")
            raise

    def upsert_model_run(
        self, conn: duckdb.DuckDBPyConnection, model_run: "ModelRun", run_id: str
    ):
        """
        Inserts or updates a single model run record.
        Idempotent based on unique_id.

        Args:
            conn: An active DuckDB connection.
            model_run: The ModelRun data record.
            run_id: The ID of the parent Pilates run.
        """
        try:
            conn.execute(
                """
                INSERT INTO model_runs (
                    unique_id, run_id, openlineage_id, model, year, iteration,
                    description, created_at, completed_at, status,
                    input_record_hashes, output_record_hashes, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (unique_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    openlineage_id = excluded.openlineage_id,
                    model = excluded.model,
                    year = excluded.year,
                    iteration = excluded.iteration,
                    description = excluded.description,
                    created_at = excluded.created_at,
                    completed_at = excluded.completed_at,
                    status = excluded.status,
                    input_record_hashes = excluded.input_record_hashes,
                    output_record_hashes = excluded.output_record_hashes,
                    metadata = excluded.metadata
                """,
                [
                    model_run.unique_id,
                    run_id,
                    model_run.openlineage_id,
                    model_run.model,
                    model_run.year,
                    model_run.iteration,
                    model_run.description,
                    model_run.created_at,
                    model_run.completed_at,
                    model_run.status,
                    model_run.input_record_hashes,
                    model_run.output_record_hashes,
                    json.dumps(_convert_numpy_types(model_run.metadata)),
                ],
            )
            logger.debug(f"Upserted model run {model_run.unique_id}")
        except Exception as e:
            logger.error(f"Failed to upsert model run {model_run.unique_id}: {e}")
            raise

    def insert_openlineage_event(
        self,
        conn: duckdb.DuckDBPyConnection,
        event: "OpenLineageEventMetadata",
        run_id: str,
    ):
        """
        Inserts a single OpenLineage event record. This is append-only.

        Args:
            conn: An active DuckDB connection.
            event: The OpenLineageEvent data record.
            run_id: The ID of the parent Pilates run.
        """
        try:
            # Events are append-only, so we just insert.
            # The table has a sequence for the primary key.
            conn.execute(
                """
                INSERT INTO openlineage_events (
                    run_id, model_run_id, event_time, event_type,
                    run_uuid, job_name
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    run_id,
                    event.model_run_id,
                    event.event_time,
                    event.event_type,
                    event.run_uuid,
                    event.job_name,
                ],
            )
            logger.debug(
                f"Inserted openlineage event for model run {event.model_run_id}"
            )
        except AttributeError as e:
            logger.error(
                f"Missing openlineage event metadata: {e}. Full event metadata: {event}"
            )
            # Do not re-raise as this is not always critical
        except Exception as e:
            logger.error(f"Failed to insert openlineage event: {e}")
            # Do not re-raise as this is not always critical

    # =========================================================================
    # BATCH AND DEPRECATED UPLOAD METHODS
    # =========================================================================

    def upload_file_record(self, file_record: FileRecord, run_id: str) -> bool:
        """
        DEPRECATED: Upload a single file record to the database.
        Use `upsert_file_record` in a transaction context instead.
        """
        logger.warning(
            "`upload_file_record` is deprecated. Use `upsert_file_record` within a transaction."
        )
        try:
            conn = self._get_connection()
            conn.begin()
            self.upsert_file_record(conn, file_record, run_id)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to upload file record {file_record.unique_id}: {e}")
            conn.rollback()
            return False

    def upload_config_snapshot(self, config_snapshot: Dict[str, Any]) -> bool:
        """
        DEPRECATED: Upload a single config snapshot to the database.
        Use `upsert_config_snapshot` in a transaction context instead.
        """
        logger.warning(
            "`upload_config_snapshot` is deprecated. Use `upsert_config_snapshot` within a transaction."
        )
        try:
            conn = self._get_connection()
            conn.begin()
            self.upsert_config_snapshot(conn, config_snapshot)
            conn.commit()
            return True
        except Exception as e:
            logger.error(
                f"Failed to upload config snapshot {config_snapshot.get('snapshot_id')}: {e}"
            )
            conn.rollback()
            return False

    def upload_run_data(self, run_info: PilatesRunInfo) -> bool:
        """
        Upload complete run data to DuckDB in a single transaction.
        This method is idempotent and can be called multiple times. Records that
        already exist will be skipped or updated.

        Args:
            run_info: Complete PILATES run information object.

        Returns:
            bool: True if upload is successful.
        """
        conn = self._get_connection()
        try:
            conn.begin()

            # 1. Upsert Config Snapshot
            config_snapshot_id = self.upsert_config_snapshot(
                conn, run_info.config_snapshot
            )

            # 2. Upsert Main Run Record
            self.upsert_pilates_run(conn, run_info, config_snapshot_id)

            # 3. Upsert all File Records (in two passes)
            h5_table_records = []
            other_file_records = []
            for rec in run_info.file_records.values():
                if isinstance(rec, H5TableRecord):
                    h5_table_records.append(rec)
                else:
                    other_file_records.append(rec)

            # Pass 1: Insert all non-H5-table records first to satisfy FKs
            for file_record in other_file_records:
                self.upsert_file_record(conn, file_record, run_info.run_id)

            # Pass 2: Insert H5 table records, whose parents now exist
            for file_record in h5_table_records:
                self.upsert_file_record(conn, file_record, run_info.run_id)

            # 4. Ensure referenced file records exist (for model run FKs)
            referenced_ids = set()
            for model_run in run_info.model_runs.values():
                referenced_ids.update(model_run.input_record_hashes)
                referenced_ids.update(model_run.output_record_hashes)

            # Find which of the referenced IDs are actually missing
            if referenced_ids:
                placeholders = [f"'{rid}'" for rid in referenced_ids]
                existing_ids_df = conn.execute(
                    f"SELECT unique_id FROM file_records WHERE unique_id IN ({','.join(placeholders)})"
                ).fetchdf()
                existing_ids = set(existing_ids_df["unique_id"].tolist())
                missing_ids = referenced_ids - existing_ids

                for missing_id in missing_ids:
                    logger.warning(
                        f"Referenced file_record {missing_id} not found. Creating placeholder."
                    )
                    conn.execute(
                        """
                        INSERT INTO file_records (
                            unique_id, record_type, run_id, openlineage_id, file_path, created_at,
                            short_name, description, year, models, producing_run_id,
                            consuming_run_ids, source_file_paths, metadata, schema, exists,
                            storage_location, data_format, logical_table_name
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (unique_id) DO NOTHING
                        """,
                        [
                            missing_id,
                            "placeholder",
                            run_info.run_id,
                            str(uuid.uuid4()),
                            "unknown",
                            datetime.now().isoformat(),
                            "missing_reference",
                            f"Placeholder for missing file_record {missing_id}",
                            None,
                            [],
                            None,
                            [],
                            [],
                            "{}",
                            "{}",
                            False,
                            "unknown",
                            "unknown",
                            "unknown",
                        ],
                    )

            # 5. Upsert Model Runs
            for model_run in run_info.model_runs.values():
                self.upsert_model_run(conn, model_run, run_info.run_id)

            # 6. Insert OpenLineage Events (append-only)
            for event in run_info.openlineage_event_metadata:
                self.insert_openlineage_event(conn, event, run_info.run_id)

            conn.commit()
            logger.info(f"Successfully uploaded batch run data for {run_info.run_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to upload batch run data for {run_info.run_id}: {e}",
                exc_info=True,
            )
            try:
                conn.rollback()
            except Exception as re:
                logger.error(f"Rollback failed: {re}")
            raise DatabaseUploadError(f"Batch upload failed: {e}")

    def upload_zarr_manifest_data(self, run_id: str, manifest_data: dict) -> bool:
        """
        Uploads detailed Zarr manifest data (snapshots) to the zarr_snapshots table.

        Args:
            run_id: The PILATES run ID associated with this manifest.
            manifest_data: The parsed content of the manifest.json file.

        Returns:
            bool: True if upload successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            conn.begin()

            for snapshot_id, snapshot in manifest_data.get("snapshots", {}).items():
                full_skims = snapshot.get("full_skims", {})
                partial_skims = snapshot.get("partial_skims", {})

                conn.execute(
                    """
                    INSERT INTO zarr_snapshots (
                        snapshot_id, run_id, year, iteration, snapshot_type, model,
                        parent_snapshot_id, created_at, full_skims_path,
                        full_skims_n_variables, full_skims_n_chunks, full_skims_total_size_mb,
                        partial_skims_path, partial_skims_n_variables, partial_skims_n_chunks,
                        partial_skims_total_size_mb, changed_chunks, chunk_manifest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (snapshot_id) DO NOTHING
                    """,
                    [
                        snapshot_id,
                        run_id,
                        snapshot.get("year"),
                        snapshot.get("iteration"),
                        snapshot.get("snapshot_type"),
                        snapshot.get("model"),
                        snapshot.get("parent_snapshot"),
                        snapshot.get("created_at"),
                        full_skims.get("path"),
                        full_skims.get("n_variables"),
                        full_skims.get("n_chunks"),
                        full_skims.get("total_size_mb"),
                        partial_skims.get("path"),
                        partial_skims.get("n_variables"),
                        partial_skims.get("n_chunks"),
                        partial_skims.get("total_size_mb"),
                        full_skims.get("changed_chunks"),
                        json.dumps(full_skims.get("chunk_manifest", {})),
                    ],
                )
            conn.commit()
            logger.info(f"Successfully uploaded Zarr manifest data for run {run_id}")
            return True

        except Exception as e:
            try:
                conn.rollback()
            except:
                pass
            logger.error(f"Failed to upload Zarr manifest data for run {run_id}: {e}")
            return False

    def upload_hierarchical_config_hashes(
        self, config_snapshot_id: str, hierarchical_hashes: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Upload hierarchical configuration hashes (Phase 1).

        Args:
            config_snapshot_id: ID of the config snapshot these hashes belong to
            hierarchical_hashes: Dict from create_hierarchical_config_hashes()
                Format: {model_name: {'hash': str, 'config_data': dict, ...}}

        Returns:
            True if upload successful

        Example:
            >>> hashes = snapshot_manager.create_hierarchical_config_hashes(snapshot, enabled_models)
            >>> db.upload_hierarchical_config_hashes(snapshot_id, hashes)
        """
        try:
            conn = self._get_connection()

            for model_name, hash_info in hierarchical_hashes.items():
                config_hash = hash_info["hash"]
                config_data = hash_info["config_data"]
                config_type = hash_info.get("config_type", model_name)

                # Insert into config_snapshots (deduplicates by hash)
                conn.execute(
                    """
                    INSERT INTO model_configs (
                        config_hash, model_name, config_snapshot_id,
                        config_type, config_data
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (config_hash) DO NOTHING
                    """,
                    [
                        config_hash,
                        model_name,
                        config_snapshot_id,
                        config_type,
                        json.dumps(config_data),
                    ],
                )

            logger.info(
                f"Uploaded {len(hierarchical_hashes)} hierarchical config hashes "
                f"for snapshot {config_snapshot_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to upload hierarchical config hashes: {e}")
            raise DatabaseUploadError(f"Upload failed: {e}")

    def link_model_run_to_config_hashes(
        self, model_run_id: str, hierarchical_hashes: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Link a model run to its hierarchical config hashes.

        This creates entries in model_run_configs linking the model run
        to all relevant config layers (base + upstream models + own config).

        Args:
            model_run_id: ID of the model run
            hierarchical_hashes: Dict from create_hierarchical_config_hashes()

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()

            for model_name, hash_info in hierarchical_hashes.items():
                config_hash = hash_info["hash"]
                config_type = hash_info.get("config_type", model_name)

                conn.execute(
                    """
                    INSERT INTO model_run_configs (
                        model_run_id, config_hash, config_type
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT (model_run_id, config_type) DO UPDATE
                    SET config_hash = excluded.config_hash
                    """,
                    [model_run_id, config_hash, config_type],
                )

            logger.info(
                f"Linked model run {model_run_id} to {len(hierarchical_hashes)} config hashes"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to link model run to config hashes: {e}")
            raise DatabaseUploadError(f"Link failed: {e}")

    def find_reusable_outputs(
        self, model_name: str, config_hash: str, year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find model outputs that can be reused based on config hash.

        This is the key query for intelligent caching: find completed model runs
        with matching configuration to reuse their outputs.

        Args:
            model_name: Name of model (urbansim, activitysim, beam, atlas)
            config_hash: Configuration hash for this model
            year: Optional year filter

        Returns:
            List of file records that can be reused

        Example:
            >>> # Find reusable ActivitySim outputs
            >>> outputs = db.find_reusable_outputs('activitysim', 'abc123def456', year=2018)
            >>> if outputs:
            ...     print(f"Found {len(outputs)} reusable files!")
        """
        try:
            conn = self._get_connection()

            query = """
                SELECT fr.*
                FROM file_records fr
                JOIN model_runs mr ON fr.model_run_id = mr.unique_id
                JOIN model_run_configs mrc ON mr.unique_id = mrc.model_run_id
                WHERE mrc.config_hash = ?
                  AND mr.model = ?
                  AND mr.status = 'completed'
            """

            params = [config_hash, model_name]

            if year is not None:
                query += " AND mr.year = ?"
                params.append(year)

            results = conn.execute(query, params).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in results]

        except Exception as e:
            logger.error(f"Failed to find reusable outputs: {e}")
            raise DatabaseQueryError(f"Query failed: {e}")

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve run information by run ID."""
        try:
            conn = self._get_connection()
            result = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()

            if result:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            raise DatabaseQueryError(f"Query failed: {e}")

    def get_runs_by_config_hash(self, config_hash: str) -> List[Dict[str, Any]]:
        """Find runs with matching configuration hash."""
        try:
            conn = self._get_connection()
            results = conn.execute(
                "SELECT * FROM runs WHERE config_content_hash = ?", [config_hash]
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in results]

        except Exception as e:
            logger.error(f"Failed to get runs by config hash {config_hash}: {e}")
            raise DatabaseQueryError(f"Query failed: {e}")

    def get_dataset_by_openlineage_id(
        self, openlineage_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve dataset information by OpenLineage ID."""
        try:
            conn = self._get_connection()
            result = conn.execute(
                "SELECT * FROM file_records WHERE openlineage_id = ?", [openlineage_id]
            ).fetchone()

            if result:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Failed to get dataset {openlineage_id}: {e}")
            raise DatabaseQueryError(f"Query failed: {e}")

    def check_dataset_exists(self, openlineage_id: str) -> bool:
        """Check if a dataset exists by OpenLineage ID."""
        try:
            conn = self._get_connection()
            result = conn.execute(
                "SELECT 1 FROM file_records WHERE openlineage_id = ? LIMIT 1",
                [openlineage_id],
            ).fetchone()

            return result is not None

        except Exception as e:
            logger.error(f"Failed to check dataset existence {openlineage_id}: {e}")
            return False

    def update_file_record_storage_info(
        self,
        unique_id: str,
        storage_location: str,
        logical_table_name: str,
        data_format: str,
        file_path: str,
    ) -> bool:
        """
        Updates the storage_location, logical_table_name, data_format, and file_path for a file_record.

        Args:
            unique_id: The unique_id of the file_record to update.
            storage_location: The new storage location ('database' or 'external').
            logical_table_name: The logical table name for the file_record.
            data_format: The format of the data ('parquet', 'csv', 'h5', etc.).
            file_path: The absolute path to the file.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            conn.execute(
                """
                UPDATE file_records
                SET
                    storage_location = ?,
                    logical_table_name = ?,
                    data_format = ?,
                    file_path = ?
                WHERE unique_id = ?
                """,
                [
                    storage_location,
                    logical_table_name,
                    data_format,
                    file_path,
                    unique_id,
                ],
            )
            logger.info(
                f"Updated file_record {unique_id} to storage_location={storage_location}, "
                f"logical_table_name={logical_table_name}, data_format={data_format}"
            )
            print(
                f"Updated file_record {unique_id} to storage_location={storage_location}, "
                f"logical_table_name={logical_table_name}, data_format={data_format}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update file_record {unique_id} storage info: {e}")
            print(f"Failed to update file_record {unique_id} storage info: {e}")
            return False

    def get_model_runs_by_config(
        self, model_name: str, config_hash: str
    ) -> List[Dict[str, Any]]:
        """Find model runs with matching model and configuration."""
        try:
            conn = self._get_connection()
            results = conn.execute(
                """
                SELECT mr.* FROM model_runs mr
                JOIN runs r ON mr.run_id = r.run_id
                WHERE mr.model = ? AND r.config_content_hash = ?
            """,
                [model_name, config_hash],
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in results]

        except Exception as e:
            logger.error(
                f"Failed to get model runs for {model_name} with config {config_hash}: {e}"
            )
            raise DatabaseQueryError(f"Query failed: {e}")

    def store_urbansim_raw_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: Optional[str] = None,
    ) -> bool:
        """
        Store raw UrbanSim data in the appropriate database table.

        Args:
            table_name: Name of the raw UrbanSim table (households, persons, jobs, etc.)
            df: DataFrame containing the raw data
            file_record_id: File record unique ID
            run_id: PILATES run ID
            openlineage_id: OpenLineage dataset ID for the parent H5 file
            table_openlineage_id: OpenLineage dataset ID for this specific table. Defaults to openlineage_id.

        Returns:
            bool: True if storage successful
        """
        if table_openlineage_id is None:
            table_openlineage_id = openlineage_id
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._store_households_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "persons":
                return self._store_persons_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "jobs":
                return self._store_jobs_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "blocks":
                return self._store_blocks_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "buildings":
                return self._store_buildings_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "parcels":
                return self._store_parcels_raw_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            else:
                # Store in generic table for other raw data
                return self._store_generic_data(
                    conn,
                    f"{table_name}_raw",
                    df,
                    file_record_id,
                    run_id,
                    openlineage_id,
                    table_openlineage_id,
                )

        except Exception as e:
            logger.error(
                f"Failed to store raw UrbanSim data for table {table_name}: {e}"
            )
            return False

    def store_activitysim_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: Optional[str] = None,
    ) -> bool:
        """
        Store ActivitySim data in the appropriate database table.

        Args:
            table_name: Name of the ActivitySim table (households, persons, land_use, etc.)
            df: DataFrame containing the data
            file_record_id: File record unique ID
            run_id: PILATES run ID
            year: Simulation year
            iteration: Simulation iteration
            openlineage_id: OpenLineage dataset ID for the parent file
            table_openlineage_id: OpenLineage dataset ID for this specific table. Defaults to openlineage_id.

        Returns:
            bool: True if storage successful
        """
        if table_openlineage_id is None:
            table_openlineage_id = openlineage_id
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._store_households_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "persons":
                return self._store_persons_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            elif table_name == "land_use":
                return self._store_land_use_data(
                    conn,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )
            else:
                # Store in generic table for other ActivitySim inputs (e.g., skims)
                return self._store_generic_data(
                    conn,
                    table_name,
                    df,
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_openlineage_id,
                )

        except Exception as e:
            logger.error(
                f"Failed to store ActivitySim data for table {table_name}: {e}"
            )
            return False

    def _store_households_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store households data in activitysim_households table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns
            expected_cols = [
                "household_id",
                "TAZ",
                "persons",
                "income",
                "cars",
                "HHT",
                "workers",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            # Households are constant across iterations - unique on (run_id, year, household_id)
            # First iteration inserts, subsequent iterations skip (reuse same data)
            conn.register("households_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_households ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM households_temp_data
                ON CONFLICT (run_id, year, household_id) DO NOTHING
            """
            )
            conn.unregister("households_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} household records")
            return True

        except Exception as e:
            logger.error(f"Failed to store households data: {e}")
            return False

    def _store_persons_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store persons data in activitysim_persons table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns
            expected_cols = [
                "person_id",
                "household_id",
                "TAZ",
                "age",
                "worker",
                "student",
                "ptype",
                "pemploy",
                "pstudent",
                "member_id",
                "workplace_taz",
                "school_taz",
                "home_x",
                "home_y",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            # Persons are constant across iterations - unique on (run_id, year, person_id)
            # First iteration inserts, subsequent iterations skip (reuse same data)
            conn.register("persons_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_persons ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM persons_temp_data
                ON CONFLICT (run_id, year, person_id) DO NOTHING
            """
            )
            conn.unregister("persons_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} person records")
            return True

        except Exception as e:
            logger.error(f"Failed to store persons data: {e}")
            return False

    def _store_land_use_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store land use data in activitysim_land_use table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns - land use tables have many columns
            expected_cols = [
                "TAZ",
                "TOTPOP",
                "TOTHH",
                "TOTEMP",
                "TOTACRE",
                "area_type",
                "employment_density",
                "pop_density",
                "hh_density",
                "AGE0004",
                "AGE0519",
                "AGE2044",
                "AGE4564",
                "AGE64P",
                "AGE62P",
                "HHINCQ1",
                "HHINCQ2",
                "HHINCQ3",
                "HHINCQ4",
                "EMPRES",
                "RETEMPN",
                "FPSEMPN",
                "HEREMPN",
                "AGREMPN",
                "MWTEMPN",
                "OTHEMPN",
                "HSENROLL",
                "COLLFTE",
                "COLLPTE",
                "PRKCST",
                "OPRKCST",
            ]

            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            # Land use is constant across iterations - unique on (run_id, year, TAZ)
            # First iteration inserts, subsequent iterations skip (reuse same data)
            conn.register("land_use_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_land_use ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM land_use_temp_data
                ON CONFLICT (run_id, year, TAZ) DO NOTHING
            """
            )
            conn.unregister("land_use_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} land use records")
            return True

        except Exception as e:
            logger.error(f"Failed to store land use data: {e}")
            return False

    def _store_generic_data(
        self,
        conn,
        table_name: str,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store generic ActivitySim data in JSON format.

        This stores data that can vary per iteration (e.g., skims updated by BEAM).
        Uses DELETE+INSERT instead of UPSERT because we want to replace the entire JSON blob.
        """
        try:
            # Convert DataFrame to JSON for storage
            data_json = df.to_json(orient="records")

            # Delete existing data for this specific table/year/iteration combination
            conn.execute(
                "DELETE FROM activitysim_data_generic WHERE run_id = ? AND year = ? AND iteration = ? AND table_name = ?",
                [run_id, year, iteration, table_name],
            )
            conn.execute(
                """
                INSERT INTO activitysim_data_generic (
                    file_record_id, run_id, year, iteration, openlineage_id, table_name, data_json, table_openlineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    file_record_id,
                    run_id,
                    year,
                    iteration,
                    openlineage_id,
                    table_name,
                    data_json,
                    table_openlineage_id,
                ],
            )

            logger.info(
                f"Stored {len(df)} records for table {table_name} in generic storage"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store generic data for table {table_name}: {e}")
            return False

    # Raw UrbanSim data storage methods
    def _store_households_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw households data in urbansim_households_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = [
                "household_id",
                "building_id",
                "persons",
                "income",
                "cars",
                "block_id",
                "age_of_head",
                "children",
                "workers",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("households_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_households_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM households_raw_temp_data
                ON CONFLICT (run_id, year, household_id) DO NOTHING
            """
            )
            conn.unregister("households_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw household records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw households data: {e}")
            return False

    def _store_persons_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw persons data in urbansim_persons_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = [
                "person_id",
                "household_id",
                "age",
                "worker",
                "student",
                "race_id",
                "sex",
                "work_zone_id",
                "school_zone_id",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("persons_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_persons_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM persons_raw_temp_data
                ON CONFLICT (run_id, year, person_id) DO NOTHING
            """
            )
            conn.unregister("persons_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw person records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw persons data: {e}")
            return False

    def _store_jobs_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw jobs data in urbansim_jobs_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = ["job_id", "building_id", "sector_id", "home_based_status"]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("jobs_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_jobs_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM jobs_raw_temp_data
                ON CONFLICT (run_id, year, job_id) DO NOTHING
            """
            )
            conn.unregister("jobs_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw job records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw jobs data: {e}")
            return False

    def _store_blocks_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw blocks data in urbansim_blocks_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = [
                "block_id",
                "block_group_id",
                "zone_id",
                "taz_zone_id",
                "square_meters_land",
                "x",
                "y",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Convert index to column if needed
            if "block_id" not in df_copy.columns and df_copy.index.name:
                df_copy["block_id"] = df_copy.index

            # Ensure block_id is string for consistency
            if "block_id" in df_copy.columns:
                df_copy["block_id"] = df_copy["block_id"].astype(str)

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("blocks_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_blocks_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM blocks_raw_temp_data
                ON CONFLICT (run_id, year, block_id) DO NOTHING
            """
            )
            conn.unregister("blocks_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw block records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw blocks data: {e}")
            return False

    def _store_buildings_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw buildings data in urbansim_buildings_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = [
                "building_id",
                "parcel_id",
                "building_type_id",
                "sqft",
                "year_built",
                "stories",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("buildings_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_buildings_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM buildings_raw_temp_data
                ON CONFLICT (run_id, year, building_id) DO NOTHING
            """
            )
            conn.unregister("buildings_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw building records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw buildings data: {e}")
            return False

    def _store_parcels_raw_data(
        self,
        conn,
        df: pd.DataFrame,
        file_record_id: str,
        run_id: str,
        year: int,
        iteration: int,
        openlineage_id: str,
        table_openlineage_id: str,
    ) -> bool:
        """Store raw parcels data in urbansim_parcels_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["year"] = year
            df_copy["iteration"] = iteration
            df_copy["openlineage_id"] = openlineage_id
            df_copy["table_openlineage_id"] = table_openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = [
                "parcel_id",
                "zone_id",
                "land_value",
                "total_sqft",
                "county_id",
            ]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = [
                "file_record_id",
                "run_id",
                "year",
                "iteration",
                "openlineage_id",
                "table_openlineage_id",
            ] + expected_cols
            df_insert = df_copy[insert_cols]

            # Bulk insert using DuckDB's DataFrame integration with UPSERT
            conn.register("parcels_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_parcels_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM parcels_raw_temp_data
                ON CONFLICT (run_id, year, parcel_id) DO NOTHING
            """
            )
            conn.unregister("parcels_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw parcel records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw parcels data: {e}")
            return False

    def query_hybrid_table(self, logical_table_name: str) -> duckdb.DuckDBPyRelation:
        """
        Constructs and executes a dynamic SQL query to fetch data from both
        the database and external Parquet files, ensuring schemas are aligned.

        Args:
            logical_table_name: The logical name of the table to query.

        Returns:
            A DuckDB relation object representing the combined result set.
        """
        conn = self._get_connection()

        # 1. Determine the canonical schema from the `uploaded_` table.
        # This is the schema all parts of the UNION must conform to.
        try:
            uploaded_table_schema_desc = conn.execute(
                f"DESCRIBE uploaded_{logical_table_name}"
            ).fetchall()
            # Making a set for quick lookups and a list to preserve order
            uploaded_cols_ordered = [row[0] for row in uploaded_table_schema_desc]
        except duckdb.Error as e:
            # If the uploaded table doesn't exist, we cannot proceed.
            logger.error(
                f"Cannot query hybrid table '{logical_table_name}': The base table 'uploaded_{logical_table_name}' does not exist. {e}"
            )
            raise DatabaseQueryError(f"Base table for {logical_table_name} not found.")

        # The base query is always on the uploaded table.
        # We explicitly list columns to ensure consistent ordering.
        col_list_str = ", ".join([f'"{c}"' for c in uploaded_cols_ordered])
        all_queries = [f"SELECT {col_list_str} FROM uploaded_{logical_table_name}"]

        # 2. Find all external files and build a schema-aligned query for each.
        external_files_df = conn.execute(
            """
            SELECT unique_id, run_id, file_path, "year", iteration
            FROM file_records 
            WHERE logical_table_name = ? AND storage_location = 'external' AND data_format = 'parquet'
            """,
            [logical_table_name],
        ).fetchdf()

        for _, record in external_files_df.iterrows():
            file_path = record["file_path"]
            try:
                # Get the schema of the current parquet file
                parquet_schema_desc = conn.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{file_path}')"
                ).fetchall()
                parquet_cols = {row[0] for row in parquet_schema_desc}
            except duckdb.Error as e:
                logger.warning(
                    f"Could not read schema for external file {file_path}, skipping. Error: {e}"
                )
                continue

            # 3. Build the projection list for this specific parquet file
            # This aligns its output with the canonical schema.
            projection = []
            for col in uploaded_cols_ordered:
                if col in parquet_cols:
                    # Column exists in the parquet file, select it directly.
                    projection.append(f'"{col}"')
                # Check for metadata columns that exist in the file_records table
                elif col == "run_id":
                    projection.append(f"'{record['run_id']}' AS run_id")
                elif col == "file_record_id":
                    projection.append(f"'{record['unique_id']}' AS file_record_id")
                elif col == "year":
                    projection.append(f"{record['year']} AS year")
                elif col == "iteration":
                    iteration_val = (
                        "NULL" if pd.isna(record["iteration"]) else record["iteration"]
                    )
                    projection.append(f"{iteration_val} AS iteration")
                elif col == "sub_iteration":
                    projection.append(
                        f"0 AS sub_iteration"
                    )  # Default sub_iteration if not present
                else:
                    # Column is in the canonical schema but not this parquet file. Project NULL.
                    projection.append(f'NULL AS "{col}"')

            all_queries.append(
                f"SELECT {', '.join(projection)} FROM read_parquet('{file_path}')"
            )

        # 4. Combine all queries into a single UNION ALL statement.
        final_query = "\nUNION ALL\n".join(all_queries)

        print(f"Executing Hybrid Query:\n{final_query}")  # For debugging
        return conn.sql(final_query)

    def retrieve_activitysim_data(
        self,
        table_name: str,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve ActivitySim data from database and return as DataFrame.

        Args:
            table_name: Name of the ActivitySim table
            openlineage_id: Specific OpenLineage ID to retrieve
            config_hash: Configuration hash to filter by
            year: Year to filter by

        Returns:
            DataFrame with the requested data or None if not found
        """
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._retrieve_households_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "persons":
                return self._retrieve_persons_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "land_use":
                return self._retrieve_land_use_data(
                    conn, openlineage_id, config_hash, year
                )
            else:
                return self._retrieve_generic_data(
                    conn, table_name, openlineage_id, config_hash, year
                )

        except Exception as e:
            logger.error(
                f"Failed to retrieve ActivitySim data for table {table_name}: {e}"
            )
            return None

    def _retrieve_households_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve households data from database."""
        try:
            base_query = """
                SELECT household_id, TAZ, persons, income, cars, HHT, workers 
                FROM activitysim_households 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} household records")
                return result
            else:
                logger.warning("No household data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve households data: {e}")
            return None

    def _retrieve_persons_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve persons data from database."""
        try:
            base_query = """
                SELECT person_id, household_id, TAZ, age, worker, student, ptype, pemploy, 
                       pstudent, member_id, workplace_taz, school_taz, home_x, home_y
                FROM activitysim_persons 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} person records")
                return result
            else:
                logger.warning("No person data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve persons data: {e}")
            return None

    def _retrieve_land_use_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve land use data from database."""
        try:
            base_query = """
                SELECT TAZ, TOTPOP, TOTHH, TOTEMP, TOTACRE, area_type, employment_density, 
                       pop_density, hh_density, AGE0004, AGE0519, AGE2044, AGE4564, AGE64P, 
                       AGE62P, HHINCQ1, HHINCQ2, HHINCQ3, HHINCQ4, EMPRES, RETEMPN, FPSEMPN, 
                       HEREMPN, AGREMPN, MWTEMPN, OTHEMPN, HSENROLL, COLLFTE, COLLPTE, PRKCST, OPRKCST
                FROM activitysim_land_use 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} land use records")
                return result
            else:
                logger.warning("No land use data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve land use data: {e}")
            return None

    def _retrieve_generic_data(
        self,
        conn,
        table_name: str,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve generic data from JSON storage."""
        try:
            base_query = """
                SELECT data_json 
                FROM activitysim_data_generic 
                WHERE table_name = ?
            """
            params = [table_name]

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC LIMIT 1"

            result = conn.execute(base_query, params).fetchone()

            if result:
                data_json = result[0]
                df = pd.read_json(data_json, orient="records")
                logger.info(f"Retrieved {len(df)} records for table {table_name}")
                return df
            else:
                logger.warning(
                    f"No data found for table {table_name} matching criteria"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve generic data for table {table_name}: {e}")
            return None

    def retrieve_urbansim_raw_data(
        self,
        table_name: str,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve raw UrbanSim data from database and return as DataFrame.

        Args:
            table_name: Name of the raw UrbanSim table (households, persons, jobs, etc.)
            openlineage_id: Specific OpenLineage ID to retrieve
            config_hash: Configuration hash to filter by
            year: Year to filter by

        Returns:
            DataFrame with the requested raw data or None if not found
        """
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._retrieve_households_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "persons":
                return self._retrieve_persons_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "jobs":
                return self._retrieve_jobs_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "blocks":
                return self._retrieve_blocks_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "buildings":
                return self._retrieve_buildings_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            elif table_name == "parcels":
                return self._retrieve_parcels_raw_data(
                    conn, openlineage_id, config_hash, year
                )
            else:
                logger.warning(f"Unknown raw UrbanSim table: {table_name}")
                return None

        except Exception as e:
            logger.error(
                f"Failed to retrieve raw UrbanSim data for table {table_name}: {e}"
            )
            return None

    def _retrieve_households_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw households data from database."""
        try:
            base_query = """
                SELECT household_id, building_id, persons, income, cars, block_id,
                       age_of_head, children, workers
                FROM urbansim_households_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw household records")
                return result
            else:
                logger.warning("No raw household data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw households data: {e}")
            return None

    def _retrieve_persons_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw persons data from database."""
        try:
            base_query = """
                SELECT person_id, household_id, age, worker, student, race_id, sex,
                       work_zone_id, school_zone_id
                FROM urbansim_persons_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw person records")
                return result
            else:
                logger.warning("No raw person data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw persons data: {e}")
            return None

    def _retrieve_jobs_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw jobs data from database."""
        try:
            base_query = """
                SELECT job_id, building_id, sector_id, home_based_status
                FROM urbansim_jobs_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw job records")
                return result
            else:
                logger.warning("No raw job data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw jobs data: {e}")
            return None

    def _retrieve_blocks_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw blocks data from database."""
        try:
            base_query = """
                SELECT block_id, block_group_id, zone_id, taz_zone_id,
                       square_meters_land, x, y
                FROM urbansim_blocks_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw block records")
                return result
            else:
                logger.warning("No raw block data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw blocks data: {e}")
            return None

    def _retrieve_buildings_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw buildings data from database."""
        try:
            base_query = """
                SELECT building_id, parcel_id, building_type_id, sqft, year_built, stories
                FROM urbansim_buildings_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw building records")
                return result
            else:
                logger.warning("No raw building data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw buildings data: {e}")
            return None

    def _retrieve_parcels_raw_data(
        self,
        conn,
        openlineage_id: str = None,
        config_hash: str = None,
        year: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve raw parcels data from database."""
        try:
            base_query = """
                SELECT parcel_id, zone_id, land_value, total_sqft, county_id
                FROM urbansim_parcels_raw 
                WHERE 1=1
            """
            params = []

            if openlineage_id:
                base_query += " AND openlineage_id = ?"
                params.append(openlineage_id)

            if config_hash:
                base_query += " AND run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                params.append(config_hash)

            base_query += " ORDER BY created_at DESC"

            result = conn.execute(base_query, params).fetchdf()

            if len(result) > 0:
                logger.info(f"Retrieved {len(result)} raw parcel records")
                return result
            else:
                logger.warning("No raw parcel data found matching criteria")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve raw parcels data: {e}")
            return None

    def export_data_dictionary(
        self, output_path: str, format: str = "markdown", include_stats: bool = True
    ) -> bool:
        """
        Export complete data dictionary documenting database schema.

        Args:
            output_path: Path where documentation should be written
            format: Output format - 'markdown', 'json', 'csv', or 'html'
            include_stats: Include row counts and data statistics

        Returns:
            bool: True if export successful

        The data dictionary includes:
        - All tables with descriptions
        - All columns with types and descriptions
        - Foreign key relationships
        - Index information
        - Row counts and data ranges (if include_stats=True)
        """
        try:
            conn = self._get_connection()

            # Query schema information from DuckDB system tables
            schema_query = """
                SELECT
                    t.table_name,
                    t.comment as table_comment,
                    c.column_name,
                    c.data_type,
                    c.comment as column_comment,
                    c.is_nullable,
                    c.column_default
                FROM duckdb_tables() t
                LEFT JOIN duckdb_columns() c
                    ON t.table_name = c.table_name
                    AND t.schema_name = c.schema_name
                WHERE t.schema_name = 'main'
                    AND t.table_name NOT LIKE 'duckdb_%'
                    AND t.table_name NOT LIKE 'sqlite_%'
                ORDER BY t.table_name, c.column_index
            """

            schema_df = conn.execute(schema_query).fetchdf()

            # Get foreign key constraints
            fk_query = """
                SELECT
                    fk_table,
                    fk_columns,
                    pk_table,
                    pk_columns
                FROM duckdb_constraints()
                WHERE constraint_type = 'FOREIGN KEY'
            """
            try:
                fk_df = conn.execute(fk_query).fetchdf()
            except:
                # Older DuckDB versions may not support this
                fk_df = pd.DataFrame()

            # Get row counts if requested
            row_counts = {}
            if include_stats:
                for table_name in schema_df["table_name"].unique():
                    try:
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"
                        ).fetchone()[0]
                        row_counts[table_name] = count
                    except:
                        row_counts[table_name] = None

            # Export based on format
            if format == "markdown":
                return self._export_markdown_dictionary(
                    output_path, schema_df, fk_df, row_counts
                )
            elif format == "json":
                return self._export_json_dictionary(
                    output_path, schema_df, fk_df, row_counts
                )
            elif format == "csv":
                return self._export_csv_dictionary(
                    output_path, schema_df, fk_df, row_counts
                )
            elif format == "html":
                return self._export_html_dictionary(
                    output_path, schema_df, fk_df, row_counts
                )
            else:
                logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            logger.error(f"Failed to export data dictionary: {e}")
            return False

    def _export_markdown_dictionary(
        self,
        output_path: str,
        schema_df: pd.DataFrame,
        fk_df: pd.DataFrame,
        row_counts: Dict,
    ) -> bool:
        """Export data dictionary as Markdown."""
        try:
            with open(output_path, "w") as f:
                f.write("# PILATES Database Data Dictionary\n\n")
                f.write(
                    "Auto-generated schema documentation for the PILATES database.\n\n"
                )
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                f.write("## Table of Contents\n\n")

                # Group by table category
                metadata_tables = [
                    "runs",
                    "config_snapshots",
                    "file_records",
                    "model_runs",
                    "openlineage_events",
                ]
                urbansim_tables = [
                    t
                    for t in schema_df["table_name"].unique()
                    if "urbansim" in t and "_raw" in t
                ]
                activitysim_tables = [
                    t for t in schema_df["table_name"].unique() if "activitysim" in t
                ]

                f.write("### Metadata Tables\n")
                for table in metadata_tables:
                    if table in schema_df["table_name"].values:
                        f.write(f"- [{table}](#{table.replace('_', '-')})\n")

                f.write("\n### UrbanSim Raw Data Tables\n")
                for table in urbansim_tables:
                    f.write(f"- [{table}](#{table.replace('_', '-')})\n")

                f.write("\n### ActivitySim Processed Data Tables\n")
                for table in activitysim_tables:
                    f.write(f"- [{table}](#{table.replace('_', '-')})\n")

                f.write("\n---\n\n")

                # Document each table
                for table_name in schema_df["table_name"].unique():
                    table_data = schema_df[schema_df["table_name"] == table_name]
                    table_comment = table_data.iloc[0]["table_comment"]

                    f.write(f"## {table_name}\n\n")

                    if table_comment:
                        f.write(f"**Description:** {table_comment}\n\n")

                    if table_name in row_counts and row_counts[table_name] is not None:
                        f.write(f"**Row Count:** {row_counts[table_name]:,}\n\n")

                    # Column table
                    f.write("| Column | Type | Nullable | Description |\n")
                    f.write("|--------|------|----------|-------------|\n")

                    for _, row in table_data.iterrows():
                        col_name = row["column_name"]
                        col_type = row["data_type"]
                        nullable = "Yes" if row["is_nullable"] == "YES" else "No"
                        col_desc = row["column_comment"] or ""

                        # Escape pipe characters in description
                        col_desc = col_desc.replace("|", "\\|")

                        f.write(
                            f"| {col_name} | {col_type} | {nullable} | {col_desc} |\n"
                        )

                    # Foreign keys
                    if not fk_df.empty:
                        table_fks = fk_df[fk_df["fk_table"] == table_name]
                        if not table_fks.empty:
                            f.write("\n**Foreign Keys:**\n\n")
                            for _, fk in table_fks.iterrows():
                                f.write(
                                    f"- `{fk['fk_columns']}` → `{fk['pk_table']}.{fk['pk_columns']}`\n"
                                )

                    f.write("\n---\n\n")

                logger.info(f"Markdown data dictionary exported to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to export markdown dictionary: {e}")
            return False

    def _export_json_dictionary(
        self,
        output_path: str,
        schema_df: pd.DataFrame,
        fk_df: pd.DataFrame,
        row_counts: Dict,
    ) -> bool:
        """Export data dictionary as JSON."""
        try:
            tables = {}

            for table_name in schema_df["table_name"].unique():
                table_data = schema_df[schema_df["table_name"] == table_name]

                columns = []
                for _, row in table_data.iterrows():
                    columns.append(
                        {
                            "name": row["column_name"],
                            "type": row["data_type"],
                            "nullable": row["is_nullable"] == "YES",
                            "default": row["column_default"],
                            "description": row["column_comment"],
                        }
                    )

                # Get foreign keys for this table
                fks = []
                if not fk_df.empty:
                    table_fks = fk_df[fk_df["fk_table"] == table_name]
                    for _, fk_row in table_fks.iterrows():
                        fks.append(
                            {
                                "column": fk_row["fk_columns"],
                                "references_table": fk_row["pk_table"],
                                "references_column": fk_row["pk_columns"],
                            }
                        )

                tables[table_name] = {
                    "description": table_data.iloc[0]["table_comment"],
                    "row_count": row_counts.get(table_name),
                    "columns": columns,
                    "foreign_keys": fks,
                }

            dictionary = {
                "database": "PILATES",
                "generated_at": datetime.now().isoformat(),
                "tables": tables,
            }

            with open(output_path, "w") as f:
                json.dump(dictionary, f, indent=2, default=str)

            logger.info(f"JSON data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export JSON dictionary: {e}")
            return False

    def _export_csv_dictionary(
        self,
        output_path: str,
        schema_df: pd.DataFrame,
        fk_df: pd.DataFrame,
        row_counts: Dict,
    ) -> bool:
        """Export data dictionary as CSV."""
        try:
            # Add row counts to schema
            schema_with_counts = schema_df.copy()
            schema_with_counts["row_count"] = schema_with_counts["table_name"].map(
                row_counts
            )

            # Export to CSV
            schema_with_counts.to_csv(output_path, index=False)

            logger.info(f"CSV data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV dictionary: {e}")
            return False

    def _export_html_dictionary(
        self,
        output_path: str,
        schema_df: pd.DataFrame,
        fk_df: pd.DataFrame,
        row_counts: Dict,
    ) -> bool:
        """Export data dictionary as HTML."""
        try:
            html = []
            html.append("<!DOCTYPE html>")
            html.append("<html><head>")
            html.append("<title>PILATES Database Data Dictionary</title>")
            html.append("<style>")
            html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
            html.append("h1 { color: #2c3e50; }")
            html.append(
                "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }"
            )
            html.append(
                "table { border-collapse: collapse; width: 100%; margin: 20px 0; }"
            )
            html.append(
                "th { background-color: #3498db; color: white; padding: 12px; text-align: left; }"
            )
            html.append("td { border: 1px solid #ddd; padding: 10px; }")
            html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
            html.append(
                ".table-desc { font-style: italic; color: #555; margin: 10px 0; }"
            )
            html.append(".row-count { color: #16a085; font-weight: bold; }")
            html.append(
                ".fk-section { margin-top: 15px; padding: 10px; background-color: #ecf0f1; }"
            )
            html.append("</style></head><body>")

            html.append("<h1>PILATES Database Data Dictionary</h1>")
            html.append(
                f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
            )

            # Table of contents
            html.append("<h2>Table of Contents</h2>")
            html.append("<ul>")
            for table_name in schema_df["table_name"].unique():
                html.append(f'<li><a href="#{table_name}">{table_name}</a></li>')
            html.append("</ul>")

            # Each table
            for table_name in schema_df["table_name"].unique():
                table_data = schema_df[schema_df["table_name"] == table_name]
                table_comment = table_data.iloc[0]["table_comment"]

                html.append(f'<h2 id="{table_name}">{table_name}</h2>')

                if table_comment:
                    html.append(f'<p class="table-desc">{table_comment}</p>')

                if table_name in row_counts and row_counts[table_name] is not None:
                    html.append(
                        f'<p class="row-count">Row Count: {row_counts[table_name]:,}</p>'
                    )

                html.append("<table>")
                html.append(
                    "<tr><th>Column</th><th>Type</th><th>Nullable</th><th>Description</th></tr>"
                )

                for _, row in table_data.iterrows():
                    html.append("<tr>")
                    html.append(f"<td><strong>{row['column_name']}</strong></td>")
                    html.append(f"<td>{row['data_type']}</td>")
                    nullable = "Yes" if row["is_nullable"] == "YES" else "No"
                    html.append(f"<td>{nullable}</td>")
                    html.append(f"<td>{row['column_comment'] or ''}</td>")
                    html.append("</tr>")

                html.append("</table>")

                # Foreign keys
                if not fk_df.empty:
                    table_fks = fk_df[fk_df["fk_table"] == table_name]
                    if not table_fks.empty:
                        html.append('<div class="fk-section">')
                        html.append("<strong>Foreign Keys:</strong><ul>")
                        for _, fk in table_fks.iterrows():
                            html.append(
                                f"<li><code>{fk['fk_columns']}</code> → <code>{fk['pk_table']}.{fk['pk_columns']}</code></li>"
                            )
                        html.append("</ul></div>")

            html.append("</body></html>")

            with open(output_path, "w") as f:
                f.write("\n".join(html))

            logger.info(f"HTML data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export HTML dictionary: {e}")
            return False

    def store_generic_table(
        self, table_name: str, df: pd.DataFrame, sort_by: List[str] = None
    ) -> bool:
        """
        Stores a pandas DataFrame into a specified table in DuckDB.

        OPTIMIZATION: This method sorts the data during insertion.
        In DuckDB, physical sort order dictates query performance for filters.

        Args:
            table_name: The name of the target table in the database.
            df: The pandas DataFrame containing the data to upload.
            sort_by: Optional list of columns to sort by. If None, auto-detects
                     hierarchical keys (run_id, year, zone/geo identifiers).

        Returns:
            True if the upload was successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            insert_cols = df.columns.tolist()

            # 1. Clean Columns: Ensure DataFrame columns match SQL identifier rules
            # (User's schema generator lowers case, so we might need to map here
            # if DF has CamelCase)
            df.columns = [c.lower() for c in df.columns]
            insert_cols = df.columns.tolist()

            # 2. Register Temp View (Zero-Copy)
            temp_table_name = f"{table_name}_temp_{uuid.uuid4().hex[:8]}"
            conn.register(temp_table_name, df)

            # 3. Determine Sort Order (The "Cluster" Key)
            if sort_by:
                sort_clause = f"ORDER BY {', '.join(sort_by)}"
            else:
                # Auto-detect standard ABM hierarchy
                # Priority: Run -> Time -> Space -> Agent
                candidates = []
                # We always want to cluster by run and year first
                if "run_id" in insert_cols:
                    candidates.append("run_id")
                if "year" in insert_cols:
                    candidates.append("year")
                if "iteration" in insert_cols:
                    candidates.append("iteration")

                # Spatial clustering (pick the most granular one available)
                if "block_id" in insert_cols:
                    candidates.append("block_id")
                elif "zone_id" in insert_cols:
                    candidates.append("zone_id")
                elif "taz" in insert_cols:
                    candidates.append("taz")
                elif "origin" in insert_cols:
                    candidates.append("origin")

                # We usually STOP at spatial clustering for the physical sort,
                # because Agent IDs (household_id) are often random/hashed
                # and don't provide good range-skipping.

                if candidates:
                    sort_clause = f"ORDER BY {', '.join(candidates)}"
                else:
                    sort_clause = ""

            # 4. Insert with Sort
            # The ORDER BY here forces DuckDB to write the file in that order
            query = f"""
                INSERT INTO {table_name} ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} 
                FROM {temp_table_name}
                {sort_clause}
            """

            logger.debug(f"Executing insert for {table_name} with: {sort_clause}")
            conn.execute(query)

            # Cleanup
            conn.unregister(temp_table_name)
            return True

        except Exception as e:
            logger.error(f"Failed to insert data into table {table_name}: {e}")
            return False

    def close(self):
        """Close DuckDB connection with explicit checkpoint."""
        if self.connection:
            try:
                # 1. COMMIT the active transaction to make all changes permanent.
                self.connection.commit()

                # 2. Force merge of WAL to main DB file.
                self.connection.execute("CHECKPOINT;")
            except Exception as e:
                logger.warning(f"Commit/Checkpoint on close failed: {e}")

            self.connection.close()
            self.connection = None
            logger.info("DuckDB connection closed")
