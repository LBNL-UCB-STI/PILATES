"""
DuckDB implementation of the PILATES database manager.

This module provides DuckDB-specific implementation for storing and retrieving
PILATES run data, optimized for analytical queries and local/cloud deployment.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

import duckdb

from pilates.generic.records import PilatesRunInfo, OpenLineageEventMetadata
from pilates.utils.database import (
    DatabaseManager,
    DatabaseUploadError,
    DatabaseQueryError,
)

logger = logging.getLogger(__name__)


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

    def _get_connection(self):
        """Get or create database connection."""
        if self.connection is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            # Use check_same_thread=False for thread safety in certain scenarios
            # Note: DuckDB connections should still be used carefully with threading
            self.connection = duckdb.connect(self.database_path)
            logger.info(f"Connected to DuckDB database at {self.database_path}")
        return self.connection

    def initialize_database(self) -> bool:
        """
        Initialize DuckDB database with PILATES schema.

        Creates all necessary tables for storing PILATES run data.
        """
        try:
            conn = self._get_connection()

            # Create runs table - main run metadata
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP,
                    start_year INTEGER,
                    end_year INTEGER,
                    models_used VARCHAR[], -- Array of model names
                    settings_hash VARCHAR,
                    code_version VARCHAR,
                    hostname VARCHAR,
                    config_snapshot_id VARCHAR,
                    config_content_hash VARCHAR,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create config_snapshots table - configuration data
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config_snapshots (
                    snapshot_id VARCHAR PRIMARY KEY,
                    created_timestamp TIMESTAMP,
                    config_content_hash VARCHAR,
                    git_hashes JSON, -- Git hashes for different components
                    config_files JSON, -- All config file contents
                    pilates_settings JSON, -- Relevant PILATES settings
                    beam_config VARCHAR,
                    asim_subdir VARCHAR,
                    region VARCHAR
                )
            """
            )

            # Create file_records table - dataset information
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_records (
                    unique_id VARCHAR PRIMARY KEY, -- File hash
                    run_id VARCHAR,
                    openlineage_id VARCHAR UNIQUE,
                    file_path VARCHAR,
                    created_at TIMESTAMP,
                    short_name VARCHAR,
                    description VARCHAR,
                    year INTEGER,
                    models VARCHAR[], -- Models that touched this file
                    producing_run_id VARCHAR,
                    consuming_run_ids VARCHAR[], -- Multiple runs can consume
                    source_file_paths VARCHAR[],
                    metadata JSON,
                    schema JSON,
                    exists BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Create model_runs table - individual model execution info
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_runs (
                    unique_id VARCHAR PRIMARY KEY,
                    run_id VARCHAR,
                    openlineage_id VARCHAR UNIQUE,
                    model VARCHAR,
                    year INTEGER,
                    iteration INTEGER,
                    description VARCHAR,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status VARCHAR,
                    input_record_hashes VARCHAR[],
                    output_record_hashes VARCHAR[],
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Create openlineage_events table - lightweight event metadata
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS openlineage_events (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    model_run_id VARCHAR,
                    event_time TIMESTAMP,
                    event_type VARCHAR, -- START, COMPLETE, FAIL
                    run_uuid VARCHAR, -- OpenLineage run UUID
                    job_name VARCHAR, -- Formatted job name with year/iteration
                    FOREIGN KEY (run_id) REFERENCES runs(run_id),
                    FOREIGN KEY (model_run_id) REFERENCES model_runs(unique_id)
                )
            """
            )

            # Create sequences for auto-incrementing IDs
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_households_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_persons_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_jobs_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_blocks_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_buildings_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS urbansim_parcels_raw_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS activitysim_households_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS activitysim_persons_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS activitysim_land_use_id_seq START 1"
            )
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS activitysim_data_generic_id_seq START 1"
            )

            # Create raw UrbanSim data storage tables
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_households_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_households_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim household data columns
                    household_id INTEGER,
                    building_id INTEGER,
                    persons INTEGER,
                    income FLOAT,
                    cars INTEGER,
                    block_id VARCHAR,
                    
                    -- Additional raw columns that might be present
                    age_of_head INTEGER,
                    children INTEGER,
                    workers INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_persons_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_persons_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim person data columns
                    person_id INTEGER,
                    household_id INTEGER,
                    age INTEGER,
                    worker INTEGER,
                    student INTEGER,
                    race_id INTEGER,
                    sex INTEGER,
                    work_zone_id INTEGER,
                    school_zone_id INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_jobs_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_jobs_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim job data columns
                    job_id INTEGER,
                    building_id INTEGER,
                    sector_id VARCHAR,
                    home_based_status INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_blocks_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_blocks_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim block data columns
                    block_id VARCHAR,
                    block_group_id VARCHAR,
                    zone_id VARCHAR,
                    taz_zone_id VARCHAR,
                    square_meters_land FLOAT,
                    x FLOAT,
                    y FLOAT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_buildings_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_buildings_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim building data columns
                    building_id INTEGER,
                    parcel_id INTEGER,
                    building_type_id INTEGER,
                    sqft INTEGER,
                    year_built INTEGER,
                    stories INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS urbansim_parcels_raw (
                    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_parcels_raw_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Raw UrbanSim parcel data columns
                    parcel_id INTEGER,
                    zone_id VARCHAR,
                    land_value FLOAT,
                    total_sqft FLOAT,
                    county_id VARCHAR,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    table_type VARCHAR DEFAULT 'raw',
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Create ActivitySim processed data storage tables
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activitysim_households (
                    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_households_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Household data columns (will be dynamically created based on actual data)
                    household_id INTEGER,
                    TAZ VARCHAR,
                    persons INTEGER,
                    income FLOAT,
                    cars INTEGER,
                    HHT INTEGER,
                    workers INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activitysim_persons (
                    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_persons_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Person data columns (will be dynamically created based on actual data)
                    person_id INTEGER,
                    household_id INTEGER,
                    TAZ VARCHAR,
                    age INTEGER,
                    worker INTEGER,
                    student INTEGER,
                    ptype INTEGER,
                    pemploy INTEGER,
                    pstudent INTEGER,
                    member_id INTEGER,
                    workplace_taz INTEGER,
                    school_taz INTEGER,
                    home_x FLOAT,
                    home_y FLOAT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activitysim_land_use (
                    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_land_use_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    
                    -- Land use data columns (will be dynamically created based on actual data)
                    TAZ VARCHAR,
                    TOTPOP INTEGER,
                    TOTHH INTEGER,
                    TOTEMP INTEGER,
                    TOTACRE FLOAT,
                    area_type INTEGER,
                    employment_density FLOAT,
                    pop_density FLOAT,
                    hh_density FLOAT,
                    
                    -- Common socioeconomic columns
                    AGE0004 INTEGER,
                    AGE0519 INTEGER,
                    AGE2044 INTEGER,
                    AGE4564 INTEGER,
                    AGE64P INTEGER,
                    AGE62P INTEGER,
                    HHINCQ1 INTEGER,
                    HHINCQ2 INTEGER,
                    HHINCQ3 INTEGER,
                    HHINCQ4 INTEGER,
                    EMPRES INTEGER,
                    RETEMPN INTEGER,
                    FPSEMPN INTEGER,
                    HEREMPN INTEGER,
                    AGREMPN INTEGER,
                    MWTEMPN INTEGER,
                    OTHEMPN INTEGER,
                    HSENROLL INTEGER,
                    COLLFTE INTEGER,
                    COLLPTE INTEGER,
                    PRKCST FLOAT,
                    OPRKCST FLOAT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Generic table for other ActivitySim input tables (accessibility, zones, etc.)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activitysim_data_generic (
                    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_data_generic_id_seq'),
                    file_record_id VARCHAR,
                    run_id VARCHAR,
                    openlineage_id VARCHAR,
                    table_name VARCHAR,
                    
                    -- JSON storage for flexible schema
                    data_json JSON,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Create indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_records_openlineage_id ON file_records(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_records_short_name ON file_records(short_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_runs_model ON model_runs(model)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_runs_year ON model_runs(year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_openlineage_events_run_uuid ON openlineage_events(run_uuid)"
            )

            # Create indexes for raw UrbanSim data tables
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_households_raw_openlineage_id ON urbansim_households_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_households_raw_run_id ON urbansim_households_raw(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_persons_raw_openlineage_id ON urbansim_persons_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_persons_raw_run_id ON urbansim_persons_raw(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_jobs_raw_openlineage_id ON urbansim_jobs_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_jobs_raw_run_id ON urbansim_jobs_raw(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_blocks_raw_openlineage_id ON urbansim_blocks_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_blocks_raw_run_id ON urbansim_blocks_raw(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_buildings_raw_openlineage_id ON urbansim_buildings_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_buildings_raw_run_id ON urbansim_buildings_raw(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_parcels_raw_openlineage_id ON urbansim_parcels_raw(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usim_parcels_raw_run_id ON urbansim_parcels_raw(run_id)"
            )

            # Create indexes for ActivitySim processed data tables
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_households_openlineage_id ON activitysim_households(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_households_run_id ON activitysim_households(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_persons_openlineage_id ON activitysim_persons(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_persons_run_id ON activitysim_persons(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_land_use_openlineage_id ON activitysim_land_use(openlineage_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_land_use_run_id ON activitysim_land_use(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_generic_table_name ON activitysim_data_generic(table_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asim_generic_openlineage_id ON activitysim_data_generic(openlineage_id)"
            )

            logger.info("DuckDB database initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB database: {e}")
            return False

    def upload_run_data(self, run_info: PilatesRunInfo) -> bool:
        """
        Upload complete run data to DuckDB.

        Args:
            run_info: Complete PILATES run information

        Returns:
            bool: True if upload successful
        """
        try:
            conn = self._get_connection()

            # Start transaction
            conn.begin()

            # 1. Upload config snapshot if present
            config_snapshot_id = None
            if run_info.config_snapshot:
                config_snapshot_id = run_info.config_snapshot.get("snapshot_id")

                # Check if config snapshot already exists
                existing = conn.execute(
                    "SELECT snapshot_id FROM config_snapshots WHERE snapshot_id = ?",
                    [config_snapshot_id],
                ).fetchone()

                if not existing:
                    conn.execute(
                        """
                        INSERT INTO config_snapshots (
                            snapshot_id, created_timestamp, config_content_hash,
                            git_hashes, config_files, pilates_settings,
                            beam_config, asim_subdir, region
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        [
                            config_snapshot_id,
                            run_info.config_snapshot.get("created_timestamp"),
                            run_info.config_snapshot.get("config_content_hash"),
                            json.dumps(run_info.config_snapshot.get("git_hashes", {})),
                            json.dumps(
                                run_info.config_snapshot.get("config_files", {})
                            ),
                            json.dumps(
                                run_info.config_snapshot.get("pilates_settings", {})
                            ),
                            run_info.config_snapshot.get("beam_config"),
                            run_info.config_snapshot.get("asim_subdir"),
                            run_info.config_snapshot.get("region"),
                        ],
                    )
                    logger.info(f"Uploaded config snapshot {config_snapshot_id}")

            # 2. Upload main run record
            conn.execute(
                """
                INSERT INTO runs (
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

            # 3. Upload file records
            for file_record in run_info.file_records.values():
                conn.execute(
                    """
                    INSERT INTO file_records (
                        unique_id, run_id, openlineage_id, file_path, created_at,
                        short_name, description, year, models, producing_run_id,
                        consuming_run_ids, source_file_paths, metadata, schema, exists
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (unique_id) DO NOTHING
                """,
                    [
                        file_record.unique_id,
                        run_info.run_id,
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
                        json.dumps(file_record.metadata),
                        json.dumps(file_record.schema),
                        file_record.exists,
                    ],
                )

            # 4. Upload model runs
            for model_run in run_info.model_runs.values():
                conn.execute(
                    """
                    INSERT INTO model_runs (
                        unique_id, run_id, openlineage_id, model, year, iteration,
                        description, created_at, completed_at, status,
                        input_record_hashes, output_record_hashes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (unique_id) DO NOTHING
                """,
                    [
                        model_run.unique_id,
                        run_info.run_id,
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
                    ],
                )

            # 5. Upload OpenLineage event metadata
            for event_metadata in run_info.openlineage_event_metadata:
                conn.execute(
                    """
                    INSERT INTO openlineage_events (
                        run_id, model_run_id, event_time, event_type,
                        run_uuid, job_name
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        run_info.run_id,
                        event_metadata.model_run_id,
                        event_metadata.event_time,
                        event_metadata.event_type,
                        event_metadata.run_uuid,
                        event_metadata.job_name,
                    ],
                )

            # Commit transaction
            conn.commit()

            logger.info(f"Successfully uploaded run data for {run_info.run_id}")
            return True

        except Exception as e:
            try:
                conn.rollback()
            except:
                pass
            logger.error(f"Failed to upload run data for {run_info.run_id}: {e}")
            raise DatabaseUploadError(f"Upload failed: {e}")

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
        openlineage_id: str,
    ) -> bool:
        """
        Store raw UrbanSim data in the appropriate database table.

        Args:
            table_name: Name of the raw UrbanSim table (households, persons, jobs, etc.)
            df: DataFrame containing the raw data
            file_record_id: File record unique ID
            run_id: PILATES run ID
            openlineage_id: OpenLineage dataset ID

        Returns:
            bool: True if storage successful
        """
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._store_households_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "persons":
                return self._store_persons_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "jobs":
                return self._store_jobs_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "blocks":
                return self._store_blocks_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "buildings":
                return self._store_buildings_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "parcels":
                return self._store_parcels_raw_data(
                    conn, df, file_record_id, run_id, openlineage_id
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
        openlineage_id: str,
    ) -> bool:
        """
        Store ActivitySim data in the appropriate database table.

        Args:
            table_name: Name of the ActivitySim table (households, persons, land_use, etc.)
            df: DataFrame containing the data
            file_record_id: File record unique ID
            run_id: PILATES run ID
            openlineage_id: OpenLineage dataset ID

        Returns:
            bool: True if storage successful
        """
        try:
            conn = self._get_connection()

            if table_name == "households":
                return self._store_households_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "persons":
                return self._store_persons_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            elif table_name == "land_use":
                return self._store_land_use_data(
                    conn, df, file_record_id, run_id, openlineage_id
                )
            else:
                # Store in generic table for other ActivitySim inputs
                return self._store_generic_data(
                    conn, table_name, df, file_record_id, run_id, openlineage_id
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
        openlineage_id: str,
    ) -> bool:
        """Store households data in activitysim_households table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM activitysim_households WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("households_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_households ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM households_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store persons data in activitysim_persons table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM activitysim_persons WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("persons_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_persons ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM persons_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store land use data in activitysim_land_use table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM activitysim_land_use WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("land_use_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO activitysim_land_use ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM land_use_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store generic ActivitySim data in JSON format."""
        try:
            # Convert DataFrame to JSON for storage
            data_json = df.to_json(orient="records")

            # Insert data
            conn.execute(
                "DELETE FROM activitysim_data_generic WHERE openlineage_id = ?",
                [openlineage_id],
            )
            conn.execute(
                """
                INSERT INTO activitysim_data_generic (
                    file_record_id, run_id, openlineage_id, table_name, data_json
                ) VALUES (?, ?, ?, ?, ?)
            """,
                [file_record_id, run_id, openlineage_id, table_name, data_json],
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
        openlineage_id: str,
    ) -> bool:
        """Store raw households data in urbansim_households_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_households_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("households_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_households_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM households_raw_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store raw persons data in urbansim_persons_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_persons_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("persons_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_persons_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM persons_raw_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store raw jobs data in urbansim_jobs_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

            # Handle column mapping and missing columns for raw data
            expected_cols = ["job_id", "building_id", "sector_id", "home_based_status"]
            for col in expected_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # Select only the columns we need in the correct order
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_jobs_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("jobs_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_jobs_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM jobs_raw_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store raw blocks data in urbansim_blocks_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_blocks_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("blocks_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_blocks_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM blocks_raw_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store raw buildings data in urbansim_buildings_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_buildings_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("buildings_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_buildings_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM buildings_raw_temp_data
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
        openlineage_id: str,
    ) -> bool:
        """Store raw parcels data in urbansim_parcels_raw table using efficient bulk loading."""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy["file_record_id"] = file_record_id
            df_copy["run_id"] = run_id
            df_copy["openlineage_id"] = openlineage_id

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
            insert_cols = ["file_record_id", "run_id", "openlineage_id"] + expected_cols
            df_insert = df_copy[insert_cols]

            # Delete existing data for this openlineage_id
            conn.execute(
                "DELETE FROM urbansim_parcels_raw WHERE openlineage_id = ?",
                [openlineage_id],
            )

            # Bulk insert using DuckDB's DataFrame integration
            conn.register("parcels_raw_temp_data", df_insert)
            conn.execute(
                f"""
                INSERT INTO urbansim_parcels_raw ({', '.join(insert_cols)})
                SELECT {', '.join(insert_cols)} FROM parcels_raw_temp_data
            """
            )
            conn.unregister("parcels_raw_temp_data")

            logger.info(f"Bulk stored {len(df_copy)} raw parcel records")
            return True

        except Exception as e:
            logger.error(f"Failed to store raw parcels data: {e}")
            return False

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

    def close(self):
        """Close DuckDB connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("DuckDB connection closed")
