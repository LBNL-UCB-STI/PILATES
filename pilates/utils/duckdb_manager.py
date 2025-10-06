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

            # Create sequence for openlineage_events
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS openlineage_events_id_seq START 1"
            )

            # Create openlineage_events table - lightweight event metadata
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS openlineage_events (
                    id INTEGER PRIMARY KEY DEFAULT nextval('openlineage_events_id_seq'),
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

            # Create schema_version table for tracking database schema evolution
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description VARCHAR,
                    pilates_version VARCHAR
                )
            """
            )

            # Record current schema version
            current_version = 1
            pilates_version = "1.0.0"  # Update this with actual PILATES version
            conn.execute(
                """
                INSERT INTO schema_version (version, description, pilates_version)
                SELECT ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = ?)
            """,
                [
                    current_version,
                    "Initial schema with metadata, provenance, and dual storage support",
                    pilates_version,
                    current_version,
                ],
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

            # Add documentation comments to schema
            self._add_schema_documentation()

            # Create summary views for easy querying
            self.create_summary_views()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB database: {e}")
            return False

    def _add_schema_documentation(self) -> None:
        """
        Add SQL COMMENT statements documenting all tables and columns.

        This provides in-database documentation that can be queried and
        is preserved during database exports/imports.
        """
        try:
            conn = self._get_connection()

            # ============================================================
            # METADATA TABLES
            # ============================================================

            # runs table
            conn.execute("COMMENT ON TABLE runs IS 'Top-level PILATES simulation runs with configuration and execution metadata'")
            conn.execute("COMMENT ON COLUMN runs.run_id IS 'Unique identifier for this PILATES run (UUID format)'")
            conn.execute("COMMENT ON COLUMN runs.created_at IS 'Timestamp when the run was initiated'")
            conn.execute("COMMENT ON COLUMN runs.start_year IS 'First simulation year in this run'")
            conn.execute("COMMENT ON COLUMN runs.end_year IS 'Final simulation year in this run'")
            conn.execute("COMMENT ON COLUMN runs.models_used IS 'Array of model names executed in this run (e.g., urbansim, activitysim, beam, atlas)'")
            conn.execute("COMMENT ON COLUMN runs.settings_hash IS 'SHA256 hash of settings dictionary for deduplication and comparison'")
            conn.execute("COMMENT ON COLUMN runs.code_version IS 'Git commit hash of PILATES code used for this run'")
            conn.execute("COMMENT ON COLUMN runs.hostname IS 'Machine/server where this run was executed'")
            conn.execute("COMMENT ON COLUMN runs.config_snapshot_id IS 'Foreign key to config_snapshots table containing complete configuration state'")
            conn.execute("COMMENT ON COLUMN runs.config_content_hash IS 'Hash of complete configuration snapshot including all config files'")
            conn.execute("COMMENT ON COLUMN runs.uploaded_at IS 'Timestamp when run metadata was uploaded to database'")

            # config_snapshots table
            conn.execute("COMMENT ON TABLE config_snapshots IS 'Complete configuration snapshots for reproducibility, including all config files and git hashes'")
            conn.execute("COMMENT ON COLUMN config_snapshots.snapshot_id IS 'Unique identifier for this configuration snapshot (UUID)'")
            conn.execute("COMMENT ON COLUMN config_snapshots.created_timestamp IS 'When this configuration snapshot was created'")
            conn.execute("COMMENT ON COLUMN config_snapshots.config_content_hash IS 'SHA256 hash of complete configuration for deduplication'")
            conn.execute("COMMENT ON COLUMN config_snapshots.git_hashes IS 'JSON object with git commit hashes for PILATES and each model component (beam, activitysim, etc.)'")
            conn.execute("COMMENT ON COLUMN config_snapshots.config_files IS 'JSON object containing complete contents of all configuration files (BEAM .conf, ActivitySim .yaml, etc.)'")
            conn.execute("COMMENT ON COLUMN config_snapshots.pilates_settings IS 'JSON object with relevant PILATES settings from settings.yaml'")
            conn.execute("COMMENT ON COLUMN config_snapshots.beam_config IS 'Name of BEAM configuration file used (e.g., sfbay-pilates-base-omx.conf)'")
            conn.execute("COMMENT ON COLUMN config_snapshots.asim_subdir IS 'ActivitySim configuration subdirectory used'")
            conn.execute("COMMENT ON COLUMN config_snapshots.region IS 'Geographic region for this configuration (e.g., sfbay, austin, seattle)'")

            # file_records table
            conn.execute("COMMENT ON TABLE file_records IS 'Individual datasets with complete provenance lineage, linking files across model stages'")
            conn.execute("COMMENT ON COLUMN file_records.unique_id IS 'File content hash (SHA256) serving as unique identifier'")
            conn.execute("COMMENT ON COLUMN file_records.run_id IS 'Foreign key to runs table indicating which run produced/used this file'")
            conn.execute("COMMENT ON COLUMN file_records.openlineage_id IS 'OpenLineage dataset UUID for cross-system lineage tracking'")
            conn.execute("COMMENT ON COLUMN file_records.file_path IS 'Absolute or relative path to the file on disk'")
            conn.execute("COMMENT ON COLUMN file_records.created_at IS 'Timestamp when file was created or first recorded'")
            conn.execute("COMMENT ON COLUMN file_records.short_name IS 'Human-readable identifier for this dataset (e.g., urbansim_h5, activitysim_beam_plans)'")
            conn.execute("COMMENT ON COLUMN file_records.description IS 'Detailed description of file contents and purpose'")
            conn.execute("COMMENT ON COLUMN file_records.year IS 'Simulation year this file corresponds to (NULL if not year-specific)'")
            conn.execute("COMMENT ON COLUMN file_records.models IS 'Array of model names that have touched/processed this file'")
            conn.execute("COMMENT ON COLUMN file_records.producing_run_id IS 'OpenLineage ID of the model run that produced this file'")
            conn.execute("COMMENT ON COLUMN file_records.consuming_run_ids IS 'Array of OpenLineage IDs for model runs that consumed this file'")
            conn.execute("COMMENT ON COLUMN file_records.source_file_paths IS 'Array of file paths that were inputs to create this file (immediate lineage)'")
            conn.execute("COMMENT ON COLUMN file_records.metadata IS 'JSON object with additional file metadata (size, format, row count, etc.)'")
            conn.execute("COMMENT ON COLUMN file_records.schema IS 'JSON array describing file schema (column names, types, descriptions)'")
            conn.execute("COMMENT ON COLUMN file_records.exists IS 'Boolean indicating if file still exists on disk'")

            # model_runs table
            conn.execute("COMMENT ON TABLE model_runs IS 'Individual model execution records tracking each model component run with inputs/outputs'")
            conn.execute("COMMENT ON COLUMN model_runs.unique_id IS 'Unique identifier for this model execution (hash of model+year+iteration+timestamp)'")
            conn.execute("COMMENT ON COLUMN model_runs.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN model_runs.openlineage_id IS 'OpenLineage run UUID for this specific model execution'")
            conn.execute("COMMENT ON COLUMN model_runs.model IS 'Model component name (urbansim, activitysim, beam, atlas, etc.)'")
            conn.execute("COMMENT ON COLUMN model_runs.year IS 'Simulation year for this model execution'")
            conn.execute("COMMENT ON COLUMN model_runs.iteration IS 'Inner iteration number (for supply-demand loops)'")
            conn.execute("COMMENT ON COLUMN model_runs.description IS 'Human-readable description of this model run stage'")
            conn.execute("COMMENT ON COLUMN model_runs.created_at IS 'Timestamp when model execution started'")
            conn.execute("COMMENT ON COLUMN model_runs.completed_at IS 'Timestamp when model execution finished'")
            conn.execute("COMMENT ON COLUMN model_runs.status IS 'Execution status: completed, failed, or running'")
            conn.execute("COMMENT ON COLUMN model_runs.input_record_hashes IS 'Array of file_records.unique_id values for input files'")
            conn.execute("COMMENT ON COLUMN model_runs.output_record_hashes IS 'Array of file_records.unique_id values for output files'")

            # openlineage_events table
            conn.execute("COMMENT ON TABLE openlineage_events IS 'Lightweight OpenLineage event metadata for integration with external lineage systems'")
            conn.execute("COMMENT ON COLUMN openlineage_events.id IS 'Auto-incrementing event ID'")
            conn.execute("COMMENT ON COLUMN openlineage_events.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN openlineage_events.model_run_id IS 'Foreign key to specific model execution'")
            conn.execute("COMMENT ON COLUMN openlineage_events.event_time IS 'Timestamp when event occurred'")
            conn.execute("COMMENT ON COLUMN openlineage_events.event_type IS 'Event type: START, COMPLETE, or FAIL'")
            conn.execute("COMMENT ON COLUMN openlineage_events.run_uuid IS 'OpenLineage run UUID (matches model_runs.openlineage_id)'")
            conn.execute("COMMENT ON COLUMN openlineage_events.job_name IS 'Formatted job name with model, year, and iteration'")

            # ============================================================
            # RAW URBANSIM DATA TABLES
            # ============================================================

            # urbansim_households_raw table
            conn.execute("COMMENT ON TABLE urbansim_households_raw IS 'Raw household data directly from UrbanSim H5 outputs, preserving original structure'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.file_record_id IS 'Foreign key to file_records for provenance tracking'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.openlineage_id IS 'OpenLineage dataset ID for this data'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.household_id IS 'Unique household identifier from UrbanSim'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.building_id IS 'Building where household resides (links to buildings table)'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.persons IS 'Number of people in household'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.income IS 'Annual household income in dollars'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.cars IS 'Number of vehicles owned by household'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.block_id IS 'Census block ID where household resides'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.age_of_head IS 'Age of household head'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.children IS 'Number of children in household'")
            conn.execute("COMMENT ON COLUMN urbansim_households_raw.workers IS 'Number of workers in household'")

            # urbansim_persons_raw table
            conn.execute("COMMENT ON TABLE urbansim_persons_raw IS 'Raw person-level data from UrbanSim H5 outputs'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.file_record_id IS 'Foreign key to file_records for provenance tracking'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.person_id IS 'Unique person identifier from UrbanSim'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.household_id IS 'Household this person belongs to (links to households table)'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.age IS 'Person age in years'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.worker IS 'Worker status (0=non-worker, 1=worker)'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.student IS 'Student status (0=non-student, 1=student)'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.race_id IS 'Race/ethnicity category identifier'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.sex IS 'Sex (1=male, 2=female)'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.work_zone_id IS 'TAZ where person works'")
            conn.execute("COMMENT ON COLUMN urbansim_persons_raw.school_zone_id IS 'TAZ where person attends school'")

            # urbansim_jobs_raw table
            conn.execute("COMMENT ON TABLE urbansim_jobs_raw IS 'Raw employment/job data from UrbanSim H5 outputs'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.job_id IS 'Unique job identifier'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.building_id IS 'Building where job is located'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.sector_id IS 'Employment sector/industry category'")
            conn.execute("COMMENT ON COLUMN urbansim_jobs_raw.home_based_status IS 'Whether job is home-based (0=no, 1=yes)'")

            # urbansim_blocks_raw table
            conn.execute("COMMENT ON TABLE urbansim_blocks_raw IS 'Raw census block geographic data from UrbanSim'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.block_id IS 'Census block FIPS code'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.block_group_id IS 'Census block group FIPS code'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.zone_id IS 'Zone identifier (may be TAZ or other)'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.taz_zone_id IS 'Traffic Analysis Zone identifier'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.square_meters_land IS 'Land area in square meters'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.x IS 'X coordinate in local projection'")
            conn.execute("COMMENT ON COLUMN urbansim_blocks_raw.y IS 'Y coordinate in local projection'")

            # urbansim_buildings_raw table
            conn.execute("COMMENT ON TABLE urbansim_buildings_raw IS 'Raw building data from UrbanSim including built environment characteristics'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.building_id IS 'Unique building identifier'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.parcel_id IS 'Parcel where building is located'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.building_type_id IS 'Building type category (residential, commercial, etc.)'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.sqft IS 'Building square footage'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.year_built IS 'Year building was constructed'")
            conn.execute("COMMENT ON COLUMN urbansim_buildings_raw.stories IS 'Number of stories/floors'")

            # urbansim_parcels_raw table
            conn.execute("COMMENT ON TABLE urbansim_parcels_raw IS 'Raw parcel/lot data from UrbanSim land use model'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.parcel_id IS 'Unique parcel identifier'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.zone_id IS 'Zone where parcel is located'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.land_value IS 'Assessed land value in dollars'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.total_sqft IS 'Total parcel area in square feet'")
            conn.execute("COMMENT ON COLUMN urbansim_parcels_raw.county_id IS 'County FIPS code'")

            # ============================================================
            # PROCESSED ACTIVITYSIM DATA TABLES
            # ============================================================

            # activitysim_households table
            conn.execute("COMMENT ON TABLE activitysim_households IS 'Processed household data ready for ActivitySim consumption, transformed from UrbanSim outputs'")
            conn.execute("COMMENT ON COLUMN activitysim_households.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN activitysim_households.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN activitysim_households.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN activitysim_households.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN activitysim_households.household_id IS 'Unique household identifier (matches UrbanSim)'")
            conn.execute("COMMENT ON COLUMN activitysim_households.TAZ IS 'Traffic Analysis Zone where household resides (ActivitySim format)'")
            conn.execute("COMMENT ON COLUMN activitysim_households.persons IS 'Number of persons (mapped from UrbanSim PERSONS column)'")
            conn.execute("COMMENT ON COLUMN activitysim_households.income IS 'Annual household income in dollars'")
            conn.execute("COMMENT ON COLUMN activitysim_households.cars IS 'Vehicle ownership (mapped from UrbanSim auto_ownership)'")
            conn.execute("COMMENT ON COLUMN activitysim_households.HHT IS 'Household type category for ActivitySim'")
            conn.execute("COMMENT ON COLUMN activitysim_households.workers IS 'Number of workers (mapped from UrbanSim num_workers)'")

            # activitysim_persons table
            conn.execute("COMMENT ON TABLE activitysim_persons IS 'Processed person-level data for ActivitySim, including synthetic person attributes'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.person_id IS 'Unique person identifier (matches UrbanSim)'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.household_id IS 'Household this person belongs to'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.TAZ IS 'Home TAZ (matches household TAZ)'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.age IS 'Person age in years'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.worker IS 'Worker status for ActivitySim'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.student IS 'Student status for ActivitySim'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.ptype IS 'Person type category (full-time worker, part-time, student, etc.)'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.pemploy IS 'Employment status category'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.pstudent IS 'Student enrollment category'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.member_id IS 'Person number within household (mapped from UrbanSim PNUM)'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.workplace_taz IS 'TAZ of workplace location'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.school_taz IS 'TAZ of school location'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.home_x IS 'X coordinate of home location'")
            conn.execute("COMMENT ON COLUMN activitysim_persons.home_y IS 'Y coordinate of home location'")

            # activitysim_land_use table
            conn.execute("COMMENT ON TABLE activitysim_land_use IS 'Processed land use/zonal data for ActivitySim including demographics, employment, and accessibility'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.TAZ IS 'Traffic Analysis Zone identifier (mapped from UrbanSim ZONE)'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.TOTPOP IS 'Total population in TAZ'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.TOTHH IS 'Total households in TAZ'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.TOTEMP IS 'Total employment in TAZ'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.TOTACRE IS 'Total acres in TAZ'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.area_type IS 'Area type category (urban core, suburban, rural, etc.)'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.employment_density IS 'Jobs per acre'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.pop_density IS 'Population per acre'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.hh_density IS 'Households per acre'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE0004 IS 'Population aged 0-4 years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE0519 IS 'Population aged 5-19 years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE2044 IS 'Population aged 20-44 years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE4564 IS 'Population aged 45-64 years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE64P IS 'Population aged 65+ years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGE62P IS 'Population aged 62+ years'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HHINCQ1 IS 'Households in income quartile 1 (lowest)'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HHINCQ2 IS 'Households in income quartile 2'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HHINCQ3 IS 'Households in income quartile 3'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HHINCQ4 IS 'Households in income quartile 4 (highest)'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.EMPRES IS 'Residential employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.RETEMPN IS 'Retail employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.FPSEMPN IS 'Financial and professional services employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HEREMPN IS 'Health, education, and recreational employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.AGREMPN IS 'Agricultural and natural resources employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.MWTEMPN IS 'Manufacturing, wholesale, and transportation employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.OTHEMPN IS 'Other employment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.HSENROLL IS 'High school enrollment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.COLLFTE IS 'College full-time enrollment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.COLLPTE IS 'College part-time enrollment'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.PRKCST IS 'Daily parking cost in dollars'")
            conn.execute("COMMENT ON COLUMN activitysim_land_use.OPRKCST IS 'Off-peak parking cost in dollars'")

            # activitysim_data_generic table
            conn.execute("COMMENT ON TABLE activitysim_data_generic IS 'Generic storage for additional ActivitySim input tables (accessibility, skims, etc.) in JSON format'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.id IS 'Auto-incrementing database row ID'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.file_record_id IS 'Foreign key to file_records'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.run_id IS 'Foreign key to parent PILATES run'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.openlineage_id IS 'OpenLineage dataset ID'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.table_name IS 'Name of the ActivitySim table stored here'")
            conn.execute("COMMENT ON COLUMN activitysim_data_generic.data_json IS 'Complete table data serialized as JSON for flexible schema support'")

            logger.info("Schema documentation comments added successfully")

        except Exception as e:
            # Log warning but don't fail initialization if comments fail
            logger.warning(f"Failed to add schema documentation: {e}")

    def create_summary_views(self) -> bool:
        """
        Create simplified database views for non-technical users.

        These views provide easy-to-query summaries without needing to
        understand the full database schema or write complex joins.

        Returns:
            bool: True if views created successfully
        """
        try:
            conn = self._get_connection()

            # View 1: Run Summary
            # Simple overview of all simulation runs
            conn.execute("""
                CREATE OR REPLACE VIEW run_summary AS
                SELECT
                    r.run_id,
                    r.created_at as run_date,
                    r.start_year,
                    r.end_year,
                    r.end_year - r.start_year + 1 as num_years,
                    array_to_string(r.models_used, ', ') as models,
                    cs.region,
                    cs.beam_config as beam_configuration,
                    r.hostname as server,
                    (SELECT COUNT(*) FROM file_records WHERE file_records.run_id = r.run_id) as file_count,
                    (SELECT COUNT(*) FROM model_runs WHERE model_runs.run_id = r.run_id) as model_execution_count,
                    r.code_version
                FROM runs r
                LEFT JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
                ORDER BY r.created_at DESC
            """)

            conn.execute("""
                COMMENT ON VIEW run_summary IS
                'Simplified summary of all PILATES runs with key metadata and counts'
            """)

            # View 2: Data Lineage Summary
            # Easy view of file relationships
            conn.execute("""
                CREATE OR REPLACE VIEW data_lineage_summary AS
                SELECT
                    f.short_name as dataset_name,
                    f.description,
                    f.year,
                    f.file_path,
                    array_to_string(f.models, ' → ') as processing_chain,
                    array_length(f.source_file_paths, 1) as num_input_files,
                    r.run_id,
                    r.created_at as run_date,
                    cs.region
                FROM file_records f
                JOIN runs r ON f.run_id = r.run_id
                LEFT JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
                ORDER BY f.created_at DESC
            """)

            conn.execute("""
                COMMENT ON VIEW data_lineage_summary IS
                'Complete data lineage showing how files were processed across model stages'
            """)

            # View 3: Model Performance Summary
            # How long each model takes to run
            conn.execute("""
                CREATE OR REPLACE VIEW model_performance_summary AS
                SELECT
                    mr.model,
                    mr.year,
                    mr.run_id,
                    COUNT(*) as execution_count,
                    AVG(EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0) as avg_runtime_minutes,
                    MIN(EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0) as min_runtime_minutes,
                    MAX(EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0) as max_runtime_minutes,
                    SUM(CASE WHEN mr.status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                    SUM(CASE WHEN mr.status = 'failed' THEN 1 ELSE 0 END) as failed_runs
                FROM model_runs mr
                WHERE mr.completed_at IS NOT NULL AND mr.created_at IS NOT NULL
                GROUP BY mr.model, mr.year, mr.run_id
                ORDER BY mr.model, mr.year
            """)

            conn.execute("""
                COMMENT ON VIEW model_performance_summary IS
                'Model execution performance metrics including runtime and success rates'
            """)

            # View 4: Household Demographics Summary
            # Aggregated household statistics per run
            conn.execute("""
                CREATE OR REPLACE VIEW household_demographics_summary AS
                SELECT
                    h.run_id,
                    COUNT(DISTINCT h.household_id) as total_households,
                    AVG(h.income) as avg_income,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY h.income) as median_income,
                    AVG(h.cars) as avg_vehicles_per_hh,
                    AVG(h.persons) as avg_household_size,
                    AVG(h.workers) as avg_workers_per_hh,
                    SUM(CASE WHEN h.cars = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_zero_vehicle,
                    SUM(CASE WHEN h.cars >= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_multi_vehicle
                FROM activitysim_households h
                GROUP BY h.run_id
            """)

            conn.execute("""
                COMMENT ON VIEW household_demographics_summary IS
                'Aggregated household demographic statistics by run'
            """)

            # View 5: TAZ Level Summary
            # Key metrics at the TAZ level for mapping and analysis
            conn.execute("""
                CREATE OR REPLACE VIEW taz_summary AS
                SELECT
                    lu.run_id,
                    lu.TAZ,
                    lu.TOTPOP as population,
                    lu.TOTHH as households,
                    lu.TOTEMP as employment,
                    lu.TOTACRE as acres,
                    ROUND(lu.pop_density, 2) as pop_per_acre,
                    ROUND(lu.employment_density, 2) as jobs_per_acre,
                    ROUND(lu.hh_density, 2) as hh_per_acre,
                    lu.area_type,
                    CASE
                        WHEN lu.TOTHH > 0 THEN ROUND(lu.TOTEMP * 1.0 / lu.TOTHH, 2)
                        ELSE NULL
                    END as jobs_per_household,
                    ROUND(lu.PRKCST, 2) as daily_parking_cost
                FROM activitysim_land_use lu
                WHERE lu.TOTPOP > 0  -- Filter to populated TAZs only
            """)

            conn.execute("""
                COMMENT ON VIEW taz_summary IS
                'Traffic Analysis Zone level summary statistics for mapping and spatial analysis'
            """)

            # View 6: Run Comparison Helper
            # Makes it easy to compare key metrics across runs
            conn.execute("""
                CREATE OR REPLACE VIEW run_comparison AS
                SELECT
                    r.run_id,
                    r.created_at as run_date,
                    cs.region,
                    r.start_year,
                    r.end_year,
                    (SELECT COUNT(DISTINCT h.household_id) FROM activitysim_households h WHERE h.run_id = r.run_id) as total_households,
                    (SELECT AVG(h.income) FROM activitysim_households h WHERE h.run_id = r.run_id) as avg_income,
                    (SELECT AVG(h.cars) FROM activitysim_households h WHERE h.run_id = r.run_id) as avg_vehicles,
                    (SELECT SUM(lu.TOTPOP) FROM activitysim_land_use lu WHERE lu.run_id = r.run_id) as total_population,
                    (SELECT SUM(lu.TOTEMP) FROM activitysim_land_use lu WHERE lu.run_id = r.run_id) as total_employment
                FROM runs r
                LEFT JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
                ORDER BY r.created_at DESC
            """)

            conn.execute("""
                COMMENT ON VIEW run_comparison IS
                'Side-by-side comparison of key metrics across different runs'
            """)

            # View 7: Employment by Sector
            # Summarize employment distribution
            conn.execute("""
                CREATE OR REPLACE VIEW employment_by_sector AS
                SELECT
                    lu.run_id,
                    SUM(lu.EMPRES) as residential,
                    SUM(lu.RETEMPN) as retail,
                    SUM(lu.FPSEMPN) as financial_professional,
                    SUM(lu.HEREMPN) as health_education_recreation,
                    SUM(lu.AGREMPN) as agricultural,
                    SUM(lu.MWTEMPN) as manufacturing_warehouse_transport,
                    SUM(lu.OTHEMPN) as other,
                    SUM(lu.TOTEMP) as total_employment
                FROM activitysim_land_use lu
                GROUP BY lu.run_id
            """)

            conn.execute("""
                COMMENT ON VIEW employment_by_sector IS
                'Employment distribution by industry sector'
            """)

            # View 8: Recent Activity Log
            # Show recent model executions for monitoring
            conn.execute("""
                CREATE OR REPLACE VIEW recent_activity AS
                SELECT
                    mr.created_at as execution_time,
                    mr.model,
                    mr.year,
                    mr.iteration,
                    mr.status,
                    ROUND(EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0, 1) as runtime_minutes,
                    r.run_id,
                    cs.region
                FROM model_runs mr
                JOIN runs r ON mr.run_id = r.run_id
                LEFT JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
                WHERE mr.created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
                ORDER BY mr.created_at DESC
            """)

            conn.execute("""
                COMMENT ON VIEW recent_activity IS
                'Recent model executions from the last 7 days for monitoring purposes'
            """)

            logger.info("Summary views created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create summary views: {e}")
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
                "recommendations": []
            }

            # Check 1: Orphaned file records (run_id doesn't exist in runs)
            orphaned = conn.execute("""
                SELECT COUNT(*) as count
                FROM file_records fr
                LEFT JOIN runs r ON fr.run_id = r.run_id
                WHERE r.run_id IS NULL
            """).fetchone()[0]

            if orphaned > 0:
                report["errors"].append(f"{orphaned} file records reference non-existent runs")

            # Check 2: Model runs without completion time
            incomplete = conn.execute("""
                SELECT COUNT(*) as count
                FROM model_runs
                WHERE status = 'completed' AND completed_at IS NULL
            """).fetchone()[0]

            if incomplete > 0:
                report["warnings"].append(f"{incomplete} model runs marked complete but missing completion time")

            # Check 3: Duplicate household IDs within a run
            dupes = conn.execute("""
                SELECT run_id, household_id, COUNT(*) as dup_count
                FROM activitysim_households
                GROUP BY run_id, household_id
                HAVING COUNT(*) > 1
            """).fetchall()

            if dupes:
                report["errors"].append(f"{len(dupes)} duplicate household_id values found")

            # Check 4: Missing foreign key relationships
            missing_hh = conn.execute("""
                SELECT COUNT(*) as count
                FROM activitysim_persons p
                LEFT JOIN activitysim_households h
                    ON p.household_id = h.household_id AND p.run_id = h.run_id
                WHERE h.household_id IS NULL
            """).fetchone()[0]

            if missing_hh > 0:
                report["errors"].append(f"{missing_hh} persons reference non-existent households")

            # Statistics
            report["statistics"] = {
                "total_runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "total_file_records": conn.execute("SELECT COUNT(*) FROM file_records").fetchone()[0],
                "total_model_runs": conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0],
                "failed_model_runs": conn.execute("SELECT COUNT(*) FROM model_runs WHERE status = 'failed'").fetchone()[0],
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
                report["recommendations"].append("Address data integrity errors before using database for analysis")
            if report["statistics"].get("failed_model_runs", 0) > 0:
                report["recommendations"].append("Investigate failed model runs using recent_activity view")

            return report

        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return {
                "generated_at": datetime.now().isoformat(),
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "statistics": {},
                "recommendations": []
            }

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
        self, output_path: str, schema_df: pd.DataFrame, fk_df: pd.DataFrame, row_counts: Dict
    ) -> bool:
        """Export data dictionary as Markdown."""
        try:
            with open(output_path, "w") as f:
                f.write("# PILATES Database Data Dictionary\n\n")
                f.write("Auto-generated schema documentation for the PILATES database.\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## Table of Contents\n\n")

                # Group by table category
                metadata_tables = ["runs", "config_snapshots", "file_records", "model_runs", "openlineage_events"]
                urbansim_tables = [t for t in schema_df["table_name"].unique() if "urbansim" in t and "_raw" in t]
                activitysim_tables = [t for t in schema_df["table_name"].unique() if "activitysim" in t]

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

                        f.write(f"| {col_name} | {col_type} | {nullable} | {col_desc} |\n")

                    # Foreign keys
                    if not fk_df.empty:
                        table_fks = fk_df[fk_df["fk_table"] == table_name]
                        if not table_fks.empty:
                            f.write("\n**Foreign Keys:**\n\n")
                            for _, fk in table_fks.iterrows():
                                f.write(f"- `{fk['fk_columns']}` → `{fk['pk_table']}.{fk['pk_columns']}`\n")

                    f.write("\n---\n\n")

                logger.info(f"Markdown data dictionary exported to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to export markdown dictionary: {e}")
            return False

    def _export_json_dictionary(
        self, output_path: str, schema_df: pd.DataFrame, fk_df: pd.DataFrame, row_counts: Dict
    ) -> bool:
        """Export data dictionary as JSON."""
        try:
            tables = {}

            for table_name in schema_df["table_name"].unique():
                table_data = schema_df[schema_df["table_name"] == table_name]

                columns = []
                for _, row in table_data.iterrows():
                    columns.append({
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"],
                        "description": row["column_comment"]
                    })

                # Get foreign keys for this table
                fks = []
                if not fk_df.empty:
                    table_fks = fk_df[fk_df["fk_table"] == table_name]
                    for _, fk_row in table_fks.iterrows():
                        fks.append({
                            "column": fk_row["fk_columns"],
                            "references_table": fk_row["pk_table"],
                            "references_column": fk_row["pk_columns"]
                        })

                tables[table_name] = {
                    "description": table_data.iloc[0]["table_comment"],
                    "row_count": row_counts.get(table_name),
                    "columns": columns,
                    "foreign_keys": fks
                }

            dictionary = {
                "database": "PILATES",
                "generated_at": datetime.now().isoformat(),
                "tables": tables
            }

            with open(output_path, "w") as f:
                json.dump(dictionary, f, indent=2, default=str)

            logger.info(f"JSON data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export JSON dictionary: {e}")
            return False

    def _export_csv_dictionary(
        self, output_path: str, schema_df: pd.DataFrame, fk_df: pd.DataFrame, row_counts: Dict
    ) -> bool:
        """Export data dictionary as CSV."""
        try:
            # Add row counts to schema
            schema_with_counts = schema_df.copy()
            schema_with_counts["row_count"] = schema_with_counts["table_name"].map(row_counts)

            # Export to CSV
            schema_with_counts.to_csv(output_path, index=False)

            logger.info(f"CSV data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV dictionary: {e}")
            return False

    def _export_html_dictionary(
        self, output_path: str, schema_df: pd.DataFrame, fk_df: pd.DataFrame, row_counts: Dict
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
            html.append("h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }")
            html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
            html.append("th { background-color: #3498db; color: white; padding: 12px; text-align: left; }")
            html.append("td { border: 1px solid #ddd; padding: 10px; }")
            html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
            html.append(".table-desc { font-style: italic; color: #555; margin: 10px 0; }")
            html.append(".row-count { color: #16a085; font-weight: bold; }")
            html.append(".fk-section { margin-top: 15px; padding: 10px; background-color: #ecf0f1; }")
            html.append("</style></head><body>")

            html.append("<h1>PILATES Database Data Dictionary</h1>")
            html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

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
                    html.append(f'<p class="row-count">Row Count: {row_counts[table_name]:,}</p>')

                html.append("<table>")
                html.append("<tr><th>Column</th><th>Type</th><th>Nullable</th><th>Description</th></tr>")

                for _, row in table_data.iterrows():
                    html.append("<tr>")
                    html.append(f"<td><strong>{row['column_name']}</strong></td>")
                    html.append(f"<td>{row['data_type']}</td>")
                    nullable = "Yes" if row['is_nullable'] == 'YES' else "No"
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
                            html.append(f"<li><code>{fk['fk_columns']}</code> → <code>{fk['pk_table']}.{fk['pk_columns']}</code></li>")
                        html.append("</ul></div>")

            html.append("</body></html>")

            with open(output_path, "w") as f:
                f.write("\n".join(html))

            logger.info(f"HTML data dictionary exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export HTML dictionary: {e}")
            return False

    def close(self):
        """Close DuckDB connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("DuckDB connection closed")
