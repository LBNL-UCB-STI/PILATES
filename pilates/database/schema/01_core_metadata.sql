-- Core metadata tables for PILATES database

-- Main run metadata
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
);
COMMENT ON TABLE runs IS 'Top-level PILATES simulation runs with configuration and execution metadata';
COMMENT ON COLUMN runs.run_id IS 'Unique identifier for this PILATES run (UUID format)';
COMMENT ON COLUMN runs.created_at IS 'Timestamp when the run was initiated';
COMMENT ON COLUMN runs.start_year IS 'First simulation year in this run';
COMMENT ON COLUMN runs.end_year IS 'Final simulation year in this run';
COMMENT ON COLUMN runs.models_used IS 'Array of model names executed in this run (e.g., urbansim, activitysim, beam, atlas)';
COMMENT ON COLUMN runs.settings_hash IS 'SHA256 hash of settings dictionary for deduplication and comparison';
COMMENT ON COLUMN runs.code_version IS 'Git commit hash of PILATES code used for this run';
COMMENT ON COLUMN runs.hostname IS 'Machine/server where this run was executed';
COMMENT ON COLUMN runs.config_snapshot_id IS 'Foreign key to config_snapshots table containing complete configuration state';
COMMENT ON COLUMN runs.config_content_hash IS 'Hash of complete configuration snapshot including all config files';
COMMENT ON COLUMN runs.uploaded_at IS 'Timestamp when run metadata was uploaded to database';

-- Configuration data
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
);
COMMENT ON TABLE config_snapshots IS 'Complete configuration snapshots for reproducibility, including all config files and git hashes';
COMMENT ON COLUMN config_snapshots.snapshot_id IS 'Unique identifier for this configuration snapshot (UUID)';
COMMENT ON COLUMN config_snapshots.created_timestamp IS 'When this configuration snapshot was created';
COMMENT ON COLUMN config_snapshots.config_content_hash IS 'SHA256 hash of complete configuration for deduplication';
COMMENT ON COLUMN config_snapshots.git_hashes IS 'JSON object with git commit hashes for PILATES and each model component (beam, activitysim, etc.)';
COMMENT ON COLUMN config_snapshots.config_files IS 'JSON object containing complete contents of all configuration files (BEAM .conf, ActivitySim .yaml, etc.)';
COMMENT ON COLUMN config_snapshots.pilates_settings IS 'JSON object with relevant PILATES settings from settings.yaml';
COMMENT ON COLUMN config_snapshots.beam_config IS 'Name of BEAM configuration file used (e.g., sfbay-pilates-base-omx.conf)';
COMMENT ON COLUMN config_snapshots.asim_subdir IS 'ActivitySim configuration subdirectory used';
COMMENT ON COLUMN config_snapshots.region IS 'Geographic region for this configuration (e.g., sfbay, austin, seattle)';

-- Individual model execution info
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
    metadata JSON,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
COMMENT ON TABLE model_runs IS 'Individual model execution records tracking each model component run with inputs/outputs';
COMMENT ON COLUMN model_runs.unique_id IS 'Unique identifier for this model execution (hash of model+year+iteration+timestamp)';
COMMENT ON COLUMN model_runs.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN model_runs.openlineage_id IS 'OpenLineage run UUID for this specific model execution';
COMMENT ON COLUMN model_runs.model IS 'Model component name (urbansim, activitysim, beam, atlas, etc.)';
COMMENT ON COLUMN model_runs.year IS 'Simulation year for this model execution';
COMMENT ON COLUMN model_runs.iteration IS 'Inner iteration number (for supply-demand loops)';
COMMENT ON COLUMN model_runs.description IS 'Human-readable description of this model run stage';
COMMENT ON COLUMN model_runs.created_at IS 'Timestamp when model execution started';
COMMENT ON COLUMN model_runs.completed_at IS 'Timestamp when model execution finished';
COMMENT ON COLUMN model_runs.status IS 'Execution status: completed, failed, or running';
COMMENT ON COLUMN model_runs.input_record_hashes IS 'Array of file_records.unique_id values for input files';
COMMENT ON COLUMN model_runs.output_record_hashes IS 'Array of file_records.unique_id values for output files';
COMMENT ON COLUMN model_runs.metadata IS 'JSON object with runtime execution metadata including container command, runtime parameters, and other execution details';

-- Dataset information
CREATE TABLE IF NOT EXISTS file_records (
    unique_id VARCHAR PRIMARY KEY, -- File content hash
    record_type VARCHAR NOT NULL, -- 'file', 'h5_container', 'h5_table'
    run_id VARCHAR NOT NULL,
    model_run_id VARCHAR, -- Foreign key to model_runs table
    openlineage_id VARCHAR UNIQUE,
    file_path VARCHAR,
    created_at TIMESTAMP,
    short_name VARCHAR NOT NULL,
    description VARCHAR,
    year INTEGER,
    iteration INTEGER, -- Simulation iteration this file corresponds to
    sub_iteration INTEGER, -- Sub-iteration number (if applicable)
    models VARCHAR[], -- Models that touched this file
    producing_run_id VARCHAR,
    consuming_run_ids VARCHAR[], -- Multiple runs can consume
    source_file_paths VARCHAR[],
    metadata JSON,
    schema JSON,
    exists BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(unique_id),
    UNIQUE (run_id, year, iteration, sub_iteration, short_name)
);
COMMENT ON TABLE file_records IS 'Individual datasets with complete provenance lineage, linking files across model stages. Records are uniquely identified by a content hash, and logically unique by run, year, iteration, sub-iteration, and short name.';
COMMENT ON COLUMN file_records.unique_id IS 'File content hash (SHA256) serving as the unique identifier and primary key.';
COMMENT ON COLUMN file_records.record_type IS 'Type of record: file, h5_container, or h5_table.';
COMMENT ON COLUMN file_records.run_id IS 'Foreign key to runs table indicating which PILATES run produced/used this file.';
COMMENT ON COLUMN file_records.model_run_id IS 'Foreign key to model_runs table, linking to the specific model execution that produced this file.';
COMMENT ON COLUMN file_records.openlineage_id IS 'OpenLineage dataset UUID for cross-system lineage tracking.';
COMMENT ON COLUMN file_records.file_path IS 'Absolute or relative path to the file on disk.';
COMMENT ON COLUMN file_records.created_at IS 'Timestamp when file was created or first recorded.';
COMMENT ON COLUMN file_records.short_name IS 'Human-readable identifier for this dataset (e.g., urbansim_h5, activitysim_beam_plans).';
COMMENT ON COLUMN file_records.description IS 'Detailed description of file contents and purpose.';
COMMENT ON COLUMN file_records.year IS 'Simulation year this file corresponds to.';
COMMENT ON COLUMN file_records.iteration IS 'Simulation iteration this file corresponds to (e.g., for supply-demand loops).';
COMMENT ON COLUMN file_records.sub_iteration IS 'Sub-iteration number (if applicable, e.g., for sub-models or internal loops).';
COMMENT ON COLUMN file_records.models IS 'Array of model names that have touched/processed this file.';
COMMENT ON COLUMN file_records.producing_run_id IS 'OpenLineage ID of the model run that produced this file.';
COMMENT ON COLUMN file_records.consuming_run_ids IS 'Array of OpenLineage IDs for model runs that consumed this file.';
COMMENT ON COLUMN file_records.source_file_paths IS 'Array of file paths that were inputs to create this file (immediate lineage).';
COMMENT ON COLUMN file_records.metadata IS 'JSON object with additional file metadata (size, format, row count, etc.).';
COMMENT ON COLUMN file_records.schema IS 'JSON array describing file schema (column names, types, descriptions).';
COMMENT ON COLUMN file_records.exists IS 'Boolean indicating if file still exists on disk.';-- H5 file-table relationships
CREATE TABLE IF NOT EXISTS h5_table_records (
    unique_id VARCHAR PRIMARY KEY,
    h5_file_unique_id VARCHAR,
    table_name VARCHAR,
    FOREIGN KEY (unique_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (h5_file_unique_id) REFERENCES file_records(unique_id)
);
COMMENT ON TABLE h5_table_records IS 'Stores the relationship between H5 container files and the tables within them';
COMMENT ON COLUMN h5_table_records.unique_id IS 'Unique ID of the table record, foreign key to file_records.unique_id';
COMMENT ON COLUMN h5_table_records.h5_file_unique_id IS 'Unique ID of the parent H5 container file, foreign key to file_records.unique_id';
COMMENT ON COLUMN h5_table_records.table_name IS 'Name of the table within the H5 file';

-- Versioned Zarr manifest data
CREATE TABLE IF NOT EXISTS zarr_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    year INTEGER,
    iteration INTEGER,
    sub_iteration INTEGER, -- Simulation sub-iteration of the snapshot
    snapshot_type VARCHAR,
    model VARCHAR,
    parent_snapshot_id VARCHAR,
    created_at TIMESTAMP,
    full_skims_path VARCHAR,
    full_skims_n_variables INTEGER,
    full_skims_n_chunks INTEGER,
    full_skims_total_size_mb FLOAT,
    partial_skims_path VARCHAR,
    partial_skims_n_variables INTEGER,
    partial_skims_n_chunks INTEGER,
    partial_skims_total_size_mb FLOAT,
    changed_chunks INTEGER,
    chunk_manifest JSON,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
COMMENT ON TABLE zarr_snapshots IS 'Detailed metadata for each versioned Zarr skim snapshot';
COMMENT ON COLUMN zarr_snapshots.snapshot_id IS 'Unique identifier for this Zarr snapshot';
COMMENT ON COLUMN zarr_snapshots.run_id IS 'Foreign key to runs table';
COMMENT ON COLUMN zarr_snapshots.year IS 'Simulation year of the snapshot';
COMMENT ON COLUMN zarr_snapshots.iteration IS 'Simulation iteration of the snapshot';
COMMENT ON COLUMN zarr_snapshots.sub_iteration IS 'Simulation sub-iteration of the snapshot (if applicable).';
COMMENT ON COLUMN zarr_snapshots.snapshot_type IS 'Type of snapshot (e.g., initialization, merged)';
COMMENT ON COLUMN zarr_snapshots.model IS 'Model that produced this snapshot';
COMMENT ON COLUMN zarr_snapshots.parent_snapshot_id IS 'ID of the previous snapshot in the lineage';
COMMENT ON COLUMN zarr_snapshots.created_at IS 'Timestamp when the snapshot was created';
COMMENT ON COLUMN zarr_snapshots.full_skims_path IS 'Relative path to the full skims Zarr store';
COMMENT ON COLUMN zarr_snapshots.full_skims_n_variables IS 'Number of variables in the full skims Zarr store';
COMMENT ON COLUMN zarr_snapshots.full_skims_n_chunks IS 'Number of chunks in the full skims Zarr store';
COMMENT ON COLUMN zarr_snapshots.full_skims_total_size_mb IS 'Total size of the full skims Zarr store in MB';
COMMENT ON COLUMN zarr_snapshots.partial_skims_path IS 'Relative path to the partial skims Zarr store (if applicable)';
COMMENT ON COLUMN zarr_snapshots.partial_skims_n_variables IS 'Number of variables in the partial skims Zarr store (if applicable)';
COMMENT ON COLUMN zarr_snapshots.partial_skims_n_chunks IS 'Number of chunks in the partial skims Zarr store (if applicable)';
COMMENT ON COLUMN zarr_snapshots.partial_skims_total_size_mb IS 'Total size of the partial skims Zarr store in MB (if applicable)';
COMMENT ON COLUMN zarr_snapshots.changed_chunks IS 'Number of chunks changed from the parent snapshot';
COMMENT ON COLUMN zarr_snapshots.chunk_manifest IS 'JSON object containing the chunk-level manifest for the full skims';


-- Sequence for openlineage_events
CREATE SEQUENCE IF NOT EXISTS openlineage_events_id_seq START 1;

-- Lightweight event metadata
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
);
COMMENT ON TABLE openlineage_events IS 'Lightweight OpenLineage event metadata for integration with external lineage systems';
COMMENT ON COLUMN openlineage_events.id IS 'Auto-incrementing event ID';
COMMENT ON COLUMN openlineage_events.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN openlineage_events.model_run_id IS 'Foreign key to specific model execution';
COMMENT ON COLUMN openlineage_events.event_time IS 'Timestamp when event occurred';
COMMENT ON COLUMN openlineage_events.event_type IS 'Event type: START, COMPLETE, or FAIL';
COMMENT ON COLUMN openlineage_events.run_uuid IS 'OpenLineage run UUID (matches model_runs.openlineage_id)';
COMMENT ON COLUMN openlineage_events.job_name IS 'Formatted job name with model, year, and iteration';

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR,
    pilates_version VARCHAR
);
COMMENT ON TABLE schema_version IS 'Tracks the evolution of the database schema over time';
COMMENT ON COLUMN schema_version.version IS 'Integer representing the schema version number';
COMMENT ON COLUMN schema_version.applied_at IS 'Timestamp when this schema version was applied';
COMMENT ON COLUMN schema_version.description IS 'A brief description of the changes in this schema version';
COMMENT ON COLUMN schema_version.pilates_version IS 'The version of the PILATES application that introduced this schema';

-- Record current schema version
INSERT INTO schema_version (version, description, pilates_version)
SELECT 1, 'Initial schema with metadata, provenance, and dual storage support', '1.0.0'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 1);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_content_hash);
CREATE INDEX IF NOT EXISTS idx_file_records_openlineage_id ON file_records(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_file_records_short_name ON file_records(short_name);
CREATE INDEX IF NOT EXISTS idx_model_runs_model ON model_runs(model);
CREATE INDEX IF NOT EXISTS idx_model_runs_year ON model_runs(year);
CREATE INDEX IF NOT EXISTS idx_openlineage_events_run_uuid ON openlineage_events(run_uuid);
