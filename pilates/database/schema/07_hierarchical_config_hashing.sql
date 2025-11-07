-- Hierarchical configuration hashing tables (Phase 1)
-- These tables support intelligent caching and output reuse by tracking
-- configuration at multiple granularity levels

-- Model-specific configuration hashes
CREATE TABLE IF NOT EXISTS model_configs (
    config_hash VARCHAR PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    config_snapshot_id VARCHAR NOT NULL,
    config_type VARCHAR NOT NULL, -- 'base', 'urbansim', 'activitysim', 'beam', 'atlas'
    config_data JSON NOT NULL, -- Filtered config relevant to this model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_snapshot_id) REFERENCES config_snapshots(snapshot_id)
);

COMMENT ON TABLE model_configs IS 'Model-specific configuration hashes for intelligent caching. Each hash represents the configuration affecting a specific model, including upstream dependencies.';
COMMENT ON COLUMN model_configs.config_hash IS 'SHA256 hash of model-specific configuration (unique identifier)';
COMMENT ON COLUMN model_configs.model_name IS 'Model name: base, urbansim, atlas, activitysim, or beam';
COMMENT ON COLUMN model_configs.config_snapshot_id IS 'Foreign key to complete configuration snapshot';
COMMENT ON COLUMN model_configs.config_type IS 'Configuration type matching model_name';
COMMENT ON COLUMN model_configs.config_data IS 'JSON containing only configuration affecting this model (for inspection/debugging)';
COMMENT ON COLUMN model_configs.created_at IS 'When this config hash was first seen';

-- Link model runs to their hierarchical configs
CREATE TABLE IF NOT EXISTS model_run_configs (
    model_run_id VARCHAR,
    config_hash VARCHAR,
    config_type VARCHAR NOT NULL, -- 'base', 'urbansim', etc.
    PRIMARY KEY (model_run_id, config_type),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(unique_id),
    FOREIGN KEY (config_hash) REFERENCES model_configs(config_hash)
);

COMMENT ON TABLE model_run_configs IS 'Links model runs to their hierarchical configuration hashes. Each model run has multiple hashes (base + upstream models + own config).';
COMMENT ON COLUMN model_run_configs.model_run_id IS 'Foreign key to model run';
COMMENT ON COLUMN model_run_configs.config_hash IS 'Foreign key to model config hash';
COMMENT ON COLUMN model_run_configs.config_type IS 'Which configuration layer this hash represents (base, urbansim, activitysim, beam, atlas)';

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_model_configs_model_hash
    ON model_configs(model_name, config_hash);

CREATE INDEX IF NOT EXISTS idx_model_configs_snapshot
    ON model_configs(config_snapshot_id);

CREATE INDEX IF NOT EXISTS idx_model_run_configs_hash
    ON model_run_configs(config_hash);

CREATE INDEX IF NOT EXISTS idx_model_run_configs_type
    ON model_run_configs(config_type);

-- Update schema version
INSERT INTO schema_version (version, description, pilates_version)
SELECT 2, 'Added hierarchical config hashing for intelligent caching', '1.1.0'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 2);
