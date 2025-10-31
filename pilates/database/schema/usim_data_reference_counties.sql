-- Auto-generated schema for table: usim_data_reference_counties
-- Source: usim_data_reference_counties
-- Description: Table '/counties' from H5 file 'usim_data_reference'
-- IMPORTANT: This is a placeholder. Review and adjust data types, add primary keys, and add indexes as needed.
-- Suggested Primary Key: county_id

CREATE SEQUENCE IF NOT EXISTS usim_data_reference_counties_id_seq START 1;

CREATE TABLE IF NOT EXISTS usim_data_reference_counties (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('usim_data_reference_counties_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR,
    file_record_id VARCHAR,
    year INTEGER,
    iteration INTEGER,
    county_id VARCHAR,
    pct_area DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE usim_data_reference_counties IS 'Table ''/counties'' from H5 file ''usim_data_reference''';
CREATE INDEX IF NOT EXISTS idx_usim_data_reference_counties_run_id ON usim_data_reference_counties(run_id);
CREATE INDEX IF NOT EXISTS idx_usim_data_reference_counties_year_iter ON usim_data_reference_counties(year, iteration);

COMMENT ON COLUMN usim_data_reference_counties.county_id IS 'TODO: Add description.';
COMMENT ON COLUMN usim_data_reference_counties.pct_area IS 'TODO: Add description.';