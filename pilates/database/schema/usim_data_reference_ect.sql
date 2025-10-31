-- Auto-generated schema for table: usim_data_reference_ect
-- Source: usim_data_reference_ect
-- Description: Table '/ect' from H5 file 'usim_data_reference'
-- IMPORTANT: This is a placeholder. Review and adjust data types, add primary keys, and add indexes as needed.

CREATE SEQUENCE IF NOT EXISTS usim_data_reference_ect_id_seq START 1;

CREATE TABLE IF NOT EXISTS usim_data_reference_ect (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('usim_data_reference_ect_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR,
    file_record_id VARCHAR,
    year INTEGER,
    iteration INTEGER,
    total_number_of_jobs DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE usim_data_reference_ect IS 'Table ''/ect'' from H5 file ''usim_data_reference''';
CREATE INDEX IF NOT EXISTS idx_usim_data_reference_ect_run_id ON usim_data_reference_ect(run_id);
CREATE INDEX IF NOT EXISTS idx_usim_data_reference_ect_year_iter ON usim_data_reference_ect(year, iteration);

COMMENT ON COLUMN usim_data_reference_ect.total_number_of_jobs IS 'TODO: Add description.';