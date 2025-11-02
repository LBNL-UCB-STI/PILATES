-- Raw UrbanSim data storage tables

-- Sequences for auto-incrementing IDs
CREATE SEQUENCE IF NOT EXISTS urbansim_households_raw_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS urbansim_persons_raw_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS urbansim_jobs_raw_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS urbansim_blocks_raw_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS urbansim_buildings_raw_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS urbansim_parcels_raw_id_seq START 1;

-- urbansim_parcels_raw
CREATE TABLE IF NOT EXISTS urbansim_parcels_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_parcels_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,

    -- Raw UrbanSim parcel data columns
    parcel_id INTEGER NOT NULL,
    zone_id VARCHAR,

    land_value FLOAT,
    total_sqft FLOAT,
    county_id VARCHAR,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_type VARCHAR DEFAULT 'raw',

    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, year, parcel_id)
);
COMMENT ON TABLE urbansim_parcels_raw IS 'Raw parcel/lot data from UrbanSim land use model';
COMMENT ON COLUMN urbansim_parcels_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_parcels_raw.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN urbansim_parcels_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_parcels_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_parcels_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_parcels_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_parcels_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_parcels_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_parcels_raw.parcel_id IS 'Unique parcel identifier';
COMMENT ON COLUMN urbansim_parcels_raw.zone_id IS 'Zone where parcel is located';
COMMENT ON COLUMN urbansim_parcels_raw.land_value IS 'Assessed land value in dollars';
COMMENT ON COLUMN urbansim_parcels_raw.total_sqft IS 'Total parcel area in square feet';
COMMENT ON COLUMN urbansim_parcels_raw.county_id IS 'County FIPS code';
COMMENT ON COLUMN urbansim_parcels_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_parcels_raw.table_type IS 'Type of table (e.g., raw, processed)';

-- urbansim_buildings_raw
CREATE TABLE IF NOT EXISTS urbansim_buildings_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_buildings_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,

    -- Raw UrbanSim building data columns
    building_id INTEGER NOT NULL,
    parcel_id INTEGER,

    building_type_id INTEGER,
    sqft INTEGER,
    year_built INTEGER,
    stories INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_type VARCHAR DEFAULT 'raw',

    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (run_id, year, parcel_id) REFERENCES urbansim_parcels_raw(run_id, year, parcel_id),
    UNIQUE (run_id, year, building_id)
);
COMMENT ON TABLE urbansim_buildings_raw IS 'Raw building data from UrbanSim including built environment characteristics';
COMMENT ON COLUMN urbansim_buildings_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_buildings_raw.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN urbansim_buildings_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_buildings_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_buildings_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_buildings_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_buildings_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_buildings_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_buildings_raw.building_id IS 'Unique building identifier';
COMMENT ON COLUMN urbansim_buildings_raw.parcel_id IS 'Parcel where building is located';
COMMENT ON COLUMN urbansim_buildings_raw.building_type_id IS 'Building type category (residential, commercial, etc.)';
COMMENT ON COLUMN urbansim_buildings_raw.sqft IS 'Building square footage';
COMMENT ON COLUMN urbansim_buildings_raw.year_built IS 'Year building was constructed';
COMMENT ON COLUMN urbansim_buildings_raw.stories IS 'Number of stories/floors';
COMMENT ON COLUMN urbansim_buildings_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_buildings_raw.table_type IS 'Type of table (e.g., raw, processed)';


-- urbansim_households_raw
CREATE TABLE IF NOT EXISTS urbansim_households_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_households_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Raw UrbanSim household data columns
    household_id INTEGER NOT NULL,
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
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (run_id, year, building_id) REFERENCES urbansim_buildings_raw(run_id, year, building_id),
    UNIQUE (run_id, year, household_id)
);
COMMENT ON TABLE urbansim_households_raw IS 'Raw household data directly from UrbanSim H5 outputs, preserving original structure';
COMMENT ON COLUMN urbansim_households_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_households_raw.file_record_id IS 'Foreign key to file_records for provenance tracking';
COMMENT ON COLUMN urbansim_households_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_households_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_households_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_households_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_households_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_households_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_households_raw.household_id IS 'Unique household identifier from UrbanSim';
COMMENT ON COLUMN urbansim_households_raw.building_id IS 'Building where household resides (links to buildings table)';
COMMENT ON COLUMN urbansim_households_raw.persons IS 'Number of people in household';
COMMENT ON COLUMN urbansim_households_raw.income IS 'Annual household income in dollars';
COMMENT ON COLUMN urbansim_households_raw.cars IS 'Number of vehicles owned by household';
COMMENT ON COLUMN urbansim_households_raw.block_id IS 'Census block ID where household resides';
COMMENT ON COLUMN urbansim_households_raw.age_of_head IS 'Age of household head';
COMMENT ON COLUMN urbansim_households_raw.children IS 'Number of children in household';
COMMENT ON COLUMN urbansim_households_raw.workers IS 'Number of workers in household';
COMMENT ON COLUMN urbansim_households_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_households_raw.table_type IS 'Type of table (e.g., raw, processed)';

-- urbansim_persons_raw
CREATE TABLE IF NOT EXISTS urbansim_persons_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_persons_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Raw UrbanSim person data columns
    person_id INTEGER NOT NULL,
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
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (run_id, year, household_id) REFERENCES urbansim_households_raw(run_id, year, household_id),
    UNIQUE (run_id, year, person_id)
);
COMMENT ON TABLE urbansim_persons_raw IS 'Raw person-level data from UrbanSim H5 outputs';
COMMENT ON COLUMN urbansim_persons_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_persons_raw.file_record_id IS 'Foreign key to file_records for provenance tracking';
COMMENT ON COLUMN urbansim_persons_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_persons_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_persons_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_persons_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_persons_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_persons_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_persons_raw.person_id IS 'Unique person identifier from UrbanSim';
COMMENT ON COLUMN urbansim_persons_raw.household_id IS 'Household this person belongs to (links to households table)';
COMMENT ON COLUMN urbansim_persons_raw.age IS 'Person age in years';
COMMENT ON COLUMN urbansim_persons_raw.worker IS 'Worker status (0=non-worker, 1=worker)';
COMMENT ON COLUMN urbansim_persons_raw.student IS 'Student status (0=non-student, 1=student)';
COMMENT ON COLUMN urbansim_persons_raw.race_id IS 'Race/ethnicity category identifier';
COMMENT ON COLUMN urbansim_persons_raw.sex IS 'Sex (1=male, 2=female)';
COMMENT ON COLUMN urbansim_persons_raw.work_zone_id IS 'TAZ where person works';
COMMENT ON COLUMN urbansim_persons_raw.school_zone_id IS 'TAZ where person attends school';
COMMENT ON COLUMN urbansim_persons_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_persons_raw.table_type IS 'Type of table (e.g., raw, processed)';

-- urbansim_jobs_raw
CREATE TABLE IF NOT EXISTS urbansim_jobs_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_jobs_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Raw UrbanSim job data columns
    job_id INTEGER NOT NULL,
    building_id INTEGER,
    
    sector_id VARCHAR,
    home_based_status INTEGER,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_type VARCHAR DEFAULT 'raw',
    
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (run_id, year, building_id) REFERENCES urbansim_buildings_raw(run_id, year, building_id),
    UNIQUE (run_id, year, job_id)
);
COMMENT ON TABLE urbansim_jobs_raw IS 'Raw employment/job data from UrbanSim H5 outputs';
COMMENT ON COLUMN urbansim_jobs_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_jobs_raw.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN urbansim_jobs_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_jobs_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_jobs_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_jobs_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_jobs_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_jobs_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_jobs_raw.job_id IS 'Unique job identifier';
COMMENT ON COLUMN urbansim_jobs_raw.building_id IS 'Building where job is located';
COMMENT ON COLUMN urbansim_jobs_raw.sector_id IS 'Employment sector/industry category';
COMMENT ON COLUMN urbansim_jobs_raw.home_based_status IS 'Whether job is home-based (0=no, 1=yes)';
COMMENT ON COLUMN urbansim_jobs_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_jobs_raw.table_type IS 'Type of table (e.g., raw, processed)';

-- urbansim_blocks_raw
CREATE TABLE IF NOT EXISTS urbansim_blocks_raw (
    id INTEGER PRIMARY KEY DEFAULT nextval('urbansim_blocks_raw_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER, -- UrbanSim does not typically have iterations for raw outputs
    sub_iteration INTEGER, -- UrbanSim does not typically have sub-iterations for raw outputs
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Raw UrbanSim block data columns
    block_id VARCHAR NOT NULL,
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
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, year, block_id)
);
COMMENT ON TABLE urbansim_blocks_raw IS 'Raw census block geographic data from UrbanSim';
COMMENT ON COLUMN urbansim_blocks_raw.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN urbansim_blocks_raw.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN urbansim_blocks_raw.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN urbansim_blocks_raw.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN urbansim_blocks_raw.iteration IS 'Simulation iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_blocks_raw.sub_iteration IS 'Simulation sub-iteration (not typically used for UrbanSim raw outputs)';
COMMENT ON COLUMN urbansim_blocks_raw.openlineage_id IS 'OpenLineage dataset ID for the parent H5 file';
COMMENT ON COLUMN urbansim_blocks_raw.table_openlineage_id IS 'OpenLineage dataset ID for this specific table within the H5 file';
COMMENT ON COLUMN urbansim_blocks_raw.block_id IS 'Census block FIPS code';
COMMENT ON COLUMN urbansim_blocks_raw.block_group_id IS 'Census block group FIPS code';
COMMENT ON COLUMN urbansim_blocks_raw.zone_id IS 'Zone identifier (may be TAZ or other)';
COMMENT ON COLUMN urbansim_blocks_raw.taz_zone_id IS 'Traffic Analysis Zone identifier';
COMMENT ON COLUMN urbansim_blocks_raw.square_meters_land IS 'Land area in square meters';
COMMENT ON COLUMN urbansim_blocks_raw.x IS 'X coordinate in local projection';
COMMENT ON COLUMN urbansim_blocks_raw.y IS 'Y coordinate in local projection';
COMMENT ON COLUMN urbansim_blocks_raw.created_at IS 'Timestamp when this record was created';
COMMENT ON COLUMN urbansim_blocks_raw.table_type IS 'Type of table (e.g., raw, processed)';



-- Indexes for raw UrbanSim data tables
CREATE INDEX IF NOT EXISTS idx_usim_households_raw_openlineage_id ON urbansim_households_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_households_raw_run_year ON urbansim_households_raw(run_id, year);
CREATE INDEX IF NOT EXISTS idx_usim_persons_raw_openlineage_id ON urbansim_persons_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_persons_raw_run_year ON urbansim_persons_raw(run_id, year);
CREATE INDEX IF NOT EXISTS idx_usim_jobs_raw_openlineage_id ON urbansim_jobs_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_jobs_raw_run_year ON urbansim_jobs_raw(run_id, year);
CREATE INDEX IF NOT EXISTS idx_usim_blocks_raw_openlineage_id ON urbansim_blocks_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_blocks_raw_run_year ON urbansim_blocks_raw(run_id, year);
CREATE INDEX IF NOT EXISTS idx_usim_buildings_raw_openlineage_id ON urbansim_buildings_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_buildings_raw_run_year ON urbansim_buildings_raw(run_id, year);
CREATE INDEX IF NOT EXISTS idx_usim_parcels_raw_openlineage_id ON urbansim_parcels_raw(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_usim_parcels_raw_run_year ON urbansim_parcels_raw(run_id, year);
