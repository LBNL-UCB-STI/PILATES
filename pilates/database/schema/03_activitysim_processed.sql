-- Processed ActivitySim data storage tables

-- Sequences for auto-incrementing IDs
CREATE SEQUENCE IF NOT EXISTS activitysim_households_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS activitysim_persons_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS activitysim_land_use_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS activitysim_data_generic_id_seq START 1;

-- activitysim_households
CREATE TABLE IF NOT EXISTS activitysim_households (
    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_households_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER, -- ActivitySim may have sub-iterations
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    household_id INTEGER NOT NULL,
    TAZ VARCHAR,
    persons INTEGER,
    income FLOAT,
    cars INTEGER,
    HHT INTEGER,
    workers INTEGER,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (run_id, year, household_id) REFERENCES urbansim_households_raw(run_id, year, household_id),
    UNIQUE (run_id, year, household_id)
);
COMMENT ON TABLE activitysim_households IS 'Processed household data ready for ActivitySim consumption, transformed from UrbanSim outputs. Households are constant across iterations within a year.';
COMMENT ON COLUMN activitysim_households.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN activitysim_households.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN activitysim_households.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN activitysim_households.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN activitysim_households.iteration IS 'Simulation iteration when this record was first created (households remain constant across iterations within a year)';
COMMENT ON COLUMN activitysim_households.sub_iteration IS 'Simulation sub-iteration (if applicable)';
COMMENT ON COLUMN activitysim_households.openlineage_id IS 'OpenLineage dataset ID for the parent file';
COMMENT ON COLUMN activitysim_households.table_openlineage_id IS 'OpenLineage dataset ID for this specific table';
COMMENT ON COLUMN activitysim_households.household_id IS 'Unique household identifier (matches UrbanSim)';
COMMENT ON COLUMN activitysim_households.TAZ IS 'Traffic Analysis Zone where household resides (ActivitySim format)';
COMMENT ON COLUMN activitysim_households.persons IS 'Number of persons (mapped from UrbanSim PERSONS column)';
COMMENT ON COLUMN activitysim_households.income IS 'Annual household income in dollars';
COMMENT ON COLUMN activitysim_households.cars IS 'Vehicle ownership (mapped from UrbanSim auto_ownership)';
COMMENT ON COLUMN activitysim_households.HHT IS 'Household type category for ActivitySim';
COMMENT ON COLUMN activitysim_households.workers IS 'Number of workers (mapped from UrbanSim num_workers)';
COMMENT ON COLUMN activitysim_households.created_at IS 'Timestamp when this record was created';

-- activitysim_persons
CREATE TABLE IF NOT EXISTS activitysim_persons (
    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_persons_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER, -- ActivitySim may have sub-iterations
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Person data columns
    person_id INTEGER NOT NULL,
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
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (household_id) REFERENCES activitysim_households(id),
    FOREIGN KEY (run_id, year, person_id) REFERENCES urbansim_persons_raw(run_id, year, person_id),
    UNIQUE (run_id, year, person_id)
);
COMMENT ON TABLE activitysim_persons IS 'Processed person-level data for ActivitySim, including synthetic person attributes. Persons are constant across iterations within a year.';
COMMENT ON COLUMN activitysim_persons.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN activitysim_persons.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN activitysim_persons.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN activitysim_persons.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN activitysim_persons.iteration IS 'Simulation iteration when this record was first created (persons remain constant across iterations within a year)';
COMMENT ON COLUMN activitysim_persons.sub_iteration IS 'Simulation sub-iteration (if applicable)';
COMMENT ON COLUMN activitysim_persons.openlineage_id IS 'OpenLineage dataset ID for the parent file';
COMMENT ON COLUMN activitysim_persons.table_openlineage_id IS 'OpenLineage dataset ID for this specific table';
COMMENT ON COLUMN activitysim_persons.person_id IS 'Unique person identifier (matches UrbanSim)';
COMMENT ON COLUMN activitysim_persons.household_id IS 'Household this person belongs to';
COMMENT ON COLUMN activitysim_persons.TAZ IS 'Home TAZ (matches household TAZ)';
COMMENT ON COLUMN activitysim_persons.age IS 'Person age in years';
COMMENT ON COLUMN activitysim_persons.worker IS 'Worker status for ActivitySim';
COMMENT ON COLUMN activitysim_persons.student IS 'Student status for ActivitySim';
COMMENT ON COLUMN activitysim_persons.ptype IS 'Person type category (full-time worker, part-time, student, etc.)';
COMMENT ON COLUMN activitysim_persons.pemploy IS 'Employment status category';
COMMENT ON COLUMN activitysim_persons.pstudent IS 'Student enrollment category';
COMMENT ON COLUMN activitysim_persons.member_id IS 'Person number within household (mapped from UrbanSim PNUM)';
COMMENT ON COLUMN activitysim_persons.workplace_taz IS 'TAZ of workplace location';
COMMENT ON COLUMN activitysim_persons.school_taz IS 'TAZ of school location';
COMMENT ON COLUMN activitysim_persons.home_x IS 'X coordinate of home location';
COMMENT ON COLUMN activitysim_persons.home_y IS 'Y coordinate of home location';
COMMENT ON COLUMN activitysim_persons.created_at IS 'Timestamp when this record was created';

-- activitysim_land_use
CREATE TABLE IF NOT EXISTS activitysim_land_use (
    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_land_use_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER, -- ActivitySim may have sub-iterations
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    
    -- Land use data columns
    TAZ VARCHAR NOT NULL,
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
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, year, TAZ)
);
COMMENT ON TABLE activitysim_land_use IS 'Processed land use/zonal data for ActivitySim including demographics, employment, and accessibility. Land use data is constant across iterations within a year.';
COMMENT ON COLUMN activitysim_land_use.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN activitysim_land_use.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN activitysim_land_use.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN activitysim_land_use.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN activitysim_land_use.iteration IS 'Simulation iteration when this record was first created (land use remains constant across iterations within a year)';
COMMENT ON COLUMN activitysim_land_use.sub_iteration IS 'Simulation sub-iteration (if applicable)';
COMMENT ON COLUMN activitysim_land_use.openlineage_id IS 'OpenLineage dataset ID for the parent file';
COMMENT ON COLUMN activitysim_land_use.table_openlineage_id IS 'OpenLineage dataset ID for this specific table';
COMMENT ON COLUMN activitysim_land_use.TAZ IS 'Traffic Analysis Zone identifier (mapped from UrbanSim ZONE)';
COMMENT ON COLUMN activitysim_land_use.TOTPOP IS 'Total population in TAZ';
COMMENT ON COLUMN activitysim_land_use.TOTHH IS 'Total households in TAZ';
COMMENT ON COLUMN activitysim_land_use.TOTEMP IS 'Total employment in TAZ';
COMMENT ON COLUMN activitysim_land_use.TOTACRE IS 'Total acres in TAZ';
COMMENT ON COLUMN activitysim_land_use.area_type IS 'Area type category (urban core, suburban, rural, etc.)';
COMMENT ON COLUMN activitysim_land_use.employment_density IS 'Jobs per acre';
COMMENT ON COLUMN activitysim_land_use.pop_density IS 'Population per acre';
COMMENT ON COLUMN activitysim_land_use.hh_density IS 'Households per acre';
COMMENT ON COLUMN activitysim_land_use.AGE0004 IS 'Population aged 0-4 years';
COMMENT ON COLUMN activitysim_land_use.AGE0519 IS 'Population aged 5-19 years';
COMMENT ON COLUMN activitysim_land_use.AGE2044 IS 'Population aged 20-44 years';
COMMENT ON COLUMN activitysim_land_use.AGE4564 IS 'Population aged 45-64 years';
COMMENT ON COLUMN activitysim_land_use.AGE64P IS 'Population aged 65+ years';
COMMENT ON COLUMN activitysim_land_use.AGE62P IS 'Population aged 62+ years';
COMMENT ON COLUMN activitysim_land_use.HHINCQ1 IS 'Households in income quartile 1 (lowest)';
COMMENT ON COLUMN activitysim_land_use.HHINCQ2 IS 'Households in income quartile 2';
COMMENT ON COLUMN activitysim_land_use.HHINCQ3 IS 'Households in income quartile 3';
COMMENT ON COLUMN activitysim_land_use.HHINCQ4 IS 'Households in income quartile 4 (highest)';
COMMENT ON COLUMN activitysim_land_use.EMPRES IS 'Residential employment';
COMMENT ON COLUMN activitysim_land_use.RETEMPN IS 'Retail employment';
COMMENT ON COLUMN activitysim_land_use.FPSEMPN IS 'Financial and professional services employment';
COMMENT ON COLUMN activitysim_land_use.HEREMPN IS 'Health, education, and recreational employment';
COMMENT ON COLUMN activitysim_land_use.AGREMPN IS 'Agricultural and natural resources employment';
COMMENT ON COLUMN activitysim_land_use.MWTEMPN IS 'Manufacturing, wholesale, and transportation employment';
COMMENT ON COLUMN activitysim_land_use.OTHEMPN IS 'Other employment';
COMMENT ON COLUMN activitysim_land_use.HSENROLL IS 'High school enrollment';
COMMENT ON COLUMN activitysim_land_use.COLLFTE IS 'College full-time enrollment';
COMMENT ON COLUMN activitysim_land_use.COLLPTE IS 'College part-time enrollment';
COMMENT ON COLUMN activitysim_land_use.PRKCST IS 'Daily parking cost in dollars';
COMMENT ON COLUMN activitysim_land_use.OPRKCST IS 'Off-peak parking cost in dollars';
COMMENT ON COLUMN activitysim_land_use.created_at IS 'Timestamp when this record was created';

-- Generic table for other ActivitySim input tables
CREATE TABLE IF NOT EXISTS activitysim_data_generic (
    id INTEGER PRIMARY KEY DEFAULT nextval('activitysim_data_generic_id_seq'),
    file_record_id VARCHAR,
    run_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER, -- ActivitySim may have sub-iterations
    openlineage_id VARCHAR,
    table_openlineage_id VARCHAR,
    table_name VARCHAR NOT NULL,
    
    -- JSON storage for flexible schema
    data_json JSON,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, year, iteration, table_name)
);
COMMENT ON TABLE activitysim_data_generic IS 'Generic storage for additional ActivitySim input tables (accessibility, skims, etc.) in JSON format. Data can vary per iteration (e.g., skims updated by BEAM).';
COMMENT ON COLUMN activitysim_data_generic.id IS 'Auto-incrementing database row ID';
COMMENT ON COLUMN activitysim_data_generic.file_record_id IS 'Foreign key to file_records';
COMMENT ON COLUMN activitysim_data_generic.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN activitysim_data_generic.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN activitysim_data_generic.iteration IS 'Simulation iteration this data corresponds to';
COMMENT ON COLUMN activitysim_data_generic.sub_iteration IS 'Simulation sub-iteration (if applicable)';
COMMENT ON COLUMN activitysim_data_generic.openlineage_id IS 'OpenLineage dataset ID for the parent file';
COMMENT ON COLUMN activitysim_data_generic.table_openlineage_id IS 'OpenLineage dataset ID for this specific table';
COMMENT ON COLUMN activitysim_data_generic.table_name IS 'Name of the ActivitySim table stored here';
COMMENT ON COLUMN activitysim_data_generic.data_json IS 'Complete table data serialized as JSON for flexible schema support';
COMMENT ON COLUMN activitysim_data_generic.created_at IS 'Timestamp when this record was created';

-- Indexes for ActivitySim processed data tables
CREATE INDEX IF NOT EXISTS idx_asim_households_openlineage_id ON activitysim_households(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_asim_households_run_year_iter ON activitysim_households(run_id, year, iteration);
CREATE INDEX IF NOT EXISTS idx_asim_persons_openlineage_id ON activitysim_persons(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_asim_persons_run_year_iter ON activitysim_persons(run_id, year, iteration);
CREATE INDEX IF NOT EXISTS idx_asim_land_use_openlineage_id ON activitysim_land_use(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_asim_land_use_run_year_iter ON activitysim_land_use(run_id, year, iteration);
CREATE INDEX IF NOT EXISTS idx_asim_generic_table_name ON activitysim_data_generic(table_name);
CREATE INDEX IF NOT EXISTS idx_asim_generic_openlineage_id ON activitysim_data_generic(openlineage_id);
CREATE INDEX IF NOT EXISTS idx_asim_generic_run_year_iter ON activitysim_data_generic(run_id, year, iteration);
