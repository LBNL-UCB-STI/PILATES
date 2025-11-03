-- Auto-generated schema
-- Description: Mutable ATLAS input file
-- IMPORTANT: This is a placeholder. Review and adjust data types, add primary keys, and add indexes as needed.

CREATE SEQUENCE IF NOT EXISTS new_vehicles_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_blocks_csv_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_householdv_input_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_jobs_csv_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_persons_csv_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_residential_csv_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_vehicles2_output_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS atlas_vehicles_input_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS householdv_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS modeaccessibility_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS new_vehicle_annual_medians_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS new_vehicle_representative_vehicle_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS new_vehicles_biannual_values_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS used_vehicles_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS vehicle_type_mapping_baseline_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS vehicles_id_seq START 1;


CREATE TABLE IF NOT EXISTS modeaccessibility (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('modeaccessibility_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    unnamed__0 BIGINT,
    geoid BIGINT NOT NULL,
    bike BIGINT,
    rail BIGINT,
    bus BIGINT,
    UNIQUE (run_id, year, geoid),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE modeaccessibility IS 'Mutable ATLAS input file: modeaccessibility.csv';
COMMENT ON COLUMN modeaccessibility.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN modeaccessibility.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN modeaccessibility.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN modeaccessibility.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN modeaccessibility.unnamed__0 IS 'Original index or unnamed column from source file.';
COMMENT ON COLUMN modeaccessibility.geoid IS 'Geographic identifier (e.g., TAZ, block group).';
COMMENT ON COLUMN modeaccessibility.bike IS 'Bike accessibility score/measure.';
COMMENT ON COLUMN modeaccessibility.rail IS 'Rail accessibility score/measure.';
COMMENT ON COLUMN modeaccessibility.bus IS 'Bus accessibility score/measure.';

CREATE TABLE IF NOT EXISTS atlas_blocks_csv (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_blocks_csv_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    block_id BIGINT NOT NULL,
    tract_id BIGINT,
    y DOUBLE,
    cousub BIGINT,
    square_meters_land BIGINT,
    x DOUBLE,
    sum_acres DOUBLE,
    state_id BIGINT,
    taz_zone_id BIGINT,
    county_id BIGINT,
    mpo_id BIGINT,
    employment_capacity DOUBLE,
    residential_unit_capacity DOUBLE,
    block_group_id BIGINT,
    zone_id BIGINT,
    node_id BIGINT,
    UNIQUE (run_id, year, block_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE atlas_blocks_csv IS 'ATLAS blocks input CSV';
COMMENT ON COLUMN atlas_blocks_csv.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_blocks_csv.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_blocks_csv.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_blocks_csv.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_blocks_csv.block_id IS 'Unique identifier for the block.';
COMMENT ON COLUMN atlas_blocks_csv.tract_id IS 'Census tract identifier.';
COMMENT ON COLUMN atlas_blocks_csv.y IS 'Y-coordinate of the block centroid.';
COMMENT ON COLUMN atlas_blocks_csv.cousub IS 'County subdivision identifier.';
COMMENT ON COLUMN atlas_blocks_csv.square_meters_land IS 'Land area in square meters.';
COMMENT ON COLUMN atlas_blocks_csv.x IS 'X-coordinate of the block centroid.';
COMMENT ON COLUMN atlas_blocks_csv.sum_acres IS 'Total area in acres.';
COMMENT ON COLUMN atlas_blocks_csv.state_id IS 'State identifier.';
COMMENT ON COLUMN atlas_blocks_csv.taz_zone_id IS 'Traffic Analysis Zone identifier.';
COMMENT ON COLUMN atlas_blocks_csv.county_id IS 'County identifier.';
COMMENT ON COLUMN atlas_blocks_csv.mpo_id IS 'Metropolitan Planning Organization identifier.';
COMMENT ON COLUMN atlas_blocks_csv.employment_capacity IS 'Capacity for employment in the block.';
COMMENT ON COLUMN atlas_blocks_csv.residential_unit_capacity IS 'Capacity for residential units in the block.';
COMMENT ON COLUMN atlas_blocks_csv.block_group_id IS 'Census block group identifier.';
COMMENT ON COLUMN atlas_blocks_csv.zone_id IS 'General zone identifier.';
COMMENT ON COLUMN atlas_blocks_csv.node_id IS 'Node identifier (e.g., in a network).';

CREATE TABLE IF NOT EXISTS atlas_householdv_input (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_householdv_input_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    household_id INTEGER NOT NULL,
    nvehicles BIGINT,
    data_year BIGINT,
    newhhflag DOUBLE,
    UNIQUE (run_id, year, household_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE atlas_householdv_input IS 'ATLAS household vehicle counts for year 2023';
COMMENT ON COLUMN atlas_householdv_input.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_householdv_input.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_householdv_input.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_householdv_input.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_householdv_input.household_id IS 'Unique household identifier.';
COMMENT ON COLUMN atlas_householdv_input.nvehicles IS 'Number of vehicles owned by the household.';
COMMENT ON COLUMN atlas_householdv_input.data_year IS 'Year of the data.';
COMMENT ON COLUMN atlas_householdv_input.newhhflag IS 'Flag indicating if its a new household.';


CREATE TABLE IF NOT EXISTS atlas_jobs_csv (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_jobs_csv_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    job_id BIGINT NOT NULL,
    lcm_county_id BIGINT,
    agg_sector BIGINT,
    sector_id VARCHAR,
    block_id BIGINT,
    UNIQUE (run_id, year, job_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE atlas_jobs_csv IS 'ATLAS jobs input CSV';
COMMENT ON COLUMN atlas_jobs_csv.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_jobs_csv.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_jobs_csv.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_jobs_csv.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_jobs_csv.job_id IS 'Unique job identifier.';
COMMENT ON COLUMN atlas_jobs_csv.lcm_county_id IS 'County identifier for job location.';
COMMENT ON COLUMN atlas_jobs_csv.agg_sector IS 'Aggregated employment sector.';
COMMENT ON COLUMN atlas_jobs_csv.sector_id IS 'Detailed employment sector identifier.';
COMMENT ON COLUMN atlas_jobs_csv.block_id IS 'Block where the job is located.';

CREATE TABLE IF NOT EXISTS atlas_persons_csv (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_persons_csv_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    person_id INTEGER NOT NULL,
    sex DOUBLE,
    member_id BIGINT,
    school_zone_id BIGINT,
    work_at_home DOUBLE,
    age_group VARCHAR,
    worker DOUBLE,
    school_block_id BIGINT,
    relate BIGINT,
    work_block_id BIGINT,
    person_age VARCHAR,
    education_group VARCHAR,
    edu DOUBLE,
    school_taz BIGINT,
    workplace_taz BIGINT,
    work_zone_id BIGINT,
    school_id BIGINT,
    p_hispanic VARCHAR,
    household_id INTEGER,
    age DOUBLE,
    person_sex VARCHAR,
    hours DOUBLE,
    mar BIGINT,
    hispanic_1 DOUBLE,
    student DOUBLE,
    hispanic DOUBLE,
    race VARCHAR,
    earning DOUBLE,
    race_id DOUBLE,
    UNIQUE (run_id, year, person_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE atlas_persons_csv IS 'ATLAS persons input CSV';
COMMENT ON COLUMN atlas_persons_csv.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_persons_csv.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_persons_csv.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_persons_csv.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_persons_csv.person_id IS 'Unique person identifier.';
COMMENT ON COLUMN atlas_persons_csv.sex IS 'Sex of the person.';
COMMENT ON COLUMN atlas_persons_csv.member_id IS 'Member ID within a household.';
COMMENT ON COLUMN atlas_persons_csv.school_zone_id IS 'Zone where the person attends school.';
COMMENT ON COLUMN atlas_persons_csv.work_at_home IS 'Flag for working from home (1=yes, 0=no).';
COMMENT ON COLUMN atlas_persons_csv.age_group IS 'Age group category.';
COMMENT ON COLUMN atlas_persons_csv.worker IS 'Worker status (1=worker, 0=non-worker).';
COMMENT ON COLUMN atlas_persons_csv.school_block_id IS 'Block where the person attends school.';
COMMENT ON COLUMN atlas_persons_csv.relate IS 'Relationship to household head.';
COMMENT ON COLUMN atlas_persons_csv.work_block_id IS 'Block where the person works.';
COMMENT ON COLUMN atlas_persons_csv.person_age IS 'Age of the person.';
COMMENT ON COLUMN atlas_persons_csv.education_group IS 'Education level group.';
COMMENT ON COLUMN atlas_persons_csv.edu IS 'Education level.';
COMMENT ON COLUMN atlas_persons_csv.school_taz IS 'Traffic Analysis Zone where the person attends school.';
COMMENT ON COLUMN atlas_persons_csv.workplace_taz IS 'Traffic Analysis Zone where the person works.';
COMMENT ON COLUMN atlas_persons_csv.work_zone_id IS 'Zone where the person works.';
COMMENT ON COLUMN atlas_persons_csv.school_id IS 'School identifier.';
COMMENT ON COLUMN atlas_persons_csv.p_hispanic IS 'Hispanic origin flag.';
COMMENT ON COLUMN atlas_persons_csv.household_id IS 'Household identifier.';
COMMENT ON COLUMN atlas_persons_csv.age IS 'Age in years.';
COMMENT ON COLUMN atlas_persons_csv.person_sex IS 'Sex of the person.';
COMMENT ON COLUMN atlas_persons_csv.hours IS 'Hours worked.';
COMMENT ON COLUMN atlas_persons_csv.mar IS 'Marital status.';
COMMENT ON COLUMN atlas_persons_csv.hispanic_1 IS 'Hispanic origin flag (possibly duplicate of hispanic).';
COMMENT ON COLUMN atlas_persons_csv.student IS 'Student status (1=student, 0=non-student).';
COMMENT ON COLUMN atlas_persons_csv.hispanic IS 'Hispanic origin flag.';
COMMENT ON COLUMN atlas_persons_csv.race IS 'Race category.';
COMMENT ON COLUMN atlas_persons_csv.earning IS 'Earnings.';
COMMENT ON COLUMN atlas_persons_csv.race_id IS 'Race identifier.';


CREATE TABLE IF NOT EXISTS atlas_residential_csv (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_residential_csv_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    unit_id BIGINT NOT NULL,
    acs_18_value DOUBLE,
    block_id BIGINT,
    acs_13_value DOUBLE,
    acs_18_rent DOUBLE,
    acs_13_rent DOUBLE,
    building_type_id BIGINT,
    lcm_county_id BIGINT,
    year_built BIGINT,
    block_group_id BIGINT,
    UNIQUE (run_id, year, unit_id),
    FOREIGN KEY (run_id, year, block_id) REFERENCES atlas_blocks_csv(run_id, year, block_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE atlas_residential_csv IS 'ATLAS residential units input CSV';
COMMENT ON COLUMN atlas_residential_csv.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_residential_csv.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_residential_csv.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_residential_csv.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_residential_csv.unit_id IS 'Unique residential unit identifier.';
COMMENT ON COLUMN atlas_residential_csv.acs_18_value IS 'ACS 2018 property value.';
COMMENT ON COLUMN atlas_residential_csv.block_id IS 'Block where the residential unit is located.';
COMMENT ON COLUMN atlas_residential_csv.acs_13_value IS 'ACS 2013 property value.';
COMMENT ON COLUMN atlas_residential_csv.acs_18_rent IS 'ACS 2018 rent.';
COMMENT ON COLUMN atlas_residential_csv.acs_13_rent IS 'ACS 2013 rent.';
COMMENT ON COLUMN atlas_residential_csv.building_type_id IS 'Building type identifier.';
COMMENT ON COLUMN atlas_residential_csv.lcm_county_id IS 'County identifier.';
COMMENT ON COLUMN atlas_residential_csv.year_built IS 'Year the unit was built.';
COMMENT ON COLUMN atlas_residential_csv.block_group_id IS 'Census block group identifier.';

CREATE TABLE IF NOT EXISTS atlas_vehicles2_output (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_vehicles2_output_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    household_id INTEGER NOT NULL,
    vehicle_id BIGINT NOT NULL,
    bodytype VARCHAR,
    pred_power VARCHAR,
    ownlease VARCHAR,
    modelyear BIGINT,
    adopt_fuel VARCHAR,
    adopt_veh VARCHAR,
    acquire_year DOUBLE,
    vehicle_tag VARCHAR,
    data_year BIGINT,
    newhhflag DOUBLE,
    maindriver_id INTEGER,
    vintage_category VARCHAR,
    vehicletypeid VARCHAR,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, household_id) REFERENCES urbansim_households_raw(run_id, year, household_id),
    UNIQUE (run_id, year, household_id, vehicle_id)
);

COMMENT ON TABLE atlas_vehicles2_output IS 'ATLAS vehicles2 CSV with vehicleTypeId';
COMMENT ON COLUMN atlas_vehicles2_output.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_vehicles2_output.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_vehicles2_output.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_vehicles2_output.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_vehicles2_output.household_id IS 'Identifier for the household owning the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.vehicle_id IS 'Unique identifier for the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN atlas_vehicles2_output.pred_power IS 'Predicted power of the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.ownlease IS 'Ownership status of the vehicle (Own or Lease).';
COMMENT ON COLUMN atlas_vehicles2_output.modelyear IS 'Model year of the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.adopt_fuel IS 'Type of fuel the vehicle uses (e.g., Gasoline, BEV).';
COMMENT ON COLUMN atlas_vehicles2_output.adopt_veh IS 'Adopted vehicle type.';
COMMENT ON COLUMN atlas_vehicles2_output.acquire_year IS 'Year the vehicle was acquired.';
COMMENT ON COLUMN atlas_vehicles2_output.vehicle_tag IS 'A tag or label for the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.data_year IS 'The year of the data record.';
COMMENT ON COLUMN atlas_vehicles2_output.newhhflag IS 'Flag indicating if the household is new.';
COMMENT ON COLUMN atlas_vehicles2_output.maindriver_id IS 'Identifier for the main driver of the vehicle.';
COMMENT ON COLUMN atlas_vehicles2_output.vintage_category IS 'Category for the vehicles vintage.';
COMMENT ON COLUMN atlas_vehicles2_output.vehicletypeid IS 'Identifier for the vehicle type.';

CREATE TABLE IF NOT EXISTS atlas_vehicles_input (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('atlas_vehicles_input_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    household_id INTEGER NOT NULL,
    vehicle_id BIGINT NOT NULL,
    bodytype VARCHAR,
    pred_power VARCHAR,
    ownlease VARCHAR,
    modelyear BIGINT,
    adopt_fuel VARCHAR,
    adopt_veh VARCHAR,
    acquire_year DOUBLE,
    vehicle_tag VARCHAR,
    data_year BIGINT,
    newhhflag DOUBLE,
    maindriver_id DOUBLE,
    vintage_category VARCHAR,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, household_id, vehicle_id)
);

COMMENT ON TABLE atlas_vehicles_input IS 'ATLAS vehicles CSV before vehicleTypeId addition';
COMMENT ON COLUMN atlas_vehicles_input.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN atlas_vehicles_input.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN atlas_vehicles_input.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_vehicles_input.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN atlas_vehicles_input.household_id IS 'Identifier for the household owning the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.vehicle_id IS 'Unique identifier for the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN atlas_vehicles_input.pred_power IS 'Predicted power of the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.ownlease IS 'Ownership status of the vehicle (Own or Lease).';
COMMENT ON COLUMN atlas_vehicles_input.modelyear IS 'Model year of the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.adopt_fuel IS 'Type of fuel the vehicle uses (e.g., Gasoline, BEV).';
COMMENT ON COLUMN atlas_vehicles_input.adopt_veh IS 'Adopted vehicle type.';
COMMENT ON COLUMN atlas_vehicles_input.acquire_year IS 'Year the vehicle was acquired.';
COMMENT ON COLUMN atlas_vehicles_input.vehicle_tag IS 'A tag or label for the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.data_year IS 'The year of the data record.';
COMMENT ON COLUMN atlas_vehicles_input.newhhflag IS 'Flag indicating if the household is new.';
COMMENT ON COLUMN atlas_vehicles_input.maindriver_id IS 'Identifier for the main driver of the vehicle.';
COMMENT ON COLUMN atlas_vehicles_input.vintage_category IS 'Category for the vehicles vintage.';

CREATE TABLE IF NOT EXISTS householdv (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('householdv_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    household_id INTEGER NOT NULL,
    nvehicles BIGINT,
    data_year BIGINT,
    newhhflag DOUBLE,
    UNIQUE (run_id, year, household_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE householdv IS 'ATLAS householdv_2023.csv output for year 2023';
COMMENT ON COLUMN householdv.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN householdv.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN householdv.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN householdv.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN householdv.household_id IS 'Unique household identifier.';
COMMENT ON COLUMN householdv.nvehicles IS 'Number of vehicles in the household.';
COMMENT ON COLUMN householdv.data_year IS 'The year of the data record.';
COMMENT ON COLUMN householdv.newhhflag IS 'Flag indicating if the household is new.';


CREATE TABLE IF NOT EXISTS new_vehicle_annual_medians (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('new_vehicle_annual_medians_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    data_year BIGINT NOT NULL,
    fueltype VARCHAR NOT NULL,
    bodytype VARCHAR NOT NULL,
    price DOUBLE,
    accel DOUBLE,
    range DOUBLE,
    mpge DOUBLE,
    refueltime_mins BIGINT,
    cpmile DOUBLE,
    bev_energy DOUBLE,
    annmain DOUBLE,
    tax_credit BIGINT,
    rebate DOUBLE,
    int_volume DOUBLE,
    total_sales DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, data_year, fueltype, bodytype)
);

COMMENT ON TABLE new_vehicle_annual_medians IS 'Mutable ATLAS input file: new_vehicle_annual_medians.csv';
COMMENT ON COLUMN new_vehicle_annual_medians.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN new_vehicle_annual_medians.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN new_vehicle_annual_medians.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicle_annual_medians.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicle_annual_medians.data_year IS 'The year of the data record.';
COMMENT ON COLUMN new_vehicle_annual_medians.fueltype IS 'Type of fuel (e.g., Gasoline, BEV).';
COMMENT ON COLUMN new_vehicle_annual_medians.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN new_vehicle_annual_medians.price IS 'Median price of the vehicle.';
COMMENT ON COLUMN new_vehicle_annual_medians.accel IS 'Median acceleration.';
COMMENT ON COLUMN new_vehicle_annual_medians.range IS 'Median range of the vehicle.';
COMMENT ON COLUMN new_vehicle_annual_medians.mpge IS 'Median miles per gallon equivalent.';
COMMENT ON COLUMN new_vehicle_annual_medians.refueltime_mins IS 'Median refueling time in minutes.';
COMMENT ON COLUMN new_vehicle_annual_medians.cpmile IS 'Median cost per mile.';
COMMENT ON COLUMN new_vehicle_annual_medians.bev_energy IS 'Median battery electric vehicle energy consumption.';
COMMENT ON COLUMN new_vehicle_annual_medians.annmain IS 'Median annual maintenance cost.';
COMMENT ON COLUMN new_vehicle_annual_medians.tax_credit IS 'Median tax credit available.';
COMMENT ON COLUMN new_vehicle_annual_medians.rebate IS 'Median rebate available.';
COMMENT ON COLUMN new_vehicle_annual_medians.int_volume IS 'Median interior volume.';
COMMENT ON COLUMN new_vehicle_annual_medians.total_sales IS 'Total sales for the vehicle category.';

CREATE TABLE IF NOT EXISTS new_vehicle_representative_vehicle (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('new_vehicle_representative_vehicle_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    data_year BIGINT NOT NULL,
    fueltype VARCHAR NOT NULL,
    bodytype VARCHAR NOT NULL,
    price DOUBLE,
    accel DOUBLE,
    range BIGINT,
    mpge DOUBLE,
    refueltime_mins BIGINT,
    cpmile DOUBLE,
    bev_energy DOUBLE,
    annmain DOUBLE,
    tax_credit DOUBLE,
    rebate DOUBLE,
    int_volume BIGINT,
    total_sales DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, data_year, fueltype, bodytype)
);

COMMENT ON TABLE new_vehicle_representative_vehicle IS 'Mutable ATLAS input file: new_vehicle_representative_vehicle.csv';
COMMENT ON COLUMN new_vehicle_representative_vehicle.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN new_vehicle_representative_vehicle.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN new_vehicle_representative_vehicle.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicle_representative_vehicle.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicle_representative_vehicle.data_year IS 'The year of the data record.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.fueltype IS 'Type of fuel (e.g., Gasoline, BEV).';
COMMENT ON COLUMN new_vehicle_representative_vehicle.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN new_vehicle_representative_vehicle.price IS 'Price of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.accel IS 'Acceleration of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.range IS 'Range of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.mpge IS 'Miles per gallon equivalent of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.refueltime_mins IS 'Refueling time in minutes of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.cpmile IS 'Cost per mile of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.bev_energy IS 'Battery electric vehicle energy consumption of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.annmain IS 'Annual maintenance cost of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.tax_credit IS 'Tax credit available for the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.rebate IS 'Rebate available for the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.int_volume IS 'Interior volume of the representative vehicle.';
COMMENT ON COLUMN new_vehicle_representative_vehicle.total_sales IS 'Total sales for the representative vehicle category.';


CREATE TABLE IF NOT EXISTS new_vehicles (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('new_vehicles_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    data_year BIGINT NOT NULL,
    fueltype VARCHAR NOT NULL,
    bodytype VARCHAR NOT NULL,
    price DOUBLE,
    accel DOUBLE,
    range BIGINT,
    mpge DOUBLE,
    refueltime_mins BIGINT,
    cpmile DOUBLE,
    bev_energy BIGINT,
    annmain BIGINT,
    tax_credit BIGINT,
    rebate BIGINT,
    sales DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, data_year, fueltype, bodytype)
);

COMMENT ON TABLE new_vehicles IS 'Mutable ATLAS input file: new_vehicles.csv';
COMMENT ON COLUMN new_vehicles.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN new_vehicles.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN new_vehicles.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicles.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicles.data_year IS 'The year of the data record.';
COMMENT ON COLUMN new_vehicles.fueltype IS 'Type of fuel (e.g., Gasoline, BEV).';
COMMENT ON COLUMN new_vehicles.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN new_vehicles.price IS 'Price of the new vehicle.';
COMMENT ON COLUMN new_vehicles.accel IS 'Acceleration of the new vehicle.';
COMMENT ON COLUMN new_vehicles.range IS 'Range of the new vehicle.';
COMMENT ON COLUMN new_vehicles.mpge IS 'Miles per gallon equivalent of the new vehicle.';
COMMENT ON COLUMN new_vehicles.refueltime_mins IS 'Refueling time in minutes of the new vehicle.';
COMMENT ON COLUMN new_vehicles.cpmile IS 'Cost per mile of the new vehicle.';
COMMENT ON COLUMN new_vehicles.bev_energy IS 'Battery electric vehicle energy consumption of the new vehicle.';
COMMENT ON COLUMN new_vehicles.annmain IS 'Annual maintenance cost of the new vehicle.';
COMMENT ON COLUMN new_vehicles.tax_credit IS 'Tax credit available for the new vehicle.';
COMMENT ON COLUMN new_vehicles.rebate IS 'Rebate available for the new vehicle.';
COMMENT ON COLUMN new_vehicles.sales IS 'Sales of the new vehicle.';


CREATE TABLE IF NOT EXISTS new_vehicles_biannual_values (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('new_vehicles_biannual_values_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    fueltype VARCHAR NOT NULL,
    bodytype VARCHAR NOT NULL,
    price DOUBLE,
    accel DOUBLE,
    range DOUBLE,
    mpge DOUBLE,
    refueltime_mins DOUBLE,
    cpmile DOUBLE,
    bev_energy DOUBLE,
    annmain DOUBLE,
    tax_credit BIGINT,
    rebate BIGINT,
    total_sales DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, fueltype, bodytype)
);

COMMENT ON TABLE new_vehicles_biannual_values IS 'Mutable ATLAS input file: new_vehicles_biannual_values_2049.csv';
COMMENT ON COLUMN new_vehicles_biannual_values.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN new_vehicles_biannual_values.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN new_vehicles_biannual_values.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicles_biannual_values.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN new_vehicles_biannual_values.fueltype IS 'Type of fuel (e.g., Gasoline, BEV).';
COMMENT ON COLUMN new_vehicles_biannual_values.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN new_vehicles_biannual_values.price IS 'Price of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.accel IS 'Acceleration of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.range IS 'Range of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.mpge IS 'Miles per gallon equivalent of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.refueltime_mins IS 'Refueling time in minutes of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.cpmile IS 'Cost per mile of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.bev_energy IS 'Battery electric vehicle energy consumption of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.annmain IS 'Annual maintenance cost of the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.tax_credit IS 'Tax credit available for the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.rebate IS 'Rebate available for the new vehicle.';
COMMENT ON COLUMN new_vehicles_biannual_values.total_sales IS 'Total sales for the vehicle category.';

CREATE TABLE IF NOT EXISTS used_vehicles (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('used_vehicles_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    model_year BIGINT NOT NULL,
    bodytype VARCHAR NOT NULL,
    fueltype VARCHAR NOT NULL,
    new_price DOUBLE,
    accel DOUBLE,
    range BIGINT,
    mpge DOUBLE,
    refueltime_mins BIGINT,
    cpmile DOUBLE,
    bev_energy DOUBLE,
    annmain DOUBLE,
    price DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, model_year, bodytype, fueltype)
);

COMMENT ON TABLE used_vehicles IS 'Mutable ATLAS input file: used_vehicles_2047.csv';
COMMENT ON COLUMN used_vehicles.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN used_vehicles.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN used_vehicles.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN used_vehicles.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN used_vehicles.model_year IS 'Model year of the used vehicle.';
COMMENT ON COLUMN used_vehicles.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN used_vehicles.fueltype IS 'Type of fuel (e.g., Gasoline, BEV).';
COMMENT ON COLUMN used_vehicles.new_price IS 'Price of the vehicle when it was new.';
COMMENT ON COLUMN used_vehicles.accel IS 'Acceleration of the used vehicle.';
COMMENT ON COLUMN used_vehicles.range IS 'Range of the used vehicle.';
COMMENT ON COLUMN used_vehicles.mpge IS 'Miles per gallon equivalent of the used vehicle.';
COMMENT ON COLUMN used_vehicles.refueltime_mins IS 'Refueling time in minutes of the used vehicle.';
COMMENT ON COLUMN used_vehicles.cpmile IS 'Cost per mile of the used vehicle.';
COMMENT ON COLUMN used_vehicles.bev_energy IS 'Battery electric vehicle energy consumption of the used vehicle.';
COMMENT ON COLUMN used_vehicles.annmain IS 'Annual maintenance cost of the used vehicle.';
COMMENT ON COLUMN used_vehicles.price IS 'Price of the used vehicle.';

CREATE TABLE IF NOT EXISTS vehicle_type_mapping_baseline (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('vehicle_type_mapping_baseline_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    unnamed__0 BIGINT,
    vehicletypeid VARCHAR NOT NULL,
    bodytype VARCHAR NOT NULL,
    modelyear BIGINT NOT NULL,
    adopt_fuel VARCHAR NOT NULL,
    sampleprobabilitywithincategory DOUBLE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, vehicletypeid, bodytype, modelyear, adopt_fuel)
);

COMMENT ON TABLE vehicle_type_mapping_baseline IS 'Mutable ATLAS input file: vehicle_type_mapping_baseline.csv';
COMMENT ON COLUMN vehicle_type_mapping_baseline.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN vehicle_type_mapping_baseline.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN vehicle_type_mapping_baseline.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN vehicle_type_mapping_baseline.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN vehicle_type_mapping_baseline.unnamed__0 IS 'Original index or unnamed column from source file.';
COMMENT ON COLUMN vehicle_type_mapping_baseline.vehicletypeid IS 'Identifier for the vehicle type.';
COMMENT ON COLUMN vehicle_type_mapping_baseline.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN vehicle_type_mapping_baseline.modelyear IS 'Model year of the vehicle.';
COMMENT ON COLUMN vehicle_type_mapping_baseline.adopt_fuel IS 'Type of fuel the vehicle uses (e.g., Gasoline, BEV).';
COMMENT ON COLUMN vehicle_type_mapping_baseline.sampleprobabilitywithincategory IS 'Sample probability within the category.';


CREATE TABLE IF NOT EXISTS vehicles (
    -- Add an auto-incrementing primary key for uniqueness
    id BIGINT PRIMARY KEY DEFAULT nextval('vehicles_id_seq'),

    -- Foreign keys and metadata to link to the main run, file, and context
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR,
    year INTEGER NOT NULL,
    iteration INTEGER, -- ATLAS does not typically have iterations for this data
    sub_iteration INTEGER, -- ATLAS does not typically have sub-iterations for this data
    household_id INTEGER NOT NULL,
    vehicle_id BIGINT NOT NULL,
    bodytype VARCHAR,
    pred_power VARCHAR,
    ownlease VARCHAR,
    modelyear BIGINT,
    adopt_fuel VARCHAR,
    adopt_veh VARCHAR,
    acquire_year DOUBLE,
    vehicle_tag VARCHAR,
    data_year BIGINT,
    newhhflag DOUBLE,
    maindriver_id DOUBLE,
    vintage_category VARCHAR,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    UNIQUE (run_id, year, household_id, vehicle_id)
);

COMMENT ON TABLE vehicles IS 'ATLAS vehicles_2023.csv output for year 2023';
COMMENT ON COLUMN vehicles.run_id IS 'Foreign key to parent PILATES run';
COMMENT ON COLUMN vehicles.year IS 'Simulation year this data corresponds to';
COMMENT ON COLUMN vehicles.iteration IS 'Simulation iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN vehicles.sub_iteration IS 'Simulation sub-iteration (not typically used for ATLAS data)';
COMMENT ON COLUMN vehicles.household_id IS 'Identifier for the household owning the vehicle.';
COMMENT ON COLUMN vehicles.vehicle_id IS 'Unique identifier for the vehicle.';
COMMENT ON COLUMN vehicles.bodytype IS 'Body type of the vehicle (e.g., Sedan, SUV).';
COMMENT ON COLUMN vehicles.pred_power IS 'Predicted power of the vehicle.';
COMMENT ON COLUMN vehicles.ownlease IS 'Ownership status of the vehicle (Own or Lease).';
COMMENT ON COLUMN vehicles.modelyear IS 'Model year of the vehicle.';
COMMENT ON COLUMN vehicles.adopt_fuel IS 'Type of fuel the vehicle uses (e.g., Gasoline, BEV).';
COMMENT ON COLUMN vehicles.adopt_veh IS 'Adopted vehicle type.';
COMMENT ON COLUMN vehicles.acquire_year IS 'Year the vehicle was acquired.';
COMMENT ON COLUMN vehicles.vehicle_tag IS 'A tag or label for the vehicle.';
COMMENT ON COLUMN vehicles.data_year IS 'The year of the data record.';
COMMENT ON COLUMN vehicles.newhhflag IS 'Flag indicating if the household is new.';
COMMENT ON COLUMN vehicles.maindriver_id IS 'Identifier for the main driver of the vehicle.';
COMMENT ON COLUMN vehicles.vintage_category IS 'Category for the vehicles vintage.';