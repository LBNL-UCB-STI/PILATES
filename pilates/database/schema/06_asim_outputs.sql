-- Auto-generated schema for table: households_asim_out
-- Description: ActivitySim output file: households - Socio-economic and demographic characteristics of households.

CREATE TABLE IF NOT EXISTS households_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    household_id BIGINT NOT NULL,
    block_id BIGINT, -- Block ID where the household resides
    home_zone_id BIGINT, -- Traffic Analysis Zone (TAZ) ID of the household's home
    income VARCHAR, -- Household income
    hhsize BIGINT, -- Household size (number of persons)
    hht BIGINT, -- Household type
    auto_ownership BIGINT, -- Number of vehicles owned by the household
    num_workers VARCHAR, -- Number of workers in the household
    sample_rate VARCHAR, -- Sample rate used for the household
    income_in_thousands NUMERIC, -- Household income in thousands
    income_segment BIGINT, -- Income segment of the household
    median_value_of_time VARCHAR, -- Median value of time for the household
    hh_value_of_time VARCHAR, -- Household value of time
    num_non_workers VARCHAR, -- Number of non-workers in the household
    num_drivers BIGINT, -- Number of drivers in the household
    num_adults BIGINT, -- Number of adults in the household
    num_children BIGINT, -- Number of children in the household
    num_young_children BIGINT, -- Number of young children in the household
    num_children_5_to_15 BIGINT, -- Number of children aged 5 to 15
    num_children_16_to_17 BIGINT, -- Number of children aged 16 to 17
    num_college_age BIGINT, -- Number of college-aged individuals
    num_young_adults BIGINT, -- Number of young adults
    non_family BOOLEAN, -- True if non-family household
    family BOOLEAN, -- True if family household
    home_is_urban BOOLEAN, -- True if home is in an urban area
    home_is_rural BOOLEAN, -- True if home is in a rural area
    hh_work_auto_savings_ratio DOUBLE, -- Household work auto savings ratio
    num_under16_not_at_school BIGINT, -- Number of household members under 16 not at school
    num_travel_active BIGINT, -- Number of travel-active household members
    num_travel_active_adults BIGINT, -- Number of travel-active adults
    num_travel_active_preschoolers BIGINT, -- Number of travel-active preschoolers
    num_travel_active_children BIGINT, -- Number of travel-active children
    num_travel_active_non_preschoolers BIGINT, -- Number of travel-active non-preschoolers
    participates_in_jtf_model BOOLEAN, -- True if participates in joint tour frequency model
    joint_tour_frequency VARCHAR, -- Joint tour frequency
    num_hh_joint_tours BIGINT, -- Number of household joint tours

    PRIMARY KEY (run_id, year, iteration, sub_iteration, household_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE households_asim_out IS 'ActivitySim output file: households - Socio-economic and demographic characteristics of households, including income, size, auto ownership, and worker counts.';

CREATE INDEX IF NOT EXISTS idx_households_asim_out_run_id ON households_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_households_asim_out_year_iter_sub_iter ON households_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_households_asim_out_household_id ON households_asim_out(household_id);
CREATE INDEX IF NOT EXISTS idx_households_asim_out_home_zone_id ON households_asim_out(home_zone_id);

-- Auto-generated schema for table: persons_asim_out
-- Description: ActivitySim output file: persons - Detailed information about individuals, including demographics, employment, education, and travel characteristics.

CREATE TABLE IF NOT EXISTS persons_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    person_id BIGINT NOT NULL,
    household_id BIGINT, -- Foreign key to households_asim_out
    age BIGINT, -- Age of the person
    pnum BIGINT, -- Person number within the household
    sex BIGINT, -- Sex of the person (e.g., 1=Male, 2=Female)
    pemploy BIGINT, -- Employment status of the person
    pstudent BIGINT, -- Student status of the person
    ptype BIGINT, -- Person type (e.g., full-time worker, part-time worker, non-worker, student)
    home_x DOUBLE, -- X-coordinate (e.g., longitude) of the person's home
    home_y DOUBLE, -- Y-coordinate (e.g., latitude) of the person's home
    age_16_to_19 BOOLEAN, -- True if age is between 16 and 19
    age_16_p BOOLEAN, -- True if age is 16 or older
    adult BOOLEAN, -- True if adult (e.g., age >= 18)
    male BOOLEAN, -- True if male
    female BOOLEAN, -- True if female
    has_non_worker BOOLEAN, -- True if household has a non-worker
    has_retiree BOOLEAN, -- True if household has a retiree
    has_preschool_kid BOOLEAN, -- True if household has a preschool kid
    has_driving_kid BOOLEAN, -- True if household has a driving-age kid
    has_school_kid BOOLEAN, -- True if household has a school-aged kid
    has_full_time BOOLEAN, -- True if household has a full-time worker
    has_part_time BOOLEAN, -- True if household has a part-time worker
    has_university BOOLEAN, -- True if household has a university student
    student_is_employed BOOLEAN, -- True if student is employed
    nonstudent_to_school BOOLEAN, -- True if non-student travels to school
    is_student BOOLEAN, -- True if person is a student
    is_gradeschool BOOLEAN, -- True if person is in gradeschool
    is_highschool BOOLEAN, -- True if person is in highschool
    is_university BOOLEAN, -- True if person is in university
    school_segment BIGINT, -- School segment
    is_worker BOOLEAN, -- True if person is a worker
    home_zone_id BIGINT, -- Home zone ID
    value_of_time VARCHAR, -- Value of time
    school_zone_id BIGINT, -- School zone ID
    school_location_logsum VARCHAR, -- Logsum for school location choice
    distance_to_school DOUBLE, -- Distance to school
    roundtrip_auto_time_to_school DOUBLE, -- Roundtrip auto travel time to school
    workplace_zone_id BIGINT, -- Workplace zone ID
    workplace_location_logsum VARCHAR, -- Logsum for workplace location choice
    distance_to_work DOUBLE, -- Distance to work
    workplace_in_cbd BOOLEAN, -- True if workplace is in Central Business District
    work_zone_area_type VARCHAR, -- Area type of the work zone
    roundtrip_auto_time_to_work DOUBLE, -- Roundtrip auto travel time to work
    work_auto_savings DOUBLE, -- Work auto savings
    work_auto_savings_ratio DOUBLE, -- Work auto savings ratio
    free_parking_at_work BOOLEAN, -- True if free parking is available at work
    cdap_activity VARCHAR, -- CDAP activity
    travel_active BOOLEAN, -- True if person is travel active
    under16_not_at_school BOOLEAN, -- True if person is under 16 and not at school
    has_preschool_kid_at_home BOOLEAN, -- True if household has a preschool kid at home
    has_school_kid_at_home BOOLEAN, -- True if household has a school kid at home
    mandatory_tour_frequency VARCHAR, -- Mandatory tour frequency
    work_and_school_and_worker BOOLEAN, -- True if person works, goes to school, and is a worker
    work_and_school_and_student BOOLEAN, -- True if person works, goes to school, and is a student
    num_mand BIGINT, -- Number of mandatory tours
    num_work_tours BIGINT, -- Number of work tours
    num_joint_tours BIGINT, -- Number of joint tours
    non_mandatory_tour_frequency BIGINT, -- Non-mandatory tour frequency
    num_non_mand BIGINT, -- Number of non-mandatory tours
    num_escort_tours BIGINT, -- Number of escort tours
    num_eatout_tours BIGINT, -- Number of eat-out tours
    num_shop_tours BIGINT, -- Number of shopping tours
    num_maint_tours BIGINT, -- Number of maintenance tours
    num_discr_tours BIGINT, -- Number of discretionary tours
    num_social_tours BIGINT, -- Number of social tours
    num_non_escort_tours BIGINT, -- Number of non-escort tours

    PRIMARY KEY (run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, household_id) REFERENCES households_asim_out(run_id, year, iteration, sub_iteration, household_id)
);

COMMENT ON TABLE persons_asim_out IS 'ActivitySim output file: persons - Detailed information about individuals, including demographics (age, sex), employment and student status, home and workplace locations, and various travel-related attributes.';

CREATE INDEX IF NOT EXISTS idx_persons_asim_out_run_id ON persons_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_year_iter_sub_iter ON persons_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_person_id ON persons_asim_out(person_id);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_household_id ON persons_asim_out(household_id);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_home_zone_id ON persons_asim_out(home_zone_id);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_school_zone_id ON persons_asim_out(school_zone_id);
CREATE INDEX IF NOT EXISTS idx_persons_asim_out_workplace_zone_id ON persons_asim_out(workplace_zone_id);

-- Auto-generated schema for table: tours_asim_out
-- Description: ActivitySim output file: tours - Detailed information about tours, including purpose, participants, and travel characteristics.

CREATE TABLE IF NOT EXISTS tours_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    tour_id BIGINT NOT NULL,
    person_id BIGINT, -- Foreign key to persons_asim_out
    household_id BIGINT, -- Foreign key to households_asim_out
    tour_type VARCHAR, -- Type of tour (e.g., work, school, shopping)
    tour_type_count BIGINT, -- Count of tours of this type for the person
    tour_type_num BIGINT, -- Number of this tour type for the person
    tour_num BIGINT, -- Tour number for the person
    tour_count BIGINT, -- Total count of tours for the person
    tour_category VARCHAR, -- Category of the tour (e.g., mandatory, non-mandatory, joint)
    number_of_participants BIGINT, -- Number of participants in the tour
    destination BIGINT, -- Destination zone ID of the tour
    origin BIGINT, -- Origin zone ID of the tour
    tdd VARCHAR, -- Tour departure and duration
    "start" VARCHAR, -- Start time of the tour
    "end" VARCHAR, -- End time of the tour
    duration VARCHAR, -- Duration of the tour
    composition VARCHAR, -- Composition of the tour (e.g., individual, joint)
    destination_logsum DOUBLE, -- Logsum for destination choice
    tour_mode VARCHAR, -- Mode of transport for the tour
    mode_choice_logsum VARCHAR, -- Logsum for mode choice
    atwork_subtour_frequency VARCHAR, -- Frequency of at-work subtours
    parent_tour_id BIGINT, -- ID of the parent tour (for subtours)
    stop_frequency VARCHAR, -- Frequency of stops during the tour
    primary_purpose VARCHAR, -- Primary purpose of the tour

    PRIMARY KEY (run_id, year, iteration, sub_iteration, tour_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, household_id) REFERENCES households_asim_out(run_id, year, iteration, sub_iteration, household_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, parent_tour_id) REFERENCES tours_asim_out (run_id, year, iteration, sub_iteration, tour_id)
);

COMMENT ON TABLE tours_asim_out IS 'ActivitySim output file: tours - Detailed information about tours, including purpose, participants, and travel characteristics, linking to persons and households.';

CREATE INDEX IF NOT EXISTS idx_tours_asim_out_run_id ON tours_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_tours_asim_out_year_iter_sub_iter ON tours_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_tours_asim_out_tour_id ON tours_asim_out(tour_id);
CREATE INDEX IF NOT EXISTS idx_tours_asim_out_person_id ON tours_asim_out(person_id);
CREATE INDEX IF NOT EXISTS idx_tours_asim_out_household_id ON tours_asim_out(household_id);
CREATE INDEX IF NOT EXISTS idx_tours_asim_out_parent_tour_id ON tours_asim_out(parent_tour_id);

-- Auto-generated schema for table: accessibility_asim_out
-- Description: ActivitySim output file: accessibility - Accessibility measures by zone.

CREATE TABLE IF NOT EXISTS accessibility_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    zone_id BIGINT NOT NULL,
    aupkretail DOUBLE, -- Auto accessibility to retail destinations during peak hours
    aupktotal DOUBLE,  -- Auto accessibility to all destinations during peak hours
    auopretail DOUBLE, -- Auto accessibility to retail destinations during off-peak hours
    auoptotal DOUBLE,  -- Auto accessibility to all destinations during off-peak hours
    trpkretail DOUBLE, -- Transit accessibility to retail destinations during peak hours
    trpktotal DOUBLE,  -- Transit accessibility to all destinations during peak hours
    tropretail DOUBLE, -- Transit accessibility to retail destinations during off-peak hours
    troptotal DOUBLE,  -- Transit accessibility to all destinations during off-peak hours
    nmretail DOUBLE,   -- Non-motorized accessibility to retail destinations
    nmtotal DOUBLE,    -- Non-motorized accessibility to all destinations

    PRIMARY KEY (run_id, year, iteration, sub_iteration, zone_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE accessibility_asim_out IS 'ActivitySim output file: accessibility - Accessibility measures by zone, including auto, transit, and non-motorized modes for retail and total destinations during peak and off-peak hours.';

CREATE INDEX IF NOT EXISTS idx_accessibility_asim_out_run_id ON accessibility_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_accessibility_asim_out_year_iter_sub_iter ON accessibility_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_accessibility_asim_out_zone_id ON accessibility_asim_out(zone_id);

-- Auto-generated schema for table: trips_asim_out
-- Description: ActivitySim output file: trips - Detailed information about individual trips, including origin, destination, mode, and purpose.

CREATE TABLE IF NOT EXISTS trips_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    trip_id BIGINT NOT NULL,
    person_id BIGINT, -- Foreign key to persons_asim_out
    household_id BIGINT, -- Foreign key to households_asim_out
    tour_id BIGINT, -- Foreign key to tours_asim_out
    primary_purpose VARCHAR, -- Primary purpose of the trip
    trip_num BIGINT, -- Trip number within a tour or for a person
    outbound BOOLEAN, -- True if the trip is outbound
    trip_count BIGINT, -- Total count of trips
    destination BIGINT, -- Destination zone ID of the trip
    origin BIGINT, -- Origin zone ID of the trip
    purpose VARCHAR, -- Purpose of the trip
    destination_logsum DOUBLE, -- Logsum for destination choice
    "depart" VARCHAR, -- Departure time of the trip
    trip_mode VARCHAR, -- Mode of transport for the trip
    mode_choice_logsum VARCHAR, -- Logsum for mode choice

    PRIMARY KEY (run_id, year, iteration, sub_iteration, trip_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, household_id) REFERENCES households_asim_out(run_id, year, iteration, sub_iteration, household_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, tour_id) REFERENCES tours_asim_out(run_id, year, iteration, sub_iteration, tour_id)
);

COMMENT ON TABLE trips_asim_out IS 'ActivitySim output file: trips - Detailed information about individual trips, including origin, destination, mode, and purpose, linking to persons, households, and tours.';

CREATE INDEX IF NOT EXISTS idx_trips_asim_out_run_id ON trips_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_trips_asim_out_year_iter_sub_iter ON trips_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_trips_asim_out_trip_id ON trips_asim_out(trip_id);
CREATE INDEX IF NOT EXISTS idx_trips_asim_out_person_id ON trips_asim_out(person_id);
CREATE INDEX IF NOT EXISTS idx_trips_asim_out_household_id ON trips_asim_out(household_id);
CREATE INDEX IF NOT EXISTS idx_trips_asim_out_tour_id ON trips_asim_out(tour_id);

-- Auto-generated schema for table: beam_plans_asim_out
-- Description: ActivitySim output file: beam_plans - Detailed trip plans generated by ActivitySim for BEAM.

CREATE TABLE IF NOT EXISTS beam_plans_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    trip_id BIGINT NOT NULL,
    tour_id BIGINT, -- Foreign key to tours_asim_out
    person_id BIGINT, -- Foreign key to persons_asim_out
    number_of_participants VARCHAR, -- Number of participants in the tour/trip
    tour_mode VARCHAR, -- Mode of transport for the tour (e.g., 'CAR', 'WALK')
    trip_mode VARCHAR, -- Mode of transport for the trip (e.g., 'CAR', 'WALK')
    planelementindex BIGINT, -- Index of the plan element
    activityelement VARCHAR, -- Type of activity element
    activitytype VARCHAR, -- Type of activity (e.g., 'Home', 'Work', 'Shop')
    x DOUBLE, -- X-coordinate (e.g., longitude) of the activity location
    y DOUBLE, -- Y-coordinate (e.g., latitude) of the activity location
    departure_time DOUBLE, -- Departure time of the trip (e.g., in hours from midnight)
    trip_dur_min DOUBLE, -- Duration of the trip in minutes
    trip_cost_dollars DOUBLE, -- Cost of the trip in dollars
    __index_level_0__ BIGINT, -- Internal index from data processing, usually not meaningful

    PRIMARY KEY (run_id, year, iteration, sub_iteration, planelementindex),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, tour_id) REFERENCES tours_asim_out(run_id, year, iteration, sub_iteration, tour_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, trip_id) REFERENCES trips_asim_out(run_id, year, iteration, sub_iteration, trip_id)
);

COMMENT ON TABLE beam_plans_asim_out IS 'ActivitySim output file: beam_plans - Detailed trip plans generated by ActivitySim, including information about tours, trips, modes, activities, and costs.';

CREATE INDEX IF NOT EXISTS idx_beam_plans_asim_out_run_id ON beam_plans_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_beam_plans_asim_out_year_iter_sub_iter ON beam_plans_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_beam_plans_asim_out_trip_id ON beam_plans_asim_out(trip_id);
CREATE INDEX IF NOT EXISTS idx_beam_plans_asim_out_person_id ON beam_plans_asim_out(person_id);
CREATE INDEX IF NOT EXISTS idx_beam_plans_asim_out_tour_id ON beam_plans_asim_out(tour_id);

-- Auto-generated schema for table: disaggregate_accessibility_asim_out
-- Description: ActivitySim output file: disaggregate_accessibility - Accessibility measures at the individual person level.

CREATE TABLE IF NOT EXISTS disaggregate_accessibility_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    person_id BIGINT NOT NULL, -- Foreign key to persons_asim_out
    workplace_location_accessibility DOUBLE, -- Accessibility to workplace locations for the person
    othdiscr_accessibility DOUBLE, -- Accessibility to other discretionary activities for the person
    shopping_accessibility DOUBLE, -- Accessibility to shopping activities for the person

    PRIMARY KEY (run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id)
);

COMMENT ON TABLE disaggregate_accessibility_asim_out IS 'ActivitySim output file: disaggregate_accessibility - Accessibility measures at the individual person level, including workplace, other discretionary, and shopping activities.';

CREATE INDEX IF NOT EXISTS idx_disaggregate_accessibility_asim_out_run_id ON disaggregate_accessibility_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_disaggregate_accessibility_asim_out_year_iter_sub_iter ON disaggregate_accessibility_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_disaggregate_accessibility_asim_out_person_id ON disaggregate_accessibility_asim_out(person_id);

-- Auto-generated schema for table: joint_tour_participants_asim_out
-- Description: ActivitySim output file: joint_tour_participants - Details of participants in joint tours.

CREATE TABLE IF NOT EXISTS joint_tour_participants_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    tour_id BIGINT NOT NULL, -- Foreign key to tours_asim_out
    household_id BIGINT, -- Foreign key to households_asim_out
    person_id BIGINT NOT NULL, -- Foreign key to persons_asim_out
    participant_num BIGINT, -- Number of the participant within the joint tour
    participant_id BIGINT, -- Unique identifier for the participant (often same as person_id)

    PRIMARY KEY (run_id, year, iteration, sub_iteration, tour_id, person_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, tour_id) REFERENCES tours_asim_out(run_id, year, iteration, sub_iteration, tour_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, household_id) REFERENCES households_asim_out(run_id, year, iteration, sub_iteration, household_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id)
);

COMMENT ON TABLE joint_tour_participants_asim_out IS 'ActivitySim output file: joint_tour_participants - Records the participation of individuals in joint tours, linking to specific tours, households, and persons.';

CREATE INDEX IF NOT EXISTS idx_joint_tour_participants_asim_out_run_id ON joint_tour_participants_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_joint_tour_participants_asim_out_year_iter_sub_iter ON joint_tour_participants_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_joint_tour_participants_asim_out_tour_id ON joint_tour_participants_asim_out(tour_id);
CREATE INDEX IF NOT EXISTS idx_joint_tour_participants_asim_out_household_id ON joint_tour_participants_asim_out(household_id);
CREATE INDEX IF NOT EXISTS idx_joint_tour_participants_asim_out_person_id ON joint_tour_participants_asim_out(person_id);

-- Auto-generated schema for table: land_use_asim_out
-- Description: ActivitySim output file: land_use - Land use characteristics by traffic analysis zone (TAZ).

CREATE TABLE IF NOT EXISTS land_use_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    zone_id BIGINT NOT NULL,
    geometry VARCHAR, -- Geographic geometry of the zone (e.g., WKT or GeoJSON)
    county_id BIGINT, -- Identifier for the county
    tothh VARCHAR, -- Total households in the zone
    totpop VARCHAR, -- Total population in the zone
    totacre VARCHAR, -- Total acres in the zone
    totemp VARCHAR, -- Total employment in the zone
    age0519 VARCHAR, -- Population aged 5-19
    retempn VARCHAR, -- Retail employment
    fpsempn VARCHAR, -- Financial, professional, and other services employment
    herempn VARCHAR, -- Health, education, and recreation employment
    othempn VARCHAR, -- Other employment
    agrempn VARCHAR, -- Agricultural employment
    mwtempn VARCHAR, -- Manufacturing, wholesale, and transportation employment
    prkcst VARCHAR, -- Parking cost
    oprkcst VARCHAR, -- Off-peak parking cost
    area_type BIGINT, -- Type of area (e.g., urban, suburban, rural)
    hsenroll VARCHAR, -- High school enrollment
    collfte VARCHAR, -- College full-time equivalent enrollment
    collpte VARCHAR, -- College part-time equivalent enrollment
    topology BIGINT, -- Topological information
    terminal BIGINT, -- Terminal information
    _original_zone_id BIGINT, -- Original zone ID before any mapping
    household_density VARCHAR, -- Household density
    employment_density VARCHAR, -- Employment density
    density_index VARCHAR, -- Density index
    totenr_univ VARCHAR, -- Total university enrollment

    PRIMARY KEY (run_id, year, iteration, sub_iteration, zone_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE land_use_asim_out IS 'ActivitySim output file: land_use - Land use characteristics by traffic analysis zone (TAZ), including population, households, employment, and various demographic and economic indicators.';

CREATE INDEX IF NOT EXISTS idx_land_use_asim_out_run_id ON land_use_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_land_use_asim_out_year_iter_sub_iter ON land_use_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_land_use_asim_out_zone_id ON land_use_asim_out(zone_id);
CREATE INDEX IF NOT EXISTS idx_land_use_asim_out_county_id ON land_use_asim_out(county_id);

-- Auto-generated schema for table: non_mandatory_tour_destination_accessibility_asim_out
-- Description: ActivitySim output file: non_mandatory_tour_destination_accessibility - Accessibility measures for non-mandatory tour destinations, including person, household, and land-use attributes.

CREATE TABLE IF NOT EXISTS non_mandatory_tour_destination_accessibility_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    proto_person_id BIGINT NOT NULL, -- Original person ID from the input data (Foreign key to persons_asim_out)
    proto_household_id BIGINT, -- Original household ID from the input data (Foreign key to households_asim_out)
    educ BIGINT, -- Education level of the person
    grade BIGINT, -- Grade level of the person
    military BIGINT, -- Military status of the person
    pnum BIGINT, -- Person number within household
    pstudent BIGINT, -- Student status of the person
    dap VARCHAR, -- Daily activity pattern
    age BIGINT, -- Age of person
    hours BIGINT, -- Hours worked by the person
    pemploy BIGINT, -- Employment status of the person
    ptype BIGINT, -- Person type
    sex BIGINT, -- Sex of person
    weeks BIGINT, -- Weeks worked by the person
    age_16_to_19 BOOLEAN, -- True if age is between 16 and 19
    age_16_p BOOLEAN, -- True if age is 16 or older
    adult BOOLEAN, -- True if adult
    male BOOLEAN, -- True if male
    female BOOLEAN, -- True if female
    has_non_worker BOOLEAN, -- True if household has a non-worker
    has_retiree BOOLEAN, -- True if household has a retiree
    has_preschool_kid BOOLEAN, -- True if household has a preschool kid
    has_driving_kid BOOLEAN, -- True if household has a driving-age kid
    has_school_kid BOOLEAN, -- True if household has a school-aged kid
    has_full_time BOOLEAN, -- True if household has a full-time worker
    has_part_time BOOLEAN, -- True if household has a part-time worker
    has_university BOOLEAN, -- True if household has a university student
    student_is_employed BOOLEAN, -- True if student is employed
    nonstudent_to_school BOOLEAN, -- True if non-student travels to school
    is_student BOOLEAN, -- True if person is a student
    is_gradeschool BOOLEAN, -- True if person is in gradeschool
    is_highschool BOOLEAN, -- True if person is in highschool
    is_university BOOLEAN, -- True if person is in university
    school_segment BIGINT, -- School segment
    is_worker BOOLEAN, -- True if person is a worker
    home_zone_id BIGINT, -- Home zone ID
    value_of_time VARCHAR, -- Value of time
    hht BIGINT, -- Household type
    auto_ownership BIGINT, -- Auto ownership
    bldgsz BIGINT, -- Building size
    family BOOLEAN, -- True if family household
    hh_value_of_time VARCHAR, -- Household value of time
    hhsize INTEGER, -- Household size
    hinccat1 BIGINT, -- Household income category 1
    home_is_rural BOOLEAN, -- True if home is rural
    home_is_urban BOOLEAN, -- True if home is urban
    household_serial_no BIGINT, -- Household serial number
    hworkers BIGINT, -- Household workers
    income VARCHAR, -- Household income
    income_in_thousands NUMERIC, -- Household income in thousands
    income_segment INTEGER, -- Household income segment
    median_value_of_time VARCHAR, -- Median value of time
    non_family BOOLEAN, -- True if non-family household
    num_adults BIGINT, -- Number of adults
    num_children BIGINT, -- Number of children
    num_children_16_to_17 BIGINT, -- Number of children aged 16-17
    num_children_5_to_15 BIGINT, -- Number of children aged 5-15
    num_college_age BIGINT, -- Number of college-aged individuals
    num_drivers INTEGER, -- Number of drivers
    num_non_workers INTEGER, -- Number of non-workers
    num_workers INTEGER, -- Number of workers
    num_young_adults INTEGER, -- Number of young adults
    num_young_children INTEGER, -- Number of young children
    persons INTEGER, -- Number of persons
    veh INTEGER, -- Number of vehicles
    geometry VARCHAR, -- Geometry
    county_id BIGINT, -- County ID
    tothh VARCHAR, -- Total households
    totpop VARCHAR, -- Total population
    totacre VARCHAR, -- Total acres
    totemp VARCHAR, -- Total employment
    age0519 VARCHAR, -- Population aged 0-59
    retempn VARCHAR, -- Retail employment
    fpsempn VARCHAR, -- Financial, professional, and other services employment
    herempn VARCHAR, -- Health, education, and recreation employment
    othempn VARCHAR, -- Other employment
    agrempn VARCHAR, -- Agricultural employment
    mwtempn VARCHAR, -- Manufacturing, wholesale, and transportation employment
    prkcst VARCHAR, -- Parking cost
    oprkcst VARCHAR, -- Off-peak parking cost
    area_type BIGINT, -- Area type
    hsenroll VARCHAR, -- High school enrollment
    collfte VARCHAR, -- College full-time equivalent
    collpte VARCHAR, -- College part-time equivalent
    topology BIGINT, -- Topology
    terminal BIGINT, -- Terminal
    _original_zone_id BIGINT, -- Original zone ID
    household_density VARCHAR, -- Household density
    employment_density VARCHAR, -- Employment density
    density_index VARCHAR, -- Density index
    totenr_univ VARCHAR, -- Total university enrollment
    tour_num BIGINT, -- Tour number
    logsums DOUBLE, -- Logsums
    tour_type VARCHAR, -- Tour type
    purpose BIGINT, -- Purpose of tour
    person_num BIGINT, -- Person number
    tour_category VARCHAR, -- Tour category
    proto_tour_id BIGINT, -- Original tour ID

    PRIMARY KEY (run_id, year, iteration, sub_iteration, proto_person_id, proto_tour_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, proto_person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, proto_household_id) REFERENCES households_asim_out(run_id, year, iteration, sub_iteration, household_id)
);

COMMENT ON TABLE non_mandatory_tour_destination_accessibility_asim_out IS 'ActivitySim output file: non_mandatory_tour_destination_accessibility - Provides accessibility measures for non-mandatory tour destinations, enriched with detailed person, household, and land-use attributes.';

CREATE INDEX IF NOT EXISTS idx_non_mandatory_tour_destination_accessibility_asim_out_run_id ON non_mandatory_tour_destination_accessibility_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_non_mandatory_tour_destination_accessibility_asim_out_year_iter_sub_iter ON non_mandatory_tour_destination_accessibility_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_non_mandatory_tour_destination_accessibility_asim_out_proto_person_id ON non_mandatory_tour_destination_accessibility_asim_out(proto_person_id);
CREATE INDEX IF NOT EXISTS idx_non_mandatory_tour_destination_accessibility_asim_out_proto_household_id ON non_mandatory_tour_destination_accessibility_asim_out(proto_household_id);
CREATE INDEX IF NOT EXISTS idx_non_mandatory_tour_destination_accessibility_asim_out_proto_tour_id ON non_mandatory_tour_destination_accessibility_asim_out(proto_tour_id);

-- Auto-generated schema for table: person_windows_asim_out
-- Description: ActivitySim output file: person_windows - Time windows for person activities.

CREATE TABLE IF NOT EXISTS person_windows_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    person_id BIGINT NOT NULL, -- Foreign key to persons_asim_out
    _4 BIGINT, -- Activity status or availability during hour 4 (e.g., 4 AM)
    _5 BIGINT, -- Activity status or availability during hour 5
    _6 BIGINT, -- Activity status or availability during hour 6
    _7 BIGINT, -- Activity status or availability during hour 7
    _8 BIGINT, -- Activity status or availability during hour 8
    _9 BIGINT, -- Activity status or availability during hour 9
    _10 BIGINT, -- Activity status or availability during hour 10
    _11 BIGINT, -- Activity status or availability during hour 11
    _12 BIGINT, -- Activity status or availability during hour 12 (noon)
    _13 BIGINT, -- Activity status or availability during hour 13 (1 PM)
    _14 BIGINT, -- Activity status or availability during hour 14
    _15 BIGINT, -- Activity status or availability during hour 15
    _16 BIGINT, -- Activity status or availability during hour 16
    _17 BIGINT, -- Activity status or availability during hour 17
    _18 BIGINT, -- Activity status or availability during hour 18
    _19 BIGINT, -- Activity status or availability during hour 19
    _20 BIGINT, -- Activity status or availability during hour 20
    _21 BIGINT, -- Activity status or availability during hour 21
    _22 BIGINT, -- Activity status or availability during hour 22
    _23 BIGINT, -- Activity status or availability during hour 23
    _24 BIGINT, -- Activity status or availability during hour 24 (midnight)

    PRIMARY KEY (run_id, year, iteration, sub_iteration, person_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id),
    FOREIGN KEY (run_id, year, iteration, sub_iteration, person_id) REFERENCES persons_asim_out(run_id, year, iteration, sub_iteration, person_id)
);

COMMENT ON TABLE person_windows_asim_out IS 'ActivitySim output file: person_windows - Records the activity status or availability of each person during different hours of the day.';

CREATE INDEX IF NOT EXISTS idx_person_windows_asim_out_run_id ON person_windows_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_person_windows_asim_out_year_iter_sub_iter ON person_windows_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_person_windows_asim_out_person_id ON person_windows_asim_out(person_id);

-- Auto-generated schema for table: school_destination_size_asim_out
-- Description: ActivitySim output file: school_destination_size - Measures of school destination attractiveness by zone.

CREATE TABLE IF NOT EXISTS school_destination_size_asim_out (
    run_id VARCHAR NOT NULL,
    file_record_id VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    sub_iteration INTEGER NOT NULL,
    zone_id BIGINT NOT NULL,
    university VARCHAR, -- Size or attractiveness of university destinations in the zone
    gradeschool VARCHAR, -- Size or attractiveness of gradeschool destinations in the zone
    highschool VARCHAR, -- Size or attractiveness of highschool destinations in the zone

    PRIMARY KEY (run_id, year, iteration, sub_iteration, zone_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)
);

COMMENT ON TABLE school_destination_size_asim_out IS 'ActivitySim output file: school_destination_size - Provides measures of school destination attractiveness or size for universities, gradeschools, and highschools within each traffic analysis zone.';

CREATE INDEX IF NOT EXISTS idx_school_destination_size_asim_out_run_id ON school_destination_size_asim_out(run_id);
CREATE INDEX IF NOT EXISTS idx_school_destination_size_asim_out_year_iter_sub_iter ON school_destination_size_asim_out(year, iteration, sub_iteration);
CREATE INDEX IF NOT EXISTS idx_school_destination_size_asim_out_zone_id ON school_destination_size_asim_out(zone_id);

