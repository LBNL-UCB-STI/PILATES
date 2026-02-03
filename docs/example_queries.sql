-- ============================================================
-- PILATES DATABASE: EXAMPLE QUERIES
-- ============================================================
--
-- This file contains example SQL queries for common analysis tasks
-- using the PILATES database. These queries demonstrate how to:
--   - Explore run metadata and provenance
--   - Trace data lineage across model stages
--   - Compare results between different runs
--   - Query simulation data for analysis
--
-- Usage:
--   1. Connect to your DuckDB database
--   2. Copy and paste queries, modifying parameters as needed
--   3. Or use DuckDB CLI: duckdb database.duckdb < example_queries.sql
--
-- ============================================================

-- ============================================================
-- SECTION 1: EXPLORING RUNS AND METADATA
-- ============================================================

-- Q1.1: List all simulation runs with basic information
-- Shows when runs were executed, what models were used, and time period
SELECT
    run_id,
    created_at,
    start_year,
    end_year,
    array_to_string(models_used, ', ') as models,
    hostname,
    code_version
FROM runs
ORDER BY created_at DESC;


-- Q1.2: Find runs for a specific region
-- Replace 'sfbay' with your region of interest (austin, seattle, etc.)
SELECT
    r.run_id,
    r.created_at,
    cs.region,
    r.start_year,
    r.end_year,
    array_to_string(r.models_used, ', ') as models
FROM runs r
JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
WHERE cs.region = 'sfbay'
ORDER BY r.created_at DESC;


-- Q1.3: Find runs with matching configuration
-- Useful for finding runs that used the same settings
SELECT
    r.run_id,
    r.created_at,
    r.config_content_hash,
    COUNT(*) OVER (PARTITION BY r.config_content_hash) as runs_with_same_config
FROM runs r
WHERE r.config_content_hash IS NOT NULL
ORDER BY r.config_content_hash, r.created_at;


-- Q1.4: Get complete configuration for a specific run
-- Replace 'your-run-id' with actual run_id
SELECT
    cs.*
FROM config_snapshots cs
JOIN runs r ON r.config_snapshot_id = cs.snapshot_id
WHERE r.run_id = 'your-run-id';


-- Q1.5: Find all files produced by a specific run
SELECT
    short_name,
    description,
    year,
    file_path,
    array_to_string(models, ' → ') as processing_chain
FROM file_records
WHERE run_id = 'your-run-id'
ORDER BY created_at;


-- ============================================================
-- SECTION 2: DATA LINEAGE AND PROVENANCE
-- ============================================================

-- Q2.1: Trace complete lineage for a specific file
-- Shows all upstream files that contributed to this file
-- Replace 'activitysim_beam_plans' with your file of interest
WITH RECURSIVE lineage AS (
    -- Start with the file of interest
    SELECT
        unique_id,
        short_name,
        file_path,
        source_file_paths,
        array_to_string(models, ' → ') as models_chain,
        1 as depth,
        CAST(short_name AS VARCHAR) as lineage_path
    FROM file_records
    WHERE short_name = 'activitysim_beam_plans'

    UNION ALL

    -- Recursively find source files
    SELECT
        f.unique_id,
        f.short_name,
        f.file_path,
        f.source_file_paths,
        array_to_string(f.models, ' → ') as models_chain,
        l.depth + 1,
        l.lineage_path || ' ← ' || f.short_name as lineage_path
    FROM file_records f
    JOIN lineage l ON f.file_path = ANY(l.source_file_paths)
    WHERE l.depth < 10  -- Prevent infinite loops
)
SELECT
    depth,
    short_name,
    models_chain,
    lineage_path
FROM lineage
ORDER BY depth DESC, short_name;


-- Q2.2: Find all downstream consumers of a file
-- Shows what files were derived from a specific dataset
SELECT
    fr_downstream.short_name as downstream_file,
    fr_downstream.description,
    array_to_string(fr_downstream.models, ', ') as processing_models,
    fr_upstream.short_name as source_file
FROM file_records fr_upstream
JOIN file_records fr_downstream
    ON fr_upstream.file_path = ANY(fr_downstream.source_file_paths)
WHERE fr_upstream.short_name = 'urbansim_h5'
ORDER BY fr_downstream.created_at;


-- Q2.3: Find model runs for a specific model and year
SELECT
    mr.unique_id,
    mr.model,
    mr.year,
    mr.iteration,
    mr.status,
    mr.created_at,
    mr.completed_at,
    EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0 as runtime_minutes
FROM model_runs mr
WHERE mr.model = 'activitysim'
  AND mr.year = 2018
ORDER BY mr.iteration;


-- Q2.4: Trace data flow through complete workflow
-- Shows how data flows: UrbanSim → ATLAS → ActivitySim → BEAM
SELECT
    mr.year,
    mr.iteration,
    mr.model,
    mr.description,
    mr.status,
    array_length(mr.input_record_hashes, 1) as num_inputs,
    array_length(mr.output_record_hashes, 1) as num_outputs,
    EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0 as runtime_minutes
FROM model_runs mr
WHERE mr.run_id = 'your-run-id'
ORDER BY mr.created_at;


-- ============================================================
-- SECTION 3: COMPARING RUNS
-- ============================================================

-- Q3.1: Compare household characteristics across runs
-- Useful for understanding impact of different scenarios
SELECT
    r.run_id,
    cs.region,
    r.start_year,
    COUNT(DISTINCT h.household_id) as total_households,
    AVG(h.income) as avg_income,
    AVG(h.cars) as avg_cars_per_hh,
    AVG(h.persons) as avg_hh_size,
    AVG(h.workers) as avg_workers_per_hh
FROM runs r
JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
JOIN activitysim_households h ON h.run_id = r.run_id
GROUP BY r.run_id, cs.region, r.start_year
ORDER BY r.created_at DESC;


-- Q3.2: Compare land use across TAZs for different runs
SELECT
    r.run_id,
    lu.TAZ,
    lu.TOTPOP as population,
    lu.TOTHH as households,
    lu.TOTEMP as employment,
    lu.employment_density as emp_per_acre
FROM runs r
JOIN activitysim_land_use lu ON lu.run_id = r.run_id
WHERE lu.TAZ IN ('1', '2', '3')  -- Replace with TAZs of interest
ORDER BY r.run_id, lu.TAZ;


-- Q3.3: Compare model runtimes across runs
-- Helps identify performance improvements or regressions
SELECT
    r.run_id,
    mr.model,
    AVG(EXTRACT(EPOCH FROM (mr.completed_at - mr.created_at)) / 60.0) as avg_runtime_minutes,
    COUNT(*) as num_executions
FROM runs r
JOIN model_runs mr ON mr.run_id = r.run_id
WHERE mr.status = 'completed'
GROUP BY r.run_id, mr.model
ORDER BY r.created_at DESC, mr.model;


-- ============================================================
-- SECTION 4: ANALYZING SIMULATION DATA
-- ============================================================

-- Q4.1: Income distribution summary
SELECT
    CASE
        WHEN income < 25000 THEN '< $25k'
        WHEN income < 50000 THEN '$25k-$50k'
        WHEN income < 75000 THEN '$50k-$75k'
        WHEN income < 100000 THEN '$75k-$100k'
        WHEN income < 150000 THEN '$100k-$150k'
        ELSE '> $150k'
    END as income_bracket,
    COUNT(*) as num_households,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM activitysim_households
WHERE run_id = 'your-run-id'
GROUP BY income_bracket
ORDER BY MIN(income);


-- Q4.2: Vehicle ownership distribution
SELECT
    cars as num_vehicles,
    COUNT(*) as num_households,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM activitysim_households
WHERE run_id = 'your-run-id'
GROUP BY cars
ORDER BY cars;


-- Q4.3: TAZ-level summary statistics
SELECT
    TAZ,
    TOTPOP as population,
    TOTHH as households,
    TOTEMP as employment,
    ROUND(pop_density, 2) as pop_per_acre,
    ROUND(employment_density, 2) as jobs_per_acre,
    area_type,
    PRKCST as daily_parking_cost
FROM activitysim_land_use
WHERE run_id = 'your-run-id'
  AND TOTPOP > 0  -- Filter to populated TAZs
ORDER BY TOTPOP DESC
LIMIT 20;


-- Q4.4: Age distribution of population
SELECT
    SUM(AGE0004) as age_0_4,
    SUM(AGE0519) as age_5_19,
    SUM(AGE2044) as age_20_44,
    SUM(AGE4564) as age_45_64,
    SUM(AGE64P) as age_65_plus,
    SUM(AGE0004 + AGE0519 + AGE2044 + AGE4564 + AGE64P) as total_population
FROM activitysim_land_use
WHERE run_id = 'your-run-id';


-- Q4.5: Employment by sector
SELECT
    SUM(EMPRES) as residential_emp,
    SUM(RETEMPN) as retail_emp,
    SUM(FPSEMPN) as financial_professional_emp,
    SUM(HEREMPN) as health_education_emp,
    SUM(AGREMPN) as agricultural_emp,
    SUM(MWTEMPN) as manufacturing_warehouse_emp,
    SUM(OTHEMPN) as other_emp,
    SUM(EMPRES + RETEMPN + FPSEMPN + HEREMPN + AGREMPN + MWTEMPN + OTHEMPN) as total_emp
FROM activitysim_land_use
WHERE run_id = 'your-run-id';


-- Q4.6: School enrollment summary
SELECT
    SUM(HSENROLL) as high_school_enrollment,
    SUM(COLLFTE) as college_fulltime,
    SUM(COLLPTE) as college_parttime,
    SUM(HSENROLL + COLLFTE + COLLPTE) as total_enrollment
FROM activitysim_land_use
WHERE run_id = 'your-run-id';


-- ============================================================
-- SECTION 5: DATA QUALITY CHECKS
-- ============================================================

-- Q5.1: Check for missing data in key tables
SELECT
    'activitysim_households' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN household_id IS NULL THEN 1 ELSE 0 END) as missing_household_id,
    SUM(CASE WHEN TAZ IS NULL THEN 1 ELSE 0 END) as missing_taz,
    SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) as missing_income
FROM activitysim_households
WHERE run_id = 'your-run-id'

UNION ALL

SELECT
    'activitysim_persons' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN person_id IS NULL THEN 1 ELSE 0 END) as missing_person_id,
    SUM(CASE WHEN household_id IS NULL THEN 1 ELSE 0 END) as missing_household_id,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as missing_age
FROM activitysim_persons
WHERE run_id = 'your-run-id';


-- Q5.2: Validate household-person relationships
-- Should return 0 rows if all relationships are valid
SELECT
    p.person_id,
    p.household_id,
    'Person has no matching household' as issue
FROM activitysim_persons p
LEFT JOIN activitysim_households h
    ON p.household_id = h.household_id
    AND p.run_id = h.run_id
WHERE p.run_id = 'your-run-id'
  AND h.household_id IS NULL;


-- Q5.3: Check for duplicate records
SELECT
    household_id,
    COUNT(*) as duplicate_count
FROM activitysim_households
WHERE run_id = 'your-run-id'
GROUP BY household_id
HAVING COUNT(*) > 1;


-- Q5.4: Summary of data completeness
SELECT
    COUNT(DISTINCT run_id) as total_runs,
    COUNT(DISTINCT fr.run_id) as runs_with_files,
    COUNT(DISTINCT mr.run_id) as runs_with_model_executions,
    SUM(CASE WHEN r.config_snapshot_id IS NOT NULL THEN 1 ELSE 0 END) as runs_with_config
FROM runs r
LEFT JOIN file_records fr ON fr.run_id = r.run_id
LEFT JOIN model_runs mr ON mr.run_id = r.run_id;


-- ============================================================
-- SECTION 6: OPENLINEAGE EVENT TRACKING
-- ============================================================

-- Q6.1: View all events for a specific run
SELECT
    oe.event_time,
    oe.event_type,
    oe.job_name,
    mr.model,
    mr.year,
    mr.iteration,
    mr.status
FROM openlineage_events oe
JOIN model_runs mr ON oe.model_run_id = mr.unique_id
WHERE oe.run_id = 'your-run-id'
ORDER BY oe.event_time;


-- Q6.2: Find failed model executions
SELECT
    r.run_id,
    mr.model,
    mr.year,
    mr.iteration,
    mr.description,
    mr.created_at,
    mr.status
FROM model_runs mr
JOIN runs r ON mr.run_id = r.run_id
WHERE mr.status = 'failed'
ORDER BY mr.created_at DESC;


-- Q6.3: Count events by type and model
SELECT
    mr.model,
    oe.event_type,
    COUNT(*) as event_count
FROM openlineage_events oe
JOIN model_runs mr ON oe.model_run_id = mr.unique_id
GROUP BY mr.model, oe.event_type
ORDER BY mr.model, oe.event_type;


-- ============================================================
-- SECTION 7: ADVANCED ANALYSIS
-- ============================================================

-- Q7.1: Jobs-housing balance by TAZ
SELECT
    TAZ,
    TOTHH as households,
    TOTEMP as employment,
    CASE
        WHEN TOTHH > 0 THEN ROUND(TOTEMP * 1.0 / TOTHH, 2)
        ELSE NULL
    END as jobs_per_household,
    CASE
        WHEN TOTEMP * 1.0 / NULLIF(TOTHH, 0) > 2 THEN 'Job-rich'
        WHEN TOTEMP * 1.0 / NULLIF(TOTHH, 0) > 0.5 THEN 'Balanced'
        ELSE 'Housing-rich'
    END as balance_category
FROM activitysim_land_use
WHERE run_id = 'your-run-id'
  AND TOTHH > 0
ORDER BY TOTEMP DESC
LIMIT 30;


-- Q7.2: Household size vs vehicle ownership
SELECT
    persons as household_size,
    AVG(cars) as avg_vehicles,
    COUNT(*) as num_households
FROM activitysim_households
WHERE run_id = 'your-run-id'
GROUP BY persons
ORDER BY persons;


-- Q7.3: Worker density by area type
SELECT
    area_type,
    COUNT(DISTINCT TAZ) as num_tazs,
    SUM(TOTEMP) as total_employment,
    AVG(employment_density) as avg_job_density
FROM activitysim_land_use
WHERE run_id = 'your-run-id'
GROUP BY area_type
ORDER BY area_type;


-- Q7.4: Year-over-year comparison (if multiple years exist)
SELECT
    h18.TAZ,
    h18.TOTPOP as pop_2018,
    h25.TOTPOP as pop_2025,
    h25.TOTPOP - h18.TOTPOP as pop_change,
    ROUND((h25.TOTPOP - h18.TOTPOP) * 100.0 / NULLIF(h18.TOTPOP, 0), 1) as pct_change
FROM activitysim_land_use h18
JOIN activitysim_land_use h25
    ON h18.TAZ = h25.TAZ
WHERE h18.run_id = 'run-with-year-2018'
  AND h25.run_id = 'run-with-year-2025'
  AND h18.TOTPOP > 0
ORDER BY ABS(h25.TOTPOP - h18.TOTPOP) DESC
LIMIT 20;


-- ============================================================
-- TIPS FOR USING THESE QUERIES
-- ============================================================
--
-- 1. Always replace 'your-run-id' with actual run_id values
--    You can find run_ids with: SELECT run_id FROM runs;
--
-- 2. Filter by region using config_snapshots table
--    JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
--    WHERE cs.region = 'sfbay'
--
-- 3. For time-series analysis, filter by year
--    WHERE mr.year BETWEEN 2018 AND 2025
--
-- 4. Use EXPLAIN to understand query performance
--    EXPLAIN SELECT ... FROM ...
--
-- 5. Export results to CSV
--    COPY (your_query) TO 'output.csv' (HEADER, DELIMITER ',');
--
-- 6. For large result sets, add LIMIT clause
--    ... ORDER BY column LIMIT 1000;
--
-- ============================================================
