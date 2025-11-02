-- Simplified database views for non-technical users

-- View 1: Run Summary
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
ORDER BY r.created_at DESC;
COMMENT ON VIEW run_summary IS 'Simplified summary of all PILATES runs with key metadata and counts';

-- View 2: Data Lineage Summary
CREATE OR REPLACE VIEW data_lineage_summary AS
SELECT
    f.short_name as dataset_name,
    f.description,
    f.year,
    f.iteration,
    f.sub_iteration,
    f.file_path,
    array_to_string(f.models, ' → ') as processing_chain,
    array_length(f.source_file_paths, 1) as num_input_files,
    r.run_id,
    r.created_at as run_date,
    cs.region
FROM file_records f
JOIN runs r ON f.run_id = r.run_id
LEFT JOIN config_snapshots cs ON r.config_snapshot_id = cs.snapshot_id
ORDER BY f.created_at DESC;
COMMENT ON VIEW data_lineage_summary IS 'Complete data lineage showing how files were processed across model stages';

-- View 3: Model Performance Summary
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
ORDER BY mr.model, mr.year;
COMMENT ON VIEW model_performance_summary IS 'Model execution performance metrics including runtime and success rates';

-- View 4: Household Demographics Summary
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
GROUP BY h.run_id;
COMMENT ON VIEW household_demographics_summary IS 'Aggregated household demographic statistics by run';

-- View 5: TAZ Level Summary
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
WHERE lu.TOTPOP > 0;
COMMENT ON VIEW taz_summary IS 'Traffic Analysis Zone level summary statistics for mapping and spatial analysis';

-- View 6: Run Comparison Helper
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
ORDER BY r.created_at DESC;
COMMENT ON VIEW run_comparison IS 'Side-by-side comparison of key metrics across different runs';

-- View 7: Employment by Sector
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
GROUP BY lu.run_id;
COMMENT ON VIEW employment_by_sector IS 'Employment distribution by industry sector';

-- View 8: Recent Activity Log
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
ORDER BY mr.created_at DESC;
COMMENT ON VIEW recent_activity IS 'Recent model executions from the last 7 days for monitoring purposes';
