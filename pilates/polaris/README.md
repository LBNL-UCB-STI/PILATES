# POLARIS Integration Architecture Breadcrumb

This package retains the reusable SQLite supply/demand/result helpers required to couple POLARIS with UrbanSim and ATLAS under PILATES's native `@consist.define_step` execution model.

## Planned `@consist.define_step` Pipeline

When POLARIS is wired into the active PILATES workflow, it decomposes into three explicit `@consist.define_step` functions:

### 1. `polaris_preprocess`
- **Population & Fleet Projection (`synergy_schema.pipeline.polaris_projection`)**:
  - Synthesizes `PopDelta.sqlite` (and/or demand tables via declarative DuckDB `translation.sql`) directly from canonical post-ATLAS Parquet artifacts (`households.parquet`, `persons.parquet`, `vehicles.parquet`).
  - Validates the committed SQLite population delta via `verify_popdelta_commit`.
- **Zonal Employment Supply Update (`pilates.polaris.preprocessor.update_supply_zone_employment`)**:
  - Aggregates UrbanSim `jobs` by zone and NAICS sector (`aggregate_zonal_employment`) and updates the `zone` table employment columns (`employment_total`, `employment_retail`, `employment_government`, `employment_manufacturing`, `employment_services`, `employment_industrial`, `employment_other`) in `{db_name}-Supply.sqlite`.

### 2. `polaris_run`
- Executes the POLARIS integrated simulation binary/container against the prepared scenario configuration and SQLite inputs.
- Produces:
  - `{db_name}-Demand.sqlite`
  - `{db_name}-Supply.sqlite`
  - `{db_name}-Result.sqlite`
  - Native OMX highway and transit skims (`highway_skim_file.omx`, `transit_skim_file.omx`) generated directly by POLARIS.

### 3. `polaris_postprocess`
- **Work & School Location Feedback (`pilates.polaris.postprocessor.extract_work_and_school_locations`)**:
  - Extracts `work_location_id` and `school_location_id` from `Person` / `Household` in `{db_name}-Demand.sqlite`.
  - Converts 0-indexed POLARIS person indices to 1-indexed `member_id` values (`{household_id}_{person_index + 1}`).
  - Optionally maps POLARIS activity locations to TAZ `zone_id` via the `Location` table in `{db_name}-Supply.sqlite`, and updates the `persons` table (`work_zone_id`, `school_zone_id`) in the UrbanSim HDF5 datastore while preserving existing values for unsampled persons.
- **Multi-SQLite Summary & Skims (`pilates.polaris.postprocessor.summarize_polaris`)**:
  - Attaches `{db_name}-Supply.sqlite` (`AS a`) and `{db_name}-Result.sqlite` (`AS b`) onto `{db_name}-Demand.sqlite` to query `b.ZoneWaitTimes` and execute convergence/summary SQL scripts (`conv_scripts/wtf_baseline_analysis.sql`).
  - Passes POLARIS's native `.omx` skim outputs downstream without legacy binary conversion.

## Legacy Code Removed in PR 1.1

- **`skim_file_reader.py`**: Deleted because modern POLARIS writes native `.omx` skims directly, eliminating custom binary `.bin` skim parsing and HDF5 conversion in PILATES.
- **`travel_model.py`**: Deleted because imperative `polarisruntime` loop orchestration and in-place JSON string munging are superseded by Consist step orchestration.
- **Procedural SQLite & HDF5 Classes in `preprocessor.py`**: Deleted `Household`, `Person`, `Vehicle`, `Job`, `Block`, `Zone`, `Usim_Data`, `Polaris_Data`, `SortedDict` block sampling, and `preprocess_usim_for_polaris`, which are superseded by `synergy_schema.pipeline.polaris_projection` (`translation.sql`).
- **Hardcoded AV VOTT Loops in `postprocessor.py`**: Deleted `OD_record`, `O_record`, `get_factor`, `archive_and_generate_usim_skims`, and `generate_polaris_skims_for_usim`, which mutated `.bin` travel time skims in Python using hardcoded autonomous-vehicle value-of-travel-time multipliers.
