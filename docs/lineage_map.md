# PILATES Lineage Map (UrbanSim → ATLAS → ActivitySim → BEAM)

This document consolidates the expected lineage mapping across the current
step-granular workflow. It focuses on **explicit file artifacts** rather than
directory-only references wherever possible.

## Conventions

- **Coupler key**: Stable workflow key set via `log_and_set_input/output`.
- **RecordStore short_name**: File-level artifact keys produced by model
  components.
- **Path**: Canonical file path within the workspace (when known).
- **Used by**: Downstream step(s) that consume the artifact.

If an artifact is represented only by a directory, this doc calls out the
specific files that are known to exist within that directory based on current
component logic.

## Step-by-step lineage

### 1) UrbanSim preprocess (`urbansim_preprocess`)

**Primary outputs**
- Coupler: `usim_mutable_data_dir` → UrbanSim mutable data directory
- Coupler: `usim_datastore_h5` → base-year UrbanSim input H5

**RecordStore outputs (file-level)**
- `usim_data` → `usim_mutable_data_dir/<input_file_template>` (H5)
- `omx_skims` → `usim_mutable_data_dir/skims_mpo_<region_id>.omx`
- `geoid_to_zone` → `usim_mutable_data_dir/geoid_to_zone.csv`
- `usim_skims_input_updated` → `usim_mutable_data_dir/skims_mpo_<region_id>.omx`
  (only for non-start years or iterations > 0 when BEAM skims are copied)
- `hh_size` → `usim_mutable_data_dir/hsize_ct_<region_id>.csv`
- `income_rates` → `usim_mutable_data_dir/income_rates_<region_id>.csv`
- `relmap` → `usim_mutable_data_dir/relmap_<region_id>.csv`
- `schools` → `usim_mutable_data_dir/schools_2010.csv`
- `school_districts` → `usim_mutable_data_dir/blocks_school_districts_2010.csv`
- Plus additional UrbanSim input files copied from the local input folder
  (short_name set to the file base name).

**Used by**
- `urbansim_run` (inputs: `usim_mutable_data_dir`, `usim_datastore_h5`)
- NOTE: urbansim also uses hh_size, income_rates, relmap
- `activitysim_preprocess` (input: `usim_datastore_h5`)
- `atlas_preprocess` (input: `usim_datastore_h5`)

---

### 2) UrbanSim run (`urbansim_run`)

**Primary outputs**
- Coupler: `usim_datastore_h5` → forecast-year UrbanSim output H5

**RecordStore outputs (file-level)**
- `usim_forecast_output` → `usim_mutable_data_dir/<output_file_template>` (H5)

**Used by**
- `urbansim_postprocess` (inputs: run outputs)
- `activitysim_preprocess` (input: `usim_datastore_h5`)
- `atlas_preprocess` (input: `usim_datastore_h5`)

---

### 3) UrbanSim postprocess (`urbansim_postprocess`)

**Primary outputs**
- Coupler: `usim_datastore_h5` → UrbanSim H5 prepared for downstream steps

**RecordStore outputs (file-level)**
- `usim_input_archive_<year>` → `usim_mutable_data_dir/<input_template>` (H5 archive)
- `usim_input_merged_<year>` → `usim_mutable_data_dir/<input_template>` (H5 updated for next stage)

**Used by**
- `activitysim_preprocess` (input: `usim_datastore_h5`)
- `atlas_preprocess` (input: `usim_datastore_h5`)

---

### 4) ATLAS preprocess (`atlas_preprocess`) (per sub-year)

**Primary outputs**
- `atlas_mutable_input_dir` (directory); file-level outputs are enumerated below.

**RecordStore inputs (file-level)**
- `usim_h5_container` → `usim_mutable_data_dir/<output_or_input_template>` (H5)
- `beam_skims_input` → `beam_output_dir/<skims_fname>` (when `settings.atlas.beamac > 0`)
- `atlas_rdata_accessibility` → `atlas_mutable_input_dir/*.RData` (when `beamac == 0`)
- Static ATLAS inputs from `atlas_input/...` copied during Initialization
  (logged as explicit step inputs for provenance).
  NOTE: On restarts that skip Initialization, a fallback enumerates all files
  under `atlas_mutable_input_dir` as inputs; this may include prior preprocess
  outputs.
  NOTE: This restart fallback is a pragmatic backstop and should be centralized
  across steps that need non-coupler inputs (or replaced with Consist-backed
  hydration of prior run artifacts).

**RecordStore outputs (file-level)**
- `atlas_households_csv` → `atlas_mutable_input_dir/year<year>/households.csv`
- `atlas_blocks_csv` → `atlas_mutable_input_dir/year<year>/blocks.csv`
- `atlas_persons_csv` → `atlas_mutable_input_dir/year<year>/persons.csv`
- `atlas_grave_csv` → `atlas_mutable_input_dir/year<year>/grave.csv` (non-start years)
- `atlas_residential_csv` → `atlas_mutable_input_dir/year<year>/residential.csv`
- `atlas_jobs_csv` → `atlas_mutable_input_dir/year<year>/jobs.csv`
- `atlas_accessibility_csv` → `atlas_mutable_input_dir/year<year>/accessibility_<forecast_year>_tract.csv`
  (when `settings.atlas.beamac > 0`)
- Config and static inputs copied from `pilates/atlas/atlas_input` with short_name
  derived from file base name.

**Used by**
- `atlas_run` (inputs: `atlas_mutable_input_dir`, `usim_datastore_h5`)

---

### 5) ATLAS run (`atlas_run`) (per sub-year)

**Primary outputs**
- Coupler: `atlas_output_dir`

NOTE: ATLAS also has a bunch of other inputs. Here's the contents of the atlas input directory:

```
(base) [zaneedell@n0003 atlas]$ ls -R
.:
atlas_input  atlas_output

./atlas_input:
 accessbility2017.RData    adopt     modeaccessibility.csv  'sfb Baseline.csv'        vehicle_type_mapping_ESS_const_220_price.csv   vehicle_type_mapping_evMandForced2.csv   year2019
 accessbility_2015.RData   cpi.csv   psid_names.Rdat         taz_to_tract_sfbay.csv   vehicle_type_mapping_baseline.csv              year2017

./atlas_input/adopt:
baseline  ess_cons  zev_mandate

./atlas_input/adopt/baseline:
new_vehicle_annual_medians.csv          new_vehicles_biannual_values_2023.csv  new_vehicles_biannual_values_2035.csv  new_vehicles_biannual_values_2047.csv  used_vehicles_2023.csv  used_vehicles_2035.csv  used_vehicles_2047.csv
new_vehicle_representative_vehicle.csv  new_vehicles_biannual_values_2025.csv  new_vehicles_biannual_values_2037.csv  new_vehicles_biannual_values_2049.csv  used_vehicles_2025.csv  used_vehicles_2037.csv  used_vehicles_2049.csv
new_vehicles.csv                        new_vehicles_biannual_values_2027.csv  new_vehicles_biannual_values_2039.csv  used_vehicles.csv                      used_vehicles_2027.csv  used_vehicles_2039.csv
new_vehicles_biannual_values_2017.csv   new_vehicles_biannual_values_2029.csv  new_vehicles_biannual_values_2041.csv  used_vehicles_2017.csv                 used_vehicles_2029.csv  used_vehicles_2041.csv
new_vehicles_biannual_values_2019.csv   new_vehicles_biannual_values_2031.csv  new_vehicles_biannual_values_2043.csv  used_vehicles_2019.csv                 used_vehicles_2031.csv  used_vehicles_2043.csv
new_vehicles_biannual_values_2021.csv   new_vehicles_biannual_values_2033.csv  new_vehicles_biannual_values_2045.csv  used_vehicles_2021.csv                 used_vehicles_2033.csv  used_vehicles_2045.csv

./atlas_input/adopt/ess_cons:
new_vehicle_annual_medians.csv          new_vehicles_biannual_values_2023.csv  new_vehicles_biannual_values_2035.csv  new_vehicles_biannual_values_2047.csv  used_vehicles_2023.csv  used_vehicles_2035.csv  used_vehicles_2047.csv
new_vehicle_representative_vehicle.csv  new_vehicles_biannual_values_2025.csv  new_vehicles_biannual_values_2037.csv  new_vehicles_biannual_values_2049.csv  used_vehicles_2025.csv  used_vehicles_2037.csv  used_vehicles_2049.csv
new_vehicles.csv                        new_vehicles_biannual_values_2027.csv  new_vehicles_biannual_values_2039.csv  used_vehicles.csv                      used_vehicles_2027.csv  used_vehicles_2039.csv
new_vehicles_biannual_values_2017.csv   new_vehicles_biannual_values_2029.csv  new_vehicles_biannual_values_2041.csv  used_vehicles_2017.csv                 used_vehicles_2029.csv  used_vehicles_2041.csv
new_vehicles_biannual_values_2019.csv   new_vehicles_biannual_values_2031.csv  new_vehicles_biannual_values_2043.csv  used_vehicles_2019.csv                 used_vehicles_2031.csv  used_vehicles_2043.csv
new_vehicles_biannual_values_2021.csv   new_vehicles_biannual_values_2033.csv  new_vehicles_biannual_values_2045.csv  used_vehicles_2021.csv                 used_vehicles_2033.csv  used_vehicles_2045.csv

./atlas_input/adopt/zev_mandate:
new_vehicle_annual_medians.csv          new_vehicles_biannual_values_2023.csv  new_vehicles_biannual_values_2035.csv  new_vehicles_biannual_values_2047.csv  used_vehicles_2023.csv  used_vehicles_2035.csv  used_vehicles_2047.csv
new_vehicle_representative_vehicle.csv  new_vehicles_biannual_values_2025.csv  new_vehicles_biannual_values_2037.csv  new_vehicles_biannual_values_2049.csv  used_vehicles_2025.csv  used_vehicles_2037.csv  used_vehicles_2049.csv
new_vehicles.csv                        new_vehicles_biannual_values_2027.csv  new_vehicles_biannual_values_2039.csv  used_vehicles.csv                      used_vehicles_2027.csv  used_vehicles_2039.csv
new_vehicles_biannual_values_2017.csv   new_vehicles_biannual_values_2029.csv  new_vehicles_biannual_values_2041.csv  used_vehicles_2017.csv                 used_vehicles_2029.csv  used_vehicles_2041.csv
new_vehicles_biannual_values_2019.csv   new_vehicles_biannual_values_2031.csv  new_vehicles_biannual_values_2043.csv  used_vehicles_2019.csv                 used_vehicles_2031.csv  used_vehicles_2043.csv
new_vehicles_biannual_values_2021.csv   new_vehicles_biannual_values_2033.csv  new_vehicles_biannual_values_2045.csv  used_vehicles_2021.csv                 used_vehicles_2033.csv  used_vehicles_2045.csv

./atlas_input/year2017:
blocks.csv  households.csv  households_output.RData  jobs.csv  persons.csv  residential.csv  vehicles_output.RData

./atlas_input/year2019:
blocks.csv  grave.csv  households.csv  households0.RData  households1.RData  households_output.RData  jobs.csv  persons.csv  persons0.RData  persons1.RData  residential.csv  vehicles_output.RData

./atlas_output:
```
So, for instance, an atlas run has inputs of atlas_input/adopt/<scenario_name>/[new_vehicle_annual_medians.csv , new_vehicle_representative_vehicle.csv, new_vehicles.csv, new_vehicles_biannual_values_<year+1>.csv, used_vehicles_<year+1>.csv], as well as the outputs of the atlas preprocessor

**RecordStore outputs (file-level)**
- `householdv_<year>` → `atlas_output_dir/householdv_<year>.csv`
- `vehicles_<year>` → `atlas_output_dir/vehicles_<year>.csv`

**Used by**
- `atlas_postprocess` (input: run outputs)

---

### 6) ATLAS postprocess (`atlas_postprocess`) (per sub-year)

**Primary outputs**
- Coupler: `atlas_output_dir`
- Coupler: `usim_datastore_h5` → updated UrbanSim H5

**RecordStore outputs (file-level)**
- `usim_h5_updated` → `usim_mutable_data_dir/<output_template>` (H5 updated)
- `atlas_vehicles2_output` → `atlas_output_dir/vehicles2_<year>.csv`

**Used by**
- `activitysim_preprocess` (input: `usim_datastore_h5`)
- `beam_preprocess` (optional input via `atlas_output_dir`)

---

### 7) ActivitySim compile (`activitysim_compile`) (per year)

**Primary outputs**
- Coupler: `zarr_skims`

**Inputs**
- RecordStore inputs from `activitysim_preprocess`:
  `land_use_asim_in`, `households_asim_in`, `persons_asim_in`, and `omx_skims`
  (when present).

**RecordStore outputs (file-level)**
- `zarr_skims` → `asim_output_dir/cache/skims.zarr`

**Used by**
- `activitysim_run` (input: `zarr_skims`)
- `beam_run` (input: `zarr_skims`)

---

### 8) ActivitySim preprocess (`activitysim_preprocess`) (per iteration)

**Primary outputs**
- Coupler: `asim_mutable_data_dir`

**RecordStore outputs (file-level)**
- `land_use_asim_in` → `asim_mutable_data_dir/land_use.csv`
- `households_asim_in` → `asim_mutable_data_dir/households.csv`
- `persons_asim_in` → `asim_mutable_data_dir/persons.csv`
- `omx_skims` → `asim_mutable_data_dir/skims.omx` (if available)

**Used by**
- `activitysim_run` (inputs: `asim_mutable_data_dir`, `zarr_skims`)

---

### 9) ActivitySim run (`activitysim_run`) (per iteration)

**Primary outputs**
- Coupler: `asim_output_dir`

**RecordStore outputs (file-level, core set)**
- `households`
- `persons`
- `tours`
- `trips`
- `beam_plans`
- `land_use`
- Additional ActivitySim outputs captured when present:
  `disaggregate_accessibility`, `joint_tour_participants`,
  `non_mandatory_tour_destination_accessibility`, `person_windows`,
  `proto_*`, `school_*`, `workplace_*`
  (see `ActivitySimPostprocessOutputs.allowed_outputs`)

**Used by**
- `activitysim_postprocess`
- `beam_preprocess` (uses `persons`, `households`, `beam_plans`)

---

### 10) ActivitySim postprocess (`activitysim_postprocess`) (per iteration)

**Primary outputs**
- Coupler: `usim_datastore_h5` → updated UrbanSim H5

**RecordStore outputs (file-level)**
- `usim_input_*` → updated UrbanSim H5 for the forecast year
- ActivitySim output tables (same short_names as run step, if materialized)

**Used by**
- `beam_preprocess` (input: `asim_output_dir`)
- `urbansim_preprocess` in subsequent years via `usim_datastore_h5`

---

### 11) BEAM preprocess (`beam_preprocess`) (per iteration)

**Primary outputs**
- Coupler: `beam_mutable_data_dir`

**RecordStore inputs (file-level)**
- `persons`, `households`, `beam_plans` from ActivitySim
- `linkstats`, `beam_plans_out` from previous BEAM iteration (warm start)
  NOTE: First iteration uses `init.linkstats.*` from the BEAM input data directory
  (copied during Initialization). Subsequent iterations use the most recent BEAM
  output linkstats record as the warm-start input.
- `atlas_vehicles2_input` from ATLAS postprocess (`vehicles2_<year>.csv`) when vehicle ownership is enabled
  (first BEAM inner-iteration only).

**RecordStore outputs (file-level)**
- `beam_prod` and `beam_common` (BEAM config/template inputs)
- `linkstats_warmstart` (if warm start is used)
- BEAM input tables written into `beam_mutable_data_dir`
  - `persons_parquet`, `households_parquet`, `plans_parquet` when `activitysim.file_format=parquet`
  - `persons_csv_gz`, `households_csv_gz`, `plans_csv_gz` when `activitysim.file_format=csv`

**Used by**
- `beam_run` (inputs: `beam_mutable_data_dir`, `zarr_skims`)

---

### 12) BEAM run (`beam_run`) (per iteration)

**Primary outputs**
- Coupler: `beam_output_dir`

**RecordStore outputs (file-level)**
- `raw_od_skims` → `beam_output_dir/*skimsActivitySimOD_current*.omx`
- `raw_od_skims_zarr` → `beam_output_dir/*.zarr` (varies by BEAM build)
- `raw_origin_skims` → `beam_output_dir/*skimsRidehail*.csv.gz`
- `linkstats` → `beam_output_dir/linkstats*.csv.gz`
- `beam_plans_out` → `beam_output_dir/plans*.csv.gz`
- `events` → `beam_output_dir/events*.csv.gz`
- `events_parquet` → `beam_output_dir/events*.parquet`

**Used by**
- `beam_postprocess` (inputs: run outputs)
- `beam_preprocess` in next iteration (warm start)

---

### 13) BEAM postprocess (`beam_postprocess`) (per iteration)

**Primary outputs**
- Coupler: `zarr_skims` → `asim_output_dir/cache/skims.zarr`
- Coupler: `final_skims_omx` → `beam_mutable_data_dir/<region>/<skims_fname>`

**RecordStore outputs (file-level)**
- `zarr_skims` → `asim_output_dir/cache/skims.zarr`
- `final_skims_omx` → `beam_mutable_data_dir/<region>/<skims_fname>`

**Used by**
- `activitysim_run` in next iteration (`zarr_skims`)
- `urbansim_preprocess` (future years, if BEAM skims copied)
- `atlas_preprocess` (if BEAM accessibility is enabled)

---

## Gaps to close (file-level lineage)

The following artifacts are **currently produced** but are not yet tracked
as explicit file-level lineage keys in all steps:

- UrbanSim preprocess: expanded list of copied CSV inputs (additional files in
  `other_data_fnames` and region-specific inputs beyond those listed above).
- ActivitySim run/postprocess: full enumeration of output tables (currently
  a curated allowlist in `ActivitySimPostprocessOutputs`).
- BEAM preprocess: explicit file-level outputs written into
  `beam_mutable_data_dir` (currently inferred via inherited short_names).
- ATLAS run: any auxiliary outputs beyond `householdv_<year>.csv` and
  `vehicles_<year>.csv` (if produced by the container).

## TODOs from NOTE review

- TODO: Narrow ATLAS static inputs by year/scenario so only the files required
  for the current sub-year are logged as step inputs.
