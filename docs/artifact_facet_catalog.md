# Artifact Facet Catalog

This document defines the current artifact facet conventions used by PILATES when logging to Consist.

## Conventions

- `facet_schema_version`: `v1`
- `facet_index`: `true` for queryable artifacts
- Keep facet fields scalar-only (string/number/bool) for indexed queries

## BEAM

### `artifact_family = "events_parquet"`
Used for BEAM run outputs keyed like:
- `events_parquet_<year>_<iteration>`
- `events_parquet_<year>_<iteration>_sub<beam_sub_iteration>`

Fields:
- `year` (int)
- `iteration` (int)
- `beam_sub_iteration` (int, optional)

### `artifact_family = "raw_od_skims"`
Keys:
- `raw_od_skims_<year>_<iteration>`
- `raw_od_skims_<year>_<iteration>_sub<beam_sub_iteration>`

Fields:
- `year` (int)
- `iteration` (int)
- `beam_sub_iteration` (int, optional)

### `artifact_family = "raw_od_skims_zarr"`
Keys:
- `raw_od_skims_zarr_<year>_<iteration>`
- `raw_od_skims_zarr_<year>_<iteration>_sub<beam_sub_iteration>`

Fields:
- `year` (int)
- `iteration` (int)
- `beam_sub_iteration` (int, optional)

### `artifact_family = "linkstats"`
Keys:
- `linkstats_<year>_<iteration>`

Fields:
- `year` (int)
- `iteration` (int)

### `artifact_family = "linkstats_parquet"`
Keys:
- `linkstats_parquet_<year>_<iteration>`
- `linkstats_parquet_<year>_<iteration>_sub<beam_sub_iteration>`

Fields:
- `year` (int)
- `iteration` (int)
- `beam_sub_iteration` (int, optional)

### `artifact_family = "linkstats_unmodified_phys_sim_iter_parquet"`
Keys (current format):
- `linkstats_unmodified_parquet__y<year>__i<iteration>__phys_sim_iter<phys_sim_iteration>`
- `linkstats_unmodified_parquet__y<year>__i<iteration>__phys_sim_iter<phys_sim_iteration>__beam_sub_iter<beam_sub_iteration>`

Fields:
- `year` (int)
- `iteration` (int)
- `phys_sim_iteration` (int)
- `beam_sub_iteration` (int, optional)

### `artifact_family = "events_parquet_split"`
Keys:
- `events_parquet_<year>_<iteration>_type_<event_type>`

Fields:
- `year` (int)
- `iteration` (int)
- `event_type` (string)

### `artifact_family = "path_traversal_links"`
Keys:
- `path_traversal_links_<year>_<iteration>`

Fields:
- `year` (int)
- `iteration` (int)

## ActivitySim

### `artifact_family = "<table_name>"`
For outputs keyed `*_asim_out`, family is the key stem:
- `persons_asim_out` -> `artifact_family = "persons"`
- `trips_asim_out` -> `artifact_family = "trips"`
- etc.

Fields:
- `year` (int)
- `iteration` (int)

### `artifact_family = "asim_input_archived"`
For keys like:
- `asim_input_households_csv_archived`
- `asim_input_persons_csv_archived`

Fields:
- `year` (int)
- `iteration` (int)

### `artifact_family = "zarr_skims"`
For key:
- `zarr_skims`

Fields:
- `year` (int)
- `iteration` (int)

## UrbanSim

### `artifact_family = "usim_datastore_base_h5"`
Key:
- `usim_datastore_base_h5`

Fields:
- `year` (int)  # forecast year context for the step

### `artifact_family = "usim_forecast_output"`
Key:
- `usim_forecast_output`

Fields:
- `year` (int)

### `artifact_family = "usim_datastore_h5"`
Key:
- `usim_datastore_h5`

Fields:
- `year` (int)

### `artifact_family = "usim_input_archive"`
Keys:
- `usim_input_archive_<year>`

Fields:
- `year` (int)

### `artifact_family = "usim_input_merged"`
Keys:
- `usim_input_merged_<year>`

Fields:
- `year` (int)

## ATLAS

### `artifact_family = "atlas_preprocess_output"`
Logged for preprocess prepared inputs.

### `artifact_family = "atlas_run_input"`
Logged for ATLAS run input artifacts from preprocess outputs.

ATLAS facet payload fields:
- `input_group` (string)
  - one of: `adopt`, `vehicle_type_mapping`, `vehicles2`, `usim`, `global`
- `scenario` (string, optional)
- `input_year` (int, optional; parsed from key suffix `_YYYY`)
- `forecast_year` (int)

Examples:
- `adopt/baseline/new_vehicles_biannual_values_2030`
  - `input_group=adopt`, `scenario=baseline`, `input_year=2030`
- `vehicle_type_mapping_evMandForced2`
  - `input_group=vehicle_type_mapping`, `scenario=zev_mandate`

## Query Example Patterns

Python (Consist):

```python
from consist.tools import queries

rows = queries.find_artifacts_by_params(
    tracker,
    params=[
        "beam.artifact_family=linkstats_unmodified_phys_sim_iter_parquet",
        "beam.year=2030",
        "beam.iteration=7",
        "beam.phys_sim_iteration=2",
    ],
)
```

CLI:

```bash
consist artifacts \
  --param beam.artifact_family=linkstats_unmodified_phys_sim_iter_parquet \
  --param beam.year=2030 \
  --param beam.iteration=7 \
  --param beam.phys_sim_iteration=2
```

