---
title: Artifact Facet Catalog
summary: Indexed artifact facet conventions used when PILATES logs workflow artifacts to Consist.
---

# Artifact Facet Catalog

## How To Read This Catalog

- Artifact keys are the workflow contract. Facets are the query surface that make logged artifacts searchable by family, year, iteration, and model-specific dimensions.
- A facet is only listed here if current step code or tests show that PILATES publishes it as part of normal workflow logging.
- When a family is dynamic, the family name itself carries part of the meaning. Examples: `householdv_{year}`, `vehicles_{year}`, `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`), and `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`).

## Common Facet Fields

These fields appear repeatedly across published artifact families:

| Facet | What PILATES uses it for |
| --- | --- |
| `artifact_family` | Primary grouping key for artifacts that share a workflow role. |
| `year` | Forecast year associated with the publication. |
| `iteration` | Supply-demand-loop iteration associated with the artifact. |
| `facet_schema_version` | Schema marker for the facet layout currently being published. |
| `facet_index` | Flag used when the publication is intended to be queryable through indexed filtering. |

## Facet Rules

- PILATES uses artifact families to distinguish workflow role, not just file type.
- The same path type can appear under different families when the workflow role changes.
- Families that participate in restart or archive inspection are usually the ones that get the clearest year and iteration metadata.
- If a field is not needed for cross-run querying or replay, it may remain internal to the step instead of becoming a public facet.

## Family Groups

### UrbanSim datastore families

These families distinguish the current mutable datastore from archived snapshots and stage-specific handoff roles:

| Family or key | Current role |
| --- | --- |
| `USIM_DATASTORE_BASE_H5` | Base datastore handle for the current run year before downstream mutation. |
| `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5` | Canonical current mutable UrbanSim datastore handle used at stage boundaries. |
| `USIM_POPULATION_SOURCE_H5` | UrbanSim datastore selected for ActivitySim preprocess. |
| `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`) | Archived year-specific UrbanSim input snapshot family. |
| `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`) | Year-specific merged UrbanSim datastore snapshot family published for later inspection or replay. |

What to notice:

- `USIM_DATASTORE_H5` is an alias to the current-role datastore key, not a separate third datastore concept.
- The archive and merged families are dynamic because the year is part of the published family identity.

### ATLAS families

ATLAS publications are mostly year-scoped outputs that get attached to the current workflow year:

| Family or key | Current role |
| --- | --- |
| `householdv_{year}` | Year-scoped household vehicle ownership output family. |
| `vehicles_{year}` | Year-scoped vehicle inventory output family. |
| `ATLAS_VEHICLES2_OUTPUT` | Postprocessed ATLAS vehicle output handed to downstream workflow consumers. |

What to notice:

- The year-suffixed families are the durable public outputs from the ATLAS run boundary.
- `ATLAS_VEHICLES2_OUTPUT` is the postprocess publication, not just a raw runner output.

### ActivitySim families

ActivitySim uses a mix of staged-input keys and shared skim publications:

| Family or key | Current role |
| --- | --- |
| `ASIM_LAND_USE_IN` | Staged ActivitySim land-use table input for the run boundary. |
| `ASIM_HOUSEHOLDS_IN` | Staged households input for ActivitySim. |
| `ASIM_PERSONS_IN` | Staged persons input for ActivitySim. |
| `ASIM_OMX_SKIMS` | ActivitySim OMX skims input staged for the run boundary. |
| `ASIM_SHARROW_CACHE_DIR` | Optional persisted compile-cache directory. |
| `ZARR_SKIMS` | Shared skims handoff used by ActivitySim and BEAM. |

What to notice:

- The input-table keys describe what PILATES stages into the ActivitySim run, not what ActivitySim conceptually represents.
- `ZARR_SKIMS` is a cross-model handoff key, so its family-level meaning is larger than the ActivitySim compile step that may republish it.

### BEAM families

BEAM publishes the densest facet set because PILATES needs to distinguish staged inputs, run outputs, warm-start materials, and phys-sim outputs:

| Family or key | Current role |
| --- | --- |
| `BEAM_PLANS_IN`, `BEAM_HOUSEHOLDS_IN`, `BEAM_PERSONS_IN` | Staged BEAM inputs derived from earlier stages. |
| `LINKSTATS` | Canonical BEAM linkstats output publication. |
| `linkstats_*` / `linkstats_parquet_*` families | Linkstats variants with year, iteration, and phys-sim metadata for querying. |
| `LINKSTATS_WARMSTART` | Warm-start linkstats input family for BEAM preprocess. |
| `BEAM_PLANS_OUT` | Published BEAM plans output from the run boundary. |
| `BEAM_INPUT_*_ARCHIVED` | Archived BEAM inputs and warm-start artifacts used for replay or inspection. |
| `FINAL_SKIMS_OMX` | Final OMX skims publication from BEAM postprocess. |
| `BEAM_FULL_SKIMS` | Separate full-skim tracked output family. |

What to notice:

- The BEAM phys-sim publications carry the richest facet metadata in current tests: year, iteration, phys-sim iteration, and optional sub-iteration identifiers.
- Archived BEAM input families are distinct from live staged-input keys because replay logic needs an explicit archived producer role.

## Where Facets Come From

- Shared step helpers attach common logging metadata when outputs are published through tracked workflow steps.
- Model-specific step modules add family-specific facets where the workflow needs finer-grained querying.
- The coupler schema and workflow tests are the best way to verify that a family is still part of the public surface.

## Adjacent Pages

- Read [Artifact Semantics](artifact_semantics.md) first for the meaning behind the families.
- Use [Consist in PILATES](consist_in_pilates.md) for the logging and query model.
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for analysis-side querying.
