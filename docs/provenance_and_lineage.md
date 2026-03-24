# Provenance And Lineage

This page is the high-level overview of provenance and lineage in PILATES.

If you are new to the system, read this before diving into the more detailed
maps and catalogs.

## What PILATES Tracks

PILATES uses Consist to track workflow execution. At a high level, it records:

- scenarios and runs
- step executions
- input and output artifacts
- selected queryable metadata about those artifacts

In practice, the most useful units are:

- run:
  one workflow execution context
- step:
  one tracked unit of execution such as `activitysim_run` or `beam_postprocess`
- artifact:
  a workflow-facing file or directory logged under a stable key
- facet:
  small metadata attached to a run or artifact for filtering and inspection

## How Lineage Fits Into The Workflow

The workflow layers and lineage layers are closely related:

- runtime creates the scenario and tracker context
- step factories execute model phases and log workflow-facing inputs/outputs
- artifact keys and typed outputs provide stable names for those logged files

That means provenance is not a separate afterthought. It is built into the step
execution layer.

## What An Artifact Usually Looks Like

A lineage-tracked artifact in PILATES usually has:

- a stable key such as `usim_datastore_h5`, `zarr_skims`, or `linkstats`
- a filesystem path
- a description
- optional facet metadata such as year, iteration, or artifact family

The exact publication logic lives in the step modules under
`pilates/workflows/steps/*.py`.

## Facets: Why They Matter

Facets make artifacts queryable without changing cache identity.

Common examples:

- model name
- year
- iteration
- BEAM sub-iteration
- artifact family

For current conventions, see `docs/artifact_facet_catalog.md`.

## What Is Usually Most Useful To Inspect

For day-to-day debugging, the most useful provenance questions are:

1. Did a step actually run, or was it recovered from cache?
2. Which files did the step declare as inputs and outputs?
3. Which artifact key was published for the file I care about?
4. Which year and iteration did that artifact belong to?

Those questions are usually easier to answer from logged step outputs and the
run database than from raw workspace browsing alone.

## How To Inspect A Run

There are two practical inspection paths:

### 1. Inspect the run-local Consist DuckDB

Most runs create a DuckDB file under:

- `<run_dir>/.consist/<run.consist_db_filename>`

Quick health check:

```bash
python - <<'PY'
from pilates.utils.consist_analysis import print_duckdb_health
print_duckdb_health(db_path="/path/to/run/.consist/provenance.duckdb", probe_open=True)
PY
```

If you have DuckDB installed, you can also inspect:

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c "SHOW TABLES;"
```

### 2. Read the workflow-facing docs

Use these docs together:

- `docs/workflow_primer.md`
- `docs/artifact_flow.md`
- `docs/lineage_map.md`
- `docs/artifact_facet_catalog.md`

The first two explain the workflow structure; the latter two explain the actual
artifact surface.

## Where Specific Provenance Logic Lives

- scenario/run metadata:
  `pilates/runtime/launcher.py`, `pilates/runtime/scenario_runtime.py`,
  `pilates/utils/consist_config.py`
- step-level input/output logging:
  `pilates/workflows/steps/*.py` and `pilates/workflows/steps/shared.py`
- artifact keys and schema:
  `pilates/workflows/artifact_keys.py`,
  `pilates/workflows/coupler_schema.py`
- run DB inspection helpers:
  `pilates/utils/consist_analysis.py`

## Practical Limits

Not every internal temporary file is tracked as a first-class workflow artifact.

PILATES is intentionally selective:

- workflow-facing and restart-relevant outputs should be logged and named
- purely local scratch files often are not

So if you cannot find a file in lineage, first ask whether it is meant to be a
workflow contract or just an implementation detail.

## Related Docs

- `docs/workflow_primer.md`
- `docs/artifact_flow.md`
- `docs/lineage_map.md`
- `docs/artifact_facet_catalog.md`
- `docs/database-setup.md`
- `docs/database_documentation_guide.md`
