# PILATES Documentation

> [!WARNING]
> These docs are an initial pass generated and reworked with the help of an LLM.
> They have not been fully verified end-to-end against every workflow, config,
> and deployment path in this repository. Treat them as a strong starting point,
> not as guaranteed source-of-truth documentation. When behavior matters, verify
> against the code and active configs.

This page is the landing page for the current documentation set and is intended
to be easy to migrate into a static site later.

## Start Here

If you are new to PILATES, read these first:

1. `getting_started.md`
2. `cli_reference.md`
3. `configuration_reference.md`
4. `data_bootstrap.md`
5. `troubleshooting.md`

## Architecture And Workflow

- `architecture.md`
  High-level architecture overview and reading guide.
- `workflow_primer.md`
  Runtime, stages, steps, manifests, restart behavior, and workflow structure.
- `provenance_and_lineage.md`
  High-level overview of provenance, artifacts, and lineage.
- `artifact_flow.md`
  Concise map of the main workflow-facing artifact handoffs.
- `lineage_map.md`
  More detailed artifact and step lineage reference.

## Model Integration

- `model_integration_guide.md`
  Current integration architecture and contract boundaries.
- `adding_a_model.md`
  Practical checklist for adding a new model or workflow step family.

## Configuration And Operations

- `configuration_reference.md`
  Canonical config structure and key runtime options.
- `cli_reference.md`
  Supported CLI entry points and common invocation patterns.
- `data_bootstrap.md`
  Expected external data layout and pre-run validation.
- `hpc_execution.md`
  Lawrencium-style Slurm execution overview.
- `troubleshooting.md`
  Common failure modes and debugging starting points.

## Provenance And Database

- `database-setup.md`
  Enabling and validating the run-local Consist DuckDB.
- `database_documentation_guide.md`
  ERD generation and live DB inspection workflow.
- `database_schema_reference.md`
  Curated schema families and live DB caveats.
- `artifact_facet_catalog.md`
  Current artifact facet conventions.

## Domain And Specialized Guides

- `zone_id_management.md`
- `land_use_skim_alignment.md`
- `test_output_preservation.md`

## Status Notes

These docs have been substantially updated to match the current runtime shape,
but they are still early. The most trustworthy documents right now are the ones
that were recently rewritten around the current code structure:

- `architecture.md`
- `workflow_primer.md`
- `model_integration_guide.md`
- `adding_a_model.md`
- `cli_reference.md`
- `provenance_and_lineage.md`
- `data_bootstrap.md`
- `hpc_execution.md`

## Source Of Truth

When the docs and code disagree, the code wins.

The most important code entry points to cross-check are:

- `run.py`
- `pilates/runtime/launcher.py`
- `workflow_state.py`
- `pilates/workflows/stages/*.py`
- `pilates/workflows/steps/*.py`
- `pilates/workflows/catalog.py`
