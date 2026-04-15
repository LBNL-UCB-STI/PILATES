---
title: SQL and DuckDB
summary: Inspecting PILATES runs through DuckDB, helper utilities, and example SQL workflows.
---

# SQL and DuckDB

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) if you want packaged commands instead of ad hoc SQL.
- Pair this with [Database Setup](../reference/database_setup.md) and [Database Documentation Guide](../reference/database_documentation_guide.md).

## Query Surfaces

The analysis package exposes SQL through several layers:

- `AnalysisSession.config()` and `AnalysisSession.diff_configs()` use the tracker query service for configuration inspection.
- `Archive.views(epoch)` returns the archive-local view wrapper for an epoch.
- `Epoch.sql(sql)` and `Epoch.query(sql)` run SQL against the epoch views.
- `EpochTables.load(...)` reads named tables from those views.
- `export_sql_query()` runs arbitrary SQL against the tracker DB and writes CSV or Parquet.

The named epoch views currently exposed by `EpochTables` are:

- `trips`
- `persons`
- `households`
- `land_use`
- `linkstats`
- `skim_summary`
- `urbansim_persons`
- `urbansim_households`
- `urbansim_jobs`

## DuckDB Health

`get_duckdb_health()` checks whether the DB and WAL exist and can optionally probe a read-only open.
`print_duckdb_health()` prints the same payload and returns it.

In the analysis tracker path, `build_archive_mounts()` maps `workspace` to the archive run directory so archive queries point at shared storage rather than a node-local execution directory.

## Example SQL

[`docs/example_queries.sql`](../example_queries.sql) is the broader SQL example file in this repo.
It covers:

- run metadata queries
- lineage and provenance queries
- run-to-run comparisons
- direct simulation-data queries

Treat it as a schema-level reference, not as the source of truth for the analysis package API.
