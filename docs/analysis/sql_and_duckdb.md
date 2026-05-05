---
title: SQL and DuckDB
summary: Current SQL entry points for Consist-backed PILATES archive analysis.
---

# SQL and DuckDB

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) for packaged commands.
- See the repo-local `analysis/README.md` for notebook-first examples.

## Recommended Path

PILATES analysis is Consist-first. Start from an archived run directory, open it
through the analysis API, and query the archive-backed epoch views rather than
copying raw SQL against legacy table names.

```python
from pathlib import Path

from pilates_consist_analysis import open_archive

archive = open_archive(
    Path("/path/to/archive/run"),
    project_root=Path("/Users/zaneedell/git/PILATES"),
)

epoch = archive.scenario("baseline").epoch(year=2030, converged=True)
epoch.sql("SELECT * FROM {views.trips} LIMIT 10")
```

The maintained SQL surfaces are:

- `open_archive(...)`, the preferred notebook entry point for archived runs
- `Archive.views(epoch)`, for archive-local view discovery and direct view access
- `Epoch.sql(sql)` and `Epoch.query(sql)`, for epoch-scoped SQL with named views
- `EpochTables.load(...)` and table helpers such as `epoch.tables.trips(...)`
- `export_sql_query(...)`, for programmatic tracker-DB exports
- `pilates-consist-analysis export-sql`, for CLI CSV or Parquet exports

## CLI Export

Use `export-sql` when you want a reproducible file output from SQL:

```bash
pilates-consist-analysis export-sql \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --sql "SELECT * FROM runs LIMIT 20" \
  --output-path /path/to/output/runs.csv
```

For longer SQL, pass `--sql-file` instead of `--sql`.

## Named Epoch Tables

The notebook API exposes common epoch-backed tables through `epoch.tables` when
the underlying artifacts are present:

- `trips`
- `persons`
- `households`
- `land_use`
- `linkstats`
- `skim_summary`
- `urbansim_persons`
- `urbansim_households`
- `urbansim_jobs`

Prefer those helpers for common analysis. Drop to `Epoch.sql(...)` when you need
joins, projections, or diagnostics that the table helpers do not cover.

## DuckDB Health

Use the analysis package health surfaces for Consist DB inspection:

- `pilates-consist-analysis db-health`, for operator-facing CLI checks
- `AnalysisSession.inspect_db()` and `AnalysisSession.assert_db_healthy(...)`,
  for notebook or script checks after `open_run(...)`
- `Archive.summary()` and `Archive.issues(...)`, for archive-level health
  summaries after `open_archive(...)`
- `get_db_health(...)`, `get_db_health_issues(...)`, and
  `db_health_to_frame(...)`, for lower-level programmatic checks

The analysis tracker mounts the archive run directory as `workspace`, so archive
queries resolve shared archived data rather than a node-local execution
workspace.

## Deprecated Example File

[`docs/example_queries.sql`](../example_queries.sql) is now only a redirect
stub. Do not treat it as a schema reference; its old examples targeted stale
legacy tables. Use this page, [Consist Analysis CLI](consist_analysis_cli.md),
and the repo-local `analysis/README.md` for maintained SQL entry
points.
