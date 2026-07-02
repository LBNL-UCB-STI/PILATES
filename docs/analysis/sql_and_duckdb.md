---
title: SQL and DuckDB
summary: Current SQL entry points for Consist-backed PILATES archive analysis.
---

# SQL and DuckDB

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) for discovery and health checks.
- See the repo-local `analysis/README.md` for notebook-first examples.

## Recommended Path

PILATES analysis is Consist-first. Start from an archived run directory, open it
through the analysis API, and prefer faceted Ibis tables before writing raw SQL.

```python
from pathlib import Path

from pilates_consist_analysis import open_archive

archive = open_archive(
    Path("/path/to/archive/run"),
    project_root=Path("/Users/zaneedell/git/PILATES"),
)

trips = archive.table(
    "activitysim.trips",
    facets=["scenario_id", "year", "iteration"],
    where={"year": 2030},
)

trips.limit(10).to_pandas()
```

The maintained SQL surfaces are:

- `open_archive(...)`, the preferred notebook entry point for archived runs
- `Archive.table(...)`, for faceted Ibis tables over grouped artifacts
- `Archive.views(epoch)`, for archive-local view discovery and direct view access
- `Epoch.sql(sql)` and `Epoch.query(sql)`, for epoch-scoped SQL with named views
- `EpochTables.load(...)` and table helpers such as `epoch.tables.trips(...)`
For file outputs, materialize the Ibis expression or DataFrame explicitly at the
end of the notebook/script.

## Named Epoch Tables

The compatibility API exposes common epoch-backed tables through `epoch.tables`
when the underlying artifacts are present:

- `trips`
- `persons`
- `households`
- `land_use`
- `linkstats`
- `skim_summary`
- `urbansim_persons`
- `urbansim_households`
- `urbansim_jobs`

Prefer `Archive.table(...)` for new notebook work. Drop to `Epoch.sql(...)` when
you need an older epoch-scoped path, exact SQL text, joins, projections, or
diagnostics that the Ibis table path does not cover.

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
