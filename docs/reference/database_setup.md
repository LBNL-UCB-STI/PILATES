---
title: Database Setup
summary: Enabling and validating the current run-local Consist DuckDB and related database surfaces.
---

# Database Setup

## Current Setup

PILATES resolves the run-local Consist DB from the settings, not from a fixed repository path.

- `shared.database.enabled` turns the DB surface on or off.
- `shared.database.path` names the shared DB location when one is configured.
- `run.consist_db_local_run` decides whether PILATES keeps a run-local copy under the workspace archive tree.
- `run.consist_db_filename` selects the filename used inside `.consist/`.

When the local DB is enabled, PILATES resolves:

- a local DB under `workspace/.consist/<filename>`
- an archive DB under `archive/.consist/<filename>`

If local DB tracking is disabled, PILATES reuses the configured shared path for both sides. If DB tracking is disabled entirely, the resolver returns no DB path.

PILATES also exposes runtime guardrails around the DB copy and snapshot flow:

- `run.consist_db_snapshot_enabled`
- `run.consist_db_snapshot_interval_seconds`
- `run.consist_db_snapshot_on_outer_iteration`
- `run.consist_db_snapshot_keep_last`
- `run.consist_db_restore_on_start`
- `run.consist_db_restore_strict`
- `run.consist_db_seed_from_shared_on_start`
- `run.consist_db_seed_strict`

The quickest health check is `get_duckdb_health()` or `print_duckdb_health()` from `pilates.utils.consist_analysis`. Those helpers report whether the DB and WAL exist and whether a read-only open succeeds.

## Adjacent Pages

- Pair this with [Database Documentation Guide](database_documentation_guide.md).
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for direct inspection workflows.
- Use [Opening Archives](../analysis/opening_archives.md) for archive-side access.
