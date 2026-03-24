# Database Setup (Consist DuckDB)

PILATES uses Consist for provenance tracking and caching. Consist persists run
state in a DuckDB file (the "Consist DB"), typically named `provenance.duckdb`
and stored under a `.consist/` folder inside the run directory.

This guide explains how to enable that database, where it lives, and how to
validate it is being created and maintained correctly.

## Prerequisites

- Python environment with PILATES dependencies (DuckDB is used through Python).
- Optional: DuckDB CLI (`duckdb`) for ad-hoc inspection. If you do not have it,
  you can still validate the DB from Python.

There is no external database server. The Consist DB is a local file.

## Configuration

Database tracking is controlled by the nested config in:

- `shared.database.*` (whether DB tracking is enabled, and an optional shared DB path)
- `run.consist_db_*` (how the run-local DB behaves: filename, snapshots, restore, seed)

### Minimal Enablement

`shared.database.enabled` must be true, otherwise Consist DB tracking is disabled.

Example:

```yaml
shared:
  database:
    enabled: true
    type: duckdb
    path: pilates/database/sfbay_pilates_data.duckdb
run:
  consist_db_filename: provenance.duckdb
  consist_db_local_run: true
```

Notes:

- `shared.database.path` is not always the file PILATES writes during the run.
  When `run.consist_db_local_run: true`, PILATES writes the run DB under
  `.consist/<filename>` inside the run directories and uses `shared.database.path`
  primarily as a configured shared/seed path.
- When `run.consist_db_local_run: false`, the DB path resolves to the configured
  `shared.database.path` and writes directly to that shared location. This is
  only safe if you do not have concurrent writers.

### Run DB Location Rules

PILATES resolves two paths:

- local DB: `<local_run_dir>/.consist/<run.consist_db_filename>`
- archive DB: `<archive_run_dir>/.consist/<run.consist_db_filename>`

On startup, PILATES may populate the local DB in this order:

1. restore from archive snapshot (if enabled)
2. seed from `shared.database.path` (if enabled)

Key flags:

- `run.consist_db_local_run` (bool, default `true`)
  - When true, the DB is maintained on node-local storage and mirrored/snapshotted
    into the archive run directory.
- `run.consist_db_filename` (string, default `provenance.duckdb`)
  - Basename only. The `.consist/` directory is managed by PILATES.
- `run.consist_db_snapshot_enabled` (bool, default `true`)
- `run.consist_db_snapshot_interval_seconds` (int, default `600`)
- `run.consist_db_snapshot_on_outer_iteration` (bool, default `true`)
- `run.consist_db_snapshot_keep_last` (int, default `3`)
- `run.consist_db_restore_on_start` (bool, default `true`)
- `run.consist_db_seed_from_shared_on_start` (bool, default `false`)
  - When true and the local DB is missing at startup, seed it from
    `shared.database.path` (after snapshot restore is attempted).

### Snapshot Layout

When snapshots are enabled, PILATES writes under:

- `<archive_run_dir>/.consist/snapshots/latest/`
- `<archive_run_dir>/.consist/snapshots/history/`

Snapshots may include a WAL sidecar and a metadata sidecar:

- `<db>.wal`
- `<db>.snapshot_meta.json`

## First Validation

After a run starts and completes at least one step, validate:

1. A `.consist/` directory exists in the run directory.
2. A DuckDB file exists at `.consist/<filename>`.
3. DuckDB can open the file.

### Validate With DuckDB CLI (Optional)

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c "SELECT 1;"
duckdb /path/to/run/.consist/provenance.duckdb -c "SHOW TABLES;"
```

### Validate From Python (Works Without CLI)

```bash
python - <<'PY'
from pilates.utils.consist_analysis import print_duckdb_health
print_duckdb_health(db_path="/path/to/run/.consist/provenance.duckdb", probe_open=True)
PY
```

If you see an open probe error, it is often due to a missing/corrupt DB, a
stale `.wal` sidecar, or a process still writing to the DB.

## Operational Notes

- Local vs archive:
  - When local DB mode is enabled, the run writes to the local `.consist/` DB and
    snapshots/mirrors to the archive run directory for persistence.
- Restarts:
  - When a local `.consist/` DB is missing, PILATES can restore it from the latest
    archived snapshot (if present), and optionally seed from `shared.database.path`.
- Shared DB path:
  - `shared.database.path` should be treated as shared storage. Only enable direct
    writes to it (`run.consist_db_local_run: false`) when that is operationally safe.

## Related Docs

- `docs/database_documentation_guide.md`
- `docs/workflow_primer.md`
- `docs/test_output_preservation.md`
