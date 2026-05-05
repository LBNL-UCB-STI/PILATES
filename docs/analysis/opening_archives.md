---
title: Opening Archives
summary: Archive directory, DuckDB, and tracker mental model for post-run analysis.
---

# Opening Archives

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then use [SQL and DuckDB](sql_and_duckdb.md) or [Consist Analysis CLI](consist_analysis_cli.md).
- Pair this with [Workspace Layout](../reference/workspace_layout.md) for path semantics.

## Archive Shape

The analysis code treats the archive run directory as the storage root for a finished run.
It resolves the Consist database from one of these paths under that directory when no explicit `--db-path` is given:

- `.consist/snapshots/latest/provenance.duckdb`
- `.consist/provenance.duckdb`
- `.consist/snapshots/latest/consist.duckdb`
- `.consist/consist.duckdb`

`create_analysis_tracker()` builds the tracker mounts as:

- `inputs` -> the repository root or explicit project root
- `workspace` -> the archive run directory
- `scratch` -> optional output root when provided

`AnalysisSession.open()` and `open_archive()` both use that same archive resolution path.
`open_run()` returns the session itself; `open_archive()` wraps the session in an `Archive`.

## Local DB Copy For Notebooks

On HPC, it can be faster and less fragile to copy the archive DuckDB file to
node-local scratch before starting a notebook. The archive mount should still
point at the preserved archive run directory; only the DB path changes.

Use the starter notebook:

- repo-local `analysis/notebooks/local_duckdb_scratch_starter.ipynb`

Or copy from a shell first:

```bash
hpc/copy_duckdb_local.sh --src /path/to/archive/run/.consist/provenance.duckdb
```

Then pass the copied DB path to `open_archive(...)`:

```python
archive = open_archive(
    ARCHIVE_RUN_DIR,
    project_root=PROJECT_ROOT,
    db_path=LOCAL_DB_PATH,
)
```

## Fast Mental Model

If you only need the opening rule, it is:

1. point analysis at the archive run directory
2. let the helper resolve the Consist DB under `.consist/`
3. let the tracker mount that archive as `workspace`
4. build higher-level views such as sessions, archives, runsets, epochs, or comparisons from there

## What Fails Early

- Missing archive directories raise `FileNotFoundError`.
- Missing DB paths raise `FileNotFoundError`.
- Invalid tagging state can raise during session open when strict tagging or fail-on-issues is enabled.
- The default analysis access mode is `analysis`, not a write mode.
