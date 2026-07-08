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

`open_archive()` uses that same archive resolution path and returns the beginner-facing `Archive` object.
`open_run()` still returns the lower-level `AnalysisSession` for compatibility and run-discovery workflows.

## Local DB Copy For Notebooks

On HPC, it can be faster and less fragile to use node-local scratch. The
beginner path is `local_cache`, which copies the DB up front and localizes
selected artifact files when you request faceted tables.

Use the starter notebook:

- repo-local `analysis/notebooks/local_duckdb_scratch_starter.ipynb`

```python
archive = open_archive(
    ARCHIVE_RUN_DIR,
    project_root=PROJECT_ROOT,
    local_cache=LOCAL_CACHE,
)
```

The helper writes a manifest under:

```text
<local-cache>/<archive-name>/.pilates/analysis_localization_manifest.json
```

If you only want to copy the DB manually, the shell helper is still available:

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
4. ask `Archive.table(...)` for a faceted Ibis table
5. aggregate and compare over ordinary facet columns

## What Fails Early

- Missing archive directories raise `FileNotFoundError`.
- Missing DB paths raise `FileNotFoundError`.
- Invalid tagging state can raise during session open when strict tagging or fail-on-issues is enabled.
- The default analysis access mode is `analysis`, not a write mode.
