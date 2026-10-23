---
title: Run Promotion
summary: Promote completed scratch archive runs to cold storage and merge run-local Consist databases into a shared main database.
---

# Run Promotion

Use `pilates.runtime.promote_run_archive` after a cluster run finishes to copy the completed archive run directory from shared scratch (`run.output_directory`) to cold project storage (`run.recovery_archive_roots`) and merge the new run lineage into a central Consist DuckDB database.

## Prerequisites

Wait until the Slurm job finishes before promoting or merging its database. Configure the active scratch archive root and cold recovery roots in your scenario YAML, or pass `--root` on the command line:

```yaml
run:
  output_directory: /global/scratch/users/${USER}/pilates-outputs
  recovery_archive_roots:
    - /clusterfs/<project-or-nfs-root>/${USER}/pilates-outputs
```

## 1. Preview Promotion (`--dry-run`)

Run with `--dry-run` first to confirm the resolved source run directory and destination paths without copying files or mutating databases:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --dry-run
```

If you omit `--run-dir`, the helper selects the latest archive directory under `run.output_directory` matching `run.output_run_name`.

## 2. Promote the Archive

Execute the promotion by dropping `--dry-run`:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs
```

For each destination root, `promote_run_archive`:

1. Copies the archive run directory to `<root>/<run-dir-name>` while skipping transient directories (`activitysim/output/final_pipeline`, `activitysim/output/cache`, `shared_cache/numba`, and `.consist/snapshots/history`).
2. Verifies that the promoted copy contains `run_state.yaml` and `.consist/` metadata (pass `--no-recovery-copy-verify` only when you want to skip byte-level checksum verification on the promoted files).
3. Updates artifact `recovery_roots` inside the run-local Consist DB and syncs `.consist/` to the promoted archive.
4. Writes `.consist/recovery_promotion.json` in both the source and promoted directories.

## 3. Promote and Merge into the Main DB (`--merge-main-db`)

To prevent collisions with historical runs in the shared main DB, pass `--merge-main-db`. The helper resolves the single new root run, exports that root run and its child steps to a filtered shard, and merges that shard with `--merge-conflict error`:

```bash
MAIN_DB=/clusterfs/<project-or-nfs-root>/pilates-main/provenance.duckdb

# Preview the main-DB merge (still promotes the archive copy)
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --merge-main-db "$MAIN_DB" \
  --merge-dry-run

# Execute the main-DB merge reusing the already-promoted archive (--merge-only)
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --merge-main-db "$MAIN_DB" \
  --merge-only
```

Additional merge flags:

- `--merge-db-only`: Exports the new root run lineage and merges it into `MAIN_DB` without copying the archive directory to recovery roots.
- `--root-run-id <run_id>`: Specifies the root run ID when automatic root resolution finds multiple candidate runs.

If `MAIN_DB` does not exist yet, the helper initializes it from the exported shard.

## 4. Verify the Promoted Copy (`--verify-only`)

Check the promoted archive structure and run database health diagnostics without modifying metadata:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --verify-only

pilates-consist-analysis db-health \
  --archive-run-dir /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --project-root /path/to/PILATES \
  --strict \
  --fail-on-issues
```

## 5. Manual Merge Fallback

If you need to inspect, repair, or merge a promoted run database outside `promote_run_archive`, use the Consist CLI against `<promoted-run>/.consist/provenance.duckdb` (`$RUN_DB`) and your shared catalog (`$MAIN_DB`):

```bash
# Check database integrity and snapshot the main DB before manual changes
consist db doctor --db-path "$RUN_DB"
consist db doctor --db-path "$MAIN_DB"
consist db snapshot --db-path "$MAIN_DB" --out "$MAIN_DB.premerge.duckdb"

# Export a single root run lineage and merge it into the main DB
consist db export <root-run-id> --db-path "$RUN_DB" --out "$PROMOTED_RUN/.consist/new-run-only.duckdb" --include-data --json
consist db merge "$PROMOTED_RUN/.consist/new-run-only.duckdb" --db-path "$MAIN_DB" --dry-run --conflict error --json
consist db merge "$PROMOTED_RUN/.consist/new-run-only.duckdb" --db-path "$MAIN_DB" --conflict error --json
```

Use `--conflict skip` only when merging an unfiltered `$RUN_DB` that contains pre-existing seeded runs, and verify that `conflicts_detected` lists only those historical run IDs.

## 6. Seed Future Runs from the Main DB

To let subsequent cluster runs cache-hit or validate admission policies against promoted runs while isolating active writes to a run-local DuckDB shard, enable shared database seeding in your scenario YAML:

```yaml
shared:
  database:
    enabled: true
    path: /clusterfs/<project-or-nfs-root>/pilates-main/provenance.duckdb

run:
  consist_db_local_run: true
  consist_db_seed_from_shared_on_start: true
  consist_db_seed_strict: false
```

## Related Pages

- Read [Lawrencium](lawrencium.md) and [HPC Overview](hpc_overview.md) for 3-tier cluster storage setup.
- Consult [Workspace Layout](../reference/workspace_layout.md) for archive versus mutable workspace structure.
- See [Analysis Overview](../analysis/overview.md) and [SQL and DuckDB](../analysis/sql_and_duckdb.md) for querying promoted archives.
