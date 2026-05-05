---
title: Run Promotion
summary: Copy completed archive runs to NFS and merge run-local Consist DB state into a main database.
---

# Run Promotion

Use this workflow after a completed HPC run when the active archive lives on
shared scratch and the long-term copy should live on project NFS or another cold
archive root.

Promotion has two separate jobs:

1. copy the complete archive run directory to one or more recovery roots
2. make the run's Consist metadata usable from durable storage

The promotion helper always handles the first job and updates the run-local DB
so promoted output artifacts record their `recovery_roots`. When passed
`--merge-main-db`, it also handles the second job by exporting only the new root
run lineage from the run-local DB and merging that filtered shard into the main
DB.

## Prerequisites

Start from a completed archive run directory. Do not promote or merge a DB while
the Slurm job is still writing to it.

Your settings file should identify the active scratch archive root and, when
possible, the cold archive destination:

```yaml
run:
  output_directory: /global/scratch/users/${USER}/pilates-outputs
  recovery_archive_roots:
    - /clusterfs/<project-or-nfs-root>/${USER}/pilates-outputs
```

If `run.recovery_archive_roots` is not set, pass one or more `--root` values on
the command line.

## 1. Preview Promotion

Use `--dry-run` first to confirm the source run and destination path:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --dry-run
```

`--run-dir` is optional when the helper can resolve the latest matching run from
`run.output_directory` and `run.output_run_name`, but it is safer for handoff
workflows because it makes the source explicit.

## 2. Promote The Archive

Run the same command without `--dry-run`:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs
```

For each recovery root, the helper:

- copies the full archive run directory to `<root>/<run-dir-name>`
- verifies the promoted copy has the expected run state and `.consist` state
- updates artifact `recovery_roots` in the run-local Consist DB
- syncs the updated `.consist` directory into the promoted copy
- writes `.consist/recovery_promotion.json` in both the source and promoted run

That marker is the operator-visible record of what was promoted, where it was
copied, and whether each destination succeeded.

## 3. Promote And Merge Into The Main DB

If the run-local DB was seeded from the shared main DB at startup, do not merge
the whole run-local DB back into the main DB. It contains historical runs that
already exist upstream. Instead, let the promotion helper resolve the new root
run, export that root run plus its child runs to a filtered shard, and merge only
that shard:

```bash
MAIN_DB=/clusterfs/<project-or-nfs-root>/pilates-main/provenance.duckdb

python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --merge-main-db "$MAIN_DB" \
  --merge-dry-run
```

Review the JSON output. It should identify exactly one `root_run_id`, list the
expected `scoped_run_ids`, and show the Consist merge preview under
`merge_result`. `--merge-dry-run` only protects the main DB from mutation; the
archive copy and artifact `recovery_roots` update still run. If the preview
looks right, run the same command without `--merge-dry-run`:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --merge-main-db "$MAIN_DB"
```

By default the helper uses `--merge-conflict error`. That is intentional: the
filtered shard should contain only the new root run subtree, so any run-ID
conflict usually means the wrong root was selected or the run was already
merged. If root resolution is ambiguous, pass the root explicitly:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --root-run-id <new-root-run-id> \
  --merge-main-db "$MAIN_DB"
```

Automatic root selection is conservative. With `--merge-main-db`, the helper
excludes root runs already present in the main DB, then requires exactly one
remaining root. It prefers a run tagged `pilates_simulation`, then a
`pilates_orchestrator` run, and otherwise fails with candidate details instead
of guessing.

## 4. Verify The Promoted Copy

After promotion, verify the copied run without rewriting metadata:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --run-dir /global/scratch/users/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --root /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs \
  --verify-only
```

Then open the promoted archive through the analysis CLI:

```bash
pilates-consist-analysis db-health \
  --archive-run-dir /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000 \
  --project-root /Users/zaneedell/git/PILATES \
  --strict \
  --fail-on-issues
```

## 5. Manual Merge Fallback

Promotion keeps the promoted archive self-contained. The integrated
`--merge-main-db` path above is preferred when the run-local DB may contain
seeded historical runs. The manual Consist CLI commands below are useful for
inspection, recovery, or one-off maintenance.

The current Consist CLI entrypoint is:

```bash
consist db merge <shard-or-run-db.duckdb> --db-path <main-db.duckdb>
```

Despite the command wording, the input does not have to be a shard created by
`consist db export`. It can be the completed run-local DB from a promoted
PILATES archive, as long as it has the Consist schema.

### What Merge Copies

`consist db merge` reads the input DB and copies eligible rows into the target
main DB:

- run rows
- run-artifact links
- run config key-value rows
- artifacts and artifact metadata
- artifact schema rows and observations
- `global_tables` rows that are scoped by `consist_run_id` or linked by `run_id`

It does not overwrite existing run IDs. It also skips unscoped cache-style
`global_tables` rows because those rows cannot be safely attributed to a merged
run. Schema drift is checked before copying global tables: with
`--conflict error`, incompatible shared column types fail the merge; with
`--conflict skip`, incompatible global tables are skipped and reported.

### Identify The Source And Target DBs

First identify the promoted run DB. The common paths are:

- `<promoted-run>/.consist/snapshots/latest/provenance.duckdb`
- `<promoted-run>/.consist/provenance.duckdb`
- `<promoted-run>/.consist/snapshots/latest/consist.duckdb`
- `<promoted-run>/.consist/consist.duckdb`

Set shell variables so the following commands are easy to review:

```bash
PROMOTED_RUN=/clusterfs/<project-or-nfs-root>/$USER/pilates-outputs/pilates-run--seattle--my-run--20260505T120000
RUN_DB=$PROMOTED_RUN/.consist/provenance.duckdb
MAIN_DB=/clusterfs/<project-or-nfs-root>/pilates-main/provenance.duckdb
```

### Preflight Both DBs

Inspect the promoted run DB:

```bash
consist db inspect --db-path "$RUN_DB"
consist db doctor --db-path "$RUN_DB"
consist runs --db-path "$RUN_DB" --limit 20
```

Inspect the main DB:

```bash
consist db inspect --db-path "$MAIN_DB"
consist db doctor --db-path "$MAIN_DB"
```

Before a real merge, snapshot the main DB:

```bash
consist db snapshot \
  --db-path "$MAIN_DB" \
  --out "$MAIN_DB.premerge.duckdb"
```

Use a timestamped or otherwise unique snapshot path in real operations so each
pre-merge backup is preserved.

### Choose A Conflict Policy

If the run-local DB is a pure shard that was not seeded from the main DB, use
the strict path:

```bash
consist db merge "$RUN_DB" \
  --db-path "$MAIN_DB" \
  --dry-run \
  --conflict error \
  --json
```

If that dry run reports no conflicts, run the real merge:

```bash
consist db merge "$RUN_DB" \
  --db-path "$MAIN_DB" \
  --conflict error \
  --json
```

If the run-local DB was seeded from the main DB at startup, a direct strict
merge will usually find conflicts for the pre-existing seeded runs. In that case
use `--conflict skip`, but review the JSON output carefully:

```bash
consist db merge "$RUN_DB" \
  --db-path "$MAIN_DB" \
  --dry-run \
  --conflict skip \
  --json
```

The dry-run output should show:

- `runs_skipped`: pre-existing runs that were already in the main DB
- `conflicts_detected`: the same expected pre-existing run IDs
- `runs_merged`: the new completed run IDs you want to add
- `ingested_tables_merged`: any run-scoped hot tables that will be copied
- `unscoped_cache_tables_skipped`: cache tables Consist cannot safely scope
- `incompatible_global_tables_skipped`: tables skipped because of schema
  incompatibility

Only run the real merge when the skipped conflicts are expected:

```bash
consist db merge "$RUN_DB" \
  --db-path "$MAIN_DB" \
  --conflict skip \
  --json
```

### Export Only The New Run First

This is the manual equivalent of the promotion helper's `--merge-main-db` path.
When the run-local DB contains seeded historical runs, export just the new root
run lineage to a temporary shard, then merge that shard with `--conflict error`.

Find the new root run ID:

```bash
consist runs --db-path "$RUN_DB" --limit 50 --status completed
```

Then export and merge:

```bash
ROOT_RUN_ID=<new-root-run-id>
FILTERED_SHARD=$PROMOTED_RUN/.consist/new-run-only.duckdb

consist db export "$ROOT_RUN_ID" \
  --db-path "$RUN_DB" \
  --out "$FILTERED_SHARD" \
  --include-data \
  --json

consist db merge "$FILTERED_SHARD" \
  --db-path "$MAIN_DB" \
  --dry-run \
  --conflict error \
  --json

consist db merge "$FILTERED_SHARD" \
  --db-path "$MAIN_DB" \
  --conflict error \
  --json
```

`consist db export` includes child runs by default. Use `--no-children` only
when you deliberately want one exact run row and not its descendants. Use
`--include-data` when run-scoped or run-linked `global_tables` rows should be
portable with the shard.

### Verify After Merge

After the real merge, re-run maintenance checks on the main DB:

```bash
consist db inspect --db-path "$MAIN_DB"
consist db doctor --db-path "$MAIN_DB"
consist runs --db-path "$MAIN_DB" --limit 20
```

Optionally compact the main DB after a batch of merges:

```bash
consist db compact --db-path "$MAIN_DB"
```

Do not pass `--include-snapshots` for the direct promoted-run DB merge unless
you intentionally created a Consist export shard with a sibling
`shard_snapshots/` directory. The promoted PILATES archive already preserves
run-local archive state; the main DB merge is about making the run rows and
artifact metadata discoverable from the central DB.

## 6. Seed Future Runs From The Main DB

DB seeding is a startup behavior, not a post-run promotion action. Use it when a
new run should hydrate its run-local DB from a shared main DB before execution:

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

With that posture, each run writes to its run-local shard during execution. At
the end, promote the archive to NFS with `--merge-main-db` if the run should
become part of the shared historical catalog.

## Adjacent Pages

- Use [Lawrencium](lawrencium.md) for the current cluster storage posture.
- Use [Workspace Layout](../reference/workspace_layout.md) for archive versus mutable workspace semantics.
- Use [Opening Archives](../analysis/opening_archives.md) for analysis access after promotion.
