---
title: Workspace Layout
summary: Stable path semantics for local workspace, archive roots, and run-local artifacts.
---

# Workspace Layout

## Current Path Model

The normal launcher path requires `run.output_directory`. PILATES uses that
field as the parent for the durable archive run directory. If
`run.local_workspace_root` is unset, the mutable workspace is created under the
same root; if it is set, the mutable workspace is created under that separate
root. In both cases, the launcher creates a run-specific folder named from the
region, `run.output_run_name`, and the startup timestamp.

`Workspace.full_path` is the root of that run-specific mutable workspace. The
lower-level `Workspace` class still has an in-place fallback for direct test or
helper use, but that is not the supported `python run.py -c ...` runtime path.

The workspace owns model-facing mutable directories such as:

- UrbanSim mutable data
- ActivitySim mutable data, configs, output, and runtime cache
- BEAM mutable data and output
- ATLAS mutable input and output directories

Those paths are resolved from the current settings, not from ambient `cwd` assumptions. The ActivitySim and BEAM mutable directories can also be overridden explicitly when restart or replay needs to point at a specific path.

The archive root is the durable run directory that PILATES uses for the archived Consist store and for replay-aware recovery. On HPC this is the shared-scratch side selected by `run.output_directory`, not the node-local mutable workspace. When the run uses a local Consist DB, `resolve_consist_db_paths()` places the local DB under `workspace/.consist/<filename>` and mirrors the archive copy under `archive/.consist/<filename>`. If local Consist DB tracking is disabled, PILATES uses the configured shared DB path instead.

Cold recovery roots configured through `run.recovery_archive_roots` are not active execution roots. They are post-run promotion destinations that receive a copy of the full archive run directory plus an updated run-local Consist DB whose artifacts record those promoted locations as `recovery_roots`.

Snapshot and recovery artifacts live under the archive-side `.consist` tree:

- `.consist/snapshots/latest`
- `.consist/snapshots/history`
- the DB sidecar WAL file when present
- the snapshot metadata sidecar written next to the snapshot DB
- `.consist/recovery_promotion.json` after a successful promotion to one or more recovery roots

## Adjacent Pages

- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) first.
- Pair this with [Opening Archives](../analysis/opening_archives.md).
- Use [Lawrencium](../run/lawrencium.md) for cluster-specific storage posture.
