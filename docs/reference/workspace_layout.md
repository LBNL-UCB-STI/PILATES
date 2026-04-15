---
title: Workspace Layout
summary: Stable path semantics for local workspace, archive roots, and run-local artifacts.
---

# Workspace Layout

## Current Path Model

PILATES uses `Workspace.full_path` as the root of the run-specific mutable workspace. If no output path is configured, PILATES runs in place in the current working directory. Otherwise it resolves the workspace under the configured output path and folder name.

The workspace owns model-facing mutable directories such as:

- UrbanSim mutable data
- ActivitySim mutable data, configs, output, and runtime cache
- BEAM mutable data and output
- ATLAS mutable input and output directories

Those paths are resolved from the current settings, not from ambient `cwd` assumptions. The ActivitySim and BEAM mutable directories can also be overridden explicitly when restart or replay needs to point at a specific path.

The archive root is the durable run directory that PILATES uses for the archived Consist store and for replay-aware recovery. When the run uses a local Consist DB, `resolve_consist_db_paths()` places the local DB under `workspace/.consist/<filename>` and mirrors the archive copy under `archive/.consist/<filename>`. If local Consist DB tracking is disabled, PILATES uses the configured shared DB path instead.

Snapshot and recovery artifacts live under the archive-side `.consist` tree:

- `.consist/snapshots/latest`
- `.consist/snapshots/history`
- the DB sidecar WAL file when present
- the snapshot metadata sidecar written next to the snapshot DB

## Adjacent Pages

- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) first.
- Pair this with [Opening Archives](../analysis/opening_archives.md).
- Use [Lawrencium](../run/lawrencium.md) for cluster-specific storage posture.
