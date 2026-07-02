---
title: Consist Analysis CLI
summary: Public CLI entrypoints for archived-run discovery and health checks.
---

# Consist Analysis CLI

## Adjacent Pages

- Start with [Opening Archives](opening_archives.md).
- Pair this with [Run Discovery and Runsets](run_discovery_and_runsets.md).
- Use [Faceted Comparison](scenario_comparison.md) for beginner comparison workflows.

## Command Groups

The CLI is built in `analysis/src/pilates_consist_analysis/cli.py` and uses the same tracker setup as the Python API.

### Discovery and health

- `discover-runs` lists runs that match run filters and can write JSON.
- `epoch-panel` summarizes runs grouped into epochs and can write CSV or JSON.
- `db-health` runs the Consist DB health checks for the archive DB.
- `run-tagging` inspects missing run tags and parent linkage consistency.

The CLI intentionally does not expose dataset builders, SQL exporters, artifact
ingest, or paired-scenario comparison commands. Use the Python `Archive` API for
analysis tables and call `.to_pandas()` or Ibis export methods at the boundary.

## Shared Arguments

Most commands accept:

- `--archive-run-dir`
- `--project-root`
- `--db-path`
- `--output-root`
- `--hashing-strategy`
- `--access-mode`

These are the same values used by `create_analysis_tracker()`.

## Python Counterparts

If you are already in Python, prefer the smaller faceted surface:

- `open_archive(...)`
- `Archive.table(...)`
- `Archive.measure(...)`
- `delta(...)`, `difference(...)`, `delta_change(...)`, and `rank(...)`

Compatibility and lower-level surfaces are still available through:

- `AnalysisSession`
- `Archive`
- `RunIndex`
- `RunSet`
- `EpochPanel`
