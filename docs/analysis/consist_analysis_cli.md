---
title: Consist Analysis CLI
summary: Public analysis CLI and API entrypoints for archived-run inspection and export workflows.
---

# Consist Analysis CLI

## Adjacent Pages

- Start with [Opening Archives](opening_archives.md).
- Pair this with [Run Discovery and Runsets](run_discovery_and_runsets.md) and [Datasets](datasets.md).
- Use [Scenario Comparison](scenario_comparison.md) for paired-run workflows.

## Command Groups

The CLI is built in `analysis/src/pilates_consist_analysis/cli.py` and uses the same tracker setup as the Python API.

### Discovery and health

- `discover-runs` lists runs that match run filters and can write JSON.
- `epoch-panel` summarizes runs grouped into epochs and can write CSV or JSON.
- `db-health` runs the Consist DB health checks for the archive DB.
- `run-tagging` inspects missing run tags and parent linkage consistency.

### Dataset builders

- `build-linkstats-dataset` writes `linkstats_artifacts.csv`, `linkstats_summary.csv`, and `linkstats_deltas.csv`.
- `build-asim-trips-dataset` writes the ActivitySim trips dataset CSVs.
- `build-skim-dataset` writes skim artifacts, matrices, summary, and deltas CSVs.
- `equilibrium-metrics` and `activitysim-equilibrium-metrics` compute metrics from the exported dataset outputs.

### Export and inspection

- `export-bundle` writes a portable Consist bundle for explicit run IDs.
- `export-scenario-db` resolves a runset from filters and exports a scenario bundle.
- `export-sql` runs ad hoc SQL and writes CSV or Parquet.
- `export-asim-inputs` writes ActivitySim trips/persons tables for one epoch.
- `ingest-artifacts` logs files as artifacts in a dedicated analysis run and can ingest them.
- `list-run-artifacts` prints a run artifact inventory.

### Scenario comparison

- `compare-scenarios` aligns two runsets, builds comparison frames, and writes a manifest plus CSV outputs.

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

If you are already in Python, the same surfaces are available through:

- `AnalysisSession`
- `Archive`
- `RunIndex`
- `RunSet`
- `EpochPanel`
- `Comparison`
- the dataset builder functions in `analysis/src/pilates_consist_analysis`
