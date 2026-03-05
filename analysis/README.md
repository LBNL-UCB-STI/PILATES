# PILATES Consist Analysis

This directory is an analysis-focused sub-library for Consist-enabled, post-run
work on archived PILATES outputs.

The goal is to provide a stable place for research and diagnostics that:
- reuses Consist lineage/query surfaces,
- avoids hard-coded notebook workflows,
- and can eventually move to its own repository with its own dependency stack.

Today it is intentionally a scaffold: useful enough to run real analyses, but
still early in API hardening.

## Why This Exists

PILATES execution code and post-run research code have different change rates
and different dependency pressure.

Execution orchestration needs stability.
Analysis needs speed of iteration, richer scientific tooling, and flexible
dataset shaping across many runs.

This package creates a boundary:
- PILATES produces lineage-rich artifacts.
- `analysis/` consumes them through Consist and builds reproducible datasets.

## Intended Workloads

Primary intended workloads:
- cross-run analysis over scenario/year/iteration/sub-iteration,
- equilibrium diagnostics over outer and inner loops,
- calibration evaluation and drilldown diagnostics,
- packaging/sharding of analysis-ready run subsets for portable workflows.

Current focus is BEAM linkstats because that is the highest-value path already
in active use.

## Design Principles

- Consist-first access: prefer Tracker/queries/views over ad hoc file crawling.
- Reproducible outputs: each dataset build emits a machine-readable manifest.
- Canonical keys: align all outputs around a shared analysis key contract.
- Composable layers: runtime bootstrap, catalog, dataset builders, metrics, packaging.
- Migration-friendly: keep wrappers close to existing PILATES logic while APIs settle.

## LLM Quick Orientation

If you are an LLM extending this package, start here:

1. Command entrypoint: `src/pilates_consist_analysis/cli.py`
2. Tracker/bootstrap and path assumptions: `src/pilates_consist_analysis/runtime.py`
3. Dataset assembly (current linkstats path): `src/pilates_consist_analysis/datasets.py`
4. ActivitySim trips pipeline: `src/pilates_consist_analysis/activitysim_trips.py`
5. Key contract and schema expectations: `src/pilates_consist_analysis/keys.py`
6. Manifest format: `src/pilates_consist_analysis/manifest.py`
7. Equilibrium metrics baselines: `src/pilates_consist_analysis/metrics_equilibrium.py`, `src/pilates_consist_analysis/metrics_activitysim.py`
8. Bundle export integration: `src/pilates_consist_analysis/packaging.py`

When adding a new analysis family, mirror the linkstats pattern:
- discover artifacts,
- build analysis tables,
- emit manifest + canonical keys,
- add CLI command.

## Repository Layout

```text
analysis/
  pyproject.toml
  README.md
  src/pilates_consist_analysis/
    activitysim_trips.py
    cli.py
    runtime.py
    catalog.py
    datasets.py
    keys.py
    manifest.py
    metrics_activitysim.py
    metrics_equilibrium.py
    packaging.py
```

## How It Works

End-to-end flow:

1. Bootstrap a Consist tracker against an archived run directory and DB.
2. Discover runs/artifacts through query surfaces.
3. Build analysis datasets (currently linkstats artifacts/summary/deltas).
4. Compute diagnostics from those datasets.
5. Optionally export a portable bundle with Consist maintenance APIs.

Execution modes:
- Attached mode: query the archived run DB + files directly.
- Portable mode: consume data from an exported bundle (where supported).

## Data Contract

This package uses a canonical key vector to align datasets:

- `scenario_id`
- `run_id`
- `model`
- `year`
- `iteration`
- `phys_sim_iteration`
- `beam_sub_iteration`
- `seed`

Not every dataset currently populates every field. Missing fields are still
materialized so schemas are stable across outputs.

Each dataset build writes `dataset_manifest.json` with:
- source archive/db details,
- query parameters,
- produced file paths,
- row counts,
- declared key columns,
- creation timestamp.

## CLI Surface

Show all commands:

```bash
pilates-consist-analysis --help
```

Available commands:

- `discover-runs`
  - Lists runs from tracker query service with common filters.
- `build-linkstats-dataset`
  - Produces `linkstats_artifacts.csv`, `linkstats_summary.csv`, `linkstats_deltas.csv`, and `dataset_manifest.json`.
- `build-asim-trips-dataset`
  - Produces ActivitySim trips summaries:
    - `asim_trips_mode_counts.csv` (trip counts and shares by mode across year/iteration),
    - `asim_trips_purpose_mode_counts.csv`,
    - `asim_trips_depart_hour_counts.csv`,
    - `asim_trips_iteration_summary.csv`,
    - `asim_trips_mode_deltas.csv`,
    - `asim_trips_equilibrium_pairs.csv`,
    - and `dataset_manifest.json`.
- `equilibrium-metrics`
  - Computes first-pass equilibrium diagnostics from deltas.
- `activitysim-equilibrium-metrics`
  - Computes first-pass equilibrium diagnostics from `asim_trips_equilibrium_pairs.csv` (+ optional mode deltas).
- `export-bundle`
  - Wraps Consist `DatabaseMaintenance.export` for portable subsets.

## Quick Start

From repo root:

```bash
python -m pip install -e ./analysis
```

If editable install is not available in your environment, run with:

```bash
PYTHONPATH=analysis/src python -m pilates_consist_analysis --help
```

Example dataset build:

```bash
pilates-consist-analysis build-linkstats-dataset \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --year 2018 \
  --iteration 3 \
  --output-dir /tmp/linkstats_dataset
```

Example equilibrium metrics:

```bash
pilates-consist-analysis equilibrium-metrics \
  --dataset-dir /tmp/linkstats_dataset \
  --output-json /tmp/linkstats_dataset/equilibrium_metrics.json
```

Example ActivitySim trips dataset build:

```bash
pilates-consist-analysis build-asim-trips-dataset \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --namespace activitysim \
  --artifact-family trips \
  --output-dir /tmp/asim_trips_dataset
```

Example ActivitySim equilibrium metrics:

```bash
pilates-consist-analysis activitysim-equilibrium-metrics \
  --dataset-dir /tmp/asim_trips_dataset \
  --output-json /tmp/asim_trips_dataset/activitysim_equilibrium_metrics.json
```

Example export:

```bash
pilates-consist-analysis export-bundle \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --run-id <RUN_ID> \
  --out-path /tmp/analysis_bundle.duckdb \
  --include-data
```

## Assumptions

Operational assumptions:
- You are analyzing archived run outputs, not mutable in-flight workspaces.
- Archive path preserves Consist URI expectations (`workspace://` remapped to archive run dir).
- A Consist DB exists and is readable for the run.
- Linkstats analysis assumes BEAM linkstats schemas and expected columns.

Path assumptions:
- DB resolution defaults to first match in:
  - `.consist/snapshots/latest/provenance.duckdb`
  - `.consist/provenance.duckdb`
  - `.consist/snapshots/latest/consist.duckdb`
  - `.consist/consist.duckdb`

Compatibility assumptions:
- Current linkstats dataset builder wraps `pilates.utils.consist_analysis` for parity.
- Some capabilities depend on Consist version (queries/views/maintenance APIs).

## Known Limitations

- Linkstats-centric today; other artifact families are scaffold-only.
- Some “typed view” conveniences are not yet abstracted into high-level dataset APIs.
- Portable bundle completeness depends on what is ingested vs cold-file referenced.
- Equilibrium metrics are intentionally baseline diagnostics, not full hypothesis suite.

## Extending This Package

Recommended pattern for a new analysis family:

1. Add a dataset builder in `datasets.py` (or new module) returning typed DataFrames.
2. Enforce canonical keys via `ensure_canonical_key_columns`.
3. Emit a manifest with explicit query + source metadata.
4. Add metrics module(s) with deterministic, file-based outputs.
5. Register a CLI subcommand in `cli.py`.
6. Add focused tests for key contracts and metric behavior.

Keep module boundaries strict:
- runtime/bootstrap code should not contain domain metrics,
- metrics code should not own artifact discovery,
- CLI should orchestrate, not implement analysis logic.

## Roadmap

### Near Term

- Add first-class dataset builders for:
  - OpenMatrix-derived demand/skim analyses,
  - NetCDF-derived analyses,
  - multi-run calibration comparison tables.
- Add direct `analyze-equilibrium` CLI pipeline (build + metrics in one command).
- Add tests for runtime DB resolution, manifest integrity, and metric correctness.

### Mid Term

- Introduce explicit “attached” vs “portable” execution profiles in CLI.
- Add richer equilibrium diagnostics aligned to research hypotheses.
- Add reusable join/alignment primitives (replicate pairing, OD/link/hour harmonization).
- Add scenario/report templates for repeated calibration diagnostics.

### Longer Term

- Extract to separate repo once APIs and contracts stabilize.
- Maintain separate environment/dependency policy from PILATES runtime.
- Keep a thin integration bridge so PILATES and analysis package evolve independently.

## Non-Goals (for now)

- Replacing PILATES execution orchestration.
- Embedding heavy model-specific preprocessing in this package.
- Guaranteeing complete portability for every artifact family before ingestion strategy is finalized.
