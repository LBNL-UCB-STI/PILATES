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

## Prioritized TODO / Backlog

Consolidated roadmap for this package (highest priority first):

- [x] **P0: Epoch-aware multi-run API (first batch)**
  - Extend `RunSet` with converged selection by max completed outer iteration per group (default `year + scenario_id`).
  - Add epoch panel primitives (`SimulationEpoch`, `EpochPanel`) and builder helpers.
  - Add runtime run-tag validation warnings and expose them on analysis sessions.
  - Add CLI `epoch-panel` for quick multi-run epoch summaries.
- [ ] **P1: Tagging hardening + consistency**
  - Tighten scenario/year/iteration/model coverage across runs so epoch grouping is fully deterministic without fallbacks.
  - Normalize ASim↔BEAM parent linkage patterns across historical archives.
- [x] **P1: Scenario compare + epoch integration**
  - Add explicit converged-epoch selection modes to `compare-scenarios`.
  - Add guardrails for mixed-completeness epochs in cross-scenario alignment.
- [ ] **P2: Portable analysis depth**
  - Expand portable mode coverage for workflows currently tied to attached DB/runtime queries.
- [ ] **P2: Repository extraction readiness**
  - Continue API hardening and dependency boundary cleanup for eventual split into a standalone analysis repository.

## LLM Quick Orientation

If you are an LLM extending this package, start here:

1. Command entrypoint: `src/pilates_consist_analysis/cli.py`
2. Tracker/bootstrap and path assumptions: `src/pilates_consist_analysis/runtime.py`
3. Dataset assembly (current linkstats path): `src/pilates_consist_analysis/datasets.py`
4. ActivitySim trips pipeline: `src/pilates_consist_analysis/activitysim_trips.py`
5. Skim convergence pipeline: `src/pilates_consist_analysis/skim_analysis.py`
6. Notebook API/session layer: `src/pilates_consist_analysis/api.py`
7. Multi-run grouping/alignment abstraction: `src/pilates_consist_analysis/runset.py`
8. Scenario comparison layer: `src/pilates_consist_analysis/scenario_compare.py`
9. Key contract and schema expectations: `src/pilates_consist_analysis/keys.py`
10. Manifest format: `src/pilates_consist_analysis/manifest.py`
11. Equilibrium metrics baselines: `src/pilates_consist_analysis/metrics_equilibrium.py`, `src/pilates_consist_analysis/metrics_activitysim.py`
12. Bundle export integration: `src/pilates_consist_analysis/packaging.py`

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
    api.py
    cli.py
    runtime.py
    runset.py
    scenario_compare.py
    catalog.py
    datasets.py
    keys.py
    manifest.py
    metrics_activitysim.py
    metrics_equilibrium.py
    packaging.py
    skim_analysis.py
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
- `build-skim-dataset`
  - Produces skim convergence datasets from OpenMatrix metadata views.
- `db-health`
  - Runs Consist `inspect` + `doctor` checks before heavy analysis.
- `run-tagging`
  - Emits structured run-tagging coverage/linkage diagnostics (`json` or `table`), with strict and fail-on-issues modes.
- `epoch-panel`
  - Summarizes simulation epochs by year/outer iteration/scenario and can emit converged-only rows.
- `compare-scenarios`
  - Compares two run sets across `linkstats`, `asim_trips`, and/or `skims`, and computes config diffs for aligned run pairs.
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

Example skim convergence dataset:

```bash
pilates-consist-analysis build-skim-dataset \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --output-dir /tmp/skim_dataset
```

## Artifact Family Mapping Overrides

`EpochViews` uses `ARTIFACT_FAMILIES` by default, but you can override/extend it at runtime.

Merge behavior:
- overrides are merged by `model -> logical_name`,
- for an existing logical entry, override fields replace defaults (`artifact_family`, `concept_key`, etc.),
- new models/logical names are added.

Direct dict override:

```python
from pilates_consist_analysis import open_run

session = open_run(
    "/path/to/archive/run",
    project_root="/Users/zaneedell/git/PILATES",
    artifact_families={
        "beam": {
            "linkstats": {
                "artifact_family": "linkstats_custom_family",
            }
        }
    },
)
```

JSON file override:

```python
session = open_run(
    "/path/to/archive/run",
    project_root="/Users/zaneedell/git/PILATES",
    artifact_families_json_path="/tmp/artifact_families_override.json",
)
```

Environment fallback (used when `artifact_families_json_path` is not provided):

```bash
export PILATES_ANALYSIS_ARTIFACT_FAMILIES_JSON=/tmp/artifact_families_override.json
```

Example DB health gate:

```bash
pilates-consist-analysis db-health \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --strict \
  --fail-on-issues
```

Example scenario comparison:

```bash
pilates-consist-analysis compare-scenarios \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --left-name baseline \
  --right-name policy \
  --left-model beam \
  --left-status completed \
  --left-tag baseline \
  --right-model beam \
  --right-status completed \
  --right-tag policy \
  --align-on year \
  --latest-group-by year \
  --latest-group-by model \
  --dataset linkstats \
  --dataset asim_trips \
  --dataset skims \
  --output-dir /tmp/scenario_compare
```

Use `--left-run-id/--right-run-id` to override filter mode with explicit run IDs.

Example converged-epoch scenario comparison:

```bash
pilates-consist-analysis compare-scenarios \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --left-model beam \
  --left-tag baseline \
  --right-model beam \
  --right-tag policy \
  --align-on year \
  --use-converged \
  --converged-group-by year \
  --converged-group-by scenario_id \
  --dataset linkstats \
  --output-dir /tmp/scenario_compare_converged
```

When `--use-converged` is enabled, compare validates that both sides have completed epoch candidates
for each overlapping alignment key. If one side is incomplete/missing for an aligned key, it raises a
`ValueError` with guidance to refine filters, adjust `--converged-group-by`, or disable converged mode.

## Python API (Notebook-Friendly)

The package now exposes a session API:

```python
from pilates_consist_analysis import open_run

session = open_run("/path/to/archive/run", project_root="/Users/zaneedell/git/PILATES")

runs = session.runs(runset_name="all-2030", year=2030)
converged_runs = runs.converged()
panel = session.epochs(models=["activitysim", "beam"])
epoch_2030 = session.converged_epoch(year=2030, models=["activitysim", "beam"])
trips = session.trips(year=2030)
skims = session.skims(year=2030)
health = session.inspect_db()
tagging_warnings = session.tagging_warnings
tagging_report = session.run_tagging_report()
session.assert_run_tagging_consistent(strict=True)
comparison = session.compare_scenarios(
    left=["run-a1", "run-a2"],
    right=["run-b1", "run-b2"],
    datasets=["linkstats", "asim_trips", "skims"],
    use_converged=True,
    converged_group_by=["year", "scenario_id"],
)
```

Optional strict tagging enforcement at session open:

```python
session = open_run(
    "/path/to/archive/run",
    project_root="/Users/zaneedell/git/PILATES",
    strict_tagging=True,
)
```

CLI tagging report examples:

```bash
pilates-consist-analysis run-tagging \
  --archive-run-dir /path/to/archive/run \
  --project-root /Users/zaneedell/git/PILATES \
  --output-format table \
  --include-issues \
  --strict \
  --fail-on-issues
```

Epoch-scoped cross-model views (Batch 1):

```python
epoch = session.converged_epoch(year=2030, models=["activitysim", "beam", "urbansim"])
views = session.views(epoch)

trips_df = views.query("select * from {views.trips} limit 5")
linkstats_df = views.query("select run_id, count(*) as n from {views.linkstats} group by 1")
skim_meta = views.skim_summary
```

Current first-batch scope:
- view helpers currently cover core ActivitySim/BEAM/UrbanSim artifact families,
- and `skim_summary` is metadata-level (OpenMatrix matrix metadata), not full skim payload joins.

`RunSet` supports split/alignment workflows for multi-run analysis:

```python
parts = runs.split_by("model")
baseline = parts["activitysim"].latest(group_by=["year"])
policy = parts["beam"].latest(group_by=["year"])
aligned = baseline.align(policy, on="year")
config_diffs = aligned.config_diffs(namespace="beam")
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
