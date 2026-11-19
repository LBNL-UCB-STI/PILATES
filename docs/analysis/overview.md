# Analysis Overview

PILATES stores run provenance, artifact lineage, and materialized tables in `.consist/provenance.duckdb` inside each run archive. The `pilates_consist_analysis` package and `consist` API let you inspect single archives, discover runs across directories, and assemble multi-year panels.

## Opening a Single Run Archive

Point `Archive.from_dir(...)` (or `open_archive(...)`) at a completed run archive directory to inspect its stages, epochs, and artifacts:

```python
import consist
from pilates_consist_analysis import Archive, open_archive

archive = Archive.from_dir("outputs/sf_2010_2020_baseline")
# Equivalent helper: archive = open_archive("outputs/sf_2010_2020_baseline")

# High-level run metadata and stage counts
print(archive.summary())

# Forecast years and replanning iterations recorded in the archive
epochs = archive.epochs()
print(epochs)

# Registered coupler and stage artifacts
artifacts = archive.artifacts()
for art in artifacts:
    print(art.key, art.driver, art.container_uri)
```

Load any tabular artifact into a DataFrame with `consist.load_df(...)` or query archive tables through `archive.table(...)`:

```python
trips_art = archive.artifact("final_trips", year=2015)
trips_df = consist.load_df(trips_art)

# Query a canonical table or measure across epochs
vmt_by_year = archive.measure("beam_vmt_total")
households_2020 = archive.table("final_households", year=2020)
```

## Discovering Runs Across Roots (`AnalysisSession` and `RunSet`)

Use `AnalysisSession.from_roots(...)` to scan one or more output directories for `.consist/provenance.duckdb` databases and filter matching runs into a `RunSet`:

```python
from pilates_consist_analysis import AnalysisSession

session = AnalysisSession.from_roots(
    ["outputs/baseline_sweeps", "outputs/transit_scenarios"]
)

# Inspect all discovered runs
all_runs = session.runs()
print(all_runs.to_df()[["run_id", "scenario_name", "status", "start_year", "end_year"]])

# Filter into a RunSet by scenario facet, model, or completion status
completed = session.filter(
    status="completed",
    region="sfbay",
    years=[2010, 2015, 2020],
)
```

A `RunSet` groups Comparable runs so you can extract aligned epoch panels or compare configuration facets across scenarios:

```python
panel = completed.epoch_panel(measures=["beam_vmt_total", "asim_trip_count"])
print(panel.to_df())
```

## Checking Run Health and Diagnostics

Before comparing runs, validate database integrity and facet tagging with `DiagnosticReport` and the health inspection helpers:

```python
from pilates_consist_analysis import check_db_health, inspect_run_tagging

# Check a single archive for missing artifacts, failed steps, or schema warnings
report = session.DiagnosticReport(archive)
for issue in archive.issues():
    print(issue.severity, issue.step_name, issue.message)

# Audit DuckDB file health and facet tag completeness across roots
health_df = check_db_health(["outputs/baseline_sweeps"])
tagging_df = inspect_run_tagging(["outputs/baseline_sweeps"])
```

## Common Analysis Workflows

| Goal | Entry Point | Next Step |
| --- | --- | --- |
| Inspect one simulation archive | `Archive.from_dir(path)` | Call `archive.summary()`, `archive.epochs()`, and `consist.load_df(...)`. |
| Track metrics across forecast years | `archive.measure(name)` or `RunSet.epoch_panel(...)` | Aggregate by `year` and `iteration` across epochs. |
| Compare baseline vs. policy runs | `AnalysisSession.from_roots(...)` | See [Scenario Comparison](scenario_comparison.md) for outcome deltas and cache diffing. |
| Run custom SQL over DuckDB | `consist.run_query(...)` or `consist` CLI | See [SQL and DuckDB](sql_and_duckdb.md) for CLI commands and hybrid view queries. |
