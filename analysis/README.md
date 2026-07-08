# PILATES Consist Analysis

Post-run analysis helpers for archived PILATES runs.

The beginner path is intentionally small:

1. open an archive,
2. ask for an Ibis table with the facets you need,
3. aggregate outcomes,
4. compare over a facet such as `year`, `iteration`, `scenario_id`, or a policy parameter.

`scenario_id` is just another facet. A two-scenario comparison and a parameter
sweep use the same helpers.

## Quick Start

Install the analysis package from the PILATES repo root:

```bash
python -m pip install -e ./analysis
```

Open an archive and inspect the run catalog:

```python
import os
from pathlib import Path

from pilates_consist_analysis import open_archive

archive = open_archive(
    Path("/path/to/archive/run"),
    project_root=Path("/Users/zaneedell/git/PILATES"),
    local_cache=Path("/scratch") / os.environ["USER"] / "pilates-analysis",  # optional
)

display(archive.summary())
display(archive.runs().head())
```

Build a faceted table and aggregate a useful outcome:

```python
import ibis

from pilates_consist_analysis import delta, difference, rank

trips = archive.table(
    "activitysim.trips",
    facets=["year", "iteration", "pricing_policy", "trip_mode"],
    where={"year": [2020, 2030]},
)

counts = archive.measure(
    trips,
    by=["pricing_policy", "year", "iteration", "trip_mode"],
    measures={"trip_count": lambda table: table.count()},
)
total_window = ibis.window(
    group_by=[counts.pricing_policy, counts.year, counts.iteration],
)
mode_shares = counts.mutate(
    total_trips=counts.trip_count.sum().over(total_window),
    mode_share=counts.trip_count / counts.trip_count.sum().over(total_window),
)

over_time = delta(
    mode_shares,
    value="mode_share",
    over="year",
    by=["pricing_policy", "trip_mode"],
)

pricing_sweep = difference(
    mode_shares,
    value="mode_share",
    compare="pricing_policy",
    baseline="none",
    at={"year": 2030},
    by=["iteration", "trip_mode"],
)

ranked = rank(
    mode_shares,
    value="mode_share",
    by=["year", "iteration", "trip_mode"],
)

display(over_time.to_pandas())
display(pricing_sweep.to_pandas())
display(ranked.to_pandas())
```

`Archive.table(...)` returns an Ibis table expression backed by Consist grouped
views. Keep working lazily in Ibis until you need concrete rows, then call
`.to_pandas()`.

## Starter Notebooks

- `analysis/notebooks/archive_exploration_starter.ipynb`
  - Beginner teaching notebook for archive opening, faceted tables, outcome measurement, and deltas.
- `analysis/notebooks/local_duckdb_scratch_starter.ipynb`
  - HPC/local-scratch notebook using `open_archive(..., local_cache=...)`.

## Public API

Use these first:

- `open_archive(path, project_root=..., local_cache=...)`
- `Archive.summary()`
- `Archive.runs(...)`
- `Archive.table("model.logical_output", facets=[...], where={...})`
- `Archive.measure(table, by=[...], measures={...})`
- `delta(table, value=..., over=..., by=[...])`
- `difference(table, value=..., compare=..., baseline=..., at={...})`
- `delta_change(table, value=..., over=..., by=[...])`
- `rank(table, value=..., by=[...])`

Advanced or compatibility surfaces:

- `open_run(...)` and `AnalysisSession`
- `RunSet`, `EpochPanel`, `Epoch`, and `Archive.views(...)`

## Local Cache Behavior

`open_archive(..., local_cache=...)` copies the Consist DB to the cache up front.
When you request a table, the selected seed artifact is copied into the local
cache using the same archive-relative path. The archive writes a manifest at:

```text
<local-cache>/<archive-name>/.pilates/analysis_localization_manifest.json
```

This is intentionally not a whole-archive copy. It localizes only the DB and the
artifact files selected by the analysis surface.

## Artifact Family Mapping

`Archive.table("activitysim.trips")` resolves logical names through the default
artifact-family mapping in `epoch_views.py`. You can override or extend that
mapping at archive open:

```python
archive = open_archive(
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

You can also provide a JSON file with `artifact_families_json_path=...`.

## CLI Surface

Show all commands:

```bash
pilates-consist-analysis --help
```

Useful read-only commands:

- `discover-runs`
- `db-health`
- `run-tagging`
- `epoch-panel`

## Assumptions

- You are analyzing archived run outputs, not mutable in-flight workspaces.
- The archive contains a readable Consist DB under `.consist/`.
- Tabular outputs have enough schema metadata for Consist grouped Ibis views.
- `workspace://` paths resolve against the archive run directory or local cache mirror.

## Non-Goals

- Replacing PILATES execution orchestration.
- Building a custom comparison DSL.
- Making scenario-pair comparison the central abstraction.
- Copying whole archive trees to local scratch by default.
