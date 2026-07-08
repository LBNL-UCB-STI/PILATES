---
title: Faceted Comparison
summary: Comparing archived outputs across scenario, year, iteration, and policy-parameter facets.
---

# Faceted Comparison

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist in Action](consist_in_action.md) for script examples.
- Use [SQL and DuckDB](sql_and_duckdb.md) only when the Ibis table path is too narrow.

## Mental Model

Do not start with a hand-built pair. Start with a measured table:

1. choose a logical artifact table such as `activitysim.trips`
2. request the facets you want as columns
3. group by those facets
4. compare over one facet

`scenario_id` is just a facet. Parameter sweeps use the same shape as baseline
versus policy comparisons.

## Notebook Shape

```python
import ibis

from pilates_consist_analysis import delta, difference, open_archive

archive = open_archive("/path/to/archive/run", project_root="/Users/zaneedell/git/PILATES")

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

sweep = difference(
    mode_shares,
    value="mode_share",
    compare="pricing_policy",
    baseline="none",
    at={"year": 2030},
    by=["iteration", "trip_mode"],
)
```

Call `.to_pandas()` at the display or export boundary.

## Script Shape

The small read-only example script measures ActivitySim trip mode share and
compares over a chosen facet:

```bash
python examples/consist/run_comparison.py \
  /path/to/archive-run \
  --table activitysim.trips \
  --compare pricing_policy \
  --baseline none \
  --where year=2030
```

If `--baseline` is omitted, the script ranks facet values instead of computing
differences.

## Lower-Level Path

Use `RunSet` only when the question is about selecting or aligning runs
themselves. For output comparisons, keep the simpler measured-table shape above.
