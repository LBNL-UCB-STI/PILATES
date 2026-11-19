# Scenario Comparison

PILATES tracks run configurations, input bindings, and output artifacts in `.consist/provenance.duckdb`, enabling structured comparisons across policy scenarios, parameter sweeps, and warm-start branches.

## Faceted Outcome Comparison

Group related archives into a `RunSet` via `AnalysisSession.from_roots(...)` and compute outcome differences against a baseline scenario:

```python
from pilates_consist_analysis import (
    AnalysisSession,
    delta,
    delta_change,
    difference,
    rank,
)

session = AnalysisSession.from_roots(["outputs/scenarios"])
runset = session.filter(status="completed", region="sfbay")

# Load an aligned table of measures across scenarios and forecast years
vmt = runset.measure("beam_vmt_total")
mode_share = runset.table("mode_choice_summary", year=2020)

# Compute absolute and relative deltas against the baseline run
vmt_diff = difference(vmt, baseline="sf_baseline")
vmt_pct = delta_change(vmt, baseline="sf_baseline")
scenario_ranks = rank(vmt, by="value", ascending=True)
```

For a self-contained runnable script that compares scenario metrics side by side, see `examples/consist/run_comparison.py`.

## Boundary Overlap and Shared Upstream Artifacts

Policy scenarios that branch from a shared baseline often reuse prior forecast years or warm-start steps. Inspect shared vs. diverged artifacts across scenario boundaries to confirm which stages executed fresh and which hydrated from cache:

```python
overlap = session.boundary_overlap(
    left_run_id="sf_baseline_2020",
    right_run_id="sf_transit_boost_2020",
)
print(overlap.shared_steps)
print(overlap.diverged_steps)
```

See `examples/consist/cache_hit_inspection.py` for a walkthrough of cache reuse across scenario branches.

## Diagnosing Cache Hits, Misses, and Config Diffs

To investigate a cache miss or step divergence between two runs, compare their resolved configuration facets and input artifact hashes with `diff_runs` and `session.diff_configs(...)`:

```python
from pilates_consist_analysis import diff_runs

# Compare step-by-step cache signatures, input hashes, and config facets
step_diff = diff_runs(
    left_archive="outputs/scenarios/sf_baseline",
    right_archive="outputs/scenarios/sf_transit_boost",
    step_name="activitysim_demand",
    year=2015,
)
print(step_diff.changed_inputs)
print(step_diff.changed_config_keys)

# Inspect config facet differences across an entire RunSet
config_diffs = session.diff_configs("sf_baseline", "sf_transit_boost")
print(runset.config_diffs())
```

`step_diff` pinpoints whether the divergence came from an upstream coupler artifact (such as updated skims or households), a parameter change in `config.yaml`, or an adapter file edit. See [Step Contracts](../workflow/step_contracts.md) for how Consist computes step identity.

## Tracing Artifact Lineage

Trace any output artifact backward to the exact steps, configuration parameters, and upstream inputs that produced it:

```python
import consist

# Return the upstream Merkle DAG for an artifact or run
lineage_tree = consist.lineage(
    artifact_or_run_id="final_trips_2020",
    db_path="outputs/scenarios/sf_transit_boost/.consist/provenance.duckdb",
)
print(lineage_tree)
```

You can also print the lineage tree from the terminal using the Consist CLI:

```bash
consist lineage final_trips_2020 --db outputs/scenarios/sf_transit_boost/.consist/provenance.duckdb
```
