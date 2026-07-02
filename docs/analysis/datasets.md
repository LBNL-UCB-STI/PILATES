---
title: Datasets
summary: How to treat analysis-ready tables after the dataset-builder cleanup.
---

# Datasets

## Current Shape

Use `Archive.table(...)` and `Archive.measure(...)` first when you are exploring
outputs in a notebook. They keep the result lazy in Ibis and let you compare
over ordinary facets.

Use `difference(...)`, `delta(...)`, `delta_change(...)`, or `rank(...)` for
scenario and parameter-sweep comparisons.

The old linkstats, ActivitySim trips, skim, and scenario-comparison dataset
builders were removed. If you need a packaged CSV, build the measured Ibis table
you want, then call `.to_pandas().to_csv(...)` or use the Ibis backend's export
surface.
