---
title: Analysis Patterns
summary: Common post-run research and operational questions and which analysis surface answers each one.
---

# Analysis Patterns

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then branch to [Faceted Comparison](scenario_comparison.md), [Run Discovery and Runsets](run_discovery_and_runsets.md), or [SQL and DuckDB](sql_and_duckdb.md).
- Use [FAQ](../reference/faq.md) for short answers that repeat across the site.

## Question Map

- "Which runs exist and how are they tagged?" -> `RunIndex`, `discover-runs`, `db-health`, `run-tagging`
- "What outcomes changed over time, iteration, or policy parameters?" -> `Archive.table()`, `Archive.measure()`, `delta()`, `difference()`, `delta_change()`, `rank()`
- "How do two scenarios differ?" -> treat `scenario_id` as the comparison facet in `difference(...)`
- "How do parameter sweeps differ?" -> treat the parameter name as the comparison facet in `difference(...)` or `rank(...)`
- "Which runs belong together?" -> `RunSet`, `runset_from_query()`, `runset_from_run_ids()`
- "Which year/iteration/model combinations are complete?" -> `EpochPanel`, `SimulationEpoch`, `epoch-panel`
- "What tables can I query?" -> `Archive.table(...)` for faceted Ibis tables; `Epoch.sql()` for compatibility paths
- "How do I inspect raw SQL or DuckDB health?" -> [SQL and DuckDB](sql_and_duckdb.md)
- "How do I write a CSV?" -> materialize the measured Ibis table with `.to_pandas()` and write it at the boundary

## Practical Order

If the question is new, start with these surfaces in order:

1. Open the archive.
2. Inspect the run index to identify available facets.
3. Request a faceted table with `Archive.table(...)`.
4. Measure outcomes with `Archive.measure(...)`.
5. Use SQL or runsets only when the faceted table path is too narrow.
