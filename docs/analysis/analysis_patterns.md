---
title: Analysis Patterns
summary: Common post-run research and operational questions and which analysis surface answers each one.
---

# Analysis Patterns

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then branch to [Run Discovery and Runsets](run_discovery_and_runsets.md), [Datasets](datasets.md), or [SQL and DuckDB](sql_and_duckdb.md).
- Use [FAQ](../reference/faq.md) for short answers that repeat across the site.

## Question Map

- "Which runs exist and how are they tagged?" -> `RunIndex`, `discover-runs`, `db-health`, `run-tagging`
- "Which runs belong together?" -> `RunSet`, `runset_from_query()`, `runset_from_run_ids()`
- "Which year/iteration/model combinations are complete?" -> `EpochPanel`, `SimulationEpoch`, `epoch-panel`
- "What tables can I query for one epoch?" -> `Archive.views()`, `Epoch.sql()`, `EpochTables.load()`
- "What packaged CSVs should I inspect first?" -> the dataset builders in [Datasets](datasets.md)
- "How do two scenarios differ?" -> `ScenarioComparison`, `compare-scenarios`, `Archive.compare()`
- "How do I inspect raw SQL or DuckDB health?" -> [SQL and DuckDB](sql_and_duckdb.md)
- "How do I move files into analysis?" -> `list-run-artifacts`, `ingest-artifacts`, `export-bundle`, `export-asim-inputs`

## Practical Order

If the question is new, start with these surfaces in order:

1. Open the archive.
2. Build a run index or run set.
3. Resolve the epoch or dataset family you need.
4. Use SQL only when the packaged surface does not already answer the question.
