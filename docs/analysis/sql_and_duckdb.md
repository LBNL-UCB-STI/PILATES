---
title: SQL and DuckDB
summary: Inspecting PILATES runs through DuckDB, helper utilities, and example SQL workflows.
---

# SQL and DuckDB

## Purpose

Show the SQL-first and DuckDB-first analysis path for archived runs.

## Who This Is For

- Analysts who prefer direct database queries and lightweight run inspection.
- Operators verifying DB health, artifact metadata, and lineage records after a run.

## This Page Answers

- Where is the run-local or archived Consist DuckDB file and how should it be opened?
- Which helper functions exist for DB health and inspection?
- Which SQL questions are common enough to ship as examples or templates?

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) if you want packaged commands instead of ad hoc SQL.
- Pair this with [Database Setup](../reference/database_setup.md) and [Database Documentation Guide](../reference/database_documentation_guide.md).

## Source Material To Mine

- `pilates/utils/consist_analysis.py`
- `analysis/src/pilates_consist_analysis/runtime.py`
- `docs/example_queries.sql`
