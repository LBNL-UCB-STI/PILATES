---
title: Database Documentation Guide
summary: How to inspect schema docs, ERDs, and live run DB structure for PILATES outputs.
---

# Database Documentation Guide

## Current Documentation Surface

PILATES keeps two related documentation surfaces for schema inspection:

- the generated ERD assets checked into `docs/database/`
- the generated schema documentation and export files checked into the repo for schema reference and export review

The ERD generator is `pilates/database/scripts/generate_schema_erd.py`. It reads the curated SQLModel registry and can emit Mermaid, DOT, or HTML output. That makes the ERD a generated view of the current schema registry, not a hand-maintained design document.

For day-to-day use, the documentation path is:

1. Open the ERD to get the table and relationship map.
2. Read [Database Schema Reference](database_schema_reference.md) to identify the schema family.
3. Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) to inspect the live run DB or query a completed archive.

If you only need a quick check that a DB file is present and openable, use the health helpers described in [Database Setup](database_setup.md).

## Adjacent Pages

- Start with [Database Setup](database_setup.md).
- Pair this with [Database Schema Reference](database_schema_reference.md).
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for direct querying.
