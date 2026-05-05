---
title: Database Schema Reference
summary: Reference map for the current curated schema families and live run database surfaces.
---

# Database Schema Reference

## Current Schema Surface

The schema registry in `pilates.database.schema.registry` is the source of truth for which SQLModel classes PILATES registers with Consist.

It resolves schema classes in three ways:

- exact artifact keys such as `households_asim_in`, `beam_plans_out`, or `linkstats`
- prefix families such as `events_parquet_`, `linkstats_`, `householdv_`, and `vehicles_`
- split BEAM event keys, where `events_parquet_..._type_...` maps to a more specific event schema

The current curated families cover the same model groupings the workflow docs use:

- ActivitySim inputs and outputs
- ATLAS CSV artifacts
- BEAM plans, network outputs, linkstats, route history, and event splits
- UrbanSim postprocess tables and archive-friendly handoff tables

The generated schema-visualization surface is checked into the repo as ERD artifacts and can be regenerated from `pilates/database/scripts/generate_schema_erd.py`. The public docs site should treat those generated files as a reference map, not as an architecture roadmap.

## Adjacent Pages

- Start with [Database Setup](database_setup.md).
- Pair this with [Database Documentation Guide](database_documentation_guide.md).
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for practical query patterns.
