---
title: Artifact Facet Catalog
summary: Indexed artifact facet conventions used when PILATES logs workflow artifacts to Consist.
---

# Artifact Facet Catalog

## Purpose

Serve as the durable reference for queryable artifact facet conventions.

## Who This Is For

- Contributors publishing new artifact families or extending indexed metadata.
- Analysts filtering artifacts by year, iteration, artifact family, or model-specific fields.

## This Page Answers

- Which facet fields are currently part of the public query surface?
- How are BEAM, ActivitySim, UrbanSim, and ATLAS families distinguished?
- Which fields are meant for indexed querying versus internal detail?

## Adjacent Pages

- Read [Artifact Semantics](artifact_semantics.md) first for the meaning behind the families.
- Use [Consist in PILATES](consist_in_pilates.md) for the logging and query model.
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for analysis-side querying.

## Source Material To Mine

- Existing facet conventions and query examples from the old page.
- Current logging code in workflow step modules.
