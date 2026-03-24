# Hybrid Storage Proposal Status

This file is not current user or developer guidance for the active PILATES
database workflow.

The earlier content here was a design proposal for a hybrid DuckDB + external
Parquet storage model using UDTFs. It described a possible implementation
direction, not the currently documented or supported workflow in this checkout.

## Current Status

- treat the prior content as historical design exploration
- do not rely on it as evidence that hybrid storage is implemented here
- use the current database docs for the supported paths:
  - `docs/database-setup.md`
  - `docs/database_documentation_guide.md`
  - `docs/database_schema_reference.md`

## If This Work Becomes Active Again

Rewrite it as a focused design record under `docs-internal/` with:

- the exact problem statement
- the selected design
- the current implementation status
- links to the real code and tests

Until then, it should not appear as an active public documentation page.
