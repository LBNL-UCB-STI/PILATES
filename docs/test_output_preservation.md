# Test Output Preservation Guide

This guide explains how to run stub tests and preserve the database and documentation for examination.

## Overview

The stub provenance tests can now save their output (database, documentation, artifacts) for examination after the test completes. This is useful for:

- **Exploring the database schema** - See exactly what tables, views, and data are created
- **Testing documentation features** - Preview what your stakeholders will see
- **Debugging provenance tracking** - Examine run_info.json and openlineage.jsonl
- **Learning the system** - Understand data flow through concrete examples

## Quick Start

### Option 1: Using the Helper Script (Easiest)

```bash
./run_stub_test_with_output.sh
```

This runs the test and saves everything to `./test_output/`.

**Custom output directory:**
```bash
./run_stub_test_with_output.sh /path/to/output
```

### Option 2: Using Environment Variable

```bash
# Save to default location (./test_output)
PRESERVE_TEST_OUTPUT=1 python tests/test_stub_provenance_flow.py

# Save to custom location
PRESERVE_TEST_OUTPUT=/path/to/output python tests/test_stub_provenance_flow.py
```

### Option 3: Running via pytest

```bash
# Run specific test with preserved output
PRESERVE_TEST_OUTPUT=1 pytest tests/test_stub_provenance_flow.py::TestStubProvenanceFlow::test_activitysim_beam_stub_workflow -v
```

## What Gets Saved

After running the test with `PRESERVE_TEST_OUTPUT`, you'll have:

```
test_output/
├── activitysim_beam/              # From ActivitySim → BEAM test
│   ├── README.md                  # Quick start guide
│   ├── test_database.duckdb       # Complete database
│   ├── documentation/
│   │   ├── schema.html            # Interactive documentation
│   │   ├── schema.md              # Markdown documentation
│   │   ├── schema.json            # JSON schema
│   │   ├── schema.csv             # CSV for Excel
│   │   └── validation_report.json # Data quality report
│   └── artifacts/                 # Complete test workspace
│       └── stub-test/
│           ├── run_info.json      # Provenance metadata
│           ├── openlineage.jsonl  # OpenLineage events
│           └── ...                # Other test files
│
└── urbansim_atlas_activitysim_beam/  # From full workflow test
    └── (same structure as above)
```

## Exploring the Output

### View Database Documentation

**Open in browser (best for sharing):**
```bash
open test_output/activitysim_beam/documentation/schema.html
```

**Read markdown:**
```bash
cat test_output/activitysim_beam/documentation/schema.md
```

**Check data quality:**
```bash
cat test_output/activitysim_beam/documentation/validation_report.json | jq
```

### Query the Database

**Open DuckDB CLI:**
```bash
duckdb test_output/activitysim_beam/test_database.duckdb
```

**Once in DuckDB:**
```sql
-- See all summary views
.tables

-- Query run summary
SELECT * FROM run_summary;

-- Query data lineage
SELECT * FROM data_lineage_summary;

-- Query household demographics
SELECT * FROM household_demographics_summary;

-- List all tables with documentation
SELECT table_name, comment FROM duckdb_tables() WHERE comment IS NOT NULL;
```

**One-line queries:**
```bash
# View all runs
duckdb test_output/activitysim_beam/test_database.duckdb -c "SELECT * FROM run_summary"

# View data lineage
duckdb test_output/activitysim_beam/test_database.duckdb -c "SELECT * FROM data_lineage_summary"

# Count records in each table
duckdb test_output/activitysim_beam/test_database.duckdb -c \
  "SELECT table_name, COUNT(*) as records FROM runs GROUP BY table_name"
```

### Examine Provenance Artifacts

**View run metadata:**
```bash
cat test_output/activitysim_beam/artifacts/stub-test/run_info.json | jq
```

**View OpenLineage events:**
```bash
cat test_output/activitysim_beam/artifacts/stub-test/openlineage.jsonl | jq -s
```

**Check specific event:**
```bash
cat test_output/activitysim_beam/artifacts/stub-test/openlineage.jsonl | \
  jq 'select(.job.name | contains("activitysim"))'
```

## Common Use Cases

### Use Case 1: Preview Documentation for Stakeholders

```bash
# Run test
./run_stub_test_with_output.sh

# Open HTML documentation
open test_output/activitysim_beam/documentation/schema.html

# This is exactly what your stakeholders will see!
```

### Use Case 2: Test Database Queries Before Production

```bash
# Generate test database
PRESERVE_TEST_OUTPUT=1 python tests/test_stub_provenance_flow.py

# Test your queries
duckdb test_output/activitysim_beam/test_database.duckdb

# Try different summary views
SELECT * FROM run_comparison;
SELECT * FROM taz_summary;
SELECT * FROM employment_by_sector;
```

### Use Case 3: Debug Provenance Tracking

```bash
# Run test with output
./run_stub_test_with_output.sh

# Examine file records
duckdb test_output/activitysim_beam/test_database.duckdb -c \
  "SELECT short_name, description, models FROM file_records"

# Check OpenLineage events
cat test_output/activitysim_beam/artifacts/stub-test/openlineage.jsonl | jq
```

### Use Case 4: Learn the Data Model

```bash
# Generate test data
PRESERVE_TEST_OUTPUT=1 python tests/test_stub_provenance_flow.py

# Explore schema
duckdb test_output/activitysim_beam/test_database.duckdb

# In DuckDB:
.schema runs
.schema file_records
.schema activitysim_households

# Or view all documentation
SELECT column_name, data_type, comment
FROM duckdb_columns()
WHERE table_name = 'runs';
```

## Comparing Test Outputs

If you run both tests, you can compare their databases:

```bash
# Full workflow test
PRESERVE_TEST_OUTPUT=1 pytest tests/test_stub_provenance_flow.py::TestStubProvenanceFlow::test_urbansim_atlas_activitysim_beam_stub_workflow -v

# Simple workflow test
PRESERVE_TEST_OUTPUT=1 pytest tests/test_stub_provenance_flow.py::TestStubProvenanceFlow::test_activitysim_beam_stub_workflow -v

# Compare
diff test_output/urbansim_atlas_activitysim_beam/documentation/schema.md \
     test_output/activitysim_beam/documentation/schema.md
```

## Sharing Test Output

You can share the test output with colleagues:

```bash
# Zip up the output
cd test_output
zip -r activitysim_beam_test.zip activitysim_beam/

# Share the zip file
# Recipients can:
# 1. Unzip it
# 2. Open documentation/schema.html in browser
# 3. Query test_database.duckdb with DuckDB
```

## Cleaning Up

**Remove all test output:**
```bash
rm -rf test_output/
```

**Remove specific test output:**
```bash
rm -rf test_output/activitysim_beam/
```

## Continuous Integration

For CI/CD, you typically **don't** want to preserve output:

```bash
# Normal test run (no preservation)
pytest tests/test_stub_provenance_flow.py

# Only preserve on demand
PRESERVE_TEST_OUTPUT=./ci_artifacts pytest tests/test_stub_provenance_flow.py
```

## Tips and Tricks

### Tip 1: Use jq for JSON Exploration

```bash
# Pretty print run_info.json
cat test_output/activitysim_beam/artifacts/stub-test/run_info.json | jq

# Extract just file records
cat test_output/activitysim_beam/artifacts/stub-test/run_info.json | \
  jq '.file_records'

# Count model runs
cat test_output/activitysim_beam/artifacts/stub-test/run_info.json | \
  jq '.model_runs | length'
```

### Tip 2: Export Query Results

```bash
# Export to CSV
duckdb test_output/activitysim_beam/test_database.duckdb -c \
  "COPY (SELECT * FROM run_summary) TO 'run_summary.csv' (HEADER, DELIMITER ',')"

# Export to JSON
duckdb test_output/activitysim_beam/test_database.duckdb -c \
  "COPY (SELECT * FROM data_lineage_summary) TO 'lineage.json'"
```

### Tip 3: Compare Documentation Formats

```bash
# Open all formats at once
open test_output/activitysim_beam/documentation/schema.html
cat test_output/activitysim_beam/documentation/schema.md
cat test_output/activitysim_beam/documentation/schema.json | jq
```

### Tip 4: Quick Database Stats

```bash
# Get database size
du -sh test_output/activitysim_beam/test_database.duckdb

# Count all records
duckdb test_output/activitysim_beam/test_database.duckdb -c \
  "SELECT
    (SELECT COUNT(*) FROM runs) as runs,
    (SELECT COUNT(*) FROM file_records) as files,
    (SELECT COUNT(*) FROM model_runs) as model_runs"
```

## Troubleshooting

### Issue: No output directory created

**Cause:** `PRESERVE_TEST_OUTPUT` not set

**Fix:**
```bash
PRESERVE_TEST_OUTPUT=1 python tests/test_stub_provenance_flow.py
```

### Issue: Permission denied

**Cause:** Output directory not writable

**Fix:**
```bash
# Use writable location
PRESERVE_TEST_OUTPUT=~/test_output python tests/test_stub_provenance_flow.py
```

### Issue: Database file is empty

**Cause:** Test may have failed before database creation

**Fix:** Check test output for errors

### Issue: Documentation not exported

**Cause:** Database manager not available

**Fix:** Check that database was created successfully

## Advanced: Custom Preservation

You can modify `preserve_test_artifacts()` in the test file to customize what gets saved:

```python
# In tests/test_stub_provenance_flow.py

def preserve_test_artifacts(tmpdir: str, test_name: str, db_manager=None):
    # Add custom exports here
    # For example: save specific query results, create custom reports, etc.
    pass
```

## Summary

Test output preservation makes it easy to:

- ✅ Examine test databases interactively
- ✅ Preview documentation before sharing
- ✅ Debug provenance tracking
- ✅ Learn the data model
- ✅ Share examples with colleagues

**Quick commands:**
```bash
# Run and save
./run_stub_test_with_output.sh

# View documentation
open test_output/activitysim_beam/documentation/schema.html

# Query database
duckdb test_output/activitysim_beam/test_database.duckdb
```

Happy exploring! 🔍
