---
title: Test Output Preservation
summary: Preserving workflow test artifacts for inspection, review, and utility-driven debugging.
---

# Test Output Preservation

## Current Workflow

Use the preserved-output workflow when you need a stable directory tree from a test run.

The current helper script, `run_stub_test_with_output.sh`, exports `PRESERVE_TEST_OUTPUT` before running the test command. That tells the test harness to keep the generated output tree instead of cleaning it away after the run.

The preserved tree is useful when you need to inspect the files that a test actually produced, including run-local artifacts, logs, and any DB or archive sidecars that the test emitted.

Keep this workflow narrow:

- use it for debugging and review of test-generated artifacts
- do not treat it as a replacement for a full scenario archive
- do not rely on it for the production run path

## Adjacent Pages

- Read [Operations Overview](overview.md) first.
- Pair this with [SQL and DuckDB](../analysis/sql_and_duckdb.md).
- Use [Testing New Integrations](../extend/testing_new_integrations.md) for how this fits into contributor verification.
