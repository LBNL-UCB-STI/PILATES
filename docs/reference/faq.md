---
title: FAQ
summary: Short answers to repeated PILATES questions that otherwise leak across multiple pages.
---

# FAQ

## Questions

### Where do I start?

Start with [Getting Started](../start-here/getting_started.md), then read [First Run Walkthrough](../start-here/first_run_walkthrough.md). If you already know the runtime shape, skip straight to [Scenario Lifecycle](../run/scenario_lifecycle.md).

### What does Consist do in PILATES?

Consist owns the scenario and step contexts, cache identity, provenance records, and published artifacts. PILATES owns the workspace topology, restart bootstrap, and the stage loop that runs inside those Consist contexts.

### How do I reason about restart or replay?

Read [Restart and Resume](../run/restart_and_resume.md) and [Consist in PILATES](../workflow/consist_in_pilates.md). Those pages explain the replay-first path, cache-hit reuse, and the current relationship between archive state and the mutable workspace.

### Where do I inspect archived runs?

Use [Opening Archives](../analysis/opening_archives.md) for the archive and tracker path, then move to [Run Discovery and Runsets](../analysis/run_discovery_and_runsets.md) or [SQL and DuckDB](../analysis/sql_and_duckdb.md) depending on whether you want grouped analysis or ad hoc queries.

### Where do I look when I am changing a model integration?

Start with [Step Contracts](../workflow/step_contracts.md), then read [Model Integration Guide](../extend/model_integration_guide.md) and [Testing New Integrations](../extend/testing_new_integrations.md). Those pages describe the tracked step shell and the current contract surface.

### Where do I look for path questions?

Use [Workspace Layout](workspace_layout.md) for mutable workspace versus archive-root behavior and [Database Setup](database_setup.md) for run-local Consist DB placement.

## Adjacent Pages

- Use [Getting Started](../start-here/getting_started.md) for the main run path.
- Use [Consist in PILATES](../workflow/consist_in_pilates.md) for the runtime model.
- Use [Analysis Patterns](../analysis/analysis_patterns.md) for post-run question routing.
