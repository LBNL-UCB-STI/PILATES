---
title: Glossary
summary: Shared vocabulary for runs, stages, steps, artifacts, replay, runsets, and analysis terms.
---

# Glossary

## Terms

| Term | Meaning |
| --- | --- |
| Scenario | The Consist scenario that PILATES opens for one run. It carries the run metadata, step contexts, and workflow state that the launcher assembles. |
| Run | One execution instance with an archive run directory, a mutable workspace, and a Consist-backed provenance record. |
| Stage | One major slice of the workflow loop, such as land use, vehicle ownership, supply and demand, or postprocessing. |
| Step | One typed execution boundary inside a stage. A step has a catalog entry, a model component, and typed outputs. |
| Artifact | A workflow-facing value or file path that PILATES publishes, stores, or reuses through the coupler, archive, or analysis tracker. |
| Facet | Queryable metadata attached to a run or step record. PILATES uses facets to keep cache identity and analysis filters explicit. |
| Replay | Consist-driven reuse of a previously published output when the current identity matches an archived or cached result. |
| Workspace | The mutable run directory where PILATES stages model inputs, writes model outputs, and keeps the run-local Consist DB when local DB tracking is enabled. |
| Archive root / archive run directory | The durable run directory that PILATES uses for archived outputs, restart recovery, and post-run analysis. |
| Recovery root | A synonym for the archive run directory when the docs are talking about restart and replay from the durable run archive. |
| Runset | A filtered or grouped view of discovered runs in the analysis package. |
| Epoch | An analysis grouping of runs by year, outer iteration, scenario, and model. |
| AnalysisSession | The analysis entrypoint that opens an archive or run and exposes tracker-backed analysis helpers. |
| Archive | The analysis wrapper around a completed run archive. |

## Adjacent Pages

- Use [FAQ](faq.md) for short repeated answers.
- Pair this with [Run Discovery and Runsets](../analysis/run_discovery_and_runsets.md).
- Pair this with [Step Contracts](../workflow/step_contracts.md).
