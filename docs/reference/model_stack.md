---
title: Model Stack
summary: Role of UrbanSim, ATLAS, ActivitySim, and BEAM inside the PILATES workflow.
---

# Model Stack

## What PILATES Does With Each Model

PILATES treats the model families as workflow roles, not as generic simulation names.

| Model family | Current role in PILATES |
| --- | --- |
| UrbanSim | PILATES reads the base datastore, stages mutable land-use inputs, runs the land-use model, and publishes the forecast datastore and postprocess datastore handles for later stages. |
| ATLAS | PILATES reads the current or base UrbanSim datastore plus optional skims, runs the vehicle-ownership stage, and publishes year-specific ownership outputs plus the datastore handle that later stages consume. |
| ActivitySim | PILATES reads the population-source datastore and optional skims, runs the travel-demand stage, and publishes the standard ActivitySim outputs plus the archive-friendly handoff artifacts. |
| BEAM | PILATES reads plans, households, persons, config, and optional warm-start artifacts, runs traffic assignment, and publishes linkstats, plans, warm-start XML outputs, and final skims where enabled. |

PILATES also keeps a few registry-level step variants separate from the broad model families. For example, `activitysim_compile` is a tracked step variant inside the ActivitySim family, and `beam_full_skim` is a separate tracked step that publishes full-skim outputs.

If you need the real workflow-facing inputs, outputs, downstream consumers, and restart-relevant artifacts for each model family, use [Model Boundaries](model_boundaries.md). This page stays intentionally short.

## Why The Coupling Matters

The value is not only that the models run in one script. PILATES lets one model
change the operating conditions seen by the next model, then carries the
feedback forward across years and iterations. That makes it possible to ask
questions such as whether a land-use policy changes activity patterns enough to
alter congestion, whether congestion and accessibility feed back into future
location choice, or whether fleet and fuel scenarios change travel demand and
network outcomes differently across demographic groups.

Single-model runs can answer useful within-model questions, but they usually
hold the neighboring systems fixed. PILATES is designed for questions where the
interaction is the object of study: land use, vehicle ownership, daily activity
patterns, network assignment, skims, and archived provenance move together
through the same scenario.

## Reading Path

- Continue to [Model Boundaries](model_boundaries.md).
- Then use [Simulation Logic by Stage](../workflow/simulation_logic_by_stage.md).
- Then use [Artifact Flow](../workflow/artifact_flow.md).
- For developer work, continue to [Model Integration Guide](../extend/model_integration_guide.md).
