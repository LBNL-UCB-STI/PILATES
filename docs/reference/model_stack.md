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

## Reading Path

- Continue to [Simulation Logic by Stage](../workflow/simulation_logic_by_stage.md).
- Then use [Artifact Flow](../workflow/artifact_flow.md).
- For developer work, continue to [Model Integration Guide](../extend/model_integration_guide.md).
