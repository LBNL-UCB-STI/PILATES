---
title: Stages and Steps
summary: How stage policy invokes native Consist step definitions.
---

# Stages and Steps

PILATES major stages are `land_use`, `vehicle_ownership_model`,
`supply_demand_loop`, and `postprocessing`. The supply-demand loop sequences
ActivitySim and traffic assignment according to `WorkflowState`.

## One stage-to-step pattern

For every model phase, follow the same boundary:

1. Define semantic input and output roles on the native step.
2. Resolve roles once into `ResolvedStepInputs`.
3. Call `execute_step(...)` with the `StepDefinition`.
4. Let Consist run, persist, and materialize the declared artifacts.
5. Project `RunResult.outputs` into typed PILATES outputs.
6. Use those typed outputs only for the next sequencing decision.

The stage owns year/iteration order, enablement-driven sequence, and
`WorkflowState` progress. It does not add a second input-selection pass,
reconstruct historical outputs, or maintain an in-memory workflow handoff
surface.

## Where to read code

- `pilates/workflows/stages/` — sequence and state transitions.
- `pilates/workflows/steps/` — decorated callable, resolver, projector, and
  `StepDefinition` for each model phase.
- `pilates/workflows/catalog.py` — stage placement and dependency policy.
- `pilates/workflows/step_execution.py` — the common native execution path.

## Restart boundary

The only committed mid-stage restart is BEAM run completion to BEAM
postprocess. It uses the pinned successor closure and current-workspace
destinations, then respects the postprocess mutation gate. Other stages resume
at their normal durable frontier.

## Adjacent pages

- Read [Workflow Architecture](architecture.md).
- Continue to [Step Contracts](step_contracts.md).
