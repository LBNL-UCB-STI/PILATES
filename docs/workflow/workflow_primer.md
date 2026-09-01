---
title: Workflow Primer
summary: Conceptual entrypoint for the native Consist execution model in PILATES.
---

# Workflow Primer

## Reading Path

- Start with [Scenario Lifecycle](../run/scenario_lifecycle.md) for what happens during `python run.py`.
- Continue to [Architecture](architecture.md) for the ownership boundaries.
- Read [Stages and Steps](stages_and_steps.md) and [Step Contracts](step_contracts.md) for the executable model boundary.
- Read [Model Boundaries](../reference/model_boundaries.md) for each model's semantic handoffs.
- Read [Consist in PILATES](consist_in_pilates.md) for cache, provenance, and materialization behavior.

## One Mental Model

PILATES uses one native path at every model boundary:

```text
semantic roles
-> resolve once
-> execute the decorated callable through Consist
-> RunResult.outputs
-> archive/refresh/republish when enabled
-> TypedOutputProjector
-> next stage decision
```

`WorkflowStepSpec` is stage policy: it chooses placement, ordering, enablement,
and dependencies. A `StepDefinition` is the executable boundary: its decorated
Consist callable, resolver, output-path policy, and typed-output projector.

- **Launcher** owns run lifecycle, settings, `WorkflowState`, storage roots,
  tracker/scenario context, the year loop, and shutdown.
- **Stages** own year and iteration ordering plus durable state transitions.
- **Resolvers** select the current coupler artifacts once and freeze their
  destinations in `ResolvedStepInputs`.
- **Consist** binds and materializes inputs, executes or admits a cache hit,
  records identity and provenance, and returns persisted outputs.
- **PILATES** refreshes its action evidence and, when archive copying is
  enabled, archives outputs and republishes the refreshed artifacts before
  typed projection.
- **Typed projectors** validate `RunResult.outputs` at their declared current
  destinations and return the typed values consumed by the next stage.

The scenario coupler is the current semantic-role map, not a history store.
Consist run outputs and snapshot artifacts are the durable facts. A workspace
is mutable run-local layout, not an alternative execution contract.

## How A Run Moves Through The Code

1. `run.py` enters `pilates/runtime/launcher.py`.
2. The launcher prepares settings, `WorkflowState`, the enabled surface,
   workspace/storage roots, and a Consist tracker and scenario.
3. The year loop invokes the enabled stage sequence.
4. Each stage resolves its semantic roles once and calls `execute_step(...)`.
5. PILATES performs its current archive/refresh/republish bridge when enabled;
   the step projector then returns typed PILATES outputs for the following
   stage decision.
6. PILATES records durable stage progress and finalizes archive state.

Fresh execution, a Consist cache hit, and the committed restart path all join
at `RunResult.outputs -> TypedOutputProjector`. Stages do not rebuild an
in-memory execution graph or select historical output sources themselves.

The bridge is downstream PILATES behavior today; `scenario.run()` does not
itself make that archive-validation and refreshed-publication guarantee.

## Restart Boundary

The only committed mid-stage restart is `beam_run_completed` to
`beam_postprocess`. PILATES pins and validates that successor's exact input
closure, Consist hydrates it to the resolver's current destinations, and the
native postprocess step executes once. The
`beam_postprocess_in_progress` mutation gate is intentionally non-restartable.
All other restart behavior resumes at the normal durable stage/year frontier;
an interrupted ActivitySim region fails closed.

If you are tracing a file, distinguish a local workspace file from a declared
step input or a persisted Consist output. Only the declared artifacts belong to
the workflow contract.
