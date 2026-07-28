---
title: Restart and Resume
summary: PILATES restart policy and the single committed BEAM checkpoint.
---

# Restart and Resume

## What Restart Means

PILATES owns restart policy; Consist owns generic artifact identity,
materialization, provenance, run identity, cache admission, and output links.
A restart is therefore a stage/year-frontier decision made from durable
`WorkflowState`, except for one committed mid-stage boundary described below.

Fresh execution, a cache hit, and restart execution all converge on the same
native step result:

```text
resolved semantic inputs -> Consist RunResult.outputs -> TypedOutputProjector
```

A Consist cache hit is admitted only when the step identity and requested
outputs satisfy its configured cache policy. It is not a PILATES restart
decision and does not let the stage select an alternate historical source.

## Normal Frontier Resume

Outside the committed BEAM boundary, a restart resumes the normal durable
stage/year frontier. The launcher initializes the scenario and the stage
sequence resolves current semantic roles in the usual way. There is no other
mid-stage recovery contract.

In particular, an interrupted ActivitySim mid-stage region fails closed rather
than being treated as a generally resumable checkpoint. The next execution is
the normal stage-policy path selected by its durable frontier.

## The Sole Mid-Stage Boundary

`beam_run_completed -> beam_postprocess` is the sole committed mid-stage
restart boundary. At BEAM run completion, PILATES records a pinned successor
closure: completed producer run IDs, output keys, artifact identities and
forms, and the exact absolute destinations required by `beam_postprocess`.

On restart PILATES:

1. validates the pinned snapshot and every linked completed producer output;
2. verifies the closure can be hydrated from the archive;
3. hydrates it to the successor resolver's exact current destinations;
4. re-resolves `beam_postprocess` and executes its native `StepDefinition`.

The closure must be complete, non-overlapping, and cleanly materialized. An
ambiguous producer, absent output link, identity/form mismatch, unavailable
archive bytes, or wrong destination rejects the checkpoint. PILATES does not
accept a leftover workspace file or a coupler value as evidence.

`beam_postprocess_in_progress` is a mutation gate, not a checkpoint. Once that
gate has begun, the operation remains non-restartable.

## Operator Checks

When validating a restart, check:

1. the durable state and its normal stage/year frontier;
2. for the BEAM boundary only, the pinned checkpoint scope and successor
   closure;
3. the archive-visible artifacts and exact current destinations;
4. the native step's typed projected outputs after execution or cache admission.

If the pinned BEAM closure cannot be proven, stop and investigate the archive
or checkpoint. Do not attempt to reconstruct a missing boundary from local
scratch files.

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) for launch lifecycle.
- Read [Step Contracts](../workflow/step_contracts.md) for resolve-once native execution.
- Read [Consist in PILATES](../workflow/consist_in_pilates.md) for ownership and cache behavior.
