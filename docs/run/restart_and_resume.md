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

`beam_run -> beam_postprocess` is the sole mid-stage boundary. BEAM execution is split into two discrete Consist `@define_step` steps:

1. `beam_run`: Executes the compute-intensive simulation container. Its outputs are tracked by Consist and verified into the recovery root (`archive_run_dir`).
2. `beam_postprocess`: Merges raw iteration skims into the master skims and extracts link statistics and events.

### Restart Across the Boundary
If a failure occurs after `beam_run` commits but before `beam_postprocess` completes:
- Consist automatically recognizes `beam_run` as a cache hit.
- If scratch was wiped or purged, Consist's native `cache_hydration="outputs-requested"` automatically hydrates required outputs directly from the recovery root.
- `beam_postprocess` executes without re-running the hours-long container.

### Postprocess Refusal Gate (`.beam_postprocess_in_progress`)
Because `beam_postprocess` currently modifies the workspace skims store (`skims.zarr`) in place, a crash mid-postprocess leaves the store partially mutated. Rerunning postprocess over a partially mutated store would double-merge trips and silently corrupt data.

To prevent this:
- When postprocess begins, PILATES sets an in-progress marker (`.beam_postprocess_in_progress`) in both scratch (`<beam_output_dir>`) and the permanent recovery root (`<archive_run_dir>`).
- If postprocess is interrupted, any restart attempt detects the marker and raises `RuntimeError`, refusing to restart automatically.

### Operator Recovery Procedure for Interrupted Postprocess
If a run halts with `Interrupted BEAM postprocess detected`:

1. **Restore Clean Skims**:
   - For iteration 0: Copy clean `skims.zarr` from the ActivitySim output archive (`<archive_run_dir>/activitysim/output/cache/skims.zarr`) or initial scenario seed back to `<workspace>/activitysim/output/cache/skims.zarr`.
   - For iteration > 0: Copy `skims.zarr` from the previous iteration's archived artifact.
2. **Clear the In-Progress Marker(s)**:
   - Remove `<beam_output_dir>/.beam_postprocess_in_progress` (if present in scratch).
   - Remove `<archive_run_dir>/.beam_postprocess_in_progress` (if present in permanent archive).
3. **Resume the Workflow**:
   - Rerun PILATES with the restart command printed by the launcher (or `--restart` flag).
   - `beam_run` will cache-hit, and `beam_postprocess` will run fresh against the restored clean skims.

## Operator Checks

When validating a restart, check:

1. the durable state and its normal stage/year frontier;
2. for the BEAM boundary, that `beam_run` committed successfully and its outputs are visible in the recovery root;
3. that no stale `.beam_postprocess_in_progress` markers remain from an unhandled crash;
4. the native step's typed projected outputs after execution or cache admission.

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) for launch lifecycle.
- Read [Step Contracts](../workflow/step_contracts.md) for resolve-once native execution.
- Read [Consist in PILATES](../workflow/consist_in_pilates.md) for ownership and cache behavior.
