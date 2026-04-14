---
title: Restart and Resume
summary: Replay-first restart model, cache-hit behavior, and operator expectations.
---

# Restart and Resume

## What Restart Means Now

- The launcher treats a run with `state.run_info_path` as a restart run.
- It reuses the archived state path and checks whether the current state is behind the archive state.
- It can pre-run bootstrap on restart so the workspace invariants are rehydrated through the normal cached path.
- If the run is already initialized, the launcher skips bespoke restart hydration and relies on scenario replay plus Consist cache hits.

For most users, the practical meaning is simple: restart is no longer “rebuild everything by hand.” It is “re-enter the scenario with the archived state, then let declared inputs, bootstrap, and cache hits restore what the workflow still needs.”

## Preflight Checks

- The runtime checks for required local artifacts before and after restart bootstrap.
- Some missing artifacts are treated as blocking; others are deferred until bootstrap hydration.
- If `restart_strict` is enabled and required artifacts are still missing after bootstrap, the launcher raises.
- The current code logs these conditions explicitly so operators can see whether a missing path is a restart blocker or a deferred bootstrap-owned artifact.

## Input Staging And Replay

- Bootstrap seeds the coupler with workspace artifacts that later stages expect to find.
- Restart hydration can materialize archived outputs back into the local workspace when the runtime needs them.
- The legacy exact-rewind helpers can restore ActivitySim or BEAM overlay directories for explicit rewind cases.
- Those helpers are narrow tooling, not the normal launcher path.

## Legacy Helpers

- `hydrate_missing_restart_artifacts` remains available for frontier hydration and focused tests.
- `hydrate_rewind_runner_inputs` remains available for exact-rewind overlays when `allow_rewind_resume` is set.
- The launcher does not call these helpers in the normal replay-first path.

## Operator Expectations

- A restart may still execute bootstrap work if the run was not fully initialized.
- A cache hit does not mean the runtime skips all work; it means Consist can reuse matching outputs where the declared identity matches.
- If local workspace state is lost, the runtime can materialize archived outputs back into place when the archive contains what the restart needs.
- If the archive does not contain the required declared outputs, restart hydration fails rather than inventing missing inputs.

## Practical Rule

If you are deciding whether a restart is safe, check these in order:

1. the state file you are resuming from
2. the archive run directory it points at
3. whether bootstrap-owned files can be restored through the normal path
4. whether the missing files are declared workflow outputs or only local scratch

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first.
- Use [HPC Overview](hpc_overview.md) for the local-versus-archive storage topology on clusters.
- Use [Consist in PILATES](../workflow/consist_in_pilates.md) for the ownership model behind replay.
