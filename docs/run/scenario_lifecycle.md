---
title: Scenario Lifecycle
summary: High-level shape of a PILATES run from bootstrap through archive and restart.
---

# Scenario Lifecycle

## What Happens At Startup

- The launcher initializes runtime flags from the validated settings file.
- It creates or restores `WorkflowState`.
- It builds the enabled workflow surface for the active run shape.
- The launcher resolves a run-specific archive run directory and a mutable local run directory.
- It initializes the Consist tracker against the archive-side run directory while mounting the project inputs and mutable workspace separately.
- It creates the `Workspace` inside the local run directory and records the archive state path there.
- It emits a `run_context` audit event with the scenario metadata and storage topology.

In practice, this is the first important distinction to keep in mind:

- the archive run directory is the durable record of the run
- the mutable local run directory is where PILATES stages live working files during execution

## Bootstrap

- Fresh runs execute the bootstrap phase before the main scenario context.
- Restart runs can also execute bootstrap pre-scenario so workspace invariants are rehydrated through the normal cached path.
- Bootstrap seeds the coupler with bootstrap-safe artifacts needed by later stages.
- If bootstrap replay hydration does not restore the required workspace invariants, the runtime can fall back to an uncached rerun for those invariants.

## Scenario Context

- PILATES enters a Consist scenario context after bootstrap.
- The launcher declares the coupler outputs up front and then seeds bootstrap artifacts into the coupler.
- From there, the workflow runs inside the year loop, and each stage is executed in order based on the current `WorkflowState` plus the enabled workflow surface for that state.

## Year Loop

- Land use runs first when enabled.
- Vehicle ownership runs next when enabled.
- Supply/demand then runs its internal loop of ActivitySim and BEAM steps.
- Postprocessing runs last when enabled.
- After each stage, the launcher can snapshot Consist state for later analysis or recovery.

## Shutdown

- When the scenario finishes, the launcher flushes the archive queue and stops the archive worker.
- If the final snapshot did not succeed, it mirrors the local Consist database back to the archive path.
- This means the archive-side provenance store is the durable record of the run, even when the mutable local workspace is ephemeral.

## Fresh Run Versus Restart

- Fresh runs create a new run name and then build both archive and local run directories.
- Restart runs reuse the archived state path from the previous run.
- If the run is already initialized, the launcher can skip bespoke restart hydration and rely on replay plus cache hits.

If you are new to PILATES, this page is the bridge between “the CLI loaded my config” and “the workflow is now running stages inside the scenario context.”

## Adjacent Pages

- Read [CLI](cli.md) and [Configuration Reference](configuration_reference.md) next.
- For restart details, go to [Restart and Resume](restart_and_resume.md).
- For the workflow semantics behind the stage loop, go to [Consist in PILATES](../workflow/consist_in_pilates.md).
