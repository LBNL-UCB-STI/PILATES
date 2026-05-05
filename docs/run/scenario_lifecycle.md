---
title: Scenario Lifecycle
summary: High-level shape of a PILATES run from bootstrap through archive and restart.
---

# Scenario Lifecycle

## Practical Lifecycle In One Pass

The runtime is easiest to understand as a sequence of ownership handoffs:

1. The CLI loads a settings file and optional state file.
2. The launcher prepares a run context with settings, `WorkflowState`, enabled surface, storage roots, tracker, workspace, and failure-recovery hints.
3. Restart preflight checks whether required local workspace artifacts are present or can be deferred to bootstrap.
4. Bootstrap establishes or rehydrates workspace invariants before the main scenario context.
5. The launcher enters the Consist scenario context, declares the coupler schema, and seeds bootstrap-safe artifacts.
6. The year loop calls stage modules in order.
7. Stages call tracked step factories; step factories call model components and publish typed outputs.
8. Snapshot and archive cleanup run at boundaries and shutdown.

The important boundary is that the launcher coordinates lifecycle, but stages
and step factories own workflow execution. Model packages own model mechanics.

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
- Restart preflight runs before bootstrap when `data_initialized` is already true. It logs missing local artifacts as either blocking or bootstrap-deferred based on the enabled workflow surface.
- Bootstrap seeds the coupler with bootstrap-safe artifacts needed by later stages.
- If bootstrap replay hydration does not restore the required workspace invariants, the runtime can fall back to an uncached rerun for those invariants.
- After bootstrap, strict restart mode can fail fast if required restart artifacts are still missing.

Bootstrap is intentionally outside the normal model-science loop. It prepares
the local workspace and workflow-visible starting handles; it should not be read
as a fifth model stage.

## Scenario Context

- PILATES enters a Consist scenario context after bootstrap.
- The launcher declares the coupler outputs up front and then seeds bootstrap artifacts into the coupler.
- From there, the workflow runs inside the year loop, and each stage is executed in order based on the current `WorkflowState` plus the enabled workflow surface for that state.
- The scenario contract is built from the static workflow catalog filtered through the enabled workflow surface.
- The coupler schema is declared before model stages run, so later publications can be checked against the expected workflow surface.

## Year Loop

- Land use runs first when enabled.
- Vehicle ownership runs next when enabled.
- Supply/demand then runs its internal loop of ActivitySim and BEAM steps.
- Postprocessing runs last when enabled.
- After each stage, the launcher can snapshot Consist state for later analysis or recovery.

The year loop itself should stay thin. If you need to understand why a model
ran, inspect the relevant stage module. If you need to understand what a model
published, inspect the step factory and typed outputs class.

## Shutdown

- When the scenario finishes, the launcher flushes the archive queue and stops the archive worker.
- If the final snapshot did not succeed, it mirrors the local Consist database back to the archive path.
- This means the archive-side provenance store is the durable record of the run, even when the mutable local workspace is ephemeral.

## Fresh Run Versus Restart

- Fresh runs create a new run name and then build both archive and local run directories.
- Restart runs reuse the archived state path from the previous run.
- If the run is already initialized, the launcher can skip bespoke restart hydration and rely on replay plus cache hits.
- Restart does not mean the runtime manually reconstructs every intermediate file. The current path prefers declared outputs, bootstrap replay, and Consist cache hits.

If you are new to PILATES, this page is the bridge between “the CLI loaded my config” and “the workflow is now running stages inside the scenario context.”

## Adjacent Pages

- Read [CLI](cli.md) and [Configuration Reference](configuration_reference.md) next.
- For restart details, go to [Restart and Resume](restart_and_resume.md).
- For the workflow semantics behind the stage loop, go to [Consist in PILATES](../workflow/consist_in_pilates.md).
