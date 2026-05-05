---
title: Configuration Basics
summary: Mental model for how PILATES config is organized before the full reference.
---

# Configuration Basics

## What Belongs Where

- `run` holds the simulation run identity and runtime posture.
- `shared` holds inputs shared across models, such as geography, skims, and the database path.
- `infrastructure` holds the container/runtime selection.
- Model sections such as `urbansim`, `atlas`, `activitysim`, and `beam` hold model-specific paths and command templates.
- `postprocessing` holds downstream output and validation settings.

## Files Versus CLI

- The YAML file selects the region, years, enabled models, output roots, and runtime flags.
- The command line selects which YAML file to load and whether to resume from a stage file.
- `python run.py -c ...` is the normal invocation for choosing a settings file.
- `-S/--stage` points at an existing state file for restart or resume.

Those runtime flags are initialized once during launcher startup and then reused through the enabled workflow surface. PILATES no longer expects each subsystem to rebuild its own view of which stages, steps, or restart contracts are active.

## Runtime State And The Two Workspaces

Two runtime concepts show up in nearly every other doc, and it helps to name
them before reading further:

- **`WorkflowState`** is the persisted record of where a run is in its
  year/stage/iteration progression. It is reconstructed from
  `run_state.yaml` inside the archive run directory and is what restart
  resumes from. You will not normally edit it by hand.
- **Archive run directory vs. mutable workspace.** PILATES separates the
  *durable record* of a run (the archive — what gets promoted, queried, and
  shared) from the *fast scratch space* models actually read and write
  during execution (the mutable workspace).
  - On a single laptop they typically point at the same filesystem, and you
    can ignore the distinction.
  - On a cluster you usually want the archive on a durable shared filesystem
    and the workspace on faster local or scratch storage. See
    [HPC Overview](../run/hpc_overview.md) for the recommended split.

## The Practical Config Split

- `run.output_directory` sets the parent directory where the launcher creates the archive run directory for each run.
- `run.local_workspace_root` sets the parent directory for the mutable workspace when you want it to differ from the archive side. Leave it unset to colocate workspace and archive (typical on a laptop).
- `run.recovery_archive_roots` sets optional colder post-run archive destinations for promotion after the run finishes.
- `run.output_run_name` is part of the generated run-directory name.
- `run.enable_archive_copy`, `run.bootstrap_cache_enabled`, `run.restart_strict`, and the Consist DB snapshot fields change runtime behavior, not model science.
- `run.models` decides which model sections must be present and which stages the launcher enables.

## What Most Users Edit First

Most first-run edits are in the `run` section:

- where the archive run directory should live
- whether the mutable workspace should live somewhere else
- which models are enabled
- the run name prefix

Most readers do not need to change deeper model sections until the active template already loads and starts.

## Validation And Normalization

PILATES loads config with Pydantic validation. The current loader expands environment variables in `run.output_directory`, `run.local_workspace_root`, `run.recovery_archive_roots`, and the database path. It also enforces basic guardrails such as `end_year >= start_year`, a basename-only `consist_db_filename`, and the `atlas.beamac` restriction that appears in the loader.

## Adjacent Pages

- Read [First Run Walkthrough](first_run_walkthrough.md) for the concrete path.
- Use [Configuration Reference](../run/configuration_reference.md) for the complete field list.
- Use [Scenario Lifecycle](../run/scenario_lifecycle.md) to connect config to runtime behavior.
