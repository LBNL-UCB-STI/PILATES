---
title: First Run Walkthrough
summary: Concrete checklist from copied scenario config to a successful first local run.
---

# First Run Walkthrough

## Do This First

1. Pick an active local template from `scenarios/`, such as a `*-local.yaml` file.
2. Copy it to a working filename in your checkout.
3. Edit the paths that point at your machine:
   - `run.output_directory`, which is the parent directory where PILATES will create the archive run directory
   - `run.local_workspace_root` if you want the mutable workspace on a different filesystem
   - any model input folders that are still absolute or environment-specific
4. Confirm the region-specific data trees referenced by the template exist.
5. Run:

```bash
python run.py -c path/to/your-settings.yaml
```

## What The Launcher Does

- It loads the YAML file with Pydantic validation.
- It attaches the runtime flags and runtime options used by the launcher.
- It resolves a run-specific archive directory under `run.output_directory`.
- It uses `run.local_workspace_root` as the mutable workspace parent when that field is set; otherwise it uses the archive side.
- It creates the run-specific archive directory and mutable workspace, then starts the scenario lifecycle.

## What Counts As Success

A successful first run reaches the scenario lifecycle and leaves behind a run-specific archive directory plus the restart state needed for later replay or resume. If the run fails early, the launcher logs a restart command that reuses the same config file and, when available, the existing state file.

## When To Stop And Switch Pages

- Use [Configuration Basics](configuration_basics.md) if you need the config mental model first.
- Use [CLI](../run/cli.md) if you want the exact flags.
- Use [Scenario Lifecycle](../run/scenario_lifecycle.md) if you want to understand what happens after startup succeeds.
- Use [Troubleshooting](../run/troubleshooting.md) if the run stops during config loading, bootstrap, or startup validation.
