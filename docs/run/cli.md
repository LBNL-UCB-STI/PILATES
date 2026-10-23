---
title: CLI
summary: Command-line entrypoints for local PILATES runs, state-file resume, config validation, and Slurm job submission.
---

# CLI

`run.py` is the primary entrypoint for PILATES simulations. It parses command-line flags, loads and validates the YAML scenario configuration, and hands control to `pilates.runtime.launcher`.

## `python run.py` Flags

`pilates/utils/io.py:parse_args_and_settings` accepts three flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `-c`, `--config <path>` | `settings.yaml` | Path to the YAML scenario configuration file (or `base:` overlay). |
| `-S`, `--stage <path>` | `None` | Path to an existing `run_state.yaml` file inside a prior archive run directory for restart or resume. The loader verifies that this file exists before loading the config. |
| `--allow-rewind-resume` | `False` | Permits resuming when `run.start_year` in the YAML config is earlier than the `current_year` recorded in the archived `run_state.yaml`. |

## Invocation Examples

```bash
# Start a fresh run from a scenario configuration
python run.py -c scenarios/sfbay/settings-sfbay-newconfig-local.yaml

# Resume an interrupted run from its archived state file
python run.py -c scenarios/sfbay/settings-sfbay-newconfig-local.yaml \
  -S /abs/path/to/pilates-outputs/pilates-run--sfbay--.../run_state.yaml

# Resume from an earlier simulation year than the state file frontier
python run.py -c scenarios/sfbay/settings-sfbay-newconfig-local.yaml \
  -S /abs/path/to/pilates-outputs/pilates-run--sfbay--.../run_state.yaml \
  --allow-rewind-resume
```

Tracked YAML files under `scenarios/` are baseline templates. Verify local paths, workspace roots, and enabled models before starting a run, or inspect the resolved configuration with the config validation utility:

```bash
python -m pilates.config.models scenarios/sfbay/settings-sfbay-newconfig-local.yaml --validate
```

## CLI Versus YAML Configuration

- **CLI flags** select which YAML configuration file to load and whether to resume from an existing `run_state.yaml` checkpoint.
- **YAML settings** define all simulation parameters: region, years, enabled models, storage roots, container images, and Consist provenance behavior.

The launcher records both the `-c` config path and `-S` state path on the runtime settings object so that any early failure logs an exact copy-paste restart command.

## HPC Slurm Submission (`hpc/job_runner.sh`)

On Slurm clusters, submit runs through `hpc/job_runner.sh` rather than invoking `python run.py` on a login node:

```bash
# Submit a fresh cluster job
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr7

# Resume a cluster job from an archived run_state.yaml
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  -s /global/scratch/users/$USER/pilates-outputs/pilates-run--.../run_state.yaml \
  -a <slurm_account> -p lr7
```

`hpc/job_runner.sh` accepts:

- `-c <config>`: Scenario YAML file (default `settings.yaml`).
- `-s <stage_file>`: Existing `run_state.yaml` file passed through to `python run.py -S`.
- `-a`, `--account <account>`: Slurm billing account (required).
- `-p <partition>`: Partition preset (`lr6`, `lr6_m192`, `lr7`, or `lr8`; default `lr7`).
- `-H`, `--high-mem`: Selects the high-memory preset on `lr6` (`180G` / `150g`) or `lr7` (`480G` / `400g`).
- `--editable-consist <abs_path>`: Installs Consist in editable mode from the specified source checkout inside the job virtual environment.

`hpc/job_runner.sh` renders `${BEAM_MEMORY}` into a per-job `settings_<jobid>.yaml` when needed and submits `hpc/job.sh`, which bootstraps the node virtual environment and runs `python run.py`.

## Related Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) for startup, bootstrap, year-loop execution, and restart rules.
- Consult [Configuration Reference](configuration_reference.md) for YAML field definitions.
- See [HPC Overview](hpc_overview.md) and [Lawrencium](lawrencium.md) for cluster storage and environment variables.
