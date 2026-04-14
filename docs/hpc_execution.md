# HPC Execution

This page gives the practical overview of running PILATES on Lawrencium-style
Slurm HPC systems.

The public entry point is:

```bash
./hpc/job_runner.sh -c <settings.yaml> -a <slurm_account>
```

For deeper implementation detail, see `hpc/README.md`.

## Main Scripts

The current HPC workflow uses:

- `hpc/job_runner.sh`
  submit wrapper that selects partition/resources, templates settings, and
  submits the actual job
- `hpc/job.sh`
  node-side execution script that bootstraps Python and runs PILATES
- `hpc/requirements-hpc.txt`
  pinned HPC dependency set

## Required Inputs

At minimum you need:

- a valid settings file
- a Slurm account name
- a checkout of PILATES on the HPC filesystem
- required region-specific data already available to that checkout

By default, the scripts assume:

- `PILATES_DIR=/global/scratch/users/$USER/sources/PILATES`

## Basic Submission

Default `lr7` submission:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

This prints the Slurm log path after submission.

## Key `job_runner.sh` Arguments

### `-c <settings file>`

Settings template to submit.

If the path is relative, it is resolved relative to `PILATES_DIR`.

### `-s <stage file>`

Optional restart state file to resume from.

Example:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -s current_stage_restart.yaml
```

### `-p <partition>`

Supported partitions in the current script:

- `lr7`
- `lr8`

### `-a`, `--account <slurm account>`

Required. The wrapper exits if this is omitted.

### `--high-mem`, `-H`

Only applies to `lr7`.

- default `lr7`: 240G job memory
- `lr7 --high-mem`: 480G job memory

## Resource Defaults

Current defaults from `hpc/job_runner.sh`:

### `lr7`

- CPUs: `56`
- memory: `240G` by default
- memory: `480G` with `--high-mem`
- BEAM memory default:
  - `180g` normal
  - `400g` high-mem

### `lr8`

- CPUs: `128`
- memory: `700G`
- BEAM memory default: `600g`

## BEAM Memory Templating

`job_runner.sh` supports settings templates that contain `${BEAM_MEMORY}`.

The wrapper writes a generated settings file per submission and substitutes the
resolved BEAM memory value before submission.

Override explicitly:

```bash
BEAM_MEMORY=450g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -p lr7
```

## Useful Environment Overrides

Common overrides include:

- `PILATES_DIR`
- `MEMORY_LIMIT_GB`
- `BEAM_MEMORY`
- `EXPECTED_EXECUTION_DURATION`
- `CONSIST_SRC_DIR`
- `CONSIST_PYPI_PACKAGE`
- `PILATES_REQUIREMENTS_FILE`
- `PILATES_THREADS`

Examples:

```bash
EXPECTED_EXECUTION_DURATION=2-00:00:00 ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

```bash
MEMORY_LIMIT_GB=550 BEAM_MEMORY=420g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -p lr7
```

## Python Environment Behavior

The HPC path is based on `venv + pip`, not conda solving at job runtime.

`job.sh` typically:

1. creates a virtual environment if missing
2. installs dependencies from `hpc/requirements-hpc.txt`
3. caches a requirements hash to avoid unnecessary reinstalls
4. installs `consist` from local source if available, otherwise from PyPI

This is meant to reduce startup time and make HPC runs more predictable.

## Restart Behavior

To restart from a saved stage file:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -s /path/to/run_state.yaml
```

The node-side script ultimately runs the same PILATES runtime as local runs,
just through Slurm and the HPC bootstrap path.

## Legacy Config Migration

If the submitted settings file is still in the older flat format, `job.sh` can
attempt to migrate it by running:

```bash
python3 scripts/migrate_config.py <old> <old>_migrated.yaml --no-validate
```

If migration succeeds, the migrated config is used for the job.

## When HPC Runs Differ From Local Runs

The actual PILATES workflow is the same, but HPC adds operational differences:

- Slurm submission instead of direct local execution
- generated settings files for memory templating
- node-local environment/bootstrap behavior
- partition-specific memory and CPU defaults
- more frequent need for restart-safe archive paths and snapshots
- `run.output_directory` is the durable archive root on shared scratch, while `run.local_workspace_root` is the mutable node-local workspace.
- Logged artifacts are mirrored through the archive helper layer; Consist DB snapshots and mirrors are handled separately.

## Replay-First Storage Model

The current HPC contract is:

- shared scratch (`run.output_directory`) is the durable run/archive root during execution
- node-local flash (`run.local_workspace_root`) is disposable mutable workspace
- colder archival NFS should be exposed to PILATES through Consist `recovery_roots`, not through PILATES path guessing

Operationally this means:

- a fresh replay in a new local workspace should recover from the scratch archive
- restart after local workspace loss should still work when the archive root is intact
- explicit archive flushes remain the boundary for preemption/termination safety

## Related Docs

- `hpc/README.md`
- `docs/cli_reference.md`
- `docs/getting_started.md`
- `docs/troubleshooting.md`
