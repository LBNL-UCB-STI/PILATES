# HPC Job Scripts

This directory contains the Slurm submission scripts used to run PILATES on Lawrencium-style HPC environments.

## What This Setup Does

The current HPC workflow is based on `venv + pip` rather than conda solving at job runtime.

Why:

- Faster startup in queued jobs.
- More predictable installs for shared clusters.
- Easy support for local editable `consist` today, and PyPI install later.
- Explicit HPC dependency management in `hpc/requirements-hpc.txt` so cluster
  installs stay stable even if local/dev requirements evolve.

## Files

- `job_runner.sh`: submit wrapper that chooses partition resources and submits `job.sh`.
- `job.sh`: runs inside the allocated node; bootstraps Python env and executes `run.py`.
- `requirements-hpc.txt`: HPC-focused pinned/guarded dependency list.

## Quick Start

1. Clone PILATES into your scratch/source location (default expected path is `/global/scratch/users/$USER/sources/PILATES`).
2. Ensure modules are available (at least Python/GCC/PROJ as used in `job.sh`).
3. Submit from repo root:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

The command prints the Slurm log path.

## Common Submit Patterns

Default `lr7`:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

By default this requests `240G` on `lr7`.

Use high-memory `lr7` mode (`480G`):

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> --high-mem
```

Use `lr8`:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -p lr8
```

Restart from an existing stage file:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -s current_stage_restart.yaml
```

## BEAM Memory Templating

`job_runner.sh` supports settings files that contain `${BEAM_MEMORY}`.

- For `lr7 --high-mem`, default is `400g`.
- For default `lr7` (240G job memory), default is `180g`.
- For `lr8`, default is `600g`.

Override explicitly:

```bash
BEAM_MEMORY=450g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -p lr7
```

`job_runner.sh` writes a per-job generated settings file (`settings_<jobid>.yaml`) and submits that file to `job.sh`.

## Python Environment Behavior

`job.sh` uses:

- `PILATES_DIR` (default `/global/scratch/users/$USER/sources/PILATES`)
- `PILATES_VENV_PATH` (default `$PILATES_DIR/PILATES-env`)
- `PILATES_REQUIREMENTS_FILE` (default `$PILATES_DIR/hpc/requirements-hpc.txt`, fallback to `$PILATES_DIR/requirements.txt`)

Bootstrapping logic:

1. Create venv if missing.
2. Install dependencies from requirements file when needed.
3. Cache requirements hash in `$VENV_PATH/.last_requirements_hash` to skip unnecessary reinstall.

To force dependency reinstall:

```bash
rm -f /global/scratch/users/$USER/sources/PILATES/PILATES-env/.last_requirements_hash
```

## `consist` Install Behavior

`job.sh` installs `consist` as follows:

1. If local source exists at `CONSIST_SRC_DIR` (default `$PILATES_DIR/../consist`), install editable (`pip install -e`).
2. Otherwise, install from PyPI package name in `CONSIST_PYPI_PACKAGE` or fall back to `consist==0.1.3`.
3. Validate with `from consist import create_tracker`.

Override examples:

```bash
CONSIST_SRC_DIR=/global/scratch/users/$USER/sources/consist ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

```bash
CONSIST_PYPI_PACKAGE=consist==0.1.3 ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

## Dependency Gates

Version-sensitive dependencies should be managed in `hpc/requirements-hpc.txt`.

Current important gates include:

- `numpy<2.0`
- `zarr==3.1.5`
- `tables>=3.9.0,<4.0`

When changing these, keep compatibility with the active Python module version and rerun a small smoke job before broad rollout.

## Useful Runtime Overrides

Examples:

```bash
EXPECTED_EXECUTION_DURATION=2-00:00:00 ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

```bash
MEMORY_LIMIT_GB=550 BEAM_MEMORY=420g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> -p lr7
```

`job.sh` also sets thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) from `PILATES_THREADS` (default `8`).

## Run Notifications

PILATES can post compact Consist run and step milestones from the Slurm job to
Slack, Google Chat, or both. Both integrations use Incoming Webhooks, so the
webhook is bound to the channel or space selected when it is created.

### Recommended: Google Chat

For new users, use Google Chat unless you specifically need Slack. Google Chat
webhooks are easy to create per space, and PILATES uses Google Chat's webhook
threading so all updates for one run stay in one thread.

Ask Zach for the Google Chat webhook URL, then create a local
notification env file from the checked-in template:

```bash
cp hpc/run-notifications.env.template hpc/run-notifications.env
$EDITOR hpc/run-notifications.env
```

In `hpc/run-notifications.env`, change the Google Chat section so it has the
real webhook URL and notifications enabled:

```bash
export PILATES_GCHAT_NOTIFICATIONS="${PILATES_GCHAT_NOTIFICATIONS:-1}"
export PILATES_GCHAT_WEBHOOK_URL="${PILATES_GCHAT_WEBHOOK_URL:-https://chat.googleapis.com/v1/spaces/...}"
```

Replace the example URL with the full webhook URL provided by the run
coordinator.

Then submit normally:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

`hpc/run-notifications.env` is ignored by git so real webhook URLs are not
accidentally committed. `job_runner.sh` automatically loads that file when it
exists before submitting. It then uses `sbatch --export=ALL,...`, so
notification variables from that file are passed through to `job.sh` and then to
`run.py`.

Treat webhook URLs as secrets. Do not commit them to `job.sh`, scenario YAML, or
other repo-tracked config. If a webhook URL is accidentally shared, ask the
space owner to delete and recreate that webhook.


You can also export variables directly in your current shell; the defaults in
`hpc/run-notifications.env` preserve already-exported values:

```bash
export PILATES_SLACK_NOTIFICATIONS=1
export PILATES_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

```bash
export PILATES_GCHAT_NOTIFICATIONS=1
export PILATES_GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/..."
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

Optional controls:

```bash
export PILATES_SLACK_TIMEOUT_SECONDS=5
export PILATES_GCHAT_TIMEOUT_SECONDS=5
export PILATES_RUN_NOTIFICATIONS_INCLUDE_INTERNAL=0
```

By default, notifications include the scenario header and child Consist
`scenario.run(...)` steps. Internal setup traces are skipped unless
`PILATES_RUN_NOTIFICATIONS_INCLUDE_INTERNAL=1` is set.

## Legacy Config Migration

If the passed settings file is in legacy format (does not contain `run:`, `shared:`, and `infrastructure:`), `job.sh` runs:

```bash
python3 scripts/migrate_config.py <old> <old>_migrated.yaml --no-validate
```

and uses the migrated file if migration succeeds.
