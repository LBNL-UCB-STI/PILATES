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
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

The command prints the Slurm log path.

## Common Submit Patterns

Default `lr7`:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

By default this requests `240G` on `lr7`.

Use high-memory `lr7` mode (`480G`):

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> --high-mem
```

Force shared live archive output on LRC while keeping execution on node-local storage:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> --archive shared
```

Force scratch live archive output while keeping execution on node-local storage:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a <slurm_account> --archive scratch
```

Enable BEAM Java Flight Recorder profiling:

```bash
./hpc/job_runner.sh -c scenarios/breathe/settings--sfbay--2018-Baseline.yaml -a <slurm_account> -p lr7 --high-mem --beam-profile
```

Use `lr8`:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr8
```

Restart from an existing stage file:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -s current_stage_restart.yaml
```

Tracked scenario files are copy/edit starting points for the current cluster
storage posture and model wiring. They are not turnkey machine-independent
configs; review account-specific paths, data roots, output roots, and model
selections before submitting.

## BEAM Memory Templating

`job_runner.sh` supports settings files that contain `${BEAM_MEMORY}` and `${BEAM_EXTRA_JVM_ARGS}`.

- For `lr7 --high-mem`, default is `300g`.
- For default `lr7` (240G job memory), default is `180g`.
- For `lr8`, default is `600g`.

`BEAM_EXTRA_JVM_ARGS` defaults to empty. Use it to append extra JVM flags for
profiling or debugging without editing the runner code.

`--archive` controls the live run/archive topology:

- `shared`: write the archive run dir to `/clusterfs/beem-core-data-nfs/pilates-outputs`, run locally in `/local/job${SLURM_JOB_ID}/pilates-workspace`, and enable archive copying
- `scratch`: write the archive run dir to `/global/scratch/users/$USER/pilates-outputs`, run locally in `/local/job${SLURM_JOB_ID}/pilates-workspace`, and enable archive copying

In both modes, `job_runner.sh` clears `run.recovery_archive_roots`. Promotion to colder/shared recovery roots is a separate post-run mechanism and should be configured explicitly in scenario YAML when desired rather than implied by `--archive`.

Override explicitly:

```bash
BEAM_MEMORY=450g ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr7
```

Example with Java Flight Recorder override:

```bash
BEAM_MEMORY=300g \
BEAM_EXTRA_JVM_ARGS='-XX:StartFlightRecording=delay=5s,duration=30m,filename=/app/output/recording.jfr,dumponexit=true,settings=default' \
./hpc/job_runner.sh -c scenarios/breathe/settings--sfbay--2018-Baseline.yaml -a <slurm_account> -p lr7 --high-mem
```

`--beam-profile` uses a default JFR output filename under `/app/output`:

```text
/app/output/recording_<job-name>.jfr
```

Since `/app/output` is bind-mounted to the host BEAM output root, the `.jfr`
file will appear in the run's `beam/beam_output/` directory.

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
2. Otherwise, install from PyPI package name in `CONSIST_PYPI_PACKAGE` or fall back to `consist==0.1.4`.
3. Validate with `from consist import create_tracker`.

Override examples:

```bash
CONSIST_SRC_DIR=/global/scratch/users/$USER/sources/consist ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

```bash
CONSIST_PYPI_PACKAGE=consist==0.1.4 ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
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
EXPECTED_EXECUTION_DURATION=2-00:00:00 ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

```bash
MEMORY_LIMIT_GB=550 BEAM_MEMORY=420g ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr7
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

In `hpc/run-notifications.env`, paste the full Google Chat webhook URL between
the quotes and change `PILATES_GCHAT_NOTIFICATIONS` from `0` to `1`:

```bash
export PILATES_GCHAT_NOTIFICATIONS=1
export PILATES_GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/..."
```

Then submit normally:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

`hpc/run-notifications.env` is ignored by git so real webhook URLs are not
accidentally committed. `job_runner.sh` automatically loads that file when it
exists before submitting. It then uses `sbatch --export=ALL,...`, so
notification variables from that file are passed through to `job.sh` and then to
`run.py`.

On submission, `job_runner.sh` prints a non-secret status line. For Google Chat,
it should look like:

```text
Loaded run notification environment: /global/scratch/users/<user>/sources/PILATES/hpc/run-notifications.env
Run notifications: google_chat enabled=1 webhook=set; slack enabled=0 webhook=missing
```

Inside the Slurm log, PILATES should later print:

```text
PILATES run notifications enabled for: google_chat
```

If the Slurm log instead says both `PILATES_SLACK_NOTIFICATIONS` and
`PILATES_GCHAT_NOTIFICATIONS` are not enabled, the job did not receive the env
values. Check that `hpc/run-notifications.env` exists in the same checkout shown
by `PILATES_DIR`, and that the edited file says `PILATES_GCHAT_NOTIFICATIONS=1`
and has a non-empty webhook URL:

```bash
cd /global/scratch/users/$USER/sources/PILATES
source hpc/run-notifications.env
echo "gchat enabled=${PILATES_GCHAT_NOTIFICATIONS:-0}"
test -n "${PILATES_GCHAT_WEBHOOK_URL:-}" && echo "gchat webhook=set" || echo "gchat webhook=missing"
```

Treat webhook URLs as secrets. Do not commit them to `job.sh`, scenario YAML, or
other repo-tracked config. If a webhook URL is accidentally shared, ask the
space owner to delete and recreate that webhook.


You can also export variables directly in your current shell; the defaults in
`hpc/run-notifications.env` preserve already-exported values:

```bash
export PILATES_SLACK_NOTIFICATIONS=1
export PILATES_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

```bash
export PILATES_GCHAT_NOTIFICATIONS=1
export PILATES_GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/..."
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
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

## Run Publishing

PILATES also writes structured run event outputs for lightweight inspection.
This is separate from chat notifications. By default, every run archive gets:

```text
.pilates/run_events.jsonl
.pilates/run_summary.html
```

`run_events.jsonl` is the durable machine-readable event stream. The HTML file
is a small static table that can be opened directly in a browser or copied to a
shared location.

These local publishers are controlled by `hpc/run-notifications.env`:

```bash
export PILATES_RUN_EVENT_LOG=1
export PILATES_RUN_SUMMARY_HTML=1
```

### Optional Google Sheet

If a run coordinator provides a Google Sheet webhook URL, PILATES can also post
each scenario/step event as a row. This is intentionally webhook-based so HPC
jobs do not need Google OAuth credentials or a new Python dependency.

To create the webhook, copy
`hpc/google-sheet-run-publisher.apps-script.js.template` into a Google Sheet's
Apps Script project and deploy it as a web app. The deployed web app URL is the
value for `PILATES_GSHEET_WEBHOOK_URL`.

For Lawrencium jobs, deploy the web app with access that accepts unauthenticated
HTTP posts, typically `Anyone` / `Anyone with the link`, and use
`PILATES_GSHEET_SECRET` as the write guard. `Anyone in your organization/domain`
can still return HTTP 401 from HPC because the Slurm job is not signed into
Google. The Sheet itself can remain private because the script executes as the
deploying user.

Set these in `hpc/run-notifications.env`:

```bash
export PILATES_GSHEET_PUBLISH=1
export PILATES_GSHEET_WEBHOOK_URL="https://script.google.com/macros/s/.../exec"
export PILATES_GSHEET_SECRET="optional-shared-secret"
```

On submission, `job_runner.sh` prints a non-secret status line:

```text
Run publishing: archive_jsonl enabled=1; summary_html enabled=1; google_sheet enabled=1 webhook=set
```

The Sheet webhook receives a JSON payload with:

- `kind`: `pilates_run_event`
- `row`: a flat row suitable for appending to a sheet
- `event`: the full structured event record
- `secret`: optional shared secret when configured

Suggested Sheet columns for `row` are:

```text
event_time, event_type, run_kind, run_name, display_id, model, status, result,
scenario_id, year, iteration, stage, phase, submit_user, slurm_job_id,
slurm_job_name, slurm_partition, node, duration_seconds, output_count,
archive_run_dir, error
```

## Legacy Config Migration

If the passed settings file is in legacy format (does not contain `run:`, `shared:`, and `infrastructure:`), `job.sh` runs:

```bash
python3 scripts/migrate_config.py <old> <old>_migrated.yaml --no-validate
```

and uses the migrated file if migration succeeds.
