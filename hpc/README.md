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

Use `lr8`:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr8
```

Restart from an existing stage file:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -s current_stage_restart.yaml
```

Run the native structural canary with a reviewed seed manifest:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  -a <slurm_account> \
  --native-structural-canary hpc/canaries/seattle-structural-reviewed.json
```

Run the isolated BEAM-preprocess cold-to-fresh acceptance harness (this is
separate from, and must not be combined with, the structural canary):

```bash
CONSIST_SRC_DIR=/global/scratch/users/$USER/sources/consist \
./hpc/job_runner.sh \
  -c scenarios/sfbay/settings-sfbay-consist-beam-preprocess-hpc-2019-acceptance.yaml \
  -a <slurm_account> --high-mem \
  --beam-preprocess-acceptance hpc/beam-preprocess-acceptance-inputs.json
```

Before submission, copy `hpc/beam-preprocess-acceptance-inputs.json.template`
to the untracked `hpc/beam-preprocess-acceptance-inputs.json`. The two manifest
roots are distinct: `PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT` holds the
four already-selected population files, while
`PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT` is the staged SFBay BEAM
tree and must contain the configured primary file at
`scenarios/sfbay-pilates-base-calibrated.conf`. Its sibling `../common/` must
also hold BEAM's shared HOCON files; the harness stages both directories so the
scenario's relative includes remain valid. A minimal operator preflight is:

```bash
export PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT=/durable/path/to/population
export PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT=/durable/path/to/beam-input
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT/plans.parquet"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT/households.parquet"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT/persons.parquet"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_INPUT_ROOT/vehicles2_2019.csv"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT/scenarios/sfbay-pilates-base-calibrated.conf"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT/../common/akka.conf"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT/../common/metrics.conf"
test -f "$PILATES_BEAM_PREPROCESS_ACCEPTANCE_BEAM_INPUT_ROOT/../common/matsim.conf"
```

The manifest fixes the direct cohort at workflow year 2017, forecast year 2019,
and inner iteration 0. `vehicles2_2019.csv` is therefore an exact forecast-year
ATLAS input, not a workflow-start-year input. The harness
creates one evidence root under
`/global/scratch/users/$USER/pilates-boundary-promotions/<job-id>/` by default;
override that parent with `PILATES_BEAM_PREPROCESS_ACCEPTANCE_ROOT`.

It retains `generated-settings.yaml`, `submitted-input-manifest.json`,
`effective-input-manifest.json`, `body-executions.jsonl`,
`persisted-runs/{cold,fresh}.json`, `phases/{cold,fresh}.json`,
`semantic-validation.json`, the shared `provenance.duckdb`, and
`consist-runs/`. The phase records identify the distinct requested Run IDs and
the fresh Run's actual persisted cache source; they also retain ordinary
action-v2 input binding/identity records, selectors, linked artifacts,
canonical BEAM adapter/config identity, requested staging paths, hydrated
output paths, and semantic file/directory details. `beam_preprocess` uses an
ordinary `BindingResult`, so this bundle's authority is the persisted Run,
full Run snapshot, and linked artifacts rather than a strict-binding invocation
record. It fails before the fresh phase when cold is a hit, and fails when
fresh misses, declared outputs are absent, the callable executes twice, the
fresh source is not the cold requested Run, hydration destinations differ, or
the non-workspace persisted identity/semantic-product comparison differs.

## UrbanSim HDF5 Snapshot-Reconciliation Acceptance

Use this narrow, pre-merge acceptance only to establish merge-confidence
evidence for an editable Consist checkout. It exercises the native
`urbansim_postprocess` boundary with one retained SFBay HDF5 input, checkpoints
the Tracker, and reconciles the completed run in a separate Python process.
It is not a cache hit, does not execute a new cacheable native run during
reconciliation, and does not promote the HDF5 artifact or make any HDF5
consumer boundary eligible for reuse.

Before submitting, point the manifest at the retained source artifact on HPC:

```fish
set -x PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5 /global/scratch/users/zaneedell/pilates-outputs/pilates-run--sfbay--consist-sfbay-usim-base-short-canary--20260826-104715/consist-recovery/pilates-run--sfbay--consist-sfbay-usim-base-short-canary--20260826-104715_atlas_postprocess__y2019__i0__phase_postprocess_b28c6e38/urbansim/data/model_data_2019_population_source.h5
test -f "$PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5"
```

The acceptance driver, not the source filename, enforces workflow year 2017,
forecast year 2019, and iteration 0. Copy the tracked template to the
operator-owned, untracked manifest only after the preflight succeeds, then
submit with the editable Consist checkout and the ordinary node shape:

```fish
set -x CONSIST_SRC_DIR /global/scratch/users/zaneedell/sources/consist
envsubst < hpc/urbansim-h5-snapshot-acceptance-inputs.json.template > hpc/urbansim-h5-snapshot-acceptance-inputs.json
./hpc/job_runner.sh \
  -c scenarios/sfbay/settings-sfbay-consist-usim-hpc-2019-canary.yaml \
  -a ac_beamcore \
  --urbansim-h5-snapshot-acceptance hpc/urbansim-h5-snapshot-acceptance-inputs.json
```

Do not add `--high-mem` or `-s`. Record the submitted Slurm job ID and the
evidence root printed by the wrapper. The evidence bundle contains distinct
`capture.json` and `reconciliation.json` records, plus a checkpointed Tracker
snapshot and `validation.json`: capture records the strict native input binding
before the later unowned override observation; reconciliation records a
separate read-only process reopening that snapshot and comparing the persisted
identity. Treat the run as successful only when Slurm is `COMPLETED` and every
named check in `validation.json` is true, including `valid`.

This mode fails before either driver starts unless `CONSIST_SRC_DIR` was
explicitly supplied, is a Git checkout, installs editable, and the resulting
`consist` import resolves inside that checkout. It never falls back to another
installed or PyPI Consist. `runtime-environment.json` records the exact PILATES
and editable-Consist revisions, import path, and Python runtime without copying
environment variables or secrets. `effective-input-manifest.json` preserves the
validated cohort, expanded/resolved HDF5 descriptor, and trusted artifact ID
and identity without modifying the submitted manifest. The capture and
reconciliation records include their distinct process IDs; the reconciliation
record independently derives action-v2 strict-binding validity and confirms
that the persisted strict link is the trusted pre-override artifact.

After a successful job, copy the evidence root to the established NFS
`pilates-boundary-promotions` archive, generate SHA-256 manifests for both the
source and destination copies, and compare them. Preserve the PILATES and
editable-Consist revisions from the evidence record. This verifies retention
of the evidence bundle; it is neither a hash of nor a replacement for the
retained external source HDF5. Append only this verified, checksum-backed
evidence to the boundary-promotion plan. A later released-Consist replay and
separate boundary-specific cache proof remain required before any HDF5
promotion decision.

The reviewed Seattle seed is based on the first-iteration launch evidence from
the failed capture run. Its host-local `/local/job...` prefixes are normalized
by the checker, so a new Slurm allocation does not need to reproduce the old
job ID or output directory. The new canary must retain its own launch log,
Consist snapshot, generated settings, and action-v2 census; those artifacts are
evidence for that new run and are not expected to match the failed run's log
byte-for-byte.

For the initial capture-only run, create the empty seed locally before
submitting the job:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m \
  pilates.runtime.native_canary \
  --init-capture hpc/canaries/seattle-structural-capture.json
```

Copy or commit that file into the HPC checkout, then pass its path to
`--native-structural-canary`. This first run collects actual launch
observations; it is not a passing structural comparison until those
observations have been independently reviewed and converted into
`expected_launches`.

This copies the seed manifest and generated settings into
`/global/scratch/users/$USER/pilates-canaries/<job-id>/`, then exports that
copy as `PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST` to the Slurm job. Set
`PILATES_NATIVE_STRUCTURAL_CANARY_ROOT` to choose a different evidence root.
The native step boundary refreshes `evidence/action-v2.jsonl` there as steps
complete. After the job, copy the final Consist snapshot and job log under that
same root, then run the checker described in
`docs-internal/2026-08-20-native-structural-canary-harness.md`.

Tracked scenario files are copy/edit starting points for the current cluster
storage posture and model wiring. They are not turnkey machine-independent
configs; review account-specific paths, data roots, output roots, and model
selections before submitting.

## BEAM Memory Templating

`job_runner.sh` supports settings files that contain `${BEAM_MEMORY}`.

- For `lr7 --high-mem`, default is `400g`.
- For default `lr7` (240G job memory), default is `180g`.
- For `lr8`, default is `600g`.

Override explicitly:

```bash
BEAM_MEMORY=450g ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr7
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
2. Otherwise, install from PyPI package name in `CONSIST_PYPI_PACKAGE` or fall back to `consist==0.3.3`.
3. Validate with `from consist import create_tracker`.

Override examples:

```bash
CONSIST_SRC_DIR=/global/scratch/users/$USER/sources/consist ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
```

```bash
CONSIST_PYPI_PACKAGE=consist==0.3.3 ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account>
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
