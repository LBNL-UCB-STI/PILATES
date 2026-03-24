# Data Bootstrap

This page explains the external data and repo layout PILATES expects before a
first real run.

For most new users, this is the missing piece between environment setup and
actually launching `python run.py`.

## What "Bootstrap Data" Means Here

Bootstrap data includes the region-specific files that are not created by the
workflow itself, such as:

- ActivitySim config trees
- BEAM production input trees
- UrbanSim base data
- ATLAS static inputs when that model is enabled

These inputs are usually staged into the workspace during initialization or
referenced by preprocess steps.

## Expected Repo Layout

The current scenario configs generally assume:

- ActivitySim configs under:
  `pilates/activitysim/configs/<region>/`
- BEAM production data under:
  `pilates/beam/production/<region>/`
- UrbanSim base data under:
  `pilates/urbansim/data/`

On some developer machines these paths are real checkouts. On others they are
symlinks into separate data repos. Both are fine as long as the paths resolve.

## Region-Specific Inputs

### Seattle

Typical paths:

- `pilates/activitysim/configs/seattle`
- `pilates/beam/production/seattle`
- Seattle UrbanSim H5 in `pilates/urbansim/data/`

### SF Bay

Typical paths:

- `pilates/activitysim/configs/sfbay`
- `pilates/beam/production/sfbay`
- SF Bay UrbanSim H5 in `pilates/urbansim/data/`

## Where These Inputs Usually Come From

The repo already contains a Lawrencium-oriented setup guide:

- [lawrencium-setup.md](/Users/zaneedell/git/PILATES/lawrencium-setup.md)

That file currently includes concrete commands for:

- downloading UrbanSim base data
- cloning ActivitySim config repos
- cloning BEAM data repos

Even if you are not on Lawrencium, it is still the best in-repo source of truth
for which companion repos and datasets PILATES expects.

## Quick Validation Before First Run

For a Seattle local scenario, this is a reasonable minimum check:

```bash
python - <<'PY'
from pathlib import Path

paths = [
    "scenarios/seattle/settings-seattle-newconfig-local.yaml",
    "pilates/activitysim/configs/seattle",
    "pilates/beam/production/seattle",
]
missing = [p for p in paths if not Path(p).exists()]
print("missing:", missing)
PY
```

If the ActivitySim or BEAM paths are missing, the run will fail early.

## What Initialization Does And Does Not Do

PILATES bootstrap and initialization can:

- copy or stage required inputs into the run workspace
- seed coupler keys for bootstrap-safe artifacts
- reuse cached bootstrap outputs when configured

Initialization does not solve missing source data on disk. If the upstream
region-specific repos are absent, bootstrap will still fail.

## Common Patterns

### Separate data repos checked out in place

You clone the companion repos directly under the expected paths, for example:

- `pilates/activitysim/configs/seattle`
- `pilates/beam/production/seattle`

### Symlinks into external checkouts

You keep large data repos elsewhere and symlink them into the PILATES tree.

This is common on shared systems and is fine as long as:

- the symlink targets exist
- the runtime user can read them
- the scenario config expects the same layout

## ATLAS Notes

If ATLAS is enabled, it also depends on static inputs that end up in the
mutable ATLAS input directory. Some are copied during initialization and some
are later consumed by the ATLAS preprocess/run/postprocess steps.

If you are debugging ATLAS startup failures, check:

- configured scenario and year-specific ATLAS inputs
- contents of the ATLAS mutable input directory
- `docs/lineage_map.md` for the current high-level ATLAS input/output flow

## Signs Your Bootstrap Inputs Are Wrong

Typical early failures include:

- missing `pilates/activitysim/configs/<region>`
- missing `pilates/beam/production/<region>`
- missing UrbanSim base H5
- file-not-found errors during preprocess
- path-resolution failures when configs contain developer-specific absolute paths

For troubleshooting patterns, see `docs/troubleshooting.md`.

## Related Docs

- `docs/getting_started.md`
- `docs/configuration_reference.md`
- `docs/workflow_primer.md`
- `docs/troubleshooting.md`
- [lawrencium-setup.md](/Users/zaneedell/git/PILATES/lawrencium-setup.md)
