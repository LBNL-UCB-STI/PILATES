# Supply-demand manifest deletion follow-up

## Context

Task 0 resolved the recovery spike with the `compatibility-backend-track`
verdict. That means the supply-demand manifest path is still part of the live
compatibility surface and is not yet ready for delete-track treatment.

## Temporary guard

`workflow.manifests.disabled_stages` must not be used to disable:

- `supply_demand`
- `activity_demand`
- `activity_demand_preprocess`
- `activity_demand_run`
- `activity_demand_postprocess`
- `activity_demand_compile`

The code now fails fast if one of those stage names is disabled. The guard is
temporary and should stay in place until the tracker-native recovery path is
ready to replace the YAML-backed fallback cleanly.

## Deletion prerequisites

Before this guard can be removed, the tracker-native recovery path needs to be
in place for the supply-demand loop and its ActivitySim sub-stages, with the
delete-track workflow validated end to end. The minimum prerequisites are:

- Tracker-native discovery of supply-demand runs by forecast year and iteration,
  without using `year_*_iteration_*.yaml` filenames as the epoch index.
- Tracker-native recovery of the ActivitySim and BEAM run IDs currently read
  from YAML manifests for resume parent seeding.
- Tracker-native or coupler-native reconstruction of the ActivitySim
  postprocess handoff mapping currently recovered from manifest `outputs`.
- Regression coverage for manifest-free restart/resume through at least the
  ActivitySim postprocess and BEAM boundary cases covered by the current
  manifest-backed tests.

At that point, the manifest-backed fallback can be retired without losing
restart behavior or stage-boundary coverage.

## Why this is temporary

The current implementation still depends on YAML-backed supply-demand recovery
for compatibility. Disabling it now would erase the fallback before the
replacement path is fully available, so the guard exists only to prevent an
early deletion step.
