# Tracker-Native Supply-Demand Recovery Spike

## Manifest Inputs To Replace

These are the values currently read from `year_*_iteration_*.yaml` manifests in the supply-demand resume path:

- `year` from the manifest filename pattern `year_<year>_iteration_<iteration>.yaml`
  - Parsed by `_SUPPLY_DEMAND_MANIFEST_NAME` in `pilates/workflows/stages/supply_demand_resume.py`.
  - Used as the fallback epoch when seeding resume state and when matching tracker runs.
- `iteration` from the same manifest filename pattern
  - Parsed alongside `year` from the filename.
  - Used as the fallback iteration when seeding resume state and when matching tracker runs.
- `activitysim_run.run_id`
  - Read by `_manifest_activitysim_run_id(...)`.
  - Used to load tracker outputs with `load_tracker_run_outputs(...)` and to seed `Scenario.remember_restored_run_id(...)`.
- `beam_run.run_id`
  - Read by `_manifest_step_run_id(..., "beam_run")`.
  - Used during `seed_supply_demand_parent_run_ids_for_resume(...)` to seed BEAM parent links.
- `activitysim_postprocess.outputs.processed_outputs`
  - Read by `_manifest_handoff_mapping(...)`.
  - Used to reconstruct the BEAM-facing ActivitySim output mapping, including promotion of `asim_input_skims_zarr_archived` to `zarr_skims`.

## Existing Non-Manifest Recovery Surfaces

- `_find_matching_run_for_resume_target(...)`
  - Calls `tracker.find_matching_run(**target)` and is the main tracker-native discovery path when a year/iteration target is already known.
  - It still needs the target epoch from somewhere else.
- `_find_tracker_run_by_id(...)`
  - Scans `tracker.run_set(...)` and returns the run whose `id` matches exactly.
  - This is the exact-ID recovery path used after a manifest supplies a run_id.
- `_run_id_epoch(...)`
  - Extracts `year` and `iteration` from a run_id pattern like `__y<year>__i<iteration>__`.
  - It is only a fallback parser; it does not discover the run on its own.
- `load_tracker_run_outputs(...)`
  - Calls `tracker.get_run_outputs(run_id)` and canonicalizes the output keys.
  - This is the tracker-backed data hydration path for exact run_ids.
- `_supplement_from_coupler(...)`
  - Fills missing BEAM-facing ActivitySim outputs from the current coupler when the values already resolve to real workspace paths.
  - This is a completion step, not a discovery step.
- Final coupler-only branch in `_restore_activity_demand_outputs_for_resume(...)`
  - If manifest restore is unavailable, the function can still recover the BEAM-facing ActivitySim outputs directly from the hydrated coupler.
  - This is the last fallback, and it only works when all required outputs are already present in the coupler.

## Verdict Matrix

| Requirement | Current non-YAML source | Proven by test | Gap |
| --- | --- | --- | --- |
| enumerate ActivitySim run IDs by year/iteration | `seed_supply_demand_parent_run_ids_for_resume(...)` can use tracker exact-ID lookup after it has year/iteration from the resume target and `_run_id_epoch(...)` fallback | `tests/test_stage_contracts.py::test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids` | Still depends on `year_*_iteration_*.yaml` filenames to supply the epoch; no tracker-native year/iteration index is present. |
| enumerate BEAM run IDs by year/iteration | `seed_supply_demand_parent_run_ids_for_resume(...)` plus `_find_matching_run_for_resume_target(...)` and `_run_id_epoch(...)` | `tests/test_restart_stage_boundary_matrix.py::test_beam_restart_recovery_readiness_diagnostic_reports_matchable_run` | Tracker matching works once the epoch is known, but the epoch still comes from the manifest filename path. |
| load ActivitySim postprocess handoff outputs | `load_tracker_run_outputs(...)` can load canonicalized outputs after an exact run ID is known; `_supplement_from_coupler(...)` can fill missing required keys from already-hydrated coupler values | `tests/test_restart_replay_archive_sources.py::test_restart_restores_activitysim_handoff_from_scratch_archive_before_nfs_promotion` | Exact run_id is still required before tracker outputs can be loaded, and manifest output data is still the only path that explicitly rebuilds the processed-output mapping when typed restored outputs are incomplete. |
| materialize BEAM-facing ActivitySim paths | `_supplement_from_coupler(...)` and the final coupler-only branch in `_restore_activity_demand_outputs_for_resume(...)` | `tests/test_restart_stage_boundary_matrix.py::test_restart_traffic_assignment_boundary_restores_activitysim_outputs` and `tests/test_restart_stage_boundary_matrix.py::test_restart_mid_iteration_traffic_assignment_preserves_promoted_warmstart` | This only completes already-hydrated outputs; it does not replace manifest-driven discovery. |
| seed parent run IDs without manifest filenames | none today | not covered | No manifest-free tracker cursor or index exists for seeding parent run IDs by year/iteration. |

## Verdict

Blocker: tracker data can already hydrate exact run IDs and complete restored ActivitySim outputs, but it does not yet provide a manifest-free way to discover or seed supply-demand runs by year/iteration. The `year_*_iteration_*.yaml` filename still carries the epoch needed to enumerate resume candidates and seed parent run IDs.

compatibility-backend-track
