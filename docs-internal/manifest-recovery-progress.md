# Manifest Recovery Refactor Progress

## Current Phase

shared-executor

## Task Status

| Task | Status | Latest SHA | Test Evidence | Review Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 0. Tracker-native recovery spike | done | af1a01a20cc7608b0e01f7196a47d684f292488a | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_stage_contracts.py::test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids tests/test_restart_replay_archive_sources.py -q` -> 3 passed | spec: passed; quality: approved; minor citation fixed | Verdict: compatibility-backend-track. |
| 1. Characterization tests | done | dc19a53140c2ddfb1e277ba7849140e51fd8fda9 | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py -q` -> 62 passed, 1 warning | spec: passed; quality: passed | Added missing no-output rerun characterization; existing tests cover stale pruning, manifest restore/coupler replay, and restored run-id seeding. |
| 2. Recovery store module | done | pending | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_recovery_store.py tests/test_step_manifest_archive.py -q` -> 5 passed; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py -q` -> 62 passed, 1 warning; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/recovery.py` -> all checks passed | spec: passed; quality: passed after fixes | Added recovery store boundary module and no-op/YAML store tests. |
| 3. Shared executor loop | in-progress | pending | pending | pending | Requires gpt-5.5 worker/review. |
| 4. Stage-level recovery policy | pending | pending | pending | pending | Not started. |
| 5. Postprocessing opt-out | pending | pending | pending | pending | Not started. |
| 6. Land-use opt-out | pending | pending | pending | pending | Not started. |
| 7. Supply-demand guard and follow-up | pending | pending | pending | pending | Not started. |
| 8. Retire direct ManifestConfig usage in migrated stages | pending | pending | pending | pending | Not started. |

## Open Risks And Blockers

- Supply-demand may still need YAML manifests as the epoch/run-id discovery index.
- ActivitySim postprocess handoff restoration may still need manifest output data.
- The branch has unrelated untracked local files; keep implementation scope narrow.
- Task 0 confirmed compatibility-backend-track because current tracker/coupler APIs do not provide a manifest-free year/iteration index for supply-demand resume seeding.

## Review Findings

| Source | Status | Finding | Resolution |
| --- | --- | --- | --- |
| Task 0 spec review | fixed | Matrix row mixed `_manifest_handoff_mapping(...)` into non-YAML source column. | Removed manifest-backed helper from non-YAML source column and re-reviewed clean. |
| Task 0 quality review | fixed | ActivitySim run-id enumeration citation was loose. | Replaced with `tests/test_stage_contracts.py::test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids`. |
| Task 1 spec review | fixed | No actionable findings. | Existing coverage map accepted; no patch needed. |
| Task 1 quality review | fixed | No actionable findings. | No patch needed. |
| Task 2 spec review | fixed | No actionable findings. | No patch needed. |
| Task 2 quality review | fixed | `StepRecoveryStore.load()` overpromised mapping-shaped manifest entries before validating raw YAML values. | Changed load/cache types to `Mapping[str, Any]`, validate entry mapping before restore decisions, and added malformed-entry coverage. |
| Task 2 quality re-review | fixed | `_manifest` private-field annotation still implied mapping-shaped entries. | Changed `_manifest` annotation to `dict[str, Any]`; focused tests and `ty` pass. |
