# Manifest Recovery Refactor Progress

## Current Phase

land-use-optout

## Task Status

| Task | Status | Latest SHA | Test Evidence | Review Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 0. Tracker-native recovery spike | done | af1a01a20cc7608b0e01f7196a47d684f292488a | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_stage_contracts.py::test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids tests/test_restart_replay_archive_sources.py -q` -> 3 passed | spec: passed; quality: approved; minor citation fixed | Verdict: compatibility-backend-track. |
| 1. Characterization tests | done | dc19a53140c2ddfb1e277ba7849140e51fd8fda9 | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py -q` -> 62 passed, 1 warning | spec: passed; quality: passed | Added missing no-output rerun characterization; existing tests cover stale pruning, manifest restore/coupler replay, and restored run-id seeding. |
| 2. Recovery store module | done | be7e43979f2dc820700d3f212ff9e335d29202eb | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_recovery_store.py tests/test_step_manifest_archive.py -q` -> 5 passed; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py -q` -> 62 passed, 1 warning; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/recovery.py` -> all checks passed | spec: passed; quality: passed after fixes | Added recovery store boundary module and no-op/YAML store tests. |
| 3. Shared executor loop | done | 38bd93c4d63c209c1e00dff5c47deca2e91124b9 | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py tests/test_recovery_store.py -q` -> 68 passed, 1 warning; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/orchestration.py pilates/workflows/recovery.py` -> all checks passed | spec: passed; quality: passed after fix | Collapsed manifest/native loops into shared recovery-store executor; `run_manifested_steps(...)` is now a wrapper. |
| 4. Stage-level recovery policy | done | 8c77efdc1c8e620983b84be4055ee758a2636e43 | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_recovery_store.py tests/test_architecture_guardrails.py tests/test_workflow_invariants.py::test_run_workflow_disabled_manifest_stage_uses_noop_recovery_store -q` -> 17 passed; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/recovery.py pilates/workflows/orchestration.py pilates/config/models.py` -> all checks passed; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_scenario_configs.py -q` -> 4 passed, 3 skipped; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py tests/test_recovery_store.py -q` -> 71 passed, 1 warning | spec: passed after fix; quality: passed | Added `workflow.manifests.disabled_stages`, wired recovery-store policy helper into the executor, and added direct `ManifestConfig` import guardrail. |
| 5. Postprocessing opt-out | done | 1cb369cb94b5ec4b9bdce571957770db6ff3e09c | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_architecture_guardrails.py tests/test_stage_contracts.py tests/test_postprocessing_manifest_persistence.py -q` -> 81 passed, 3 warnings; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/orchestration.py pilates/workflows/stages/postprocessing.py` -> all checks passed | spec: passed; quality: passed | Postprocessing uses policy-built manifest config; disabled stage runs without writing manifest YAML. |
| 6. Land-use opt-out | review | pending | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_architecture_guardrails.py tests/test_stage_contracts.py tests/test_land_use_manifest_persistence.py -q` -> 81 passed, 3 warnings; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check pilates/workflows/stages/land_use.py` -> all checks passed; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_architecture_guardrails.py tests/test_postprocessing_manifest_persistence.py tests/test_land_use_manifest_persistence.py tests/test_stage_contracts.py -q` -> 83 passed, 3 warnings; `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m ty check tests/test_land_use_manifest_persistence.py` -> all checks passed | spec: finding fixed, re-review pending; quality: passed | Land use uses policy-built manifest config; disabled path runs all UrbanSim steps, writes no manifest YAML, and preserves handoff mapping from sibling baseline couplers. |
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
| Task 3 spec review | fixed | No actionable findings. | No patch needed. |
| Task 3 quality review | fixed | Pruning stale/downstream manifest entries did not clear outputs hydrated into `outputs_holder` during stale detection. | Clear holder attributes for all expanded stale entries, keep local manifest snapshot pruned, and added cache-hit regression coverage. |
| Task 3 quality re-review | fixed | No actionable findings. | P1 fix accepted. |
| Task 4 spec review | fixed | `workflow.manifests.disabled_stages` was not wired into the live executor path. | `run_workflow(...)` and `run_manifested_steps(...)` now select stores through `recovery_store_for_stage(...)`; added disabled-stage runtime coverage. |
| Task 4 quality review | fixed | No actionable findings before policy wiring fix. | Re-review requested after executor wiring. |
| Task 4 spec re-review | fixed | No actionable findings. | No patch needed. |
| Task 4 quality re-review | fixed | No actionable findings. | No patch needed. |
| Task 5 spec review | fixed | No actionable findings. | No patch needed. |
| Task 5 quality review | fixed | No actionable findings. | No patch needed. |
| Task 6 spec review | fixed | Handoff equivalence test cloned the disabled-path coupler after the default run mutated it. | Capture baseline coupler before either branch and run default/disabled paths from sibling baseline copies. |
| Task 6 quality review | fixed | No actionable findings. | No patch needed. |
