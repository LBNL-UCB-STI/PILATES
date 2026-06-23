# Manifest Recovery Refactor Progress

## Current Phase

characterization

## Task Status

| Task | Status | Latest SHA | Test Evidence | Review Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 0. Tracker-native recovery spike | done | 8132551609dbed380fbcac39d350f19653691da0 | `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest tests/test_stage_contracts.py::test_seed_supply_demand_parent_run_ids_for_resume_replays_manifest_run_ids tests/test_restart_replay_archive_sources.py -q` -> 3 passed | spec: passed; quality: approved; minor citation fixed | Verdict: compatibility-backend-track. |
| 1. Characterization tests | in-progress | pending | pending | pending | Next task. |
| 2. Recovery store module | pending | pending | pending | pending | Not started. |
| 3. Shared executor loop | pending | pending | pending | pending | Requires gpt-5.5 worker/review. |
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
