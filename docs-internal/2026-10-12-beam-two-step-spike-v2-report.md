# BEAM Two-Step Spike v2: Final Report

## 1. Net Line Delta
- `pilates/workflows/beam_checkpoint.py`: -738 lines (fully removed).
- `tests/test_beam_checkpoint.py`: -798 lines (fully removed).
- Checkpoint caller cleanup across `launcher.py`, `supply_demand.py`, `supply_demand_beam.py`, `steps/beam.py`, `test_restart_stage_boundary_matrix.py`, `test_run_bootstrap_phase.py`, `test_step_inputs_e2e.py`: -1,109 lines.
- Native restart, mutation refusal, and atomic staging logic added: +178 lines.
- **Net Delta**: -2,467 lines across existing code.
- New scenario suite (`tests/test_beam_two_step_consist_scenarios.py`): +597 lines.

## 2. Scenario Results Table
All 7 scenarios exercise real stage code (`_run_beam_steps`) with real `@define_step` definitions; only `BeamRunner.run_container` is stubbed (at most two monkeypatches per test).

| # | Scenario | Result | Evidence |
|---|---|---|---|
| 1 | Crash during `beam_run` | PASS | `test_scenario_1_crash_during_beam_run`: Container raised; rerun succeeded without consuming corrupted state. |
| 2 | Crash after `beam_run` commits | PASS | `test_scenario_2_crash_after_beam_run_commits_cache_hits_and_runs_postprocess`: Container ran once; restart cache-hit `beam_run` and executed postprocess. |
| 3 | Crash during postprocess | PASS | `test_scenario_3_crash_partway_through_postprocess_refuses_restart`: Interrupted postprocess marker detected; restart refused with `RuntimeError`. |
| 4 | Scratch wiped after `beam_run` | PASS | `test_scenario_4_scratch_wiped_after_beam_run_committed_exposes_hydration_gap`: Scratch wiped; outputs restored from recovery root to declared paths. |
| 5 | BEAM config changed | PASS | `test_scenario_5_beam_config_changed_causes_cache_miss`: HOCON config hash miss triggered fresh container rerun. |
| 6 | Two committed runs (same identity) | PASS | `test_scenario_6_two_committed_runs_same_identity_exposes_recency_gap`: Two stochastic runs committed; `find_matching_run` selected Run 2 by timestamp recency. |
| 7 | Archive copy failure | PASS | `test_scenario_7_archive_copy_failure_raises`: Staging copy failure raised loudly (`OSError`); archive marked incomplete. |

## 3. Postprocess Write Audit & Restart Decision
| Operation | Target Location | Mechanism | Classification |
|---|---|---|---|
| `_merge_beam_skims_to_zarr` | `workspace/activitysim/output/cache/skims.zarr` | `xarray.Dataset.to_zarr(mode='a'/'w')` | **(b) In-place change to existing model data** |
| `split_events_parquet_by_type` | `workspace/beam/output/.../ITERS/it.0/*.parquet` | `pandas.DataFrame.to_parquet` | (a) Declared output |
| `write_zarr_skim_as_omx_new` | `workspace/beam/local_mutable_data/<region>/skims.omx` | `omx.open_file('w')` | (a) Declared output |
| `_materialize_native_outputs` | `year-{y}-iteration-{i}/.pilates-consist-outputs/...` | `shutil.copy2` / `copytree` atomic temp swap | (a) Declared output |

**Decision**: Because `skims.zarr` mutates existing ActivitySim data in-place without snapshot isolation, an interrupted postprocess leaves partially merged skims. The restart path **refuses execution** upon detecting `.beam_postprocess_in_progress` in the regional output directory or archive recovery root. Auto-resume is prohibited.

## 4. Scenario 6 Findings: Stochastic BEAM & Recency Gap
- BEAM is stochastic: multiple runs under identical input artifacts, config, and code produce divergent output bytes.
- Consist's `find_matching_run` resolves candidates via `RunOrdering.recent_run_order_by()`, sorting strictly by `epoch_us(created_at) DESC, id DESC`.
- **Gap Exposed**: A cache hit unconditionally returns the newest committed run. If an earlier simulation phase (or downstream consumer) already read outputs from Run 1, a subsequent cache lookup will switch to Run 2, breaking lineage consistency and reproducibility across coupled stages.

## 5. Missing Capabilities & Minimal API Sketches
1. **[Consist] Declared-Destination Output Hydration**
   - *Problem*: `cache_hydration="outputs-all"` stores files under `.pilates-consist-outputs/`, but external tools expect files in tool-specific folder hierarchies (`ITERS/it.0/0.events.parquet`).
   - *API Sketch*: `CacheOptions(hydrate_declared_destinations=True)` or `tracker.hydrate_outputs_to_declared_paths(run_id)`.
   - *Value*: Generic; benefits any containerized or third-party workflow tool relying on fixed filesystem layouts.
2. **[Consist] Lineage-Consistent Run Pinning**
   - *Problem*: No mechanism to pin a specific cached run when multiple runs match an identical Merkle identity.
   - *API Sketch*: `tracker.find_matching_run(..., pin_run_id=...)` or `tracker.scenario(..., pinned_runs={model: run_id})`.
   - *Value*: Generic; essential for stochastic simulation pipelines where Merkle identity alone cannot distinguish alternative valid realizations.
3. **[PILATES] Isolated Skim Staging**
   - *Problem*: Direct in-place mutation of `skims.zarr` forces the postprocess refusal gate.
   - *API Sketch*: Stage skim updates to a temporary zarr group and atomically swap pointers only upon successful completion.

## 6. Recommendation: Keep a Thin Wrapper (~120 Lines)
- **Do not abandon**: Fully deleting `beam_checkpoint.py` (739 lines) succeeded, eliminating brittle state deserialization and custom checkpoint caches.
- **Do not delete entirely yet**: Without Consist-native declared-destination hydration and postprocess skim isolation, PILATES requires a minimal ~120-line orchestration layer in `pilates/workflows/stages/supply_demand_beam.py` to:
  1. Guard against in-place corruption via the `.beam_postprocess_in_progress` refusal gate.
  2. Execute verified post-run archival via `archive_run_output_files(..., verify=True)`.
  3. Re-hydrate nested iteration outputs from recovery roots when local scratch is wiped.
