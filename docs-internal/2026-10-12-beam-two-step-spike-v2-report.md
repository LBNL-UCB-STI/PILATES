# BEAM Two-Step Spike v2: Final Report

## 1. Net Line Delta
- `pilates/workflows/beam_checkpoint.py`: -738 lines (fully deleted).
- `tests/test_beam_checkpoint.py`: -798 lines (fully deleted).
- `pilates/workflows/stages/supply_demand_beam.py`: -450 net lines (-614 deleted, +164 added).
- `tests/test_restart_stage_boundary_matrix.py`: -369 net lines (-544 deleted, +175 added).
- Checkpoint caller cleanups across `launcher.py`, `supply_demand.py`, `steps/beam.py`, `test_run_bootstrap_phase.py`, `test_step_inputs_e2e.py`: -197 net lines.
- **Net Across Existing Code**: -2,552 lines.
- New scenario test suite (`tests/test_beam_two_step_consist_scenarios.py`): +596 lines.
- Overall net reduction across the repository: **-1,814 lines**.

## 2. Scenario Results Table
All 7 scenarios exercise real stage code (`_run_beam_steps`) with real `@define_step` definitions; only `BeamRunner.run_container` and canonical fault-injection points are stubbed (at most two monkeypatches per test).

| # | Scenario | Result | Evidence |
|---|---|---|---|
| 1 | Crash during `beam_run` | PASS | `test_scenario_1_crash_during_beam_run`: Container raised; rerun succeeded without consuming corrupted state. |
| 2 | Crash after `beam_run` commits | PASS | `test_scenario_2_crash_after_beam_run_commits_cache_hits_and_runs_postprocess`: Container ran once; restart cache-hit `beam_run` and executed postprocess. |
| 3 | Crash during postprocess | PASS | `test_scenario_3_crash_partway_through_postprocess_refuses_restart`: Interrupted postprocess marker detected; restart refused with `RuntimeError`. |
| 4 | Scratch wiped after `beam_run` | FAIL (Gap) | `test_scenario_4_scratch_wiped_after_beam_run_committed_exposes_hydration_gap`: Scratch wiped; restart cache-hit `beam_run` but failed with missing output at declared destination. Consist `cache_hydration="outputs-all"` materializes to tracker run_dir, not declared external paths. |
| 5 | BEAM config changed | PASS | `test_scenario_5_beam_config_changed_causes_cache_miss`: HOCON config hash miss triggered fresh container rerun. |
| 6 | Two committed runs (same identity) | GAP EXPOSED | `test_scenario_6_two_committed_runs_same_identity_exposes_recency_gap`: Two stochastic runs committed; `find_matching_run` selected Run 2 by timestamp recency (`epoch_us` DESC), diverging from previously consumed run. |
| 7 | Archive copy failure | PASS | `test_scenario_7_archive_copy_failure_raises`: Archive copy failure raised loudly (`RuntimeError`); archive not treated as present. |

## 3. Postprocess Write Audit & Restart Decision
| Operation | Target Location | Mechanism | Classification |
|---|---|---|---|
| `_merge_beam_skims_to_zarr` | `workspace/activitysim/output/cache/skims.zarr` | `xarray.Dataset.to_zarr(mode='a')` | **(b) In-place change to existing model data** |
| `split_events_parquet_by_type` | `workspace/beam/output/.../ITERS/it.0/*.parquet` | `pandas.DataFrame.to_parquet` | (a) Declared output |
| `write_zarr_skim_as_omx_new` | `workspace/beam/local_mutable_data/<region>/skims.omx` | `omx.open_file('w')` | (a) Declared output |
| `_materialize_native_outputs` | `.pilates-consist-outputs/beam_run/...` | `shutil.copy2` / `copytree` atomic temp swap | (a) Declared output |

**Decision**: Because `skims.zarr` mutates existing ActivitySim data in-place without snapshot isolation, an interrupted postprocess leaves partially merged skims. The restart path **refuses execution** upon detecting `.beam_postprocess_in_progress` in the regional output directory or archive recovery root. Auto-resume is prohibited.

## 4. Scenario 6 Findings: Stochastic BEAM & Recency Gap
- BEAM is stochastic: multiple runs under identical input artifacts, config, and code produce divergent output bytes.
- Consist's `find_matching_run` resolves candidates via `RunOrdering.recent_run_order_by()`, sorting strictly by `epoch_us(created_at) DESC, id DESC`.
- **Gap Exposed**: A cache hit unconditionally returns the newest committed run. If an earlier simulation phase (or downstream consumer) already read outputs from Run 1, a subsequent cache lookup switches to Run 2, breaking lineage consistency and reproducibility across coupled stages.

## 5. Missing Capabilities & Minimal API Sketches
1. **[Consist] Declared-Destination Output Hydration** (Scenario 4)
   - *Problem*: `cache_hydration="outputs-all"` stages outputs inside `tracker.run_dir / "outputs" / <preview>`, but external CLI tools require files at declared workspace paths (`ITERS/it.0/0.events.parquet`).
   - *API Sketch*: `CacheOptions(hydrate_declared_destinations=True)` or `@define_step(cache_hydration="declared-destinations")` to call `hydrate_run_outputs_to_destinations`.
   - *Value*: Generic; benefits any containerized or legacy workflow tool relying on fixed filesystem hierarchies.
2. **[Consist] Lineage-Consistent Run Pinning** (Scenario 6)
   - *Problem*: No mechanism to pin a specific cached run when multiple runs match an identical Merkle identity.
   - *API Sketch*: `tracker.find_matching_run(..., pin_run_id=...)` or `tracker.scenario(..., pinned_runs={model: run_id})`.
   - *Value*: Generic; essential for stochastic simulation pipelines where Merkle identity alone cannot distinguish alternative valid realizations.
3. **[PILATES] Isolated Skim Staging** (Scenario 3 audit)
   - *Problem*: Direct in-place mutation of `skims.zarr` forces the postprocess refusal gate.
   - *API Sketch*: Stage skim updates to a temporary zarr group and atomically swap pointers only upon successful completion.
4. **[Consist] Pinned Producer Closures for Multi-Phase Steps**
   - *Problem*: Checkpoint closure tests in `test_restart_stage_boundary_matrix.py` verified frozen producer lineage for zarr outputs. Consist Coupler re-evaluates bindings dynamically.
   - *API Sketch*: `tracker.freeze_step_closure(step_name)` to lock input/output artifact bindings across restarts.
   - *Value*: Generic; preserves exact multi-step producer lineage under restarts.

## 6. Recommendation: Keep a Thin Wrapper (~120 Lines)
- **Do not abandon**: Deleting `beam_checkpoint.py` (738 lines) eliminated brittle JSON checkpoint deserialization and custom checkpoint caches.
- **Do not delete entirely yet**: Native Consist primitives alone cannot survive scratch wipes (Scenario 4) or ensure deterministic stochastic lineage (Scenario 6). PILATES requires a minimal ~120-line orchestration layer in `pilates/workflows/stages/supply_demand_beam.py` to:
  1. Enforce the `.beam_postprocess_in_progress` refusal gate against in-place skim corruption.
  2. Execute verified post-run archival via `archive_run_output_files(..., verify=True)`.
  3. Bridge declared-destination hydration from recovery roots when local scratch is wiped until Consist supports it natively.
