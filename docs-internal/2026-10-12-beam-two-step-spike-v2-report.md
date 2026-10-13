# BEAM Two-Step Spike v2: Final Report

## 1. Net Line Delta
- `pilates/workflows/beam_checkpoint.py`: -738 lines (deleted).
- `tests/test_beam_checkpoint.py`: -798 lines (deleted).
- `pilates/workflows/stages/supply_demand_beam.py`: -450 net lines (-614 deleted, +164 added).
- `tests/test_restart_stage_boundary_matrix.py`: -369 net lines.
- `pilates/workflows/steps/beam.py`: -32 lines (native `cache_hydration="outputs-requested"`).
- Other checkpoint caller cleanups: -197 net lines.
- New scenario tests (`tests/test_beam_two_step_consist_scenarios.py`): +596 lines.
- **Overall Net Reduction**: **-1,988 lines** of custom checkpoint and recovery code deleted.

## 2. Scenario Results Table
All 7 scenarios exercise real stage code (`_run_beam_steps`) with real `@define_step` definitions and $\le 2$ monkeypatches per test.

| # | Scenario | Result | Evidence |
|---|---|---|---|
| 1 | Crash during `beam_run` | PASS | Container raised; rerun succeeded without consuming corrupt state. |
| 2 | Crash after `beam_run` commits | PASS | Restart cache-hit `beam_run` and ran postprocess; container ran once. |
| 3 | Crash during postprocess | PASS | In-progress marker detected; refusal gate raised `RuntimeError` to prevent double-merge corruption. |
| 4 | Scratch wiped after `beam_run` | PASS | `cache_hydration="outputs-requested"` natively restored declared outputs from recovery roots. |
| 5 | BEAM config changed | PASS | HOCON config hash miss triggered fresh container rerun. |
| 6 | Two runs, same identity | PASS | Exposed Consist recency gap: `find_matching_run` picks newest run by timestamp (`epoch_us` DESC). |
| 7 | Archive copy failure | PASS | Archive copy failure raised loudly (`verify=True`). |

## 3. Postprocess Write Audit & Idempotency Analysis
**Status: Not Settled**. Postprocess is not yet idempotent because `_merge_beam_skims_to_zarr` modifies ActivitySim's `skims.zarr` in place.

### Why Swap-Based Staging Fails Idempotency
Attempting to make in-place mutation safe via temporary directory swaps introduced six severe failure modes:
1. **Post-Swap Double Merge**: A crash after swapping but before Consist commits leaves `skims.zarr` containing merged data. On restart, postprocess runs again and appends trips into the already-merged store (`mode="a"`). Array shapes/dtypes match, hiding the corruption from file size checks.
2. **Mid-Loop Iteration Double Merge**: Postprocess loops over BEAM iterations. A crash between iterations 0 and 1 commits iteration 0's swap; restart merges iteration 0 a second time.
3. **Verification Blind Spots**: Detecting double merges requires comparing array values against an uninterrupted baseline; byte sizes and keys remain identical.
4. **Window with Missing Store**: Renaming live store $\rightarrow$ backup $\rightarrow$ staging leaves a window where no live store exists. A crash during this window leaves no recoverable store.
5. **Disk Space Accumulation**: Abandoned `.tmp_<pid>_<ts>` and `.old_<pid>_<ts>` copies pile up multi-GB directories on scratch without cleanup.
6. **Swallowed Errors**: Staging fallbacks return unmodified paths, masking merge failures as successes.

### The Architectural Fix: Separate Stores
True idempotency requires separating input and output artifacts:
- **Input**: Immutable pre-postprocess skims resolved via Consist coupler. Postprocess never writes to it.
- **Output**: A new store at a declared output path, merging all iterations in a single pass and published once.
- **Consumers**: ActivitySim and downstream models bind dynamically to the newly produced artifact.

**Decision**: Until postprocess reads and writes separate stores, the refusal gate and `.beam_postprocess_in_progress` marker are maintained. Auto-resume after an interrupted postprocess remains strictly prohibited.

## 4. Scenario 6 Analysis: Stochastic BEAM (Consist Backlog)
BEAM is stochastic; identical Merkle identity can yield divergent output bytes. Consist orders candidates newest first (`recent_run_order_by()`).

### Production Paths
1. **Forced Rerun / Overwrite**: Runs invoked with `CacheOptions(mode="overwrite")`.
2. **Retrying Downstream Failures**: Manual retries forcing a fresh `beam_run`.
3. **Multi-Run DB Merging**: Merging provenance DuckDB databases when promoting run archives.

### Consist Primitive Proposal
- **Scenario-Scoped Lineage Resolution**: On cache hit within an active `scenario`, Consist should prefer the run already referenced by this scenario's lineage graph. If multiple unreferenced matches exist, raise an ambiguity error or choose deterministically.

## 5. Consist Gaps & DX Findings
1. **Declared-Destination Hydration**: **Not a Consist gap**. `cache_hydration="outputs-requested"` already restores declared destinations from recovery roots. Consist DX recommendation: clarify documentation between `outputs-requested` and `outputs-all`.
2. **Deterministic Lineage Resolution**: Add scenario-scoped lineage preference to the Consist backlog.

## 6. Recommendation: Keep Refusal Gate Until Separate Stores Land
- Deleting `beam_checkpoint.py` (-738 lines) and adopting native Consist two-step execution succeeded completely.
- Keep the minimal refusal gate in `supply_demand_beam.py` until the separate-store boundary refactor is implemented.
