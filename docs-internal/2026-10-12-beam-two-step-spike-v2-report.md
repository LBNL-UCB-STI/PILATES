# BEAM Two-Step Spike v2: Final Report

## 1. Net Line Delta
- `pilates/workflows/beam_checkpoint.py`: -738 lines (deleted).
- `tests/test_beam_checkpoint.py`: -798 lines (deleted).
- `pilates/workflows/stages/supply_demand_beam.py`: -518 net lines (deleted checkpoint logic, progress markers, and refusal gate).
- `tests/test_restart_stage_boundary_matrix.py`: -369 net lines.
- `pilates/beam/postprocessor.py`: +45 lines (copy-on-write staging for zarr skims merge).
- `pilates/workflows/steps/beam.py`: -32 lines (removed redundant `OutputSet`, set `cache_hydration="outputs-requested"`).
- Other checkpoint caller cleanups: -197 net lines.
- New test suite (`tests/test_beam_two_step_consist_scenarios.py`): +632 lines.
- **Overall Net Reduction**: **-1,975 lines** of complex custom restart code deleted.

## 2. Scenario Results (All 7 Pass)
All scenarios exercise real stage code (`_run_beam_steps`) with real `@define_step` definitions and $\le 2$ monkeypatches per test.

| # | Scenario | Result | Evidence |
|---|---|---|---|
| 1 | Crash during `beam_run` | PASS | Container raised; rerun succeeded without consuming corrupt state. |
| 2 | Crash after `beam_run` commits | PASS | Restart cache-hit `beam_run` and executed postprocess; container ran once. |
| 3 | Crash during postprocess | PASS | Copy-on-write skims prevented in-place corruption. Restart cache-hit `beam_run`, completed postprocess, and matched uninterrupted run. |
| 4 | Scratch wiped after `beam_run` | PASS | `cache_hydration="outputs-requested"` natively restored declared outputs from recovery roots. |
| 5 | BEAM config changed | PASS | HOCON config hash miss triggered fresh container rerun. |
| 6 | Two runs, same identity | PASS | Exposed Consist recency gap: `find_matching_run` picks newest run by timestamp (`epoch_us` DESC). |
| 7 | Archive copy failure | PASS | Archive copy failure raised loudly (`verify=True`). |

## 3. Postprocess Audit & Copy-on-Write Resolution
The audit identified `_merge_beam_skims_to_zarr` as the sole in-place mutation (`xarray.Dataset.to_zarr(mode='a')`). All other writes target declared outputs.

Rather than maintaining a refusal gate and `.beam_postprocess_in_progress` markers, we implemented copy-on-write staging:
1. `_merge_beam_skims_to_zarr` duplicates `skims.zarr` to an isolated PID/timestamped staging path before modification.
2. All additions and metadata consolidations run strictly on the staging store.
3. On success, staging is atomically swapped into place via POSIX directory rename. Any failure leaves the original store byte-identical to its pre-step state.
4. The refusal gate and progress markers were deleted (56 lines removed from `supply_demand_beam.py`). Postprocess is completely idempotent.

## 4. Analysis of Scenario 6 (Same Identity, Divergent Bytes)
BEAM is stochastic; identical Merkle identity (code + config + inputs) can produce divergent outputs. Consist orders matching candidates newest first (`recent_run_order_by()`).

### Production Paths
1. **Forced Rerun / Overwrite**: Pipeline invoked with `CacheOptions(mode="overwrite")`, committing a second run with the same hash.
2. **Retrying after Downstream Failure**: A user manually retries a failed stage by re-running `beam_run` instead of relying on cache lookup.
3. **Archive / Database Merging**: Promoting archives or merging provenance DuckDB files across multiple workers or clusters.

### Consist Primitive Proposal
Instead of a per-call `pin_run_id` parameter:
- **Scenario-Scoped Lineage Resolution**: On cache lookup within an active `scenario`, Consist prefers the run already referenced by this scenario's lineage. If multiple matching runs exist and none is referenced yet, raise an ambiguous-match error or select deterministically.

## 5. Consist Gaps & DX Findings
1. **Declared-Destination Hydration**: **Not a Consist gap**. `cache_hydration="outputs-requested"` already restores cached outputs to declared `output_paths` and probes `recovery_roots` natively. The earlier failure occurred because `output_sets` was configured alongside `outputs-all`.
   - *Consist DX Lesson*: The names `outputs-requested` (hydrates declared destinations) and `outputs-all` (copies to tracker `run_dir`) are counterintuitive. Consist should document or alias them as `destination-declared` vs `tracker-local`.
2. **Deterministic Lineage Resolution**: Consist should introduce scenario-scoped lineage preference to prevent stochastic runs from silently swapping upstream realizations between coupled steps.

## 6. Recommendation: Pure Consist Steps (Delete All Checkpointing)
- **Delete entirely**: No thin wrapper or custom checkpoint logic is needed. `beam_checkpoint.py` is gone.
- Combining Consist native `cache_hydration="outputs-requested"`, copy-on-write skims staging, and standard `consist.api.archive_run_output_files` satisfies all recovery requirements natively without custom state files.
