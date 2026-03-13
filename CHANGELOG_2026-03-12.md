# HPC Runtime Fix Changelog — 2026-03-12

## Context

Series of fixes applied to get PILATES running on the Savio HPC cluster after
upgrading to consist v0.1.0 (PyPI). Each entry documents the symptom observed,
root cause, and code change applied.

Branch: `2026-merge-to-main`

---

## Fix 1 — `LD_LIBRARY_PATH: unbound variable`

**File:** `hpc/job.sh`

**Symptom:**
```
/var/spool/slurmd/job21522993/slurm_script: line 104: LD_LIBRARY_PATH: unbound variable
```

**Root cause:** `set -euo pipefail` treats unset variables as errors. The HPC
node did not have `LD_LIBRARY_PATH` pre-set, and `module load` commands can also
internally reference unset variables.

**Change:**
- Changed `$LD_LIBRARY_PATH` → `${LD_LIBRARY_PATH:-}` in the `export` line.
- Wrapped all `module load` calls with `set +u` / `set -u` to suppress unbound
  variable errors produced by the HPC module system.

---

## Fix 2 — `ImportError: cannot import name 'CacheOptions' from 'consist.types'`

**Files:** `hpc/job.sh`, `hpc/requirements-hpc.txt`

**Symptom:**
```
ImportError: cannot import name 'CacheOptions' from 'consist.types'
```

**Root cause:** The HPC was running a stale local checkout of the consist repo
(commit `7cc56c7`) which predated the addition of `CacheOptions`,
`ExecutionOptions`, and `OutputPolicyOptions` (added in commit `03bd98a`).
`git pull` had silently failed because the HPC node had no GitHub network access
from within a running job.

**Change:**
- Removed the entire `install_consist()` function and its invocation from
  `hpc/job.sh` (which cloned/updated consist from GitHub).
- Added `consist>=0.1.0` to `hpc/requirements-hpc.txt` so consist is installed
  from PyPI automatically alongside all other Python dependencies.

---

## Fix 3 — `ImportError: pyhocon is required for BEAM canonicalization`

**File:** `hpc/requirements-hpc.txt`

**Symptom:**
```
ImportError: pyhocon is required for BEAM canonicalization. Install it with: pip install pyhocon
```

**Root cause:** consist v0.1.0's BEAM config adapter requires `pyhocon` to parse
HOCON configuration files, but `pyhocon` was not listed in the HPC requirements
file.

**Change:**
- Added `pyhocon>=0.3.60` to `hpc/requirements-hpc.txt`.

---

## Fix 4 — `RuntimeError: Cannot ingest: db_path is not configured`

**File:** `pilates/utils/consist_db_snapshot.py` — `resolve_consist_db_paths()`

**Symptom:**
```
RuntimeError: Cannot ingest: db_path is not configured
```

**Root cause:** consist v0.1.0 requires a DuckDB `db_path` to be set before it
can run `ingest` operations (triggered by the BEAM config adapter). PILATES only
created a local consist DB when `shared.database.enabled=True` (default:
`False`), so `db_path` was unconfigured in the standard deployment.

**Change:**
- `resolve_consist_db_paths()` now always creates a local run-scoped DuckDB at
  `<run_dir>/.consist/provenance.duckdb` when `consist_db_local_run=True`
  (the default), regardless of `shared.database.enabled`.
- `shared.database.enabled` now only controls whether a configured shared DB
  path is used directly when local run mode is explicitly disabled.

**Before (simplified):**
```python
db_cfg = getattr(getattr(settings, "shared", None), "database", None)
if not db_cfg or not getattr(db_cfg, "enabled", False):
    return None, None   # ← never created a DB unless shared DB was enabled
```

**After (simplified):**
```python
shared_db_enabled = db_cfg and getattr(db_cfg, "enabled", False)
if not is_local_consist_db_enabled(settings):
    if shared_db_enabled and configured_path:
        return resolved, resolved   # use shared DB only
    return None, None
# otherwise, fall through and always create a local run DB
```

---

## Fix 5 — `RuntimeError: Run missing outputs: ['plans_beam_in', 'households_beam_in', 'persons_beam_in']`

**File:** `pilates/beam/outputs.py` — `BeamPreprocessOutputs`

**Symptom:**
```
RuntimeError: Run missing outputs: ['plans_beam_in', 'households_beam_in', 'persons_beam_in']
```

**Root cause:** `BeamPreprocessOutputs` declared a `declared_outputs` class
attribute listing `(BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)`.
consist v0.1.0 enforces declared outputs strictly: if a key is declared but not
registered as a consist artifact during the step run, it raises `RuntimeError`.

These three paths are tracked via PILATES's own `prepared_inputs` dict and
coupler mechanism — not as consist artifacts — so the enforcement could never
be satisfied.

**Change:**
- Removed `declared_outputs = (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)`
  from `BeamPreprocessOutputs`.

**Before:**
```python
@dataclass
class BeamPreprocessOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "beam_mutable_data_dir"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    )
    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_mutable_data_dir",)
```

**After:**
```python
@dataclass
class BeamPreprocessOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "beam_mutable_data_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_mutable_data_dir",)
```

---

## Fix 6 — `KeyError: "Coupler missing key='plans_beam_in'. Available keys: linkstats_warmstart"`

**Files:**
- `pilates/workflows/stages/supply_demand.py` — `_derive_beam_run_input_keys()`
- `tests/test_premigration_handoff_and_restart_guards.py`
- `tests/test_stage_contracts.py`

**Symptom:**
```
KeyError: "Coupler missing key='plans_beam_in'. Available keys: linkstats_warmstart.
Did you forget to call coupler.set('plans_beam_in', ...)?"
```

**Root cause (detailed):**

When ActivitySim is **disabled**, the beam preprocessor's `_preprocess()` method
guards plans/households/persons processing behind:
```python
if self.settings.activitysim is not None:
    store += self._copy_plans_from_asim(input_records, workspace)
```
Only `_handle_linkstats()` runs unconditionally, adding `linkstats_warmstart` to
the output `RecordStore`. As a result:
- `BeamPreprocessOutputs.prepared_inputs` = `{linkstats_warmstart: path}` only
- `consist.log_output()` is called only for `linkstats_warmstart`
- After `beam_preprocess` completes, consist's `ScenarioContext` updates its
  internal `Coupler` from `result.outputs` — only `linkstats_warmstart` is there

`_derive_beam_run_input_keys()` was unconditionally building:
```python
run_input_keys = [BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN]
```
When consist's `scenario.run(fn=beam_run_step, input_keys=[...])` is called, it
calls `coupler.require(key)` for each key. The lookup for `plans_beam_in` raises
`KeyError` because only `linkstats_warmstart` is in the Coupler.

**Change:**
- `_derive_beam_run_input_keys()` now only extends `input_keys` with
  `BEAM_PLANS_IN`, `BEAM_HOUSEHOLDS_IN`, `BEAM_PERSONS_IN` when
  `activity_demand_outputs is not None` (i.e., ActivitySim ran and
  `beam_preprocess` logged these as consist artifacts via `_copy_plans_from_asim`).

**Before:**
```python
run_input_keys = [
    BEAM_PLANS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
]
if LINKSTATS_WARMSTART in beam_preprocess_inputs:
    run_input_keys.append(LINKSTATS_WARMSTART)
```

**After:**
```python
run_input_keys = []
if activity_demand_outputs is not None:
    run_input_keys.extend([BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN])
if LINKSTATS_WARMSTART in beam_preprocess_inputs:
    run_input_keys.append(LINKSTATS_WARMSTART)
```

**Behavior matrix:**

| ActivitySim | plans/households/persons in consist Coupler? | Include in `input_keys`? |
|---|---|---|
| Enabled (`activity_demand_outputs is not None`) | Yes — logged by `_copy_plans_from_asim` | Yes |
| Disabled (`activity_demand_outputs is None`) | No — static files, never logged | No |

**Tests updated:**
- `test_beam_run_input_keys_only_require_exact_warmstart_alias`: changed
  assertions from `in input_keys` to `not in input_keys` for
  plans/households/persons when `activity_demand_outputs=None`.
- `test_supply_demand_stage_beam_only_uses_default_scenario_inputs`: updated
  beam_run call filter to use `call.get("model") == "beam_run"` and asserted
  plans/households/persons are absent from `input_keys`.

---

## Files Changed

| File | Change |
|---|---|
| `hpc/job.sh` | Fix unbound `LD_LIBRARY_PATH`; add `set +u`/`set -u` around `module load`; remove `install_consist()` |
| `hpc/requirements-hpc.txt` | Add `consist>=0.1.0`, `pyhocon>=0.3.60` |
| `pilates/utils/consist_db_snapshot.py` | Always create local run DB for consist ingest |
| `pilates/beam/outputs.py` | Remove `declared_outputs` from `BeamPreprocessOutputs` |
| `pilates/workflows/stages/supply_demand.py` | Conditionalize plans/households/persons in beam_run `input_keys` on `activity_demand_outputs` |
| `tests/test_premigration_handoff_and_restart_guards.py` | Update `input_keys` assertions for ActivitySim-disabled case |
| `tests/test_stage_contracts.py` | Update beam-only test to assert correct `input_keys` behavior |

---

**Date:** 2026-03-12
**Branch:** `2026-merge-to-main`