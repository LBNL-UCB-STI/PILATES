# Pilates ‑ Refactor Road-map  
_Initial version – YYYY-MM-DD_

## Context  
This plan covers the **three largest utility modules that now mix many unrelated
responsibilities**:  

1. `workflow_state.py`  
2. `pilates/utils/provenance.py`  
3. `pilates/utils/beam.py`  

Large size and cross-cutting logic make them hard to follow, extend and test.

---

## High-level objectives  

| Priority | Goal | Why |
|----------|------|-----|
| P1 | Single-responsibility modules/classes | Easier mental model, lower change-ripple-risk |
| P2 | Eliminate code duplication (git helpers, path validation, volume mapping, etc.) | Reduce bugs & maintenance cost |
| P3 | Provide explicit, typed interfaces | Enable unit-testing & IDE support |
| P4 | Isolate all **I/O** (fs, subprocess, docker/singularity) from pure logic | Faster tests through mocks |
| P5 | Keep backwards compatibility behind a thin façade to avoid breaking CLI |

---

## Proposed refactorings (in recommended order)

### 1. Extract tiny helper modules  *(P1, P2)*
| New module | Responsibilities pulled out | Source lines to move |
|------------|----------------------------|----------------------|
| `pilates/utils/git_utils.py` | `is_git_repo`, `get_git_hash` | provenance.py: 117-146 |
| `pilates/utils/file_utils.py` | `_validate_file_path`, `_get_relative_path`, `_calculate_file_hash`, `_load_metadata` | provenance.py: 148-230 |
| `pilates/utils/container_utils.py` | `to_singularity_volumes`, `to_singularity_env`, docker/singularity command builders | workflow_state.py (last 200 lines) |

_All current call-sites keep importing through the old modules; add `from …git_utils import …` re-exports for a transition period._

### 2. Split `ProvenanceTracker` into two classes  *(P1, P3, P4)*
```
provenance/
    core.py           -> pure data model + json (no fs/git)
    file_backend.py   -> subclasses core, adds file I/O + hashing
```
Move static/path helpers to new helper modules (see step 1).  
Expose public API via `__all__ = ["ProvenanceTracker"]` that instantiates the
backend class; no caller changes.

### 3. Decompose `WorkflowState`  *(P1, P3, P4)*
```
workflow/
    state.py          -> current status, iteration logic
    path_manager.py   -> output_dir creation, copy_data_to_mutable_location
    provenance_mixin.py -> thin mixin delegating to ProvenanceTracker
```
Refactor so `state.py` _owns no disk access_.  
`path_manager.py` is injected (DI) or imported by CLI code.

### 4. Remove duplicated BEAM source dir logic  *(P2)*
`pilates/utils/beam.py::get_beam_source_dir` = single source of truth.  
All other occurrences (`workflow_state._create_output_dir`, `beam/preprocessor.py`)
should **import and call** this helper instead of re-implementing.

### 5. Introduce `settings` dataclass  *(P3)*
Create lightweight immutable `SimulationSettings` (in `pilates/config.py`) that
wraps the nested dict and offers typed accessors (`.region`, `.start_year`, …).
Gradually replace raw-dict access; start with new code only.

### 6. Unit-tests scaffold  *(P4)*
Add `tests/test_git_utils.py`, `tests/test_workflow_state.py` with pytest
fixtures stubbing filesystem and docker.

---

## Migration / Task checklist

1. [ ] Create helper modules (step 1) + move functions, add re-exports.
2. [ ] Implement new `provenance/core.py` and `file_backend.py`; update imports.
3. [ ] Slice `WorkflowState`:
   * [ ] Move file/dir creation to `path_manager.py`.
   * [ ] Add `ProvenanceMixin` that only wraps `self.provenance_tracker`.
   * [ ] Make `WorkflowState` inherit from the mixin.
4. [ ] Replace duplicate BEAM dir logic with call to `utils.beam.get_beam_source_dir`.
5. [ ] Introduce `SimulationSettings` dataclass; update only **new** helper code.
6. [ ] Add smoke tests + CI target (pytest -q).
7. [ ] Update docs & diagrams.

---

## Backwards-compatibility guardrails
* Keep public function names and parameters for at least one release.
* Emit `DeprecationWarning` from moved functions.
* Provide proxy import lines (`from .git_utils import get_git_hash`) in the old modules.

---

## Estimated effort

| Task | LoE (dev-days) |
|------|---------------|
| Helpers extraction  | 0.5 |
| Provenance split    | 1.0 |
| WorkflowState split | 1.5 |
| BEAM logic cleanup  | 0.3 |
| Settings dataclass  | 0.5 |
| Tests & docs        | 0.7 |
| **Total**           | **4–5 days** |

---

_End of plan_
