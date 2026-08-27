# First Native Boundary Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `beam_preprocess` the first complete native consumer boundary,
with a portable canonical identity closure and evidence that an equivalent
fresh-workspace invocation reuses its cached result.

**Architecture:** Treat the full SFBay native canary as closed structural
evidence only. Narrow the next change to `beam_preprocess`: its resolver must
own every material input, configuration, and staging root used by the
preprocessor; the same portable closure then contributes to this step's Consist
identity. A new identity naturally produces the required cold run; an
equivalent run from a different workspace must then hydrate outputs and skip
the callable. No other native boundary changes status in this plan.

**Tech Stack:** Python, PILATES native workflow definitions, Consist Scenario
steps and BEAM configuration adapter, pytest, Slurm/HPC acceptance run.

**Spec:**
`docs-internal/2026-08-14-native-input-contract-structural-migration-design.md`
and
`docs/superpowers/specs/2026-08-13-native-step-explicit-input-contracts-design.md`

## Evidence and decision record

The formal structural acceptance artifact is the full SFBay
UrbanSim/ATLAS/ActivitySim/BEAM canary archive:

```text
/clusterfs/beem-core-data-nfs/pilates-canaries/sfbay-usim-verify-U9AM148M-20260826
```

All retained files were copied with matching SHA-256s and rechecked with
checker revision `1c7a2ed85205e8fb62bb2e924fa7a82c26914f4b`, which reported
`native structural canary evidence: OK`.

The Seattle ActivitySim/BEAM archive at
`/clusterfs/beem-core-data-nfs/pilates-canaries/seattle-asim-beam-F487EST3-20260824`
is supporting runtime evidence only. It is schema v1 and cannot be checked by
the schema-v2 checker because later observations lack year/iteration keys.
Retain `seattle-structural-reviewed-postrun.json` as its reviewed `i0`
expectation, but do not count it as a second formal checker success.

### HDF5 persistence pre-merge integration evidence — 2026-08-27

PILATES was tested against the editable Consist
`hdf5-identity-mismatch` branch at `c01335d01aa2f803a50c1effba41b994bebe4dcf`
(Consist PR #225). The focused native regression
`test_urbansim_postprocess_binding_survives_later_h5_hash_override_after_reopen`
executes `URBANSIM_POSTPROCESS` through `execute_step()`, records a later
caller-supplied HDF5 hash override, closes the provenance database, and
reopens it. It proves that the override is a new unowned observation and that
the original strict-bound artifact retains its trusted identity after reload.
Together with the existing strict-snapshot regression, the focused gate passed
`2 passed in 13.93s`.

This is supporting pre-merge compatibility evidence, not an HDF5 promotion
approval. It does not establish a released Consist version, reconcile a fresh
HPC snapshot, or prove a cold/fresh-workspace cache hit for an HDF5 consumer.
Keep every HDF5-consuming boundary incomplete until those separate conditions
are met.

**Selected first candidate:** `beam_preprocess`.

This is the narrowest currently plausible non-HDF5 consumer boundary: it takes
named population/optional warm-start/optional vehicle roles, already uses the
BEAM configuration adapter, is outside the pinned
`beam_run_completed -> beam_postprocess` checkpoint, and has a focused
fresh-versus-hit output-path parity test. Its initial closure audit found
mutable BEAM config, zone preparation, and workspace exchange reads; those are
now resolver-owned for the native path. The contract remains incomplete only
because its portable canonical identity closure is still pending. If a later
audit exposes a model-visible read that cannot be made resolver-owned without
a behavior decision, stop this plan; leave the contract incomplete and create
a new candidate-selection decision.

## Global Constraints

- Preserve the SFBay archive as evidence; do not run another broad native
  canary to make this change.
- Do not change `run.consist_hashing_strategy`, global `CacheOptions`, or the
  existing BEAM cache versions. The cold execution comes from the new
  boundary-specific identity, not `cache_mode="off"`.
- Do not modify `beam_run_completed -> beam_postprocess`, its pinned closure,
  exact hydration destinations, or mutation gate.
- Do not promote any HDF5-consuming boundary. HDF5 promotion remains blocked
  until a released Consist persistence fix and fresh snapshot-reload
  reconciliation prove the persisted identity is trustworthy.
- Do not widen OutputSet semantics, enable broad cross-producer reuse, or
  introduce a feature flag or a second runner path.
- Canonical identity may contain semantic roles, configuration-adapter content,
  declared launch values, and repository-relative logical destinations. It
  must not contain a workspace-absolute path, scheduler allocation ID, or
  observed output path.
- Use `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q`
  for tests. Preserve unrelated files and do not stage broadly.

---

### Task 1: Prove the BEAM-preprocess closure is implementable

**Files:**

- Modify: `pilates/workflows/steps/beam.py`
- Modify: `pilates/beam/preprocessor.py`
- Modify only if a read is actually owned there:
  `pilates/beam/beam_exchange.py`, `pilates/beam/launch_config.py`, or
  `pilates/beam/config_hocon.py`
- Test: `tests/test_beam_native_step_definitions.py`
- Test: `tests/test_beam_launch_config.py`

**Consumes:** the selected input roles produced by
`_resolve_beam_preprocess_inputs(...)`: `beam_plans_in`, `beam_households_in`,
`beam_persons_in`, optional `linkstats_warmstart`, and optional or required
`atlas_vehicles2_output` according to the enabled workflow surface.

**Produces:** one frozen resolver decision containing every model-visible
preprocess input, configuration root, zone source/destination, exchange source
or destination, and writable output root. The native callable and
`BeamPreprocessor` use that decision rather than rediscovering a material path
from `Workspace`.

- [x] **Step 1: Write closure tests before changing the resolver.**

  Add focused tests that construct a legitimate resolver result, poison the
  plausible undeclared BEAM config, zone, and exchange workspace paths, and
  verify that preprocessing either uses the resolver-declared path or fails
  before side effects. Add one test for each valid optional-role surface:
  warm-start absent/present and ATLAS vehicles absent/present.

- [x] **Step 2: Run the new tests to establish the current hidden-read
  behavior.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_beam_native_step_definitions.py tests/test_beam_launch_config.py
  ```

  Record which poisoned path is still read. Do not change the input-contract
  status at this point.

  **Observed before the resolver handoff:** primary configuration was read
  under `workspace.get_beam_mutable_data_dir() / region / settings.beam.config`;
  canonical zones came from `settings.shared.geography.zones.source_file`; and
  the exchange scenario folder was rediscovered from mutable BEAM
  configuration/workspace state.

- [x] **Step 3: Move each discovered material choice into the resolver-owned
  decision.**

  Extend `_resolve_beam_preprocess_inputs(...)` with the selected
  configuration/staging values and place the immutable decision in
  `ResolvedStepInputs.metadata`. Make `_native_beam_preprocess(...)` pass that
  decision to `BeamPreprocessor.preprocess(...)`; change the preprocessor and
  any directly responsible helper to consume those values. Keep `Workspace`
  only for non-material workflow control or output projection. Do not add a
  compatibility fallback.

  **Completed:** `beam_preprocess_context` freezes validated primary config,
  exchange scenario folder, and, when zones are enabled, canonical zone source,
  identity columns, and deterministic zone-output destination. The native
  callable requires this context; the resolver path no longer rediscovers those
  material choices.

- [x] **Step 4: Re-run the closure tests and the existing output-path parity
  tests.**

  Run the command from Step 2, then:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_step_execution.py::test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit
  ```

  The resolver-owned paths must be the only viable material paths, while fresh
  and cached output projection retains the same declared destinations.

  **Verified 2026-08-27:**

  ```text
  tests/test_beam_native_step_definitions.py
  tests/test_beam_launch_config.py
  tests/test_beam_preprocessor_exchange_folder.py
  tests/test_step_execution.py::test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit
  71 passed in 15.59s
  ```

### Task 2: Define the portable BEAM-preprocess identity closure

**Files:**

- Modify: `pilates/workflows/step_consist_meta.py`
- Modify: `pilates/workflows/steps/beam.py`
- Test: `tests/test_step_definitions.py`
- Test: `tests/test_beam_launch_config.py`
- Test: `tests/test_beam_native_step_definitions.py`

**Consumes:** Task 1's frozen resolver-owned decision and the existing BEAM
configuration adapter returned by `consist_step_meta("beam_preprocess")`.

**Produces:** a canonical, deterministic configuration contribution for
`beam_preprocess` only. It includes the contract schema/version, selected
semantic-role shape, configuration-adapter identity, and allow-listed launch
choices, but excludes workspace-local paths and runtime observations.

- [x] **Step 1: Add identity-shape regressions.**

  Assert that two contexts with distinct workspace roots but the same selected
  roles, adapter content, and allow-listed launch choices resolve to the same
  `beam_preprocess` step identity. Assert that changing one material
  allow-listed choice or selected optional-role presence changes that identity.
  Assert that merely changing an output destination or scheduler/workspace
  prefix does not.

- [x] **Step 2: Run the identity tests and verify the new closure contribution
  is absent.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_step_definitions.py tests/test_beam_launch_config.py \
    tests/test_beam_native_step_definitions.py
  ```

- [x] **Step 3: Add the boundary-specific canonical identity contribution.**

  Keep the existing adapter/config/identity inputs intact. Add only the
  portable closure payload for `beam_preprocess` to the metadata that
  `scenario.run(...)` uses as step configuration. Do not add the generic
  report-only `native_input_contract` facet to identity for every native step,
  and do not alter cache options.

- [x] **Step 4: Mark only the proven boundary complete.**

  Replace `_BEAM_PREPROCESS_INPUT_CONTRACT` with
  `InputContract(status="complete", config_contract=ConfigContract.adapter("beam"))`
  only after Steps 1–3 pass. Retain every other native contract as
  `incomplete` with its current explicit reason.

- [x] **Step 5: Run the focused identity and structural regression gate.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_step_definitions.py tests/test_step_execution.py \
    tests/test_step_execution_architecture.py \
    tests/test_beam_native_step_definitions.py tests/test_beam_launch_config.py
  rtk git diff --check
  ```

  **Verified 2026-08-27:** The BEAM adapter produces the same canonical
  identity for equivalent staged configuration trees at distinct workspace
  roots and changes it when configuration content changes. The
  `beam_preprocess` config contribution contains only
  `beam_preprocess_identity_v1`, the complete `adapter:beam` contract marker,
  selected semantic roles, and allow-listed ActivitySim/zone-preparation
  scalars; scheduler IDs, resolver paths, and output destinations are omitted.

  ```text
  tests/test_step_definitions.py
  tests/test_step_execution.py
  tests/test_step_execution_architecture.py
  tests/test_beam_native_step_definitions.py
  tests/test_beam_launch_config.py
  83 passed in 22.49s
  git diff --check: clean
  ```

### Task 3: Local cold-then-hit proof with distinct workspaces

**Files:**

- Modify: `tests/test_beam_native_step_definitions.py`
- Modify only if a small reusable helper is needed:
  `tests/helpers/` (reuse an existing helper if present)

**Consumes:** the completed `beam_preprocess` definition and Task 2's
canonical closure identity.

**Produces:** a local regression that runs `beam_preprocess` once from a cold
workspace and once from a separate workspace with equivalent roles and
configuration. The first execution calls the preprocessor; the second is a
Consist cache hit, does not call it, and materializes the complete declared
output map at the second workspace's destinations.

- [x] **Step 1: Write the two-workspace acceptance test.**

  Use one tracker/archive fixture and two distinct workspace roots. Seed the
  same tracked population and optional-role artifacts, invoke `execute_step()`
  in the first workspace, and assert the callable executed. Invoke the same
  definition with equivalent settings and roles in the second workspace, and
  assert `result.meta["cache_hit"]` is true, the callable count remains one,
  every requested preprocess output exists at the second root, and projected
  output keys equal the cold result's keys.

- [x] **Step 2: Run the test and verify the intended cold/hit sequence.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_beam_native_step_definitions.py
  ```

- [x] **Step 3: Add negative identity controls.**

  In the same test module, prove that one changed selected optional-role
  presence and one changed allow-listed launch/config value each produce a
  cache miss. Do not use `cache_mode="off"` or a manually incremented cache
  version to force those misses.

- [x] **Step 4: Re-run the focused gate and inspect the cache records.**

  Re-run the command from Step 2 and inspect the test assertions for the cold
  execution, cache hit, materialized output destinations, and negative misses.

  **Verified 2026-08-27:** `test_beam_preprocess_caches_across_workspaces_and_misses_on_material_changes`
  uses one full-hash tracker and archive fixture with separate cold and fresh
  workspace roots. The cold execution is a miss and calls the preprocessor;
  the equivalent fresh-workspace execution is a cache hit, makes no second
  preprocessor call, and materializes every selected declared output below the
  fresh root. Adding `linkstats_warmstart` or changing the allow-listed
  `activitysim_file_format` each produces a cache miss without changing cache
  mode or cache version.

  ```text
  tests/test_beam_native_step_definitions.py
  45 passed in 16.99s
  ```

### Task 4: Narrow HPC acceptance and evidence update

**Files:**

- Modify: `docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md`
- Retain externally: a new, narrowly scoped BEAM-preprocess cold/hit evidence
  bundle under the established HPC archive root

**Consumes:** the local acceptance proof and the formal SFBay structural
evidence bundle.

**Produces:** one reviewer-readable result that links the cold invocation,
the equivalent fresh-workspace hit, the checker-independent model-aware output
comparison, and the relevant Consist snapshot/action-v2 rows.

**Harness prepared 2026-08-27; no HPC evidence has been run.** Submit the
dedicated `settings-sfbay-consist-beam-preprocess-hpc-2019-acceptance.yaml`
through `hpc/job_runner.sh --beam-preprocess-acceptance` with an operator-made
input manifest and explicit `CONSIST_SRC_DIR`. The one-allocation harness
retains generated settings, submitted and environment-expanded input
manifests, one shared provenance DB and Consist run directory, full persisted
Run snapshots, cold/fresh phase JSON, and semantic validation below
`pilates-boundary-promotions/<job-id>/`. `beam_preprocess` intentionally uses an
ordinary `BindingResult`: its evidence authority is therefore each persisted
Run, its action-v2 identity/configuration metadata, and its linked artifacts,
not a `RunBindingInvocation`. The validation requires distinct requested Run
IDs, a cold miss, the fresh persisted cache source/execution Run equal to the
cold requested Run, equal non-workspace action/config identities and artifact
links, normalized requested staging paths, exact fresh hydration destinations,
one body call total, and present/equivalent semantic products. This is
checker-independent cache evidence; it does not alter the structural-canary
verdict, migrate the production boundary to strict binding, or claim HDF5
promotion/persistence evidence.

- [ ] **Step 1: Run one cold BEAM-preprocess invocation on HPC.**

  Use the normal SFBay workflow wrapper narrowed to the selected boundary and
  retain the generated settings, submitted/effective manifests, run log,
  Consist snapshot, persisted ordinary Run snapshot/action-v2 identity,
  selected-role record, linked artifacts, and declared output list. The run
  must execute rather than reuse.

- [ ] **Step 2: Run the equivalent invocation from a fresh workspace.**

  Keep the code, selected roles, canonical configuration, and artifact bytes
  equivalent. Change only the run-local workspace/allocation path. Retain the
  same evidence, and show that the body was skipped and all declared outputs
  hydrated at the fresh destinations.

- [ ] **Step 3: Perform model-aware validation.**

  Compare the BEAM-preprocess semantic products and configuration/staging
  diagnostics, not raw BEAM output bytes. Record expected path/allocation
  differences separately from behavior differences.

- [ ] **Step 4: Update this evidence section with immutable locations and
  verdict.**

  Add the archive paths, the two run IDs, checker/Consist/PILATES revisions,
  the cache-hit evidence, and the output-validation verdict. Do not update the
  SFBay structural-canary verdict or reinterpret the Seattle schema-v1 bundle.

## Completion criteria

`beam_preprocess` is the sole completed boundary only when: (1) its runner and
preprocessor no longer rediscover a material configuration, zone, or exchange
path; (2) its portable canonical closure changes identity for each material
choice and ignores workspace location; (3) local and HPC evidence both show a
cold execution followed by an equivalent fresh-workspace cache hit with exact
declared-output hydration; and (4) model-aware output validation accepts the
two executions. Failure of any condition leaves the contract `incomplete` and
does not authorize a substitute boundary or broader cache change.
