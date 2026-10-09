# ActivitySim Run Portable-Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Status:** approved design; implementation has not started.

**Goal:** Close the actual model-visible input and configuration launch tree for
`activitysim_run`, while retaining Sharrow/Numba compilation as a disposable,
private, process-local preparation action rather than a portable cache product.

**Architecture:** Keep one native `activitysim_run` Consist step. Its resolver
selects the three tabular inputs and exactly one usable skim artifact, and its
identity-bearing configuration adapter describes the complete ActivitySim
configuration tree. A typed launch-tree value gives execution deterministic
staging destinations for those sources and private destinations for output,
temporary, and compiler files. The runner consumes that value; it must not
rediscover a model-visible input or configuration root from `Workspace`.

Before a real, cache-miss body execution only, a runner-private compile epoch
may prepare a fresh host-local Numba/Sharrow cache. The epoch is retained only
in process memory and private filesystem state, so neither a prior workspace nor
a different host can suppress its first required compilation. Cache hydration
does not enter the runner, preparation, or ActivitySim body.

**Tech Stack:** Python, PILATES native workflow definitions and ActivitySim
runner, Consist resolver/requested-input staging and configuration adapter,
pytest, then a focused Slurm acceptance.

**Spec:**
`docs/superpowers/specs/2026-08-28-activitysim-run-portable-closure-design.md`

## Global constraints

- Keep `activitysim_run` as one native step. Do not add an
  `activitysim_compile` `StepDefinition`, Consist run, artifact role, archive
  product, restart input, or output-set member.
- The private compiler filesystem, its process marker, cache contents, and
  host/runtime observations are not `InputContract` identity, a declared
  output, a snapshot/restart artifact, or archived evidence.
- Do not use a nonempty `shared_cache/numba` directory as proof that the first
  compile for this PILATES invocation already occurred. Each new Python process
  starts with no compile epoch.
- Run preparation only if Consist will execute the body, Sharrow-cache
  persistence is enabled, and `activitysim.num_processes > 1`. A native cache
  hit skips all runner work by construction.
- A valid resolver-selected Zarr is read-only model input and is never
  regenerated. When Zarr is missing or invalid but OMX is selected, only the
  compile-required branch creates/finalizes runtime Zarr before the body; the
  compile-skipped branch retains the existing body-first OMX-to-Zarr behavior.
- An invalid Zarr is not repaired in place. Fall back only to a resolver
  selected OMX input, otherwise fail before the container starts.
- Keep all `InputContract` statuses `incomplete`; passing local tests does not
  promote this boundary. Do not alter global cache policy, HDF5 guidance,
  `OutputSet`, broad cross-producer reuse, or the pinned BEAM checkpoint.
- Use
  `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q` for
  focused tests. Preserve unrelated work and stage only files named in each
  task.

## File structure

- Modify: `pilates/workflows/steps/activitysim.py` — typed skim decision,
  resolver-owned launch-tree destinations, materialized execution options, and
  `activitysim_run` contract reason.
- Modify: `pilates/workflows/step_consist_meta.py` — expose the exact
  configuration roots used by the existing `ActivitySimConfigAdapter` so they
  can be staged rather than rediscovered.
- Modify: `pilates/activitysim/runner.py` — launch-context model, read-only
  skim validation, private epoch registry/preparation, and container mounts.
- Modify: `tests/test_activitysim_step_definitions.py` — resolver/identity and
  launch-context tests.
- Modify: `tests/test_activitysim_runner_mounts.py` — staged mount tests.
- Modify: `tests/test_activitysim_numba_warmup.py` and
  `tests/test_activitysim_compile_run_handshake.py` — epoch and pre-body
  preparation tests.
- Modify: `tests/test_activitysim_run_zarr_archive.py` — Zarr/OMX output and
  non-regeneration tests.
- Modify after successful focused proof only:
  `docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md` —
  record a candidate-specific acceptance, without asserting promotion until
  its final closure audit passes.

---

### Task 1: Make skim selection a pure, resolver-owned decision

**Files:**

- Modify: `pilates/workflows/steps/activitysim.py`
- Modify: `pilates/activitysim/runner.py`
- Test: `tests/test_activitysim_step_definitions.py`
- Test: `tests/test_activitysim_run_zarr_archive.py`

**Consumes:** the current native resolution of `zarr_skims` and
`asim_omx_skims`, including the special prior-BEAM-handoff case.

**Produces:** an immutable, JSON/configuration-safe `ActivitySimSkimDecision`
carried in resolved metadata with `role`, `mode`, `produces_zarr`, and the
logical selected source. It is derived before binding inputs are subset, so
unselected skim alternatives cannot materialize or affect identity.

- [ ] **Step 1: Write failing selection tests.**

  Add cases beside the existing parameterized resolver tests that create a
  structurally valid Zarr directory, a missing Zarr, and an invalid Zarr
  directory. Assert:

  1. a valid selected Zarr chooses Zarr mode and `produces_zarr=False`;
  2. invalid Zarr plus available OMX selects OMX and `produces_zarr=True`;
  3. invalid Zarr without OMX raises before an execution option or runner is
     created; and
  4. the Zarr validation does not change any file, metadata, or zone flags.

  Continue to require Zarr for the prior-BEAM-handoff surface; there is no OMX
  fallback on that declared workflow surface.

- [ ] **Step 2: Run the new selection tests and record current behavior.**

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_step_definitions.py \
    tests/test_activitysim_run_zarr_archive.py
  ```

  Expected before the change: any present Zarr wins without a read-only format
  check.

- [ ] **Step 3: Add a non-mutating Zarr validator and decision type.**

  Put the validator next to the existing skim helpers in
  `pilates/activitysim/runner.py`; it may inspect directory/layout metadata but
  must not call `ensure_0_based_and_flag_zarr_skims()` or
  `finalize_activitysim_zarr_skims()`. Define `ActivitySimSkimDecision` in the
  narrowest shared module needed by both resolver and runner. In
  `_activitysim_run_resolver()`, inspect the selected Zarr candidate before
  deciding the binding subset. On rejection with OMX available, bind only OMX
  and preserve a diagnostic string outside identity-bearing metadata; otherwise
  raise a precise error naming the selected skim role and rejection reason.

  Preserve the existing metadata keys temporarily for output-path compatibility,
  but derive their values only from the decision type. Do not encode absolute
  source paths in the identity contribution.

- [ ] **Step 4: Re-run focused skim and output-projection tests.**

  Run the command from Step 2 plus:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_step_execution.py::test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit
  ```

  Verify that only the selected role appears in `BindingResult.inputs` and that
  fallback does not turn an invalid Zarr into a published output.

- [ ] **Step 5: Commit the selection contract.**

  ```bash
  rtk git add pilates/workflows/steps/activitysim.py pilates/activitysim/runner.py \
    tests/test_activitysim_step_definitions.py tests/test_activitysim_run_zarr_archive.py
  rtk git commit -m "test: define ActivitySim skim selection contract"
  ```

### Task 2: Stage the complete ActivitySim launch tree

**Files:**

- Modify: `pilates/workflows/steps/activitysim.py`
- Modify: `pilates/workflows/step_consist_meta.py`
- Modify: `pilates/activitysim/runner.py`
- Test: `tests/test_activitysim_step_definitions.py`
- Test: `tests/test_activitysim_preprocessor_config_dirs.py`
- Test: `tests/test_activitysim_runner_mounts.py`
- Test: `tests/test_resolved_binding_v1_eligibility.py`

**Consumes:** Task 1's selected role and the configuration-root ordering already
used by `_activitysim_adapter()` (`main_configs_dir`, `configs`,
`configs_extended`, `configs_mp`, and `configs_sh_compile`, deduplicated in
that order).

**Produces:** a typed `ActivitySimLaunchContext` whose input-data and config
roots are deterministic destinations under a native staged launch tree and
whose output, temporary, runtime-Zarr, and compiler paths are private
execution destinations. `ExecutionOptions` carries that context; the runner
does not call `Workspace.get_asim_mutable_data_dir()` or
`Workspace.get_asim_mutable_configs_dir()` to discover model-visible sources.

- [ ] **Step 1: Write closure tests before changing destinations.**

  Construct a resolver result with distinct staged data/config destinations,
  then poison the corresponding ambient workspace data/config trees. Assert the
  native callable creates mounts only from the staged context and that all three
  configuration directories required for normal, multiprocessing, and compile
  modes are present. Add a two-workspace regression: different workspace roots
  with identical selected roles/config contents produce the same portable
  closure payload, while a changed staged config file changes the adapter-backed
  identity.

  Add a test that a missing staged config directory fails before
  `run_container()`, rather than falling back to the original mutable config
  root. Retain the preprocessor test proving the expected configuration layout
  is copied/available.

- [ ] **Step 2: Run closure tests and observe the current ambient read.**

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_step_definitions.py \
    tests/test_activitysim_preprocessor_config_dirs.py \
    tests/test_activitysim_runner_mounts.py \
    tests/test_resolved_binding_v1_eligibility.py
  ```

  Expected before the change: the configuration adapter fingerprints roots, but
  `_activitysim_launch_context(workspace)` and runner mounts rediscover mutable
  workspace data/config paths.

- [ ] **Step 3: Create one declarative staged launch-tree layout.**

  Refactor `_activitysim_launch_context()` into a builder that receives the
  workspace only to allocate private destinations and receives explicit staged
  input/config destinations for every model-visible root. Point
  `ActivitysimRunner.declared_expected_inputs()` / resolver logical destinations
  at this tree, not the ambient mutable-data directory.

  Extract the ordered config-root discovery currently embedded in
  `_activitysim_adapter()` into one typed helper in
  `step_consist_meta.py`. Use that helper both to construct
  `ActivitySimConfigAdapter(root_dirs=...)` and to stage the same exact roots
  into the launch tree. Copy/stage configuration bytes only after Consist has
  admitted a body execution; no cache hit needs a writable local launch tree.
  Treat absence of a declared config root as a fail-closed error.

  Materialize selected tabular/skim roles at their declared launch-tree
  destinations through existing requested-input staging. Construct the final
  `ActivitySimLaunchContext` from those materialized destinations in
  `_activitysim_execution_options()` and pass it unchanged through
  `_activitysim_run_callable()` into `ActivitysimRunner.run()`.

- [ ] **Step 4: Restrict mounts to the supplied context.**

  Update `ActivitysimRunner.get_asim_docker_vols()` and warmup delegation to
  require an explicit launch context for native execution. Mount staged model
  data/configuration inputs read-only where ActivitySim permits it and mount
  only the private output/cache/temp/compiler directories writable. Keep
  legacy callers explicit during the transition; do not silently synthesize a
  context from `Workspace` in the native path.

- [ ] **Step 5: Re-run tests and audit the native definition.**

  Run the command from Step 2, then:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_compile_run_handshake.py \
    tests/test_activitysim_run_zarr_archive.py
  ```

  Inspect the `activitysim_run` resolver/callable and confirm every
  model-visible input/config path comes from the resolved context. Update the
  `InputContract.reason` to name only any remaining real closure gap; do not
  change status.

- [ ] **Step 6: Commit the launch-tree closure.**

  ```bash
  rtk git add pilates/workflows/steps/activitysim.py pilates/workflows/step_consist_meta.py \
    pilates/activitysim/runner.py tests/test_activitysim_step_definitions.py \
    tests/test_activitysim_preprocessor_config_dirs.py tests/test_activitysim_runner_mounts.py \
    tests/test_resolved_binding_v1_eligibility.py
  rtk git commit -m "feat: stage ActivitySim native launch inputs"
  ```

### Task 3: Replace stale-cache warmup skipping with a private compile epoch

**Files:**

- Modify: `pilates/activitysim/runner.py`
- Test: `tests/test_activitysim_numba_warmup.py`
- Test: `tests/test_activitysim_compile_run_handshake.py`
- Test: `tests/test_activitysim_runner_mounts.py`

**Consumes:** Task 2's explicit launch context and Task 1's immutable skim
decision.

**Produces:** a module-private, process-local compile-epoch registry keyed by
the live workflow invocation (retain the live state/context object, not a
workspace pathname). It allocates a fresh epoch cache root before the first
required preparation and hands the same root to later required executions in
that invocation. A fresh Python process always has an empty registry.

- [ ] **Step 1: Write failing predicate and epoch tests.**

  Replace the current parameter table that treats an existing
  `shared_cache/numba` as a warmup skip with tests for the exact predicate:

  ```text
  body execution AND persist_sharrow_cache_enabled(settings) AND num_processes > 1
  ```

  Cover Sharrow disabled, one process, and explicit rewind skip. For the true
  case, pre-populate a stale workspace cache and assert preparation still runs;
  call a second runner in the same live invocation and assert it reuses the
  in-memory epoch without a second compile; construct a new state/process seam
  and assert it prepares again. Assert no epoch path appears in binding inputs,
  declared outputs, or step configuration.

  Keep the native cache-hit test: prove `execute_step()` hydrates without
  calling `ActivitysimRunner.run`, the preparation helper, or a container.

- [ ] **Step 2: Run the existing compile tests to capture old behavior.**

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_numba_warmup.py \
    tests/test_activitysim_compile_run_handshake.py \
    tests/test_activitysim_runner_mounts.py
  ```

  Expected before the change: a nonempty local `shared_cache/numba` suppresses
  warmup even if it came from a prior invocation.

- [ ] **Step 3: Implement the runner-private preparation seam.**

  Add a narrow private preparation method to `ActivitysimRunner` that accepts
  the immutable skim decision and launch context, returns the body skim
  mode/path, and performs no Consist call or state transition. Replace
  `_dir_contains_files()` as the warmup gate with the explicit predicate plus
  an `ActivitySimCompileEpochRegistry` held only in Python memory. On first
  required preparation allocate a fresh epoch directory beneath the private
  launch root, pass a `replace(launch_context, shared_cache_dir=epoch_path)`
  context to warmup and the production body, and retain that exact path for
  later eligible executions in the same live invocation.

  Do not delete or rely on an old workspace cache. A compile failure aborts the
  body and must not fall back to old bytes. Preserve compile-only output
  isolation and `lineage_mode="none"`.

- [ ] **Step 4: Re-run the predicate/epoch suite.**

  Run the command from Step 2. Verify logs distinguish `not required`, `new
  private epoch`, and `reused private epoch`; they must not report an ambient
  node-local cache as reusable.

- [ ] **Step 5: Commit the epoch semantics.**

  ```bash
  rtk git add pilates/activitysim/runner.py tests/test_activitysim_numba_warmup.py \
    tests/test_activitysim_compile_run_handshake.py tests/test_activitysim_runner_mounts.py
  rtk git commit -m "fix: isolate ActivitySim compile epochs"
  ```

### Task 4: Make preparation preserve the selected skim semantics

**Files:**

- Modify: `pilates/activitysim/runner.py`
- Test: `tests/test_activitysim_numba_warmup.py`
- Test: `tests/test_activitysim_compile_run_handshake.py`
- Test: `tests/test_activitysim_run_zarr_archive.py`

**Consumes:** Tasks 1–3.

**Produces:** a normal body invocation that sees the same selected Zarr bytes
when Zarr is input, or a read-only validated runtime Zarr generated from OMX
before the body only when preparation is required.

- [ ] **Step 1: Write failing Zarr/OMX handshake tests.**

  Mock only container execution and assert all four matrix rows from the
  approved design:

  1. selected Zarr plus required preparation: compile and body use the same
     staged Zarr source, and no Zarr finalizer/write runs;
  2. selected Zarr with preparation skipped: body uses that staged input with
     no compile;
  3. selected OMX plus required preparation: preparation generates runtime
     Zarr, validates/finalizes it before the body, then body receives Zarr mode;
  4. selected OMX with preparation skipped: body remains OMX mode and the
     existing post-body finalization creates the declared Zarr output.

  Add failures for a missing/invalid generated Zarr and a failed preparation;
  both must fail before the body or output admission. Assert a selected Zarr
  is never copied into a separate compile-only Zarr whose bytes diverge from
  the body input.

- [ ] **Step 2: Run the handshake tests before implementation.**

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_numba_warmup.py \
    tests/test_activitysim_compile_run_handshake.py \
    tests/test_activitysim_run_zarr_archive.py
  ```

- [ ] **Step 3: Thread the decision through preparation and body.**

  Remove `_stage_compile_zarr_input()`'s separate-copy semantics for the
  native path. Use the staged launch-tree Zarr as the one source supplied to
  both compile and body, with mounts preserving the required read-only source
  contract. In the OMX/compile-required branch, direct warmup to create the
  production runtime Zarr destination, then use the existing finalization
  routine exactly once before switching the body to Zarr mode. In the
  OMX/compile-skipped branch, leave finalization after the normal body.

  Make the preparation return an explicit result rather than changing
  `skim_mode` through mutable arguments. Keep only the generated OMX-path Zarr
  in `ActivitySimRunOutputs.zarr_skims`; an input Zarr must never be re-enqueued
  for archive projection.

- [ ] **Step 4: Re-run handshake and output tests.**

  Run the command from Step 2 plus the focused cache-hit test in
  `tests/test_activitysim_compile_run_handshake.py`. Verify no test needs a
  host-mounted historical compile cache.

- [ ] **Step 5: Commit the skim/prepare handshake.**

  ```bash
  rtk git add pilates/activitysim/runner.py tests/test_activitysim_numba_warmup.py \
    tests/test_activitysim_compile_run_handshake.py tests/test_activitysim_run_zarr_archive.py
  rtk git commit -m "feat: prepare ActivitySim skims before parallel runs"
  ```

### Task 5: Audit the closed candidate and define its one acceptance run

**Files:**

- Modify: `tests/test_resolved_binding_v1_eligibility.py`
- Modify only if focused evidence needs an operator seam:
  `pilates/runtime/activitysim_run_acceptance.py`,
  `hpc/job_runner.sh`, `hpc/job.sh`, `tests/test_hpc_job_runner.py`, and
  `hpc/README.md`
- Modify after evidence: 
  `docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md`

**Consumes:** the completed local implementation and an operator-supplied real
SFBay ActivitySim input manifest. It must use the released Consist replay when
available; the existing editable checkout is useful implementation evidence,
not final HDF5 promotion evidence.

**Produces:** either a documented remaining ambient-read blocker (contract
stays incomplete) or a narrow cold-miss/fresh-workspace-hit candidate
acceptance plan with model-aware validation.

- [ ] **Step 1: Extend the closure audit before building an HPC harness.**

  Add a test that poisons every former mutable ActivitySim data/config root and
  proves resolver-owned staged sources are the only viable execution inputs.
  Audit `activitysim_run` in the actual callable, runner mounts, adapter, and
  output projection. If a material source remains ambient, stop here: document
  it as a blocker rather than creating a cache-evidence wrapper.

- [ ] **Step 2: Run the complete focused local gate.**

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_activitysim_step_definitions.py \
    tests/test_activitysim_preprocessor_config_dirs.py \
    tests/test_activitysim_runner_mounts.py \
    tests/test_activitysim_numba_warmup.py \
    tests/test_activitysim_compile_run_handshake.py \
    tests/test_activitysim_run_zarr_archive.py \
    tests/test_resolved_binding_v1_eligibility.py \
    tests/test_step_execution.py::test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit
  ```

- [ ] **Step 3: Build the smallest evidence wrapper only if the audit passes.**

  Follow the existing `beam_preprocess` acceptance pattern, but require an
  explicit four-role-or-fewer manifest for ActivitySim’s three tables and one
  selected skim. Use separate empty cold and fresh workspace roots with the
  same tracker/provenance store; record requested/source run IDs, body counts,
  resolved binding records, persisted execution/source IDs, output hydration
  destinations, and model-aware semantic validators. Do not run a broad
  UrbanSim/ATLAS/ActivitySim/BEAM canary or compare BEAM-like outputs bytewise.

- [ ] **Step 4: Require the exact acceptance result before changing status.**

  The cold invocation must be a miss with exactly one body execution. The
  equivalent fresh workspace invocation must be a hit with no additional body
  or compile preparation, hydrate all declared outputs, retain the strict
  binding/persisted identity relationship, and pass model-aware ActivitySim
  product checks. Archive evidence with checksums. Only then update the first
  boundary-promotion plan with facts and decide whether an independently
  reviewed contract-completeness change is warranted.

- [ ] **Step 5: Commit the audit/harness separately from evidence records.**

  ```bash
  rtk git add tests/test_resolved_binding_v1_eligibility.py \
    pilates/runtime/activitysim_run_acceptance.py hpc/job_runner.sh hpc/job.sh \
    tests/test_hpc_job_runner.py hpc/README.md
  rtk git commit -m "test: add ActivitySim run closure acceptance"
  ```

  Do not create this commit if Task 5 Step 1 finds a real ambient-read blocker;
  instead commit the focused audit test and a documented blocker.

## Final verification before requesting evidence

```bash
rtk git diff --check
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
  tests/test_activitysim_step_definitions.py \
  tests/test_activitysim_preprocessor_config_dirs.py \
  tests/test_activitysim_runner_mounts.py \
  tests/test_activitysim_numba_warmup.py \
  tests/test_activitysim_compile_run_handshake.py \
  tests/test_activitysim_run_zarr_archive.py \
  tests/test_resolved_binding_v1_eligibility.py \
  tests/test_step_execution.py::test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit
rtk git status --short
```

Expected: all focused tests pass; only task-scoped files are changed; no
contract has been promoted and no compiler/cache path appears in the portable
identity closure. The next action is an operator-reviewed, single-boundary HPC
cold-miss/fresh-workspace-hit proof, not a global cache-policy change.
