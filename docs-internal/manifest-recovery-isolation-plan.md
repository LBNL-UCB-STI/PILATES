# Manifest Recovery Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate YAML workflow manifest behavior behind a narrow recovery-store boundary so manifests can become optional by stage and eventually removable.

**Architecture:** Keep `scenario.run(...)` as the single execution path and move manifest loading, stale detection, output restore, skip policy, and persistence into a named recovery store. First run a tracker-only recovery spike so the team knows whether this is a short bridge to deletion or a longer-lived compatibility backend. Then collapse the duplicate executor unconditionally; later phases introduce a no-op recovery store for low-risk stages before touching supply-demand restart behavior.

**Tech Stack:** Python 3.11, dataclasses, `typing.Protocol`, PyYAML for the existing YAML store, pytest, PILATES workflow/stage modules, Consist scenario execution.

---

## Context And Motivation

The current workflow manifest path was useful during the migration to typed
step boundaries and Consist-backed execution. It still protects several real
behaviors:

- restart and cache-hit output hydration for typed `StepOutputs` objects
- stale-output pruning before trusting a previous manifest entry
- run-id restoration for skipped steps
- archive copying of `.workflow/*.yaml` files
- supply-demand restart handoff repair, especially ActivitySim-to-BEAM resume

At the same time, the manifests now obscure the intended architecture. PILATES
is Consist-first, and `scenario.run(...)` should be the normal authority for
step execution, cache checks, lineage, and run identity. The YAML manifests
should read as a compatibility/recovery backend, not as a second workflow engine.

The clean migration path is not to delete `run_manifested_steps(...)` in one
patch. Instead, isolate all manifest-specific behavior behind a small interface,
make the default implementation wrap today's YAML behavior, then add a no-op
implementation that can be enabled for low-risk stages. Deletion only becomes
safe after no production stage depends on the YAML store and supply-demand
restart behavior has an equivalent Consist-native recovery path.

The tracker-native recovery question should be answered before the first
refactor PR lands. That spike decides whether this plan is primarily a bridge to
manifest deletion or a deliberate isolation of a compatibility backend. The
executor-loop cleanup remains valuable either way: removing the duplicate loop
in `run_manifested_steps(...)` reduces maintenance risk even if YAML manifests
stay enabled for supply-demand longer than desired.

## Current Runtime Shape

The main manifest touch points are:

- `pilates/workflows/orchestration.py`
  - `ManifestConfig`
  - `run_manifested_steps(...)`
  - `run_workflow(...)`
  - `_detect_stale_steps(...)`
  - `_expand_stale_manifest_steps(...)`
  - `_restore_outputs_from_manifest(...)`
- `pilates/utils/step_manifest.py`
  - `load_step_manifest(...)`
  - `save_step_manifest(...)`
- stage modules that create `ManifestConfig`
  - `pilates/workflows/stages/land_use.py`
  - `pilates/workflows/stages/vehicle_ownership.py`
  - `pilates/workflows/stages/supply_demand.py`
  - `pilates/workflows/stages/supply_demand_activity.py`
  - `pilates/workflows/stages/postprocessing.py`
- supply-demand resume helpers
  - `pilates/workflows/stages/supply_demand_resume.py`

The current `run_workflow(...)` branches early: if `manifest_config is not None`,
it delegates to `run_manifested_steps(...)`; otherwise it executes the native
step loop. That is the main structural problem. The target shape is one shared
step loop with an optional recovery store.

Supply-demand manifests also have two deletion-critical roles that are separate
from the duplicate executor problem:

- `seed_supply_demand_parent_run_ids_for_resume(...)` discovers candidate
  `(year, iteration)` epochs by globbing `year_*_iteration_*.yaml` files, then
  seeds restored ActivitySim and BEAM run IDs from manifest entries or tracker
  fallbacks.
- `_restore_activity_demand_outputs_for_resume(...)` can use manifest
  `activitysim_postprocess` output data to reconstruct the BEAM-facing handoff
  before falling back to tracker outputs and coupler artifacts.

## Desired End State

The end state is:

1. `run_workflow(...)` has one executor loop.
2. A `StepRecoveryStore` protocol owns checkpoint/recovery behavior.
3. `YamlManifestRecoveryStore` implements today's YAML manifest behavior.
4. `NoManifestRecoveryStore` implements native Consist-only execution.
5. Stage modules ask for a recovery store through one small policy helper.
6. Stage-level opt-out can disable manifests for proven-low-risk stages.
7. Supply-demand is migrated last because its manifest data participates in
   restart run-id and ActivitySim/BEAM handoff recovery.
8. Any temporary guard that blocks supply-demand opt-out has an explicit removal
   criterion and tracking note, so it does not become permanent scaffolding.

Out of scope:

- replacing binding, catalog, or stage orchestration
- changing artifact key names or typed output semantics
- making Consist API changes inside this repo
- removing PyYAML, because YAML remains core to settings and state files
- changing archive worker behavior

## Proposed File Structure

- Create `pilates/workflows/recovery.py`
  - Owns `StepRecoveryStore`, `StepRecoveryDecision`,
    `YamlManifestRecoveryStore`, and `NoManifestRecoveryStore`.
  - Wraps existing manifest helpers without changing YAML format.

- Modify `pilates/workflows/orchestration.py`
  - Keeps step execution, `scenario.run(...)`, cache recovery, and output
    replay orchestration.
  - Stops owning raw manifest load/save policy once `recovery.py` exists.
  - Keeps `ManifestConfig` only as a temporary compatibility surface through
    the shared-executor migration.

- Create `pilates/workflows/recovery_policy.py`
  - Small helper module that maps settings/stage/year/iteration to a
    recovery store.
  - Starts behavior-preserving by always returning `YamlManifestRecoveryStore`
    when a manifest path is supplied.
  - Later reads a stage-level opt-out setting.

- Modify stage modules
  - Replace direct `ManifestConfig(...)` construction with policy helper calls
    only after the store boundary is proven.

- Modify tests
  - `tests/test_architecture_guardrails.py`
  - `tests/test_manifest_cache_parity.py`
  - `tests/test_workflow_invariants.py`
  - `tests/test_cache_hit_recovery.py`
  - `tests/test_postprocessing_manifest_persistence.py`
  - `tests/test_land_use_manifest_persistence.py`
  - `tests/test_stage_contracts.py`
  - `tests/test_restart_stage_boundary_matrix.py`

## Interface Sketch

The exact names can change during implementation, but the boundary should stay
this narrow:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class StepRecoveryDecision:
    skip_step: bool = False
    restored_outputs: Optional[Any] = None
    restored_run_id: Optional[str] = None
    recovery_meta: Mapping[str, Any] = field(default_factory=dict)


class StepRecoveryStore(Protocol):
    def prepare(self, *, steps: list[Any], outputs_holder: Any) -> None:
        """Load and validate any persisted recovery state before execution."""

    def before_step(self, *, step: Any, outputs_holder: Any) -> StepRecoveryDecision:
        """Return a skip/restore decision for the next step."""

    def after_step(
        self,
        *,
        step: Any,
        result: Any,
        outputs: Optional[Any],
        serialized_outputs: Mapping[str, Any],
    ) -> None:
        """Persist run metadata and outputs after a live step execution."""
```

Implementation note: the real protocol should use concrete local types such as
`StepRef`, `StepOutputsHolder`, and `Mapping[str, Any]` rather than unbounded
`Any` where imports do not create cycles. The plan should not introduce
`getattr` or `hasattr` feature probing; use explicit protocol methods and
dataclasses.

## Migration Phases

### Phase 0: Tracker-Native Recovery Spike

**Objective:** Determine whether Consist tracker data can replace the
supply-demand manifest roles before investing in stage-by-stage opt-out work.

Questions to answer:

- Can tracker queries enumerate all relevant ActivitySim and BEAM
  `(year, iteration) -> run_id` pairs without using manifest filenames as the
  seed list?
- Can `load_tracker_run_outputs(...)` or another current Consist API supply the
  complete ActivitySim postprocess handoff required by BEAM restart?
- Are run IDs reliably tagged with `model`, `year`, and `iteration`, or does the
  current code still need manifest filenames and run-id regex parsing as
  fallbacks?
- Do tracker/coupler fallback paths cover restart into both `activity_demand`
  and `traffic_assignment` boundaries?

Expected result:

- A short docs-internal spike note records the answer as either
  `delete-track` or `compatibility-backend-track`.
- If the answer is `delete-track`, Phases 3-6 should be compressed around
  replacing supply-demand resume reads directly.
- If the answer is `compatibility-backend-track`, proceed with the conservative
  isolation plan and keep supply-demand YAML-backed until Consist exposes the
  missing recovery surface.

### Phase 1: Extract A Behavior-Preserving YAML Store

**Objective:** Move manifest-specific behavior into `YamlManifestRecoveryStore`
without changing any stage behavior.

Expected result:

- All existing manifest tests still pass.
- Stage modules still produce the same `.workflow/*.yaml` files.
- `run_workflow(..., manifest_config=ManifestConfig(...))` still works.
- The internal code now reads as "native executor plus recovery store" rather
  than "native executor or manifested executor."

### Phase 2: Collapse The Dual Executor Loop

**Objective:** Remove the separate `run_manifested_steps(...)` execution loop.

Expected result:

- `run_manifested_steps(...)` can remain as a compatibility wrapper, but it
  should create a `YamlManifestRecoveryStore` and call the shared executor.
- Live execution, cache recovery, output replay, and logging all flow through
  one code path.
- Manifest skip/restore behavior happens through `before_step(...)`.
- Manifest persistence happens through `after_step(...)`.
- `ManifestConfig` remains importable only to preserve current callers during
  the migration; new production code should pass a recovery store or use the
  policy helper.

### Phase 3: Add A No-Manifest Store Behind A Stage Policy

**Objective:** Add `NoManifestRecoveryStore` and policy wiring without changing
the default behavior.

Expected result:

- A targeted test can run the postprocessing stage without writing a manifest.
- Defaults continue to use YAML manifests for all existing stages.
- The policy helper makes stage-level opt-out explicit and searchable.
- Supply-demand opt-out is blocked with a temporary, documented guard unless
  the Phase 0 spike proves tracker-native recovery is complete.

### Phase 4: Disable Manifests For Postprocessing First

**Objective:** Prove one low-risk stage can run Consist-native without YAML
manifest checkpointing.

Why postprocessing first:

- It has a single step.
- It runs after the year workflow has completed.
- It does not mediate ActivitySim/BEAM restart handoffs.
- It is easy to test with existing manifest persistence tests.

Expected result:

- Postprocessing can run with `NoManifestRecoveryStore`.
- Existing postprocessing output behavior remains unchanged.
- If the opt-out is reverted, the YAML store still works.

### Phase 5: Consider Land Use And Vehicle Ownership

**Objective:** Migrate non-supply-demand stages only after postprocessing proves
the policy path and tests are clean.

Expected result:

- Land-use and vehicle-ownership manifests become optional only if local tests
  prove output hydration, archive copy, and downstream handoff behavior remain
  equivalent.
- Any migration of these stages should be one stage per PR.

### Phase 6: Supply-Demand Last

**Objective:** Replace supply-demand manifest dependencies only after Consist or
PILATES native recovery can supply equivalent restart metadata.

Supply-demand-specific blockers:

- `supply_demand_resume.py` scans `year_*_iteration_*.yaml` manifests.
- ActivitySim and BEAM run IDs are restored from manifest entries.
- ActivitySim postprocess outputs may be hydrated from manifest data to repair
  traffic-assignment restart handoffs.
- Stale output handling prevents mixtures of fresh upstream and stale downstream
  artifacts.

Expected result before deletion:

- restart from activity-demand and traffic-assignment boundaries works without
  reading YAML manifests
- run-id restoration is sourced from Consist tracker data or an explicit
  non-YAML recovery record
- ActivitySim postprocess output hydration is proven equivalent
- golden/stub workflow and restart boundary tests pass without manifest input

## Compatibility Deadlines

- `ManifestConfig` may remain as a compatibility class while
  `run_workflow(..., manifest_config=...)` still has existing callers.
- After the stage policy helper is introduced, new production stage code should
  not construct `ManifestConfig` directly.
- By the end of the postprocessing opt-out slice, add an architecture guard that
  fails on new direct `ManifestConfig` imports in migrated production stages.
- The supply-demand opt-out guard must be labeled temporary in code and linked
  to the tracker-native recovery spike result or follow-up note.

## Detailed Task Plan

### Task 0: Spike Tracker-Native Supply-Demand Recovery

**Files:**

- Create: `docs-internal/manifest-recovery-tracker-spike.md`
- Read: `pilates/workflows/stages/supply_demand_resume.py`
- Read: `pilates/runtime/scenario_runtime.py`
- Read: `pilates/workflows/tracker_outputs.py`
- Read: `tests/test_restart_stage_boundary_matrix.py`
- Read: `tests/test_restart_replay_archive_sources.py`

- [ ] **Step 1: Trace manifest-backed restart inputs**

Document every value read from `year_*_iteration_*.yaml` manifests:

- manifest filename-derived `year`
- manifest filename-derived `iteration`
- `activitysim_run.run_id`
- `beam_run.run_id`
- `activitysim_postprocess.outputs.processed_outputs`

Write the list to `docs-internal/manifest-recovery-tracker-spike.md` under
`## Manifest Inputs To Replace`.

- [ ] **Step 2: Trace current tracker and coupler fallbacks**

Document current non-manifest paths:

- `_find_matching_run_for_resume_target(...)`
- `_find_tracker_run_by_id(...)`
- `_run_id_epoch(...)`
- `load_tracker_run_outputs(...)`
- `_supplement_from_coupler(...)`
- the final coupler-only branch in `_restore_activity_demand_outputs_for_resume(...)`

Write the result under `## Existing Non-Manifest Recovery Surfaces`.

- [ ] **Step 3: Build a verdict matrix**

Add a matrix with these rows:

| Requirement | Current non-YAML source | Proven by test | Gap |
| --- | --- | --- | --- |
| enumerate ActivitySim run IDs by year/iteration |  |  |  |
| enumerate BEAM run IDs by year/iteration |  |  |  |
| load ActivitySim postprocess handoff outputs |  |  |  |
| materialize BEAM-facing ActivitySim paths |  |  |  |
| seed parent run IDs without manifest filenames |  |  |  |

- [ ] **Step 4: Classify the migration track**

End the spike note with exactly one of these verdicts:

- `delete-track`: existing tracker/coupler behavior can replace YAML manifests
  for supply-demand with focused implementation work.
- `compatibility-backend-track`: YAML manifests still provide at least one
  required recovery input that current tracker/coupler APIs do not replace.

Include the specific blocker if the verdict is `compatibility-backend-track`.

- [ ] **Step 5: Commit**

```bash
git add docs-internal/manifest-recovery-tracker-spike.md
git commit -m "docs: evaluate tracker-native manifest recovery path"
```

### Task 1: Characterize Existing Manifest Behavior

**Files:**

- Modify: `tests/test_manifest_cache_parity.py`
- Modify: `tests/test_workflow_invariants.py`
- Modify: `tests/test_cache_hit_recovery.py`

- [ ] **Step 1: Add focused characterization tests for the recovery contract**

Add tests that explicitly cover:

- stale manifest entries are pruned with downstream dependents
- a manifest-restored step publishes recovered outputs back to the coupler
- a manifest-restored step remembers a restored run ID when the scenario
  supports restored run IDs
- a step with a manifest run ID but no declared outputs reruns

Use existing fixtures in `tests/test_manifest_cache_parity.py` and
`tests/test_workflow_invariants.py` rather than creating new workflow fixtures.

- [ ] **Step 2: Run the characterization slice**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py \
  tests/test_workflow_invariants.py \
  tests/test_cache_hit_recovery.py -q
```

Expected: PASS before production code changes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py
git commit -m "test: characterize workflow manifest recovery behavior"
```

### Task 2: Add The Recovery Store Module

**Files:**

- Create: `pilates/workflows/recovery.py`
- Test: `tests/test_manifest_cache_parity.py`

- [ ] **Step 1: Write failing unit tests for the store boundary**

Add tests that instantiate a YAML-backed store with a temporary manifest path
and prove:

- `prepare(...)` loads existing manifest data
- `before_step(...)` returns a skip decision for a step with restorable outputs
- `after_step(...)` writes the same manifest shape used today
- `NoManifestRecoveryStore` never skips and never writes a file

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py -q
```

Expected: FAIL because `pilates.workflows.recovery` does not exist.

- [ ] **Step 3: Implement `pilates/workflows/recovery.py`**

Move or wrap the existing manifest responsibilities into the new module:

- `ManifestConfig`
- `StepRecoveryDecision`
- `StepRecoveryStore`
- `YamlManifestRecoveryStore`
- `NoManifestRecoveryStore`

Keep YAML serialization in `pilates/utils/step_manifest.py`; do not duplicate
the load/save functions.

- [ ] **Step 4: Run the recovery tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pilates/workflows/recovery.py tests/test_manifest_cache_parity.py
git commit -m "refactor: introduce workflow recovery stores"
```

### Task 3: Route `run_workflow` Through The Recovery Store

**Files:**

- Modify: `pilates/workflows/orchestration.py`
- Modify: `tests/test_manifest_cache_parity.py`
- Modify: `tests/test_workflow_invariants.py`
- Modify: `tests/test_cache_hit_recovery.py`

- [ ] **Step 1: Write a failing parity test for shared execution**

Add a test that runs the same two-step workflow with:

- `manifest_config=ManifestConfig(path=...)`
- an explicit `YamlManifestRecoveryStore`

Assert both paths produce the same outputs holder state, coupler publications,
and manifest data.

- [ ] **Step 2: Run the parity test to verify it fails**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py::test_manifest_config_and_recovery_store_paths_match -q
```

Expected: FAIL because `run_workflow(...)` does not yet accept an explicit
recovery store.

- [ ] **Step 3: Add `recovery_store` to `run_workflow(...)`**

Change `run_workflow(...)` to accept:

```python
recovery_store: Optional[StepRecoveryStore] = None
```

Rules:

- if `recovery_store` is supplied, use it
- if only `manifest_config` is supplied, create `YamlManifestRecoveryStore`
- if neither is supplied, use `NoManifestRecoveryStore`
- keep `manifest_config` for backward compatibility during the migration

- [ ] **Step 4: Move manifest skip/persist logic into the shared executor loop**

The loop should:

1. call `recovery_store.prepare(...)` once before step execution
2. call `recovery_store.before_step(...)` before validating/running each step
3. skip and replay outputs only when the decision says `skip_step=True`
4. run `scenario.run(...)` for live execution
5. call `recovery_store.after_step(...)` after live execution

Keep `run_manifested_steps(...)` as a wrapper that creates a
`YamlManifestRecoveryStore` and calls `run_workflow(...)`.

- [ ] **Step 5: Run the manifest/cache parity suite**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py \
  tests/test_workflow_invariants.py \
  tests/test_cache_hit_recovery.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pilates/workflows/orchestration.py pilates/workflows/recovery.py tests/test_manifest_cache_parity.py tests/test_workflow_invariants.py tests/test_cache_hit_recovery.py
git commit -m "refactor: run workflows through recovery store"
```

### Task 4: Add Stage-Level Recovery Policy

**Files:**

- Create: `pilates/workflows/recovery_policy.py`
- Modify: `pilates/config/models.py`
- Modify: `tests/test_architecture_guardrails.py`
- Modify: `tests/test_workflow_runtime_context.py`
- Modify: `tests/test_postprocessing_manifest_persistence.py`

- [ ] **Step 1: Add tests for policy defaults**

Add tests proving:

- default policy returns YAML recovery when a manifest path is supplied
- explicit postprocessing opt-out returns `NoManifestRecoveryStore`
- all other stages remain YAML-backed by default
- production stage modules are not allowed to add new direct `ManifestConfig`
  imports once they are migrated to the policy helper

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_architecture_guardrails.py \
  tests/test_workflow_runtime_context.py \
  tests/test_postprocessing_manifest_persistence.py -q
```

Expected: FAIL because the policy helper and settings option do not exist.

- [ ] **Step 3: Add a conservative settings field**

Add a small workflow settings surface with a default that preserves current
behavior. The field can be shaped as:

```yaml
workflow:
  manifests:
    disabled_stages: []
```

Use an empty list as the default. Avoid global `enabled: false` at first because
global disablement would make supply-demand too easy to break accidentally.

- [ ] **Step 4: Implement `recovery_store_for_stage(...)`**

The helper should accept:

- stage name
- manifest path
- settings
- state
- workspace

It should return:

- `NoManifestRecoveryStore()` when `stage_name` is listed in
  `workflow.manifests.disabled_stages`
- `YamlManifestRecoveryStore(ManifestConfig(path=...))` otherwise

- [ ] **Step 5: Run policy tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_architecture_guardrails.py \
  tests/test_workflow_runtime_context.py \
  tests/test_postprocessing_manifest_persistence.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pilates/workflows/recovery_policy.py pilates/config/models.py tests/test_architecture_guardrails.py tests/test_workflow_runtime_context.py tests/test_postprocessing_manifest_persistence.py
git commit -m "feat: add stage-level manifest recovery policy"
```

### Task 5: Migrate Postprocessing To The Policy Helper

**Files:**

- Modify: `pilates/workflows/stages/postprocessing.py`
- Modify: `tests/test_postprocessing_manifest_persistence.py`

- [ ] **Step 1: Add a postprocessing opt-out test**

Add a test that sets:

```yaml
workflow:
  manifests:
    disabled_stages:
      - postprocessing
```

Assert:

- `run_postprocessing_stage(...)` completes
- no `postprocessing_year_<year>.yaml` manifest is written
- postprocessing still executes the step
- archive queue flushing behavior is unchanged

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_postprocessing_manifest_persistence.py -q
```

Expected: FAIL because `postprocessing.py` still always builds `ManifestConfig`.

- [ ] **Step 3: Use `recovery_store_for_stage(...)` in postprocessing**

Replace direct `ManifestConfig(...)` use with the policy helper. Pass the
returned store to `run_workflow(..., recovery_store=...)`.

- [ ] **Step 4: Run postprocessing tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_postprocessing_manifest_persistence.py \
  tests/test_stage_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pilates/workflows/stages/postprocessing.py tests/test_postprocessing_manifest_persistence.py
git commit -m "refactor: make postprocessing manifests optional"
```

### Task 6: Migrate Land Use As A Separate Slice

**Files:**

- Modify: `pilates/workflows/stages/land_use.py`
- Modify: `tests/test_land_use_manifest_persistence.py`
- Modify: `tests/test_output_handoff_mapping.py`

- [ ] **Step 1: Add land-use opt-out tests**

Test both paths:

- default behavior still writes `land_use_year_<year>.yaml`
- opt-out behavior completes without writing the manifest

Assert downstream handoff values for `USIM_DATASTORE_BASE_H5`,
`USIM_DATASTORE_CURRENT_H5`, `USIM_FORECAST_OUTPUT`, and
`USIM_POPULATION_SOURCE_H5` stay equivalent.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_land_use_manifest_persistence.py \
  tests/test_output_handoff_mapping.py -q
```

Expected: FAIL for the opt-out path.

- [ ] **Step 3: Use `recovery_store_for_stage(...)` in land use**

Replace direct `ManifestConfig(...)` use with policy helper wiring. Keep the
manifest path builder because the YAML store still needs it.

- [ ] **Step 4: Run land-use and handoff tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_land_use_manifest_persistence.py \
  tests/test_output_handoff_mapping.py \
  tests/test_workflow_invariants.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pilates/workflows/stages/land_use.py tests/test_land_use_manifest_persistence.py tests/test_output_handoff_mapping.py
git commit -m "refactor: make land-use manifests optional"
```

### Task 7: Leave Supply-Demand YAML-Backed And Document The Blockers

**Files:**

- Modify: `pilates/workflows/recovery_policy.py`
- Modify: `docs-internal/manifest-recovery-isolation-plan.md`
- Create: `docs-internal/supply-demand-manifest-deletion-followup.md`
- Test: `tests/test_restart_stage_boundary_matrix.py`

- [ ] **Step 1: Add a guard test against accidental supply-demand opt-out**

Add a test that setting `workflow.manifests.disabled_stages` to include
`activity_demand` or `traffic_assignment` raises a clear configuration error.

Expected message:

```text
Supply-demand manifests cannot be disabled until restart run-id and ActivitySim/BEAM handoff recovery no longer read YAML manifests.
```

- [ ] **Step 2: Run the guard test to verify failure**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_restart_stage_boundary_matrix.py -q
```

Expected: FAIL because the guard does not exist.

- [ ] **Step 3: Implement the guard in recovery policy**

Reject supply-demand stage opt-out for:

- `supply_demand`
- `activity_demand`
- `traffic_assignment`
- `activity_demand_directly_from_land_use`

Keep the error message explicit so future workers understand the blocker.

- [ ] **Step 4: Write the follow-up note for removing the guard**

Create `docs-internal/supply-demand-manifest-deletion-followup.md` with:

- the Phase 0 spike verdict
- the exact recovery inputs still provided by YAML manifests
- the tests that must pass before the guard can be removed
- the code locations that should disappear when deletion is safe:
  - manifest globbing in `seed_supply_demand_parent_run_ids_for_resume(...)`
  - manifest output hydration in `_restore_activity_demand_outputs_for_resume(...)`
  - supply-demand entries in `.workflow/year_*_iteration_*.yaml`

- [ ] **Step 5: Run restart boundary tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_restart_stage_boundary_matrix.py \
  tests/test_stage_contracts.py \
  tests/test_manifest_cache_parity.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pilates/workflows/recovery_policy.py tests/test_restart_stage_boundary_matrix.py docs-internal/manifest-recovery-isolation-plan.md docs-internal/supply-demand-manifest-deletion-followup.md
git commit -m "test: guard supply-demand manifest recovery dependency"
```

### Task 8: Retire Direct `ManifestConfig` Stage Usage

**Files:**

- Modify: `pilates/workflows/stages/postprocessing.py`
- Modify: `pilates/workflows/stages/land_use.py`
- Modify: `tests/test_architecture_guardrails.py`

- [ ] **Step 1: Add an architecture guard for direct stage imports**

Add or update a guardrail test that fails if production stage modules import
`ManifestConfig` directly after they have been migrated to
`recovery_store_for_stage(...)`.

Allowed locations for `ManifestConfig` after this task:

- `pilates/workflows/recovery.py`
- `pilates/workflows/orchestration.py`
- unmigrated stage allowlist entries for supply-demand and vehicle-ownership
  code, until those stages have their own opt-out slices
- tests that explicitly cover backward compatibility

- [ ] **Step 2: Run the guard to verify failure**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_architecture_guardrails.py -q
```

Expected: FAIL until the migrated stage modules stop importing
`ManifestConfig`.

- [ ] **Step 3: Remove direct `ManifestConfig` imports from migrated stages**

Use `recovery_store_for_stage(...)` in migrated stages and pass
`recovery_store=...` to `run_workflow(...)`. Keep manifest path builder
functions where the YAML store still needs the path.

- [ ] **Step 4: Run stage and guardrail tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_architecture_guardrails.py \
  tests/test_postprocessing_manifest_persistence.py \
  tests/test_land_use_manifest_persistence.py \
  tests/test_stage_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pilates/workflows/stages/postprocessing.py pilates/workflows/stages/land_use.py tests/test_architecture_guardrails.py
git commit -m "refactor: route migrated stages through recovery policy"
```

### Task 9: Final Verification For Each PR

**Files:**

- No required code changes.

- [ ] **Step 1: Run targeted regression tests**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_manifest_cache_parity.py \
  tests/test_workflow_invariants.py \
  tests/test_cache_hit_recovery.py \
  tests/test_restart_stage_boundary_matrix.py \
  tests/test_stage_contracts.py \
  tests/test_postprocessing_manifest_persistence.py \
  tests/test_land_use_manifest_persistence.py -q
```

Expected: PASS.

- [ ] **Step 2: Run architecture guardrails**

Run:

```bash
rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_architecture_guardrails.py -q
```

Expected: PASS.

- [ ] **Step 3: Inspect diff for behavior creep**

Run:

```bash
rtk git diff --stat
rtk git diff -- pilates/workflows/orchestration.py pilates/workflows/recovery.py pilates/workflows/recovery_policy.py
```

Expected:

- first PR is mostly extraction and tests
- no artifact key renames
- no binding/catalog rewrites
- no archive worker changes
- no supply-demand opt-out
- no new direct `ManifestConfig` imports in migrated stage modules

## Stop Conditions

Stop and re-plan if any slice:

- changes `.workflow/*.yaml` schema before the YAML store is isolated
- changes supply-demand restart behavior before postprocessing opt-out is proven
- weakens typed output validation
- adds model-specific logic to the generic recovery store
- requires broad changes in binding, catalog, or archive worker code
- makes full production runs the only proof of correctness

## Deletion Criteria

YAML manifest deletion is only in scope after all of these are true:

1. No production stage constructs `YamlManifestRecoveryStore` by default.
2. `supply_demand_resume.py` no longer reads `year_*_iteration_*.yaml`.
3. Run-id restoration is sourced from Consist tracker data or another
   non-YAML recovery surface.
4. ActivitySim postprocess output hydration works without manifest output data.
5. Restart boundary tests pass with manifests disabled.
6. Golden/stub workflow tests pass with manifests disabled.
7. Archive promotion no longer needs `.workflow/*.yaml` for expected operator
   diagnostics.

Until then, manifests should be treated as an isolated compatibility backend,
not dead code.
