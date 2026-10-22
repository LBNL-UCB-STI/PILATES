---
title: Adding a Model
summary: Native Consist integration path for a new PILATES model.
---

# Adding a Model

`scripts/new_model_scaffold.py` is a mechanical native starting aid, not a
turnkey model integration. It generates the required native-contract surfaces
with conservative incomplete adapter contracts; establish every real semantic
role, dependency, and completion status in the [Model Contract
Checklist](model_contract_checklist.md) before relying on an integration.

## Implementation order

1. Define stable semantic input and output roles.
2. Declare an `InputContract` as a first-class part of the step definition.
   Its optional `config_contract` field must be set for `complete`, which
   requires demonstrated portable identity closure; it may be absent for
   `incomplete`, which records the remaining status and evidence gap rather
   than promoting a cache entry.
3. Define typed outputs for the persisted files downstream code may trust.
4. Implement a decorated native callable for preprocess, run, or postprocess.
5. Implement one resolver that selects roles once and gives each selected input
   an exact materialization destination.
6. Implement one projector that turns `RunResult.outputs` into the typed
   output object and validates current destinations.
7. Export the `StepDefinition` and add a policy-only catalog entry.
8. Add the explicit `execute_step(...)` call in the owning stage.
9. Add focused contract, stage, and restart tests as applicable.

## Touch Map (What You Edit vs. What You Leave Alone)

### Files you create or edit (`pilates/<model>/` + 3–4 workflow files)

- **Model adapter package (`pilates/<model>/`)**: `preprocessor.py`, `runner.py`,
  `postprocessor.py`, and `outputs.py` (scaffolded by
  `scripts/new_model_scaffold.py`).
- **Native step module (`pilates/workflows/steps/<model>.py`)**: decorated
  Consist callables, semantic input resolver (`ResolvedStepInputs`), typed
  output projector, `InputContract`, and `StepDefinition` exports.
- **Step surface & catalog (`pilates/workflows/steps/__init__.py`, `pilates/workflows/catalog.py`, `pilates/generic/model_factory.py`)**:
  export the new `StepDefinition` objects in `STEP_DEFINITIONS`, add policy-only
  `WorkflowStepSpec` entries, and register the model components in
  `ModelFactory`.
- **Owning stage module (`pilates/workflows/stages/<stage>.py`)**: sequence the
  new `StepDefinition` objects with explicit `execute_step(...)` calls.
- **Semantic roles & config schema (when introducing new cross-step roles or settings)**:
  `pilates/workflows/artifact_keys.py`, `pilates/workflows/coupler_schema.py`,
  and `pilates/config/models.py`.

### Infrastructure you do NOT edit

- `pilates/workflows/step_execution.py` (central `execute_step` bridge, cache
  options, action refresh, and output archival/republication)
- `pilates/workflows/step_definition.py` and `pilates/workflows/resolved_inputs.py`
- `pilates/runtime/` (orchestrator, runtime context, archive/hydration helpers)
- `pilates/workspace.py` and `workflow_state.py`
- Unrelated model packages (`pilates/urbansim/`, `pilates/atlas/`,
  `pilates/activitysim/`, `pilates/beam/`) and unrelated stage modules

## Ownership rules

- Model adapters prepare files, run model code, and interpret local outputs.
- Consist owns identity, input materialization, cache admission, and persisted
  output lineage.
- PILATES owns semantic role selection, typed output validation, stage order,
  and restart policy.
- Stages sequence definitions; they do not rediscover inputs or reconstruct
  completed results.
- The central `execute_step` bridge owns the post-scenario action refresh,
  configured output archive, and republishing of archived artifacts to the
  coupler before typed projection. Model integrations must not recreate that
  sequence.

Use the coupler only as the current semantic-role map. Introduce a new role
only when a workflow boundary needs it; an archive location is not a role.

## Restart-sensitive work

Do not add a generic mid-stage recovery path for an interrupted model. The sole committed
mid-stage restart is `beam_run_completed -> beam_postprocess`, which restores
the exact pinned successor closure and obeys its mutation gate.

## Validation

Run the contract and architecture suites named in
`docs/workflow/step_contracts.md`, then a focused stage test. A production run
is useful after the native role, projector, and stage contracts are proven.

## Adjacent pages

- Read [Model Integration Guide](model_integration_guide.md).
- Use [Model Contract Checklist](model_contract_checklist.md).
- Review [Step Contracts](../workflow/step_contracts.md).
