---
title: Adding a Model
summary: Native Consist integration path for a new PILATES model.
---

# Adding a Model

Start with `scripts/new_model_scaffold.py`. It creates model adapter shells,
typed output classes, a decorated callable for each phase, resolvers,
projectors, `StepDefinition` objects, catalog policy entries, and explicit
stage-call snippets.

## Implementation order

1. Define stable semantic input and output roles.
2. Define typed outputs for the persisted files downstream code may trust.
3. Implement a decorated native callable for preprocess, run, or postprocess.
4. Implement one resolver that selects roles once and gives each selected input
   an exact materialization destination.
5. Implement one projector that turns `RunResult.outputs` into the typed
   output object and validates current destinations.
6. Export the `StepDefinition` and add a policy-only catalog entry.
7. Add the explicit `execute_step(...)` call in the owning stage.
8. Add focused contract, stage, and restart tests as applicable.

## Ownership rules

- Model adapters prepare files, run model code, and interpret local outputs.
- Consist owns identity, input materialization, cache admission, and persisted
  output lineage.
- PILATES owns semantic role selection, typed output validation, stage order,
  and restart policy.
- Stages sequence definitions; they do not rediscover inputs or reconstruct
  completed results.

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
