---
title: Model Contract Checklist
summary: Checklist for a native Consist model boundary in PILATES.
---

# Model Contract Checklist

## Semantic roles

- [ ] Name every workflow-visible input and output role.
- [ ] Declare static roles and schema outputs on the decorated callable.
- [ ] Keep conditional selections and dynamic outputs in the resolver or output
  path provider.
- [ ] Do not treat archive locations as new semantic roles.

## Native definition

- [ ] Define a typed `StepOutputsBase` subclass for persisted public outputs.
- [ ] Implement a resolver that selects each role once into concrete Consist
  inputs and deterministic destinations.
- [ ] Implement a decorated callable that runs only model-local work.
- [ ] Implement a projector that validates and types `RunResult.outputs` at
  current declared destinations.
- [ ] Construct and export one `StepDefinition` per model phase.

## Catalog and stage

- [ ] Add a policy-only `WorkflowStepSpec`: name, phase, stage, order,
  enablement, dependency, and dynamic semantic families.
- [ ] Register the definition in `STEP_DEFINITIONS`.
- [ ] Sequence it with an explicit `execute_step(...)` call in the owning
  stage.
- [ ] Keep source precedence in the resolver and stage order in the stage.

## Cache and restart

- [ ] Add Consist metadata/config-adapter policy only when it affects identity.
- [ ] Require cache hits to hydrate requested output destinations before reuse.
- [ ] Keep restart policy in PILATES; do not invent a second mid-stage recovery mechanism.
- [ ] Preserve the sole BEAM committed boundary and its postprocess mutation
  gate when touching supply-demand behavior.

## Tests

- [ ] Add native definition and projector tests.
- [ ] Add a stage sequence test and semantic-key contract coverage.
- [ ] Add restart coverage only if the change touches a durable boundary.
- [ ] Run the architecture guard and focused catalog/definition tests.

## Adjacent pages

- [Adding a Model](adding_a_model.md)
- [Model Integration Guide](model_integration_guide.md)
- [Step Contracts](../workflow/step_contracts.md)
