---
title: Adding a Model
summary: Hands-on checklist for wiring a new model or model phase into PILATES.
---

# Adding a Model

## Reading Path

- Read [Model Integration Guide](model_integration_guide.md) first.
- Pair this with [Model Contract Checklist](model_contract_checklist.md).
- Finish with [Testing New Integrations](testing_new_integrations.md).

## Practical Order

For most contributors, the least confusing order is:

1. define the typed outputs and public workflow keys
2. add or update the catalog entry
3. wire the step factory and holder publication
4. connect the stage that will call the step
5. add validation and tests before broadening the docs
