---
title: Output Validation
summary: Startup contract validation and runtime output validation for workflow boundaries.
---

# Output Validation

## Purpose

Explain the two validation layers that protect the public workflow contract.

## Who This Is For

- Contributors changing step contracts or typed outputs.
- Reviewers checking whether new integration work is guarded against wiring drift and semantic output errors.

## This Page Answers

- What fails at startup versus what fails after a step runs?
- When should a contributor add contract validation versus runtime output validators?
- Which validation surface should be treated as public workflow API protection?

## Adjacent Pages

- Read [Step Contracts](../workflow/step_contracts.md) first.
- Pair this with [Adding a Model](adding_a_model.md).
- Use [Testing New Integrations](testing_new_integrations.md) for the acceptance path.

## Source Material To Mine

- Existing validation backbone patterns.
- `tests/test_output_validation_backbone.py`.
- Current startup contract checks in workflow step wiring.
