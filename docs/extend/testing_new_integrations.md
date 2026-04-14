---
title: Testing New Integrations
summary: Tests and acceptance criteria for new workflow and model integration work.
---

# Testing New Integrations

## Purpose

Define the expected verification path for new tracked workflow or model-integration work.

## Who This Is For

- Contributors preparing integration changes for review.
- Maintainers enforcing a consistent definition of done for workflow-facing edits.

## This Page Answers

- Which unit, contract, and workflow tests should new integration work usually add?
- What should be checked around replay, validation, and artifact publication?
- How should contributors think about smoke tests versus full regression coverage?

## Adjacent Pages

- Pair this with [Adding a Model](adding_a_model.md).
- Use [Output Validation](output_validation.md) for contract-specific checks.
- Use [Operations Overview](../operations/overview.md) for preserved test-output workflows.

## Source Material To Mine

- Current `tests/` layout and targeted workflow tests.
- Existing test-output preservation utility docs.
