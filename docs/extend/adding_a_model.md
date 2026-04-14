---
title: Adding a Model
summary: Hands-on checklist for wiring a new model or model phase into PILATES.
---

# Adding a Model

## Purpose

Provide the practical implementation path for adding a new model family or phase to the workflow.

## Who This Is For

- Contributors building a new integration.
- Maintainers standardizing how new tracked steps should be added after the refactor.

## This Page Answers

- What files do contributors usually touch when adding a model?
- What is the recommended order for defining outputs, catalog metadata, step wiring, and stage ownership?
- Which parts of the process are required for tracked, replay-aware integrations?

## Reading Path

- Read [Model Integration Guide](model_integration_guide.md) first.
- Pair this with [Model Contract Checklist](model_contract_checklist.md).
- Finish with [Testing New Integrations](testing_new_integrations.md).

## Source Material To Mine

- Existing tracked-model implementations.
- The prior adding-a-model guide.
- Current workflow catalog, holder, and step-builder patterns.
