---
title: Model Integration Guide
summary: Architecture-oriented guide to how model components plug into the PILATES workflow.
---

# Model Integration Guide

## Purpose

Describe the current integration architecture before readers start adding or refactoring model families.

## Who This Is For

- Contributors modifying tracked workflow integrations.
- Readers who need to understand where model packages stop and workflow code begins.

## This Page Answers

- What are the main integration layers between model code and workflow orchestration?
- What belongs in model packages, step modules, stage modules, and the workflow catalog?
- How do typed outputs, the coupler, and replay-aware contracts fit into new integrations?

## Reading Path

- Read [Workflow Primer](../workflow/workflow_primer.md) and [Step Contracts](../workflow/step_contracts.md) first.
- Continue to [Adding a Model](adding_a_model.md) for the hands-on path.
- Use [Output Validation](output_validation.md) for runtime and startup guardrails.

## Source Material To Mine

- Current workflow step modules under `pilates/workflows/steps/`.
- Contract surfaces in `pilates/workflows/catalog.py` and `pilates/workflows/outputs_base.py`.
- Internal model-wiring simplification audit notes.
