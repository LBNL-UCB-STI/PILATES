---
title: Step Contracts
summary: Semantic workflow contract layer built from step specs, typed outputs, and artifact keys.
---

# Step Contracts

## Purpose

Document the public workflow contract surface that keeps producers, consumers, and replay behavior aligned.

## Who This Is For

- Contributors adding or changing tracked workflow steps.
- Readers who need the semantic contract model before editing step factories or stage bindings.

## This Page Answers

- What does `WorkflowStepSpec` define and what does it intentionally not define?
- How do typed outputs, artifact keys, input keys, and output keys work together?
- Which contract surfaces are semantic, which are runtime-binding details, and which are model-local?

## Adjacent Pages

- Read [Stages and Steps](stages_and_steps.md) first.
- Continue to [Model Integration Guide](../extend/model_integration_guide.md) and [Output Validation](../extend/output_validation.md).
- Pair this with [Artifact Semantics](artifact_semantics.md).

## Source Material To Mine

- `pilates/workflows/catalog.py`
- `pilates/workflows/outputs_base.py`
- `pilates/workflows/artifact_keys.py`
- current validation and standard-step wiring
