---
title: Stages and Steps
summary: Execution model for years, stages, substages, and workflow steps in PILATES.
---

# Stages and Steps

## Purpose

Make the execution model explicit: what a stage is, what a step is, and how year and iteration state flows through them.

## Who This Is For

- Contributors reading stage or step modules for the first time.
- Users who understand the run lifecycle but need the workflow execution vocabulary.

## This Page Answers

- What are the major stages and inner-loop substages?
- What does a stage own versus what does a step own?
- How do year, iteration, and sub-iteration affect execution and artifact meaning?

## Adjacent Pages

- Read [Workflow Primer](workflow_primer.md) first.
- Then read [Simulation Logic by Stage](simulation_logic_by_stage.md) for the science-facing meaning.
- Read [Step Contracts](step_contracts.md) for the semantic workflow boundary.

## Source Material To Mine

- `workflow_state.py`
- `pilates/workflows/stages/*.py`
- `pilates/workflows/orchestration.py`
