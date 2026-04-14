---
title: Architecture
summary: Stable layer map for the current PILATES runtime and where responsibilities live.
---

# Architecture

## Purpose

Describe the stable architectural slices of the current codebase without collapsing into implementation detail.

## Who This Is For

- Contributors locating responsibility boundaries before editing code.
- Readers who want the system map before digging into step contracts or model integration.

## This Page Answers

- What are the main runtime layers and what does each one own?
- Where do configuration, workspace, workflow orchestration, model adapters, and contracts meet?
- Which docs should a reader open next for running, extending, or analyzing PILATES?

## Reading Path

- Start here, then read [Workflow Primer](workflow_primer.md).
- Continue to [Stages and Steps](stages_and_steps.md) for execution units.
- For developer work, continue to [Model Integration Guide](../extend/model_integration_guide.md).

## Source Material To Mine

- `run.py` and `pilates/runtime/launcher.py`
- `workflow_state.py` and `pilates/workspace.py`
- `pilates/workflows/` and model packages under `pilates/`
