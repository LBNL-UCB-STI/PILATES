---
title: Configuration Basics
summary: Mental model for how PILATES config is organized before the full reference.
---

# Configuration Basics

## Purpose

Explain how to think about PILATES config before dropping into the full field-by-field reference.

## Who This Is For

- Users editing a copied scenario template for the first time.
- Contributors who need the high-level config shape but not every field yet.

## This Page Answers

- What belongs under `run`, `shared`, `infrastructure`, and model sections?
- Which settings usually need local edits first?
- Which config choices affect runtime posture versus model behavior?

## Adjacent Pages

- Read [First Run Walkthrough](first_run_walkthrough.md) for the concrete path.
- Use [Configuration Reference](../run/configuration_reference.md) for full field coverage.
- Use [Scenario Lifecycle](../run/scenario_lifecycle.md) to connect config to runtime behavior.

## Source Material To Mine

- Active nested settings files in the repo root and `scenarios/`.
- Current config-validation code and runtime loader behavior.
