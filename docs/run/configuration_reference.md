---
title: Configuration Reference
summary: Reference for the current nested PILATES settings model and its workflow effects.
---

# Configuration Reference

## Purpose

Act as the durable reference for the active nested settings schema.

## Who This Is For

- Users editing scenario templates into runnable local or HPC configs.
- Contributors who need to know which config fields affect workflow behavior, cache identity, and model enablement.

## This Page Answers

- What is the current high-level config shape?
- Which fields belong under `run`, `shared`, `infrastructure`, and model-specific sections?
- Which settings change scientific behavior versus cache identity or runtime posture?

## Adjacent Pages

- Start with [Configuration Basics](../start-here/configuration_basics.md) if you are new.
- Pair this with [CLI](cli.md) for invocation behavior.
- Use [Restart and Resume](restart_and_resume.md) for replay-specific settings.

## Source Material To Mine

- Active nested settings files at the repo root and under `scenarios/`.
- Config loading and validation code in `pilates.config.models`.
- Current runtime identity and facet handling in `pilates/utils/consist_config.py`.
