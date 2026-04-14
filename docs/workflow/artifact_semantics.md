---
title: Artifact Semantics
summary: Meaning of major workflow artifacts and why semantic names matter even when files overlap.
---

# Artifact Semantics

## Purpose

Explain the logical meaning of major handoff artifacts, not just their filenames or producers.

## Who This Is For

- Contributors reasoning about cross-step boundaries or renaming artifact keys.
- Analysts who need to understand what a published artifact actually means scientifically and operationally.

## This Page Answers

- Which artifact families represent distinct workflow roles rather than just distinct files?
- Why do some handoffs preserve multiple semantic names for related datastore or skim artifacts?
- How should developers decide whether a file belongs in the public artifact contract?

## Adjacent Pages

- Read [Artifact Flow](artifact_flow.md) for the short boundary map.
- Pair this with [Step Contracts](step_contracts.md) and [Artifact Facet Catalog](artifact_facet_catalog.md).
- Use [Lineage Map](lineage_map.md) for the per-step reference.

## Source Material To Mine

- Current artifact keys and coupler schema.
- Internal contract-centralization and wiring audit notes.
- Existing workflow step publication logic.
