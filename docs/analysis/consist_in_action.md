---
title: Consist in Action
summary: Small, read-only walkthroughs for cache-hit inspection, run comparison, and replay-aware archive inspection.
---

# Consist in Action

## Purpose

These examples are the shortest path to seeing what PILATES gets from Consist
without reading the runtime code first:

- cache-aware rerun inspection
- run or scenario comparison
- replay/restart inspection against archived outputs

All three scripts are read-only. They operate on archive run directories and
the attached Consist database under `.consist/`.

## 1. Cache-Hit Inspection For A Rerun

Compare the logical year/iteration/model boundaries present in a baseline
archive and a rerun archive:

```bash
python examples/consist/cache_hit_inspection.py \
  /path/to/baseline-archive \
  /path/to/rerun-archive \
  --scenario-id baseline
```

What it shows:

- archive-level run summaries for each archive
- logical boundary overlap on `scenario_id`, `year`, `iteration`, and `model`
- which boundaries appear in both archives versus only one side

Use this when you want a fast answer to "what reran and what stayed logically
the same?"

## 2. Scenario Or Run Comparison

Compare two scenarios inside one archive:

```bash
python examples/consist/run_comparison.py \
  /path/to/archive-run \
  --left-scenario baseline \
  --right-scenario pricing_policy \
  --year 2030 \
  --converged
```

Or compare explicit run ids:

```bash
python examples/consist/run_comparison.py \
  /path/to/archive-run \
  --left-run-id RUN_A \
  --right-run-id RUN_B
```

What it shows:

- comparison alignment summary
- dataset summary rows for the compared outputs

This is the quickest script-level entrypoint before dropping into notebooks or
the full analysis CLI.

## 3. Replay / Restart Archive Inspection

Inspect one archived run's output keys, hashes, and recovery roots:

```bash
python examples/consist/restart_replay_inspection.py \
  /path/to/archive-run \
  --scenario-id baseline \
  --model beam
```

Or inspect one explicit run id:

```bash
python examples/consist/restart_replay_inspection.py \
  /path/to/archive-run \
  --run-id RUN_ID
```

What it shows:

- archive summary
- selected run id
- archived output keys for that run
- output hashes when available
- recovery-root counts and paths

Use this when you want to sanity-check whether the archive has the material
that replay/restart tooling can recover from.

## Where To Go Next

- Read [Opening Archives](opening_archives.md) for the tracker and mount model.
- Read [Scenario Comparison](scenario_comparison.md) for the higher-level analysis surface.
- Read [Consist in PILATES](../workflow/consist_in_pilates.md) for the runtime ownership split.
