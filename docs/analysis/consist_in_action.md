---
title: Consist in Action
summary: Small, read-only walkthroughs for boundary-overlap, faceted outcome comparison, and archive inspection.
---

# Consist in Action

## Purpose

These examples are the shortest path to seeing what PILATES gets from Consist
without reading the runtime code first:

- boundary-overlap inspection
- faceted output comparison
- restart and archive inspection

All examples are read-only. They operate on archive run directories and
the attached Consist database under `.consist/`.

## 1. Boundary-Overlap Analysis For A Rerun

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

Use this to compare logical coverage between archives. It does not inspect
Consist admission records, so it cannot prove whether a run was a cache hit or
whether a cache outcome was portable.

## 2. Faceted Output Comparison

Measure ActivitySim trip mode share and compare over a facet:

```bash
python examples/consist/run_comparison.py \
  /path/to/archive-run \
  --table activitysim.trips \
  --compare pricing_policy \
  --baseline none \
  --where year=2030
```

What it shows:

- archive summary
- trip counts, total trips, and mode share by `trip_mode`
- either differences against a baseline facet value or ranks when no baseline is supplied

This is the quickest script-level entrypoint before moving into notebooks. It
works for two scenarios and for larger parameter sweeps because both are just
facet comparisons.

## 3. Restart Archive Inspection

Use the tracker-backed archive views to inspect an archived run's output keys,
hashes, and recovery roots before investigating a resume. The durable facts are
the Consist output links and the pinned BEAM successor closure, not a workspace
file selected by an inspection script.

For the supported checkpoint, confirm that the archived output links can
materialize to the current `beam_postprocess` resolver destinations. For all
other restarts, inspect the durable stage/year frontier and let normal native
step resolution choose current inputs.

## Where To Go Next

- Read [Opening Archives](opening_archives.md) for the tracker and mount model.
- Read [Faceted Comparison](scenario_comparison.md) for the beginner comparison surface.
- Read [Consist in PILATES](../workflow/consist_in_pilates.md) for the runtime ownership split.
