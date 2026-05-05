# Notebook-First Analysis Plan

## Purpose

This document describes the target architecture and delivery plan for evolving
`analysis/` from a Consist-oriented analysis scaffold into a notebook-first
analysis suite for archived PILATES runs.

The intended audience is an implementation agent or engineer working in this
repository. The goal is not to prescribe every method signature up front, but
to make the desired UX, architectural boundaries, sequencing, and acceptance
criteria explicit enough that implementation decisions can be made locally
without losing the overall direction.

## Problem Statement

The current `analysis/` package contains useful lower-level primitives, but the
startup cost for real use is still too high.

Today a user typically has to understand:

- how to point a Consist tracker at an archived run directory,
- how `workspace://` and `inputs://` mounts are remapped for analysis,
- how run metadata is recovered across historical archives,
- which helpers return DataFrames vs SQL view names vs exported files,
- artifact family naming conventions,
- and which package features are designed for CLI/export workflows rather than
  interactive exploratory analysis.

This is the wrong shape for the primary use case:

- running Jupyter notebooks on an HPC node,
- opening an archived PILATES run tree directly from shared scratch storage,
- finding runs and epochs of interest,
- loading standard output tables into pandas,
- and comparing groups of runs with minimal boilerplate.

The package should make the common notebook path obvious and easy, while still
preserving access to lower-level Consist and SQL surfaces for advanced use.

## Primary Users and Workflows

### Primary user

A researcher or developer working in a Jupyter notebook, usually on an HPC node
with access to archived PILATES outputs and the associated Consist DuckDB.

### Primary workflows

1. Open an archive and understand what is in it.
2. Find scenarios, years, iterations, and available model outputs.
3. Resolve a converged epoch for a scenario/year.
4. Load common tables from that epoch as DataFrames.
5. Compare two scenario groups or two scenario epochs.
6. Drop down to SQL, raw tracker access, or artifact-level inspection only when
   necessary.

### Secondary workflows

- DB health and run-tagging inspection.
- ad hoc artifact ingest/export and handoff packaging.
- CLI usage for scripted workflows.

Secondary workflows are still important, but they should no longer define the
top-level mental model of the package.

## Current State Summary

The package already has several strong foundations.

### Existing strengths

- `AnalysisSession` in `analysis/src/pilates_consist_analysis/api.py`
  - solid wrapper around tracker creation and basic health/tagging inspection.
- `RunSet` in `analysis/src/pilates_consist_analysis/runset.py`
  - useful grouping, filtering, latest selection, alignment, and converged
    selection.
- `EpochPanel` and `SimulationEpoch` in
  `analysis/src/pilates_consist_analysis/epochs.py`
  - good abstraction for thinking in terms of scenario/year/iteration rather
    than raw run IDs.
- `EpochViews` in `analysis/src/pilates_consist_analysis/epoch_views.py`
  - valuable mechanism for constructing epoch-scoped logical data access.
- `scenario_compare.py`
  - real start on group-to-group comparisons.
- `activitysim_trips.py`, `datasets.py`, and `skim_analysis.py`
  - real analysis builders already exist.

### Current UX/DX problems

- The main entry point is not notebook-first.
  - `open_run()` opens an archive analysis session rather than a specific run.
  - this is misleading and creates an avoidable naming trap alongside
    `AnalysisSession.open_run(run_id)`.
- The bundled notebook is oriented around handoff/export/ingest rather than
  everyday exploratory analysis.
- Common notebook questions are not first-class:
  - "what scenarios are here?"
  - "what years exist for this scenario?"
  - "what is the converged epoch for 2030?"
  - "load the trips table for that epoch"
  - "compare baseline vs policy for that year"
- `EpochViews` returns SQL view names instead of DataFrames.
  - this is powerful, but too low-level for the default user path.
- Metadata normalization logic is spread across multiple modules.
  - similar alias recovery exists in `runtime.py`, `runset.py`, `epochs.py`,
    and some dataset builders.
- Some analysis families still reach back into PILATES internals.
  - for example linkstats analysis depends on
    `pilates.utils.consist_analysis`, which weakens the promise that
    `analysis/` is a clean standalone package.

## Design Goals

### Primary goals

- Make notebook use the default experience.
- Minimize startup cost for opening and understanding an archive.
- Make scenario/year/iteration the primary selectors.
- Return DataFrames by default for common data access.
- Preserve lower-level escape hatches for SQL, tracker internals, and artifact
  inspection.
- Build on the existing architecture rather than replacing it.

### Non-goals

- Rewriting the full analysis package from scratch.
- Removing the CLI.
- Fully decoupling `analysis/` from PILATES internals in the first phase.
- Solving portable/exported-bundle mode comprehensively before the notebook
  attached-archive path is strong.

## Architectural Principles

### 1. Notebook-first, not CLI-first

The top-level public API should be optimized for import and interactive use in
Jupyter. The CLI should wrap or reuse those notebook-first abstractions where it
is helpful, but not drive their design.

### 2. Progressive disclosure

The package should expose three layers:

- high-level notebook objects for common work,
- standard table/comparison helpers for typical analyses,
- lower-level raw SQL, tracker, and artifact escape hatches.

Users should not need to start at the lowest layer.

### 3. Canonical metadata normalization

Historical archives are imperfect. The package should absorb that complexity
once and expose a canonical run catalog downstream.

### 4. Epoch-centric analysis

For most PILATES use, the correct unit is not "a run" but "a scenario/year/
outer-iteration grouping with model runs attached." The notebook API should
make epoch selection and comparison feel natural.

### 5. Keep attached-archive analysis as the mainline

The current PILATES/Consist runtime contract already supports opening archived
run trees directly. That should remain the primary path until portable mode is
materially stronger.

## Target Public Mental Model

The desired notebook mental model is:

1. Open an archive.
2. Inspect its contents.
3. Select a scenario.
4. Select an epoch.
5. Load tables as DataFrames.
6. Compare scenarios through a first-class comparison object.

Illustrative usage:

```python
from pilates_consist_analysis import open_archive

archive = open_archive("/path/to/archive")

archive.summary()
archive.scenarios()
archive.runs().head()

baseline = archive.scenario("baseline")
policy = archive.scenario("policy")

epoch = baseline.epoch(year=2030, converged=True)
trips = epoch.tables.trips()
persons = epoch.tables.persons(columns=["person_id", "household_id"])

cmp = archive.compare("baseline", "policy", year=2030, converged=True)
cmp.mode_shares()
cmp.linkstats_summary()
cmp.config_diff()
```

This flow should be easier and more obvious than any equivalent SQL-first or
artifact-first route.

## Target Object Model

The top-level notebook API should be organized around the following nouns.

### `Archive`

Represents one archived PILATES run directory and its attached Consist DB.

Responsibilities:

- open the archive safely,
- resolve the DB path,
- create the analysis tracker,
- expose health and tagging diagnostics,
- build and cache a canonical run index,
- provide scenario and epoch lookup,
- provide comparison entry points.

`Archive` should be the preferred top-level entry object. It is expected to wrap
existing `AnalysisSession` logic rather than replace it.

### `RunIndex`

Represents a canonical, cached, analysis-friendly run catalog for one archive.

Responsibilities:

- gather runs from the tracker,
- normalize key metadata fields,
- preserve provenance on where normalized values came from,
- support notebook-friendly tabular inspection,
- serve as the source of truth for scenario/year/model discovery.

This is the main architectural cleanup step because it centralizes metadata
recovery that is currently duplicated in several places.

### `Scenario`

Represents one logical scenario slice within an archive.

Responsibilities:

- provide scenario-scoped run and epoch discovery,
- provide year listing and simple summary methods,
- provide scenario-scoped comparison hooks when appropriate.

`Scenario` should feel like a convenient slice over the archive, not an
independent tracker wrapper.

### `Epoch`

Represents one scenario/year/outer-iteration grouping and its model runs.

Responsibilities:

- expose the selected epoch metadata,
- expose run IDs by model,
- provide a concise summary,
- expose a `tables` helper for DataFrame-first access,
- expose SQL access as an escape hatch.

This should wrap existing `SimulationEpoch` and `EpochViews`.

### `EpochTables`

Represents common table loaders for a selected epoch.

Responsibilities:

- return DataFrames for common logical outputs,
- hide SQL view creation details,
- optionally accept light column/row selection arguments,
- preserve a low-friction transition to raw SQL if the user needs it.

### `Comparison`

Represents a notebook-friendly comparison between two scenario slices or two
run collections.

Responsibilities:

- preserve alignment policy,
- preserve selected run sets or epochs,
- expose standard summary methods,
- expose config diffs,
- support higher-level opinionated comparison helpers.

This should build on existing scenario comparison machinery rather than replace
it.

## Target Module Layout

The current code should largely stay in place. Add a notebook-focused layer on
top.

Recommended additions:

- `analysis/src/pilates_consist_analysis/archive.py`
  - `open_archive`
  - `Archive`
- `analysis/src/pilates_consist_analysis/run_index.py`
  - canonical run index construction and caching
- `analysis/src/pilates_consist_analysis/scenario.py`
  - `Scenario`
- `analysis/src/pilates_consist_analysis/epoch_api.py`
  - `Epoch`
  - `EpochTables`
- `analysis/src/pilates_consist_analysis/comparison_api.py`
  - `Comparison`

The existing modules should remain the implementation substrate:

- `api.py`
- `runtime.py`
- `runset.py`
- `epochs.py`
- `epoch_views.py`
- `scenario_compare.py`
- `handoff.py`

This avoids disruptive refactoring while still clarifying the public API shape.

## Canonical Run Index Requirements

`RunIndex` is the most important enabling abstraction. It should centralize
field normalization and expose one canonical DataFrame.

### Required canonical fields

At minimum:

- `run_id`
- `parent_run_id`
- `scenario_id`
- `year`
- `iteration`
- `model`
- `status`
- `seed`
- `name`
- `created_at`
- `ended_at`

### Strongly recommended convenience fields

- `is_complete`
- `is_completed_status`
- `is_converged_candidate`
- `has_parent`
- `archive_run_dir`

### Strongly recommended debugging/source fields

Preserve where normalized values came from so debugging historical archives is
not opaque.

Examples:

- `scenario_id_source`
- `year_source`
- `iteration_source`
- `model_source`

Possible values might be:

- `run_attr`
- `metadata`
- `metadata.facet`
- `fallback`
- `missing`

### Required behavior

- Use the existing alias logic as the baseline.
- Do not require all historical archives to be fully normalized before the
  index is useful.
- Keep the index cheap to inspect in notebooks.
- Cache it on the `Archive` object once built.

### Important implementation note

It is acceptable for `RunIndex` to reuse helper logic from `runset.py`,
`epochs.py`, and `runtime.py` initially. It is preferable to consolidate that
logic into shared helpers as part of implementation rather than leaving the new
layer to duplicate it yet again.

## Notebook API Requirements

### `open_archive()` should become the preferred entry point

This is a user-facing naming fix and a conceptual fix.

Required behavior:

- opens the archive analysis context,
- returns an `Archive`,
- uses the same archive/db resolution behavior as the current session layer,
- preserves existing optional controls for power users,
- is documented as the preferred notebook entry point.

### `open_run()` should remain temporarily

Do not break existing users immediately.

Recommended approach:

- keep `open_run()` working as a compatibility alias,
- stop using it in new docs and notebooks,
- optionally add a light deprecation note in docstrings or comments later.

### `Archive` notebook helpers

Minimum useful methods:

- `summary()`
- `issues()`
- `runs()`
- `scenarios()`
- `years()`
- `epochs()`
- `epoch(...)`
- `scenario(scenario_id)`
- `compare(...)`

These do not all need to be implemented with extensive custom logic. Thin
wrappers over the existing session and epoch machinery are acceptable as long as
the notebook UX is clear.

### `Scenario` notebook helpers

Minimum useful methods:

- `summary()`
- `runs()`
- `years()`
- `epochs()`
- `epoch(year=..., converged=...)`

### `Epoch` notebook helpers

Minimum useful methods:

- `summary()`
- `run_ids()`
- `sql(...)`
- `tables`

### `EpochTables` notebook helpers

Minimum initial table loaders:

- `trips()`
- `persons()`
- `households()`
- `land_use()`
- `linkstats()`
- `skim_summary()`

These should return DataFrames. They may internally rely on `EpochViews`, but
that should not be the user-visible default.

## Comparison API Requirements

The current `ScenarioComparison` surface is useful but still too close to the
dataset-builder layer.

The notebook-facing `Comparison` object should emphasize:

- what is being compared,
- how it is aligned,
- which epochs or run groups are included,
- and how to retrieve common summaries quickly.

### Initial comparison helpers

The first comparison helpers should be small and opinionated:

- `summary()`
- `config_diff()`
- `mode_shares()`
- `trip_purposes()`
- `linkstats_summary()`

This is intentionally smaller than the full current analysis capability. The
goal is to make common comparisons obvious and easy, not to expose every
existing builder immediately.

### Alignment policy

The comparison object should preserve, surface, and document:

- the left and right selection,
- the alignment key,
- whether converged selection was used,
- and any warnings about missing or incomplete epoch candidates.

Those concepts already exist in `scenario_compare.py`; the notebook layer
should preserve them rather than hide them.

## HPC and Runtime Assumptions

Implementation must continue to respect the current PILATES/Consist runtime
contract.

### Required assumptions

- archived run directories remain the root unit of attached analysis,
- `workspace://` should resolve against the archived run directory for analysis,
- `inputs://` should resolve against the project root used for analysis tracker
  creation,
- the archive should typically contain or resolve a usable `.consist` DB,
- analysis should not depend on the original execution `cwd`.

### Important constraint

The analysis notebook path should assume attached archive + attached DB as the
mainline. Portable mode can continue to exist, but it should not distort the
notebook API design in early phases.

## Delivery Phases

### Phase 1: Notebook Entry Layer

Goal:

Make archive opening and archive understanding easy without forcing users into
`RunSet` or low-level session concepts.

Recommended scope:

- add `open_archive()`,
- add `Archive`,
- add `RunIndex`,
- expose archive summary/discovery helpers,
- update documentation and examples to use `open_archive()`,
- add a notebook starter focused on archive exploration.

Acceptance criteria:

- a user can open an archive in one line,
- a user can list scenarios and years without touching `RunSet`,
- a user can inspect a canonical runs DataFrame,
- existing `open_run()` behavior still works.

### Phase 2: Epoch DataFrame Access

Goal:

Make epoch selection and common table loading easy in notebooks.

Recommended scope:

- add `Scenario`,
- add `Epoch`,
- add `EpochTables`,
- wrap `EpochViews` so DataFrames are the default for common access,
- preserve raw SQL/view-name escape hatches.

Acceptance criteria:

- a user can resolve a converged epoch with a scenario/year call,
- a user can load trips/persons/linkstats/skims summary as DataFrames,
- a user does not need to write SQL for routine table access.

### Phase 3: Notebook Comparison Layer

Goal:

Make common baseline vs policy comparison workflows first-class.

Recommended scope:

- add `Comparison`,
- wrap existing scenario compare machinery,
- add a small set of opinionated comparison helpers,
- ensure alignment/warning information is easy to inspect.

Acceptance criteria:

- a user can compare two scenarios in one or two lines,
- a user can retrieve at least one ASim-focused summary and one BEAM-focused
  summary,
- config diffs remain accessible from the notebook layer.

### Phase 4: Discovery and Polishing

Goal:

Reduce archive-pointing friction further and improve package coherence.

Recommended scope:

- add optional archive discovery helpers for HPC scratch trees,
- add more notebook examples,
- reduce direct dependency on PILATES internals where practical,
- revisit CLI wrappers after the notebook API stabilizes.

Acceptance criteria:

- common archive-discovery tasks are easier,
- notebook examples are clearly the primary documentation path,
- the analysis package boundary is cleaner than it is today.

## Recommended First PR

The first PR should be intentionally narrow. It should establish the new public
shape without forcing a large refactor.

### Suggested first PR contents

- add `archive.py`,
- add `run_index.py`,
- expose `open_archive()` in `__init__.py`,
- add `Archive` with summary/discovery helpers,
- add a cached canonical run index,
- update docs and examples to lead with `open_archive()`,
- add a new notebook starter focused on archive exploration,
- leave existing session, CLI, and handoff code intact.

### Suggested first PR non-goals

- no major CLI redesign,
- no full comparison layer yet,
- no full portable-mode overhaul,
- no broad extraction of the package into a standalone repository,
- no requirement to fully decouple linkstats from PILATES internals.

This is the right first cut because it changes the user experience immediately
without demanding that all internals be perfected first.

## Documentation Requirements

The documentation needs to change alongside the code or the new layer will be
hard to discover.

### Required documentation changes

- `analysis/README.md`
  - lead with `open_archive()`, not `open_run()`,
  - describe the notebook-first object model,
  - clearly separate notebook workflows from CLI workflows.
- add a new notebook starter
  - focused on archive exploration, scenario selection, epoch resolution, and
    table loading,
  - not centered on handoff/export/ingest.

### Recommended documentation language

Avoid presenting the package primarily as:

- Consist tracker bootstrap tooling,
- a collection of CLI commands,
- or a handoff/export utility.

Lead with the actual user jobs:

- open an archive,
- find a scenario,
- load tables,
- compare runs.

## Testing Strategy

The new layer should be tested as a UX-preserving wrapper over the existing
machinery.

### Phase 1 tests

- `open_archive()` returns the expected top-level object.
- `Archive.summary()` includes health/tagging signals.
- `Archive.scenarios()` and `Archive.years()` behave correctly on stub data.
- `RunIndex` canonicalization handles missing or aliased fields.
- `Archive.runs()` returns a useful DataFrame without requiring direct `RunSet`
  manipulation.

### Phase 2 tests

- scenario selection resolves the expected epochs,
- converged epoch resolution matches existing behavior,
- `Epoch.tables.trips()` and similar loaders return DataFrames correctly,
- SQL escape hatch still works.

### Phase 3 tests

- comparison alignment information is preserved,
- standard summary helpers return stable columns,
- warnings about incomplete candidate sets are not lost.

### Important testing note

The tests should continue to use lightweight tracker and run stubs where
possible. Avoid turning early phases of this work into heavy integration-test
projects.

## Risks and Tradeoffs

### Risk: duplicated metadata logic persists

If `RunIndex` is implemented as just another place that reimplements field
recovery, the architecture will get worse rather than better.

Mitigation:

- consolidate helper logic while building `RunIndex`,
- or explicitly factor shared normalization helpers into one internal module.

### Risk: notebook layer becomes a thin rename only

If `Archive` merely renames `AnalysisSession` without changing the default UX,
the startup-cost problem will remain.

Mitigation:

- require scenario/year discovery and canonical runs inspection to be first-class
  on the new object.

### Risk: DataFrame-first helpers become too opinionated

Some users will still need SQL and raw tracker access.

Mitigation:

- preserve `sql(...)`, raw views, and tracker escape hatches,
- but keep them secondary.

### Risk: package boundary remains muddy

Some dataset builders still depend on PILATES internals. That is acceptable in
the short term, but it should not quietly spread.

Mitigation:

- keep the new notebook layer focused on API and workflow shape first,
- document direct PILATES dependencies clearly,
- treat standalone extraction as a later boundary-hardening effort.

## Explicit Decisions

These decisions should be treated as part of the plan unless strong evidence
emerges otherwise during implementation.

- Prefer `open_archive()` as the new public notebook entry point.
- Keep `open_run()` temporarily for compatibility.
- Use `Archive` as the top-level notebook object.
- Add `scenario_id` as a first-class notebook-facing selector.
- Use DataFrame-returning helpers as the default user path.
- Keep attached archive + attached DB as the primary supported notebook mode.
- Build on top of `AnalysisSession`, `RunSet`, `EpochPanel`, and `EpochViews`
  rather than replacing them.

## Implementation Readiness Checklist

Before starting code changes, the implementing agent should verify:

- where to add the new notebook-facing modules,
- which existing helper functions can be reused for canonical field recovery,
- how to expose `open_archive()` cleanly in `__init__.py`,
- how to keep `open_run()` backward-compatible,
- where to place the new exploration notebook,
- and which existing tests can be extended rather than duplicated.

Before opening the first PR, the implementing agent should be able to answer:

- Can a new user open an archive and discover scenarios/years quickly?
- Is the preferred notebook entry point obvious in the docs?
- Has metadata normalization moved toward one canonical place rather than
  further duplication?
- Did the new layer improve the default UX without breaking existing surfaces?

## Summary

The package does not need a rewrite. It needs a better top layer.

The existing session, runset, epoch, and comparison primitives are good enough
to support a strong notebook-first analysis experience. The missing step is to
compose them into a clearer public workflow:

- archive,
- scenario,
- epoch,
- tables,
- comparison.

If implementation stays disciplined around that model, the resulting package
should be materially easier for both the current author and other users to
adopt in real notebooks on HPC-backed archives.
