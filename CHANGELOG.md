# Changelog

## v1.1.0 - 2026-10-23

This release completes PILATES' cutover to Consist-native step execution, introduces iterative MATSim plan memory and standalone full-skim generation in BEAM, rebuilds scenario analysis around faceted Ibis/DuckDB queries, and simplifies the architecture and scenario configuration footprint across the repository.

### Consist-Native Step Execution, Coupler Snapshots & Restart

- Cut over all model stages (UrbanSim, Atlas, ActivitySim, and BEAM) to first-class Consist-native step execution with typed `StepDefinition` contracts (`#81`, `#87`, `#111`).
- Replaced ad-hoc disk checkpoint probes and transition-era manifest restart machinery with DuckDB coupler snapshot artifacts (`#73`, `#74`, `#80`, `#96`) and delegated snapshot materialization and pruning directly to Consist (`#101`).
- Centralized model output archiving and isolated recovery artifacts under `consist-recovery/`, while preserving scenario `parent_run_id` lineage across chained runs and archiving resolved BEAM launch configs (`#85`, `#102`, `bfee015b`).
- Folded `activitysim_compile` into `activitysim_run`, separating durable workflow artifacts from transient runtime compilation caches (`#78`).
- Added bidirectional population aliasing for forecast-year UrbanSim HDF5 snapshots (`#88`) and fixed base-year schema handling in Atlas postprocessing (`#86`).

### BEAM Plan Memory, Full Skims & Input Admission

- Implemented MATSim-compatible iterative plan memory and agent learning across supply-demand iterations, preserving freight agents (`freight_` prefix), supporting alphanumeric person IDs, and preferring ActivitySim replanning outputs on replanning iterations (`#98`).
- Added standalone full-skim workflow support (`FullSkimsCreatorApp`) with OMX, Zarr, and CSV targets, dynamic CPU/RAM parallelism heuristics, 72-hour calculation timeouts, and synthetic scenario stubs for headless runs (`#92`, `#95`, `#96`).
- Added BEAM CSV-to-ActivitySim 5-period OMX full-skim conversion (`pilates/beam/full_skim_converter.py`) with intrazonal imputation, canonical logical matrix SHA-256 hashing (`sha256:logical_matrices:<hex>`), and configurable maximum vehicular speed capping (`#94`, `#100`, `347ab19f`).
- Added execution-exact preflight admission for BEAM warm-start `linkstats` and ActivitySim bootstrap skims with JSON provenance sidecars (`pilates.skim_provenance.v1` and `pilates.linkstats_provenance.v1`) and native Consist cross-tracker verification (`#77`, `#79`, `#94`, `#95`).
- Isolated `beam_postprocess` Zarr skims into dedicated output stores and implemented copy-on-write skims staging for idempotent restarts (`#83`, `#84`, `#88`, `#91`).
- Upgraded default BEAM container image to `zaneedell/beam:1.1.0`, added Lawrencium `lr6` high-memory and `lr7` 56-core thread tuning in `hpc/job_runner.sh`, and configured `chunk_size: 0` for Seattle block-group ActivitySim skims (`#93`, `#98`, `4ad968fa`).

### Scenario Configuration Inheritance & Analysis

- Implemented declarative scenario YAML inheritance via top-level `base:` / `extends:` keys and converted regional scenario variants into concise overlays, eliminating over 7,300 lines of duplicated YAML across Seattle, SFBay, and Breathe scenarios (`#89`, `#106`).
- Rebuilt `pilates.analysis` around faceted Ibis archive queries over DuckDB (`open_archive`, `Archive.table`, `Archive.measure`, `delta`, `difference`, `delta_change`, `rank`), removing legacy paired-scenario comparison harnesses (`#75`, `#76`, `#98`).

### Architecture Simplification & Cleanup

- Decoupled synthetic population staging from `UrbansimPreprocessor` when land-use is disabled, deduplicated `CanonicalZones` staging, and enforced fail-fast input path checks (`#105`).
- Retired `ModelFactory` (`pilates/generic/model_factory.py`), the legacy `Model` base class (`pilates/generic/model.py`), `scripts/migrate_config.py`, `step_output_handoff_mapping`, dead `expected_inputs` methods, and unused `coupler_helpers.py` / `steps/shared.py` wrappers (`#82`, `#104`, `#107`, `#108`, `#109`).
- Retired finished one-off HPC migration acceptance harnesses (`#110`).
- Split monolithic `urbansim_atlas.py` into dedicated `pilates/workflows/steps/urbansim.py` and `pilates/workflows/steps/atlas.py` step modules (`#112`), and updated `scripts/new_model_scaffold.py` and `docs/extend/adding_a_model.md` (`#106`, `#107`).

## v1.0.1 - 2026-06-05

This release tightens PILATES’ Consist-backed runtime behavior and aligns archive/recovery handling across the workflow.

### Highlights

- Updated Consist dependency and environment pins to match the current Consist release line (`#68`).
- Fixed restart behavior across runtime, workflow, and analysis paths, improving replay and restart consistency (`#69`).
- Integrated Consist 0.1.5 capabilities, including capability probing, workflow binding updates, stage-contract adjustments, and broader restart/promotion support (`#70`).
- Aligned run promotion with recovery-root adoption across ActivitySim, BEAM, Atlas, and UrbanSim paths, and added supporting recovery-root tooling and docs (`#71`).

### Validation and coverage

- Expanded test coverage around restart equivalence, archive promotion, stage contracts, workflow metadata, and recovery behavior.
- Updated HPC and workflow documentation where the Consist integration changes affected operator-facing behavior.

### Notes

- This is primarily a reliability and infrastructure release rather than a new model-feature release.
- The main behavioral shifts are in restart, recovery, and archive promotion semantics.

## v1.0.0 - Consist Integration (2026-05-05)

PILATES now uses Consist for cache-aware workflow tracking, run identity, and
artifact provenance. This release establishes Consist as the source of truth
for:

- run lifecycle through Scenario and Step contexts
- artifact identity, content hashing, and recovery metadata
- cache hit/miss decisions for restart and resume
- the coupler that carries semantic role values between workflow steps

Conceptual changes:

- Snapshot artifacts are semantic workflow-boundary facts, distinct from archive
  recovery roots that describe storage motion.
- Workflow stages, steps, typed outputs, and binding rules are explicit
  contracts. See `docs/workflow/step_contracts.md` and
  `docs/workflow/artifact_semantics.md`.
- Model integration still follows the preprocessor/runner/postprocessor
  pattern, now wired through Consist-aware step contracts. See
  `docs/extend/adding_a_model.md`.

Breaking changes:

- Consist is mandatory. `shared.database.use_consist` is deprecated and ignored
  for compatibility with older local configs.
- Named scenario settings now live under `scenarios/<region>/`; the repository
  root keeps only the canonical `settings.yaml` default.

Migration and cleanup:

- Removed stale root scripts, notebooks, duplicate ATLAS input data, and
  migration-era scratch artifacts from the tracked tree.
- Promoted `scripts/verify_zone_ids.py` as the supported operator diagnostic
  for canonical zone ID checks.
- Replaced the stale `run_stub_test_with_output.sh` helper with direct
  `PRESERVE_TEST_OUTPUT=... pytest ...` documentation.

Restart and provenance:

- Added lifecycle audit diagnostics for local-to-scratch archive copies and
  Phase 2 recovery-root readiness.