# Changelog

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