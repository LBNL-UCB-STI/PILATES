# Changelog

## v2.0.0 - Consist Integration (2026-MM-DD)

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
- Captured the May 4 SFBay canary decision surface with
  `phase2_recommendation: "defer"` and zero local-to-scratch recovery-root
  writes.
- Kept H5 container families blocked from Phase 2 recovery-root registration
  until Consist has an explicit container recovery policy.
- Documented upstream Consist feature requests for semantic run matching and
  container artifact recovery policy.

Release notes:

- Update the date and version before tagging.
- Tag only after the merge target becomes the release/main branch.
