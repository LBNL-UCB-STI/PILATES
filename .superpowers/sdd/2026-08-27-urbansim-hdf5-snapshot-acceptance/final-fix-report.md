# UrbanSim HDF5 snapshot acceptance: final fix report

## Scope

This final fix wave addresses the three Major review findings for the narrow
UrbanSim HDF5 snapshot-reconciliation acceptance only. It does not submit HPC
work, change cache policy or contract status, or modify the first-boundary-
promotion plan.

## Changes

- `hpc/job.sh` now has an HDF5-acceptance-only Consist setup path. It requires
  an explicitly supplied `CONSIST_SRC_DIR` that exists as a Git checkout,
  requires `pip install -e` to succeed, verifies that `import consist` resolves
  inside that checkout, and fails without a PyPI or alternative-install
  fallback. Before capture it records `runtime-environment.json` with the
  exact PILATES and editable-Consist revisions, Consist import path, and
  Python runtime. It intentionally leaves the generic job path unchanged.
- `pilates/runtime/urbansim_h5_snapshot_acceptance.py` now writes
  `effective-input-manifest.json` after manifest expansion, HDF5 inspection,
  and trusted artifact creation. It retains the fixed cohort, resolved source
  descriptor, trusted artifact ID, and trusted identity without changing the
  supplied manifest. Capture and reconciliation records include distinct
  process IDs and a secret-free runtime record.
- Snapshot reconciliation now obtains `input_identity` from the reopened,
  completed run and independently requires action-v2 mode, exactly one strict
  input, a nonempty strict-binding identity, and no ordinary bindings. It
  separately proves the snapshot run has exactly the trusted HDF5 input,
  excludes the later override, and compares the reloaded identity with the
  pre-override trusted identity. `validation.json` includes the separate
  `persisted_strict_link_trusted` check and uses the snapshot-derived strict
  binding result rather than the capture flag.
- Focused tests cover the effective manifest and process provenance, resistance
  to a changed capture flag, absent editable checkout rejection, import-
  verification rejection, and durable runtime revision evidence. `hpc/README.md`
  documents these acceptance-specific preconditions and records.

## Verification

All local verification used
`/Users/zaneedell/miniforge3/envs/PILATES/bin/python`:

- `pytest -q tests/test_hpc_job_runner.py` — 13 passed.
- `pytest -q tests/test_urbansim_h5_snapshot_acceptance.py -k 'not capture and not reconciliation'` — 15 passed, 4 deselected.
- `pytest -q tests/test_urbansim_h5_snapshot_acceptance.py::test_capture_then_fresh_process_reconciliation_preserves_h5_identity` — 1 passed.
- `pytest -q tests/test_urbansim_h5_snapshot_acceptance.py::test_capture_retains_effective_manifest_and_distinct_process_provenance` — 1 passed.
- `pytest -q tests/test_urbansim_h5_snapshot_acceptance.py::test_reconciliation_derives_strict_binding_from_snapshot_not_capture_flag` — 1 passed.
- `pytest -q tests/test_urbansim_h5_snapshot_acceptance.py::test_reconciliation_requires_tracker_snapshot_before_opening_tracker` — 1 passed.
- `pytest -q tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_rejects_snapshot_missing_required_root_tables tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_binding_survives_later_h5_hash_override_after_reopen` — 2 passed.
- `python -m ruff check pilates/runtime/urbansim_h5_snapshot_acceptance.py tests/test_urbansim_h5_snapshot_acceptance.py tests/test_hpc_job_runner.py`, `bash -n hpc/job.sh`, `bash -n hpc/job_runner.sh`, and `git diff --check` — all passed.

## Residual concerns

No HPC submission was made, so this is local regression evidence only. The
future Slurm acceptance must use the intended editable Consist checkout and
retained source HDF5, then retain the generated evidence bundle and its
checksum-backed archive copy. Passing that run remains merge-confidence
evidence only; HDF5 cache/boundary promotion remains separately gated.
