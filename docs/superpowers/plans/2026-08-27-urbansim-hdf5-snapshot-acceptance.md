# UrbanSim HDF5 Snapshot-Reconciliation Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrow HPC acceptance that proves Consist PR #225 preserves a
real, strictly bound UrbanSim HDF5 input through a checkpointed Tracker snapshot
and separate-process read-only reconciliation.

**Architecture:** Implement one UrbanSim-specific driver with `capture` and
`reconcile` subcommands. Capture runs the existing `URBANSIM_POSTPROCESS`
resolver and wrapper using a driver-local no-op postprocessor seam, records the
PR-relevant later hash observation, and checkpoint-snapshots the Tracker.
Reconciliation is a second Python invocation that reads only that snapshot and
proves the strict HDF5 artifact identity has not changed. The existing HPC
acceptance wrapper stages the input manifest and invokes both phases; it does
not run a model or make the boundary cacheable.

**Tech Stack:** Python 3.11, Consist `Tracker` and `ArtifactIdentity`, PILATES
native step execution and HDF5 alias helpers, pytest, shell-wrapper tests,
Slurm.

**Spec:**
`docs/superpowers/specs/2026-08-27-urbansim-hdf5-snapshot-acceptance-design.md`

## File structure

- Create: `pilates/runtime/urbansim_h5_snapshot_acceptance.py` — manifest
  parsing, HDF5 preflight/descriptors, capture, read-only reconciliation, and
  reviewer-facing evidence records.
- Create: `tests/test_urbansim_h5_snapshot_acceptance.py` — real small HDF5
  fixture, native wrapper capture, failure modes, and a subprocess
  reconciliation proof.
- Create: `hpc/urbansim-h5-snapshot-acceptance-inputs.json.template` —
  environment-substituted source artifact and explicit cohort.
- Modify: `hpc/job_runner.sh` — stage one submitted manifest and submit the
  new mutually exclusive acceptance mode.
- Modify: `hpc/job.sh` — execute capture and reconciliation in two distinct
  unbuffered Python processes, then exit before `run.py`.
- Modify: `tests/test_hpc_job_runner.py` — cover submission construction,
  mode exclusions, and the two driver invocations.
- Modify: `hpc/README.md` — operator preflight, exact submission, expected
  evidence, and non-promotion interpretation.
- Modify after successful HPC evidence only:
  `docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md` —
  append immutable pre-merge evidence without changing any HDF5 contract
  status.

## Global constraints

- Use the real retained SFBay HDF5 only through an operator-supplied manifest;
  do not commit its path, copy it into the repository, or modify it.
- Require `workflow_year=2017`, `forecast_year=2019`, and `iteration=0`; do
  not accept an ambiguous `year` field.
- Run `URBANSIM_POSTPROCESS` through its actual resolver and `execute_step()`
  path. Replace only the `UrbansimPostprocessor` model-work seam inside the
  driver and restore it in a `finally` block.
- Keep the native wrapper's copied output and
  `ensure_usim_population_year_table_aliases()` validation active.
- The second phase must use `Tracker(..., db_path=snapshot_path,
  access_mode="read_only")`; it must never reopen the live capture database or
  reuse a live Tracker object.
- Do not modify `InputContract` statuses, global hashing/cache policies,
  `OutputSet` behavior, production snapshot policy, or the pinned
  `beam_run_completed -> beam_postprocess` checkpoint.
- A successful editable-Consist run is merge-confidence evidence only. HDF5
  promotion remains blocked until the same acceptance passes against a released
  Consist version and a candidate boundary subsequently proves its own
  cold-miss/fresh-workspace-hit closure.
- Use `rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q`
  for focused tests. Preserve all unrelated tracked and untracked worktree
  files; stage only files named by the task at hand.

---

### Task 1: Define the local real-HDF5 acceptance driver and its fail-closed contract

**Files:**

- Create: `pilates/runtime/urbansim_h5_snapshot_acceptance.py`
- Create: `tests/test_urbansim_h5_snapshot_acceptance.py`
- Read: `pilates/workflows/steps/urbansim_atlas.py:504-519,695-735,1106-1115`
- Read: `pilates/utils/usim_h5.py:20-58`
- Read: `tests/test_urbansim_atlas_native_step_definitions.py:389-429,1094-1249`

**Consumes:** a JSON manifest with one path under `inputs.usim_datastore_h5`;
a settings file whose `WorkflowState` is `(year=2017, forecast_year=2019,
current_inner_iter=0)`; and an otherwise empty evidence directory.

**Produces:** the following importable functions and records:

```python
@dataclass(frozen=True)
class AcceptanceManifest:
    usim_datastore_h5: Path
    workflow_year: int
    forecast_year: int
    iteration: int

def load_manifest(path: Path) -> AcceptanceManifest: ...
def describe_population_h5(path: Path, *, year: int, require_year_aliases: bool) -> dict[str, object]: ...
def run_capture(*, settings_path: Path, manifest_path: Path, evidence_root: Path) -> dict[str, object]: ...
def run_reconciliation(*, evidence_root: Path) -> dict[str, object]: ...
def validate(*, capture: Mapping[str, object], reconciliation: Mapping[str, object]) -> dict[str, object]: ...
```

The capture record is saved as `capture.json`; reconciliation writes
`reconciliation.json` and `validation.json`. `effective-input-manifest.json`
contains only resolved source metadata, the explicit cohort, and the trusted
artifact identity. The snapshot destination is
`evidence_root / "snapshots" / "tracker.duckdb"`.

- [ ] **Step 1: Write the failing manifest, HDF5-preflight, and cohort tests.**

  In `tests/test_urbansim_h5_snapshot_acceptance.py`, create a real fixture
  using `pandas.HDFStore` with root `households`, `persons`, `jobs`, and
  `blocks` tables. Assert the input validation requires a readable file and
  the complete, exact cohort:

  ```python
  def test_load_manifest_rejects_ambiguous_or_incomplete_cohort(tmp_path: Path) -> None:
      manifest = tmp_path / "inputs.json"
      manifest.write_text(
          json.dumps({"inputs": {"usim_datastore_h5": str(tmp_path / "source.h5")},
                      "cohort": {"year": 2019}}),
          encoding="utf-8",
      )
      with pytest.raises(ValueError, match="workflow_year=2017, forecast_year=2019, iteration=0"):
          acceptance.load_manifest(manifest)

  def test_describe_population_h5_rejects_missing_root_table(tmp_path: Path) -> None:
      source = _write_population_h5(tmp_path / "source.h5", tables=("households", "persons", "jobs"))
      with pytest.raises(ValueError, match="missing root population tables"):
          acceptance.describe_population_h5(source, year=2019, require_year_aliases=False)
  ```

  Also assert the valid fixture descriptor names all four root tables and does
  not create a `2019` group in the source file.

- [ ] **Step 2: Run the new tests to verify the module is absent.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_urbansim_h5_snapshot_acceptance.py -k 'manifest or describe'
  ```

  Expected: collection fails because
  `pilates.runtime.urbansim_h5_snapshot_acceptance` does not exist.

- [ ] **Step 3: Implement manifest and read-only HDF5 descriptor helpers.**

  Add `AcceptanceManifest`, `load_manifest`, `_validate_state`, and
  `describe_population_h5` in the new driver. Expand environment variables in
  manifest values, require a file, and reject any cohort mapping other than:

  ```python
  _COHORT = {"workflow_year": 2017, "forecast_year": 2019, "iteration": 0}
  _REQUIRED_ROOT_TABLES = ("households", "persons", "jobs", "blocks")
  ```

  Use `h5py.File(path, "r")` to inspect root members; do not call the alias
  writer on the submitted source. Return a JSON-safe descriptor with resolved
  path, byte size, root-table names, and (when requested) exact
  `/<forecast_year>/<table>` aliases. Save JSON through one helper that writes
  sorted, indented output and creates only the evidence parent directory.

- [ ] **Step 4: Run the focused helper tests.**

  Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the tested helper contract.**

  ```bash
  rtk git add pilates/runtime/urbansim_h5_snapshot_acceptance.py \
    tests/test_urbansim_h5_snapshot_acceptance.py
  rtk git commit -m "test: define HDF5 snapshot acceptance contract"
  ```

### Task 2: Capture the real native wrapper and reconcile its checkpoint from a fresh process

**Files:**

- Modify: `pilates/runtime/urbansim_h5_snapshot_acceptance.py`
- Modify: `tests/test_urbansim_h5_snapshot_acceptance.py`
- Read: `pilates/workflows/step_execution.py:80-125`
- Read: `pilates/workflows/steps/urbansim_atlas.py:504-519,695-735`
- Read: `tests/test_beam_checkpoint.py:659-687`

**Consumes:** Task 1's parsed manifest and HDF5 descriptor.

**Produces:** a completed native `urbansim_postprocess` run, a checkpointed
snapshot, and a separate-process reconciliation verdict whose strict input
identity equals the capture-time trusted identity.

- [ ] **Step 1: Write failing end-to-end local acceptance tests.**

  Use the real small HDF5 fixture and the SFBay 2019 settings, then call the
  capture CLI and reconciliation CLI in distinct Python processes. Assert the
  capture executes the actual wrapper and that reconciliation reads only the
  retained snapshot:

  ```python
  def test_capture_then_fresh_process_reconciliation_preserves_h5_identity(
      tmp_path: Path,
  ) -> None:
      source, manifest = _real_h5_manifest(tmp_path)
      evidence = tmp_path / "evidence"
      _run_module("capture", settings=_SETTINGS, manifest=manifest, evidence=evidence)
      _run_module("reconcile", evidence=evidence)

      validation = _read_json(evidence / "validation.json")
      assert validation == {
          **validation,
          "source_h5_valid": True,
          "strict_binding_valid": True,
          "override_is_unowned": True,
          "snapshot_created": True,
          "fresh_snapshot_readable": True,
          "persisted_identity_unchanged": True,
          "output_h5_aliases_valid": True,
          "valid": True,
      }
      assert _read_json(evidence / "capture.json")["trusted_identity"] == _read_json(
          evidence / "reconciliation.json"
      )["persisted_identity"]
  ```

  Add a negative test that removes `snapshots/tracker.duckdb`; reconciliation
  must raise an error naming that snapshot before constructing a Tracker. In
  the positive test, delete the live `provenance.duckdb` after capture but
  before reconciliation; the fresh process must still pass, proving it did not
  reopen the capture database.

- [ ] **Step 2: Run the end-to-end test and observe the missing capture/reconciliation API.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_urbansim_h5_snapshot_acceptance.py \
    -k 'capture_then_fresh_process or reconciliation'
  ```

  Expected: FAIL because the capture and reconciliation CLI behavior is not
  implemented.

- [ ] **Step 3: Implement capture with the native wrapper and the deliberate later observation.**

  In `run_capture`:

  ```python
  tracker = Tracker(
      run_dir=evidence_root / "consist-runs",
      db_path=evidence_root / "provenance.duckdb",
      hashing_strategy="full",
      allow_external_paths=True,
  )
  with tracker.start_run("urbansim-h5-acceptance-inputs", "acceptance"):
      trusted = tracker.log_artifact(source, key=USIM_DATASTORE_H5, direction="input")
  trusted_identity = str(consist.ArtifactIdentity.from_artifact(trusted))
  ```

  Construct `Workspace(settings, str(evidence_root / "workspaces"), "capture")`.
  With one `try/finally` seam replacement, assign a private
  `_AcceptanceUrbansimPostprocessor` to
  `pilates.workflows.steps.urbansim_atlas.UrbansimPostprocessor`; its
  `postprocess` method returns `None` and does no model work. Within the
  `finally`, restore the original class even when `execute_step()` fails.

  Bind `trusted` on a Scenario coupler, call
  `URBANSIM_POSTPROCESS.resolve_inputs(...)`, then call `execute_step(...)`
  with `stage="land_use"`, `year=state.year`,
  `iteration=state.current_inner_iter`, and `phase="postprocess"`. Do not
  replace the `StepDefinition`, output-path function, output projector, or
  alias helper. Assert the completed run has `input_identity.mode ==
  "action-v2"`, `strict_input_count == 1`, and a nonempty
  `strict_binding_identity`.

  After completion, create the later observation exactly as follows, then
  require it to be a distinct artifact with no owner run:

  ```python
  override = tracker.log_artifact(
      trusted,
      key=USIM_DATASTORE_H5,
      direction="input",
      content_hash=trusted.hash,
      force_hash_override=True,
  )
  assert override.id != trusted.id
  assert override.run_id is None
  assert override.meta["hash_semantics"]["source"] == "caller_supplied"
  ```

  Create `snapshots/tracker.duckdb` with
  `tracker.snapshot_db(str(snapshot_path), checkpoint=True)`, require the file
  exists, close/dispose the live Tracker database, and persist all IDs,
  identity strings, run snapshot, source/output descriptors, and snapshot
  path relative to `evidence_root` in `capture.json`.

- [ ] **Step 4: Implement read-only reconciliation and verdict construction.**

  `run_reconciliation` must load `capture.json`, resolve its snapshot relative
  to `evidence_root`, and reject an absent snapshot before constructing a
  Tracker. It opens only:

  ```python
  snapshot_tracker = Tracker(
      run_dir=evidence_root / "consist-runs",
      db_path=snapshot_path,
      allow_external_paths=True,
      access_mode="read_only",
  )
  ```

  Retrieve the completed run by recorded ID and its input links with the
  public Tracker method. Require exactly the recorded trusted artifact ID for
  `USIM_DATASTORE_H5`; reject the override ID. Build
  `persisted_identity = str(consist.ArtifactIdentity.from_artifact(artifact))`
  and compare it exactly with the capture identity. Write
  `reconciliation.json`; call `validate` to write `validation.json` with the
  eight named booleans from the spec and `valid` as their conjunction.

  The `argparse` entry point has two required subcommands:

  ```text
  capture --settings SETTINGS --manifest MANIFEST --evidence-root ROOT
  reconcile --evidence-root ROOT
  ```

  `capture` must not call reconciliation, and `reconcile` must not load the
  submitted source HDF5 or settings file.

- [ ] **Step 5: Run the focused acceptance suite.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_urbansim_h5_snapshot_acceptance.py \
    tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_rejects_snapshot_missing_required_root_tables \
    tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_binding_survives_later_h5_hash_override_after_reopen
  ```

  Expected: all tests pass, including the subprocess reconciliation case.

- [ ] **Step 6: Commit the native HDF5 snapshot proof.**

  ```bash
  rtk git add pilates/runtime/urbansim_h5_snapshot_acceptance.py \
    tests/test_urbansim_h5_snapshot_acceptance.py
  rtk git commit -m "feat: add UrbanSim HDF5 snapshot acceptance"
  ```

### Task 3: Add the mutually exclusive HPC entry point and operator manifest

**Files:**

- Create: `hpc/urbansim-h5-snapshot-acceptance-inputs.json.template`
- Modify: `hpc/job_runner.sh:30-280`
- Modify: `hpc/job.sh:120-193`
- Modify: `tests/test_hpc_job_runner.py`

**Consumes:** Task 2's module CLI and the manifest schema from Task 1.

**Produces:** one Slurm allocation that stages the submitted manifest and
generated settings below its own evidence root, then runs capture and
reconciliation as distinct unbuffered module processes.

- [ ] **Step 1: Write failing HPC-wrapper tests.**

  Add a job-runner test modeled on the existing BEAM acceptance test, but use
  `PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_ROOT`. Assert the sbatch invocation
  contains the exact selector, original manifest path, and unique evidence
  directory. Add a rejection test for combining this selector with
  `--beam-preprocess-acceptance`, `--native-structural-canary`, or `-s`.

  Add an allocated-job test with a fake `python3` that asserts the driver
  calls occur, in order, as:

  ```text
  -u -m pilates.runtime.urbansim_h5_snapshot_acceptance capture --settings <settings> --manifest <manifest> --evidence-root <evidence>
  -u -m pilates.runtime.urbansim_h5_snapshot_acceptance reconcile --evidence-root <evidence>
  ```

  Assert no call contains `run.py`.

- [ ] **Step 2: Run the wrapper tests to verify the selector is unsupported.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_hpc_job_runner.py -k 'urbansim_h5_snapshot_acceptance'
  ```

  Expected: FAIL because the selector and driver commands do not exist.

- [ ] **Step 3: Implement one exclusive acceptance selector.**

  Add `urbansim_h5_snapshot_acceptance_manifest` handling in `job_runner.sh`.
  Resolve and verify the manifest before `sbatch`; copy it and the generated
  settings into:

  ```bash
  acceptance_root="${PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_ROOT:-/global/scratch/users/$USER/pilates-boundary-promotions}"
  evidence_dir="${acceptance_root%/}/$JOB_NAME"
  ```

  Reject every pair of acceptance selectors and reject any acceptance selector
  plus `-s/--stage`. Pass the original manifest path and evidence root through
  `sbatch`, just as the BEAM harness does.

  In `job.sh`, parse the new selector separately from the BEAM selector and
  require exact argument counts. After editable-Consist installation and
  environment reporting, run the two commands from Step 1 with
  `PYTHONUNBUFFERED=1` and `python3 -u`, in that order, followed by `exit 0`.
  Do not route this mode through the legacy settings migration or `run.py`.

  Add the template exactly as:

  ```json
  {
    "inputs": {
      "usim_datastore_h5": "${PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5}"
    },
    "cohort": {"workflow_year": 2017, "forecast_year": 2019, "iteration": 0}
  }
  ```

- [ ] **Step 4: Run the complete HPC wrapper test file.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_hpc_job_runner.py
  ```

  Expected: all selector, rejection, and terminal-driver-path tests pass.

- [ ] **Step 5: Commit the HPC entry point.**

  ```bash
  rtk git add hpc/job_runner.sh hpc/job.sh \
    hpc/urbansim-h5-snapshot-acceptance-inputs.json.template \
    tests/test_hpc_job_runner.py
  rtk git commit -m "feat: add HDF5 snapshot acceptance job mode"
  ```

### Task 4: Document the pre-merge evidence handoff and execute the narrow HPC proof

**Files:**

- Modify: `hpc/README.md:88-145`
- Modify after success only:
  `docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md`
- Read: `docs/superpowers/specs/2026-08-27-urbansim-hdf5-snapshot-acceptance-design.md`

**Consumes:** the completed driver and HPC selector from Tasks 1–3; the
retained real source artifact on HPC; an editable checkout of Consist PR #225.

**Produces:** reproducible operator instructions and, only after success, one
immutable pre-merge evidence record with a checksum-verified NFS copy.

- [ ] **Step 1: Write the HPC README section.**

  Document this exact preflight pattern, replacing the value only with the
  current retained artifact path on HPC:

  ```fish
  set -x PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5 /global/scratch/users/zaneedell/pilates-outputs/pilates-run--sfbay--consist-sfbay-usim-base-short-canary--20260826-104715/consist-recovery/pilates-run--sfbay--consist-sfbay-usim-base-short-canary--20260826-104715_atlas_postprocess__y2019__i0__phase_postprocess_b28c6e38/urbansim/data/model_data_2019_population_source.h5
  test -f "$PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5"
  ```

  Explain that the driver—not the filename—enforces workflow year 2017,
  forecast year 2019, and iteration 0. Document the editable-Consist submission
  form and the two-phase evidence records. State plainly that success is not a
  cache hit or HDF5 promotion.

- [ ] **Step 2: Run the documentation and focused regression gate.**

  Run:

  ```bash
  rtk /Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest -q \
    tests/test_urbansim_h5_snapshot_acceptance.py \
    tests/test_hpc_job_runner.py \
    tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_rejects_snapshot_missing_required_root_tables \
    tests/test_urbansim_atlas_native_step_definitions.py::test_urbansim_postprocess_binding_survives_later_h5_hash_override_after_reopen
  rtk git diff --check
  ```

  Expected: all focused tests pass and the working-tree diff is whitespace
  clean.

- [ ] **Step 3: Commit the operator documentation.**

  ```bash
  rtk git add hpc/README.md
  rtk git commit -m "docs: explain HDF5 snapshot acceptance"
  ```

- [ ] **Step 4: Run the HPC acceptance against the editable Consist checkout.**

  On HPC, copy the template to an operator-owned manifest, set the source
  variable above, and submit with the ordinary node shape:

  ```fish
  set -x CONSIST_SRC_DIR /global/scratch/users/zaneedell/sources/consist
  envsubst < hpc/urbansim-h5-snapshot-acceptance-inputs.json.template > hpc/urbansim-h5-snapshot-acceptance-inputs.json
  ./hpc/job_runner.sh \
    -c scenarios/sfbay/settings-sfbay-consist-usim-hpc-2019-canary.yaml \
    -a ac_beamcore \
    --urbansim-h5-snapshot-acceptance hpc/urbansim-h5-snapshot-acceptance-inputs.json
  ```

  Record the submitted job ID and evidence root printed by the wrapper. Do not
  request `--high-mem` and do not append `-s`.

- [ ] **Step 5: Verify and archive a successful evidence bundle.**

  Require Slurm `COMPLETED` and `validation.json` with every named check and
  `valid` true. Copy the evidence root to the established NFS
  `pilates-boundary-promotions` archive, generate SHA-256 manifests for source
  and destination, and compare them before treating the archive as retained
  evidence. Capture PILATES and Consist revisions from the evidence record.

- [ ] **Step 6: Append only verified evidence to the boundary-promotion plan.**

  Add the scratch and NFS paths, Slurm outcome, revisions, validation verdict,
  and checksum result beneath the existing HDF5 pre-merge evidence heading in
  `2026-08-27-first-native-boundary-promotion.md`. State that the result is
  editable-branch merge-confidence evidence and that HDF5 contract completion
  still requires a released-version replay plus boundary-specific cache proof.

- [ ] **Step 7: Commit the verified evidence record.**

  ```bash
  rtk git add docs/superpowers/plans/2026-08-27-first-native-boundary-promotion.md
  rtk git commit -m "docs: record HDF5 snapshot acceptance evidence"
  ```

## Final verification checklist

- [ ] The local driver suite passes with a real small HDF5 and a second Python
  process opening the checkpoint read-only.
- [ ] The existing native HDF5 override regression still passes.
- [ ] The full HPC wrapper test file passes, proving the normal workflow cannot
  accidentally fall through into this acceptance mode or vice versa.
- [ ] The HPC run records a real retained SFBay HDF5 source, the editable
  Consist revision, a valid two-process verdict, and a checksum-verified NFS
  evidence copy.
- [ ] Documentation labels the run pre-merge evidence only; no HDF5 consumer
  `InputContract` is marked complete and no cache policy changes.
