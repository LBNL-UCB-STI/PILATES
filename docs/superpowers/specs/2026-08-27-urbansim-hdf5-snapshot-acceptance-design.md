# UrbanSim HDF5 Snapshot-Reconciliation Acceptance Design

**Status:** implemented and accepted against the editable Consist PR #225
checkout on 2026-08-27; released-version replay and any HDF5 boundary
promotion remain pending.

## Purpose

Establish narrow, pre-merge integration evidence that Consist PR #225 preserves a
strictly bound, real UrbanSim HDF5 artifact across a persisted Tracker snapshot
and a fresh-process reload in PILATES. The acceptance target is the retained
SFBay `model_data_2019_population_source.h5` product from the formal native
structural-canary interval, supplied by the operator through a manifest rather
than embedded in source control.

This acceptance is merge-confidence evidence for the editable Consist branch.
It is deliberately not a cache-promotion test and does not make any HDF5
consumer boundary eligible for reuse.

## Decision and non-decision

The test will run the existing native `urbansim_postprocess` boundary through
its ordinary resolver and `execute_step()` path. A driver-local,
content-preserving adapter replaces only `UrbansimPostprocessor`'s
model-specific inner work. The native wrapper itself still receives a strictly
staged `USIM_DATASTORE_H5`, copies its population-source HDF5 output, and runs
the ordinary HDF5 table-alias validation.

This is preferable to a full UrbanSim postprocess invocation. A full run
would require a matched original input/output pair and could fail for ordinary
model-data reasons that say nothing about the HDF5 persistence defect. A
Consist-only probe is also insufficient because it would omit the PILATES
resolver, staging, native wrapper, and persisted run metadata.

No `InputContract` status, cache option, global hashing setting, `OutputSet`
meaning, production snapshot policy, or BEAM checkpoint behavior changes as a
result of this work.

## Input contract

A new operator-authored manifest template has one external input and an
unambiguous cohort:

```json
{
  "inputs": {
    "usim_datastore_h5": "${PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_INPUT_H5}"
  },
  "cohort": {
    "workflow_year": 2017,
    "forecast_year": 2019,
    "iteration": 0
  }
}
```

The acceptance driver rejects a missing or non-file input, an HDF5 file that
cannot be opened, an input missing the root population tables required by the
native alias check, or settings that do not create the specified workflow
state. It records the resolved source path, file size, HDF5 table descriptor,
and trusted Consist artifact identity in the effective manifest and capture
record. The source is read only; all materialization and output copies occur
under the per-job evidence workspace.

## Execution design

The Slurm allocation invokes two separate Python processes.

### Capture process

The capture process creates a Tracker and logs the retained HDF5 as a trusted
external `USIM_DATASTORE_H5` input. It sets that artifact on a Scenario
coupler, calls the native postprocess resolver, and invokes
`execute_step()` with `URBANSIM_POSTPROCESS`.

The driver temporarily supplies the minimal adapter only at the
`UrbansimPostprocessor` seam. The adapter must not alter the resolved input,
the step definition, normal staging, declared-output paths, output projection,
or alias validation. The real wrapper must therefore materialize the input at
its resolver-owned location, create the population-source HDF5 output, and
validate that output's HDF5 population-table shape.

After the native run is complete, the driver deliberately logs a later
caller-supplied hash observation for the same HDF5 using the PR #225-relevant
override path. That observation must be a separate, unowned artifact and must
not rewrite the strict artifact already bound to the completed run. The driver
then calls `Tracker.snapshot_db(..., checkpoint=True)` and retains the
resulting snapshot as the only database source for reconciliation.

### Fresh reconciliation process

The second process starts with no live Tracker object from capture. It opens
the retained snapshot read-only, retrieves the completed native run and its
input artifact link, and compares the persisted artifact identity with the
trusted identity recorded before the later observation. It also confirms that
the persisted run reports ordinary action-v2 strict input identity with exactly
one strict input and that the linked artifact is not the later unowned
observation.

This process verifies persistence and reconciliation, not a cached execution.
It must not invoke a new cacheable native run or infer a cache hit from a
successful snapshot lookup.

## Evidence bundle and validation

The existing `pilates-boundary-promotions/<job-id>/` convention is retained.
The bundle contains at least:

- `generated-settings.yaml`, the submitted manifest, and an
  environment-expanded effective manifest;
- a revision/environment record for PILATES and the editable Consist checkout;
- capture and fresh-reconciliation records, including process identifiers and
  persisted-run snapshots;
- the checkpointed Tracker snapshot and the capture Tracker database/run
  directory needed to audit how it was produced;
- HDF5 input/output table descriptors and source/output identities; and
- `validation.json`, a reviewer-facing map of named boolean checks.

The validation verdict is true only when all of the following are true:

1. the supplied artifact is readable, structurally valid real HDF5;
2. the native `urbansim_postprocess` wrapper completed with one strict
   action-v2 input;
3. the later override is a distinct unowned observation;
4. a checkpointed snapshot was created after the completed native run;
5. a separate process reopened that snapshot read-only;
6. the persisted strict input link is the original trusted artifact;
7. its reloaded `ArtifactIdentity` equals the pre-override trusted identity;
   and
8. the wrapper-produced HDF5 passes the native population-table alias check.

The bundle should be copied to the established NFS boundary-promotion archive
and rechecked with SHA-256s after a successful Slurm job. That copy verifies
retained-evidence integrity; it is not a hash of, or replacement for, the
external retained source HDF5.

## HPC interface

The implementation adds a single mutually exclusive mode,
`--urbansim-h5-snapshot-acceptance <input-manifest>`, to `hpc/job_runner.sh`
and `hpc/job.sh`. It follows the existing editable-Consist installation,
generated-settings, per-job evidence-root, notification, and failure-reporting
conventions used by the BEAM preprocess acceptance. It is incompatible with a
normal stage, native structural-canary capture, and BEAM-preprocess acceptance.

The driver is implemented in
`pilates/runtime/urbansim_h5_snapshot_acceptance.py` and the manifest template
in `hpc/urbansim-h5-snapshot-acceptance-inputs.json.template`. The existing
SFBay UrbanSim 2019 settings may be supplied only if they initialize the
manifest cohort; the driver is the authority that enforces the cohort.

The mode requests the ordinary HPC node shape. It opens and stages one HDF5
artifact but does not execute UrbanSim, ATLAS, ActivitySim, or BEAM and does
not need a high-memory allocation.

## Failure handling

Failures are fail-closed and preserve the evidence written before the failed
assertion. Input and cohort errors fail before the Tracker is created.
HDF5-read, alias, strict-binding, override, snapshot, and reload failures name
the failed invariant and the relevant evidence path. A failed fresh
reconciliation never falls back to the capture database or a live Tracker.

## Release replay rule

A passing editable-branch acceptance means the Consist branch is supported by
this narrow PILATES integration test and may be merged on that basis. It does
not lift HDF5 promotion eligibility.

After a released Consist version contains the persistence fix, rerun this same
acceptance against that released version and archive the result. Only then may
a candidate HDF5-consuming boundary begin its own resolver-closure and
cold-miss/fresh-workspace-hit acceptance. That later boundary evidence remains
model-aware and is separate from this snapshot-reconciliation test.

## Verification plan

Before HPC use, focused local tests cover manifest/cohort rejection, real-HDF5
fixture preflight, strict-binding preservation after the later observation,
checkpoint creation, and read-only snapshot reopening from a separately
invoked Python process. HPC acceptance uses the retained SFBay HDF5 and keeps
the full evidence bundle. The final documentation records the source archive,
Slurm status, PILATES and Consist revisions, validation verdict, and durable
archive checksum check without changing the structural-canary verdict.
