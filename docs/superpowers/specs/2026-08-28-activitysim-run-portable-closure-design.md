# ActivitySim Run Portable Closure and Local Compile-Epoch Design

**Status:** approved design; implementation has not started.

## Purpose

Make `activitysim_run` the next candidate native consumer boundary without
misclassifying host-local Numba/Sharrow compiler products as portable artifacts.
The boundary must select its skims and stage every model-visible data/configuration
root deterministically, while keeping compilation as one local preparation action
before the first real multiprocessing ActivitySim execution in each PILATES launch.

This work is intentionally separate from the completed `beam_preprocess`
promotion evidence and from the HDF5 persistence gate. It must not change global
cache policy, HDF5 contract status, `OutputSet` behavior, or the pinned BEAM
postprocess checkpoint.

## Decision

`activitysim_run` remains one native Consist step. There is no
`activitysim_compile` step and no compiler-cache artifact admission.

The step has two distinct concerns:

1. **Portable boundary closure.** The resolver selects named input artifacts and
   the executable receives a deterministic, staged launch tree for its data and
   configuration roots. Those selected artifacts and the canonical ActivitySim
   configuration adapter are identity-bearing.
2. **Local execution preparation.** Before a real body execution only, the
   runner may prepare a fresh, launch-scoped Numba/Sharrow compilation epoch.
   Its files are host-local and disposable. They are not step inputs, outputs,
   identities, restart artifacts, or archived evidence.

Consist decides a cache hit before local preparation. A hit hydrates the ordinary
declared outputs and does not compile or execute ActivitySim.

## Skim selection and execution matrix

Skim selection is resolver-owned and uses a read-only format check before the
step identity is finalized.

| Selected condition | Preparation needed? | Real ActivitySim body uses | Zarr outcome |
| --- | --- | --- | --- |
| A valid published Zarr artifact is available | Only when Sharrow is enabled and multiprocessing is active | That selected Zarr, staged read-only for model consumption | No regeneration; Zarr is an identity-bearing input |
| No valid Zarr, but OMX is available; preparation is needed | Yes | Zarr generated/finalized from the selected OMX during preparation | Generated Zarr is a declared `activitysim_run` output; OMX is the identity-bearing input |
| No valid Zarr, but OMX is available; preparation is not needed | No | OMX, as in the current efficient single-process/no-Sharrow path | The body generates/finalizes Zarr afterward as a declared output |
| Neither a valid Zarr nor OMX is available | N/A | Fail before the body | No output is admitted |

An invalid Zarr must never be used merely because a path exists. If the resolver
also has an OMX candidate, it selects that OMX and records the rejected-Zarr
diagnostic outside identity-bearing configuration. If OMX is unavailable, it
fails before execution. The validation itself must not rewrite Zarr metadata,
zone flags, or files.

## Compile epoch

Compilation is required only when all three conditions hold:

- Consist has decided that the native body will execute rather than hydrate a
  cache hit;
- the ActivitySim configuration enables persisted Sharrow/Numba compilation;
  and
- `activitysim.num_processes > 1`.

For every PILATES process invocation, the first required ActivitySim execution
creates a fresh local compile epoch. Later required executions in that same
invocation reuse that epoch only on the same host/process context. A prior
workspace directory, restart artifact, archive, or another machine must never
suppress this first required compilation.

The compile epoch has a process-local marker and a fresh private filesystem
root. It is deliberately absent from `InputContract` identity, Consist artifact
logging, output projection, snapshot/restart recovery, and archive promotion.
The runner must not infer epoch validity from a pre-existing nonempty
`shared_cache/numba` directory.

## Launch-tree closure

The current runner mounts mutable ActivitySim data, configuration, output, cache,
temporary, and shared-cache roots. For a portable `activitysim_run` boundary:

- The three table inputs and the selected skim artifact are resolver-bound and
  materialized at deterministic launch-tree destinations.
- The configuration adapter supplies the exact model-visible configuration tree
  and any launch scalars. That tree is staged under the private launch root,
  not discovered from a mutable workspace/configuration root at execution time.
- Output, temporary, compile-epoch, and shared-cache roots are private execution
  destinations. They may be writable but are not identity-bearing inputs.
- `ActivitySimLaunchContext` is built from the staged/private launch tree and is
  passed through `ExecutionOptions`; the runner must not recompute model-visible
  input/config roots from `Workspace`.

This preserves native requested staging, makes container mounts explicit, and
leaves the existing output projection and downstream postprocess interfaces
intact.

## Internal preparation seam

Introduce one runner-private preparation operation between cache admission and
the normal ActivitySim body. It consumes the resolver's immutable skim decision
and launch context and returns the exact skim mode/path the body will use.

It may run the existing Numba warmup machinery, but its semantics change:

- It uses the same staged Zarr input as the body when Zarr was selected.
- In the OMX branch, when compilation is required, it generates and read-only
  validates the production runtime Zarr location before the body begins. The
  body is then invoked in Zarr mode.
- It keeps compile-only output isolated from ordinary ActivitySim model outputs.
- It does not publish an independent Consist run, call a cache lookup, or modify
  `WorkflowState` progress as though a separate model phase occurred.

When compilation is not required, the current OMX main-body behavior remains
permitted: the body creates/finalizes Zarr after it executes. This avoids an
extra invocation solely to prepare skims in configurations where compilation
provides no multiprocessing benefit.

## Error handling

- Reject an invalid selected Zarr before the body; fall back only to a resolver
  selected OMX artifact, never an ambient workspace file.
- Reject a missing or invalid generated Zarr before switching an OMX execution
  to Zarr mode.
- Fail if the launch tree lacks any required staged configuration/data path
  before starting a container.
- A compile failure aborts the pending body execution. It cannot fall back to an
  old shared compiler cache or silently switch skim sources.
- A cache hit must not require the local compile epoch, container availability,
  or a writable compile cache.

## Verification and acceptance

Focused local tests must prove:

1. read-only valid-Zarr selection and invalid-Zarr-to-OMX fallback;
2. the three-way compile predicate (body execution, Sharrow enabled,
   multiprocessing);
3. fresh compile epoch once per process invocation and no reuse from stale
   workspace cache directories;
4. Zarr selected input is not regenerated, while OMX plus required preparation
   creates/finalizes Zarr before the body enters Zarr mode;
5. OMX with compilation skipped preserves the existing body-first Zarr
   generation path;
6. cache hydration skips preparation and the ActivitySim body;
7. container mounts derive only from the staged launch context; and
8. invalid/missing skim and failed preparation paths fail before output
   admission.

Only after those tests and a closure audit pass may the step's portable identity
be exercised with a dedicated cold-miss/fresh-workspace-hit acceptance. That
later evidence must compare ActivitySim results model-aware, confirm one body
execution, and leave the contract `incomplete` if any model-visible input or
launch root remains ambient.

## Non-goals and release boundary

This design does not promote `activitysim_run`, alter another boundary's
contract, reuse compiler products across PILATES invocations or machines, or
replace the required released-Consist HDF5 snapshot-reconciliation replay.
HDF5-consuming boundaries remain separately blocked until that released replay
and their own boundary-specific cache evidence exist.
