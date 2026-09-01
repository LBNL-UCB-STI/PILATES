# PILATES Land-Use and Skim Source Guide

PILATES treats the selected durable `ZARR_SKIMS` artifact as the authoritative
skim source for an ActivitySim run. The resolver binds that artifact to the
current runtime destination; it does not infer a skim source from an old
workspace path.

## Source Precedence

For a normal ActivitySim run, a selected `ZARR_SKIMS` source is used when it is
available. An OMX skim is the compile-time source used to create the local Zarr
store only when no `ZARR_SKIMS` source is selected. A selected Zarr that cannot
be staged or validated fails the run; PILATES does not silently fall back to
OMX. The resulting Zarr from an OMX-mode run is normalized and published as the
durable skim artifact for downstream use.

This is a PILATES workflow contract, not a promise about ActivitySim's internal
table loading, recoding, or coordinate implementation. Those details belong to
the selected ActivitySim version and configuration.

## Zone Alignment

The land-use table and the skim source must use the same configured zone system.
Before a run, verify that the chosen source matches the scenario's land-use
inputs and that no stale local Zarr directory is being mistaken for the
selected durable artifact. PILATES stages the selected Zarr to its runtime
location and validates and normalizes newly generated Zarr before publication.

If zone identifiers or ordering change, regenerate the source skims for that
zone system and publish a new durable `ZARR_SKIMS` artifact. Do not repair
alignment by substituting an OMX file while a selected durable Zarr is present.

## Related Code

- `pilates/workflows/steps/activitysim.py` resolves the selected skim role.
- `pilates/activitysim/inputs.py` reads the selected `ZARR_SKIMS` coupler value.
- `pilates/activitysim/runner.py` stages and finalizes runtime Zarr skims.
