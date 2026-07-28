"""Semantic artifact-key invariants for the direct native step contracts.

StageRunner-era tests asserted mutable coupler state after broad stage execution.
The native surface instead declares the semantic keys at each Consist step
boundary, with bootstrap owning initial datastore publication.
"""

from types import SimpleNamespace

from consist import BindingResult

from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    BEAM_PERSONS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import STEP_DEFINITIONS, activitysim


def _declared_inputs(step_name: str) -> set[str]:
    metadata = STEP_DEFINITIONS[step_name].function.__consist_step__
    return set(metadata.inputs or ()) | set(metadata.input_keys or ())


def _declared_optional_inputs(step_name: str) -> set[str]:
    return set(
        STEP_DEFINITIONS[step_name].function.__consist_step__.optional_input_keys or ()
    )


def _declared_outputs(step_name: str) -> set[str]:
    return set(
        STEP_DEFINITIONS[step_name].function.__consist_step__.schema_outputs or ()
    )


def test_land_use_native_contract_publishes_urbansim_datastore_family() -> None:
    """Land use owns the current and forecast datastore artifacts, not a stage cache."""

    assert {USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT} <= _declared_outputs(
        "urbansim_run"
    )
    assert USIM_DATASTORE_H5 in _declared_inputs("urbansim_postprocess")


def test_vehicle_ownership_native_contract_carries_the_current_datastore() -> None:
    """ATLAS consumes the current datastore and publishes its typed follow-on roles."""

    assert USIM_DATASTORE_H5 in _declared_inputs("atlas_preprocess")
    assert {USIM_POPULATION_SOURCE_H5, ATLAS_VEHICLES2_OUTPUT} <= _declared_outputs(
        "atlas_postprocess"
    )


def test_activity_demand_native_contract_declares_activitysim_key_family() -> None:
    """The three direct ActivitySim steps expose their typed exchange keys."""

    assert {
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_OMX_SKIMS,
    } <= _declared_outputs("activitysim_preprocess")
    assert {
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
    } <= _declared_inputs("activitysim_run")
    assert ZARR_SKIMS in _declared_outputs("activitysim_run")
    assert {USIM_DATASTORE_H5, *ASIM_REQUIRED_RUN_OUTPUT_KEYS} <= _declared_outputs(
        "activitysim_postprocess"
    )


def test_activitysim_omx_selection_requires_zarr_publication() -> None:
    """A first-run OMX selection is frozen as a native invocation that emits Zarr."""

    resolved = ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(inputs={ASIM_OMX_SKIMS: object()}),
        optional_roles=(ASIM_OMX_SKIMS,),
        metadata={
            "activitysim_skim_mode": "omx",
            "activitysim_produces_zarr": True,
        },
    )

    output_paths = activitysim.activitysim_run.output_paths(
        settings=SimpleNamespace(
            run=None,
            activitysim=SimpleNamespace(output_tables={"tables": []}),
        ),
        state=SimpleNamespace(year=2030, forecast_year=2030, iteration=0),
        workspace=SimpleNamespace(
            get_asim_output_dir=lambda: "/tmp/activitysim-output"
        ),
        resolved_inputs=resolved,
    )

    assert activitysim._activitysim_run_produces_zarr(resolved) is True
    assert ZARR_SKIMS in output_paths


def test_traffic_assignment_native_contract_declares_beam_key_family() -> None:
    """The direct BEAM chain replaces stage-local output-holder publication."""

    assert {
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        LINKSTATS_WARMSTART,
    } <= _declared_outputs("beam_preprocess")
    assert {LINKSTATS, BEAM_PLANS_OUT} <= _declared_outputs("beam_run")
    assert ZARR_SKIMS in _declared_optional_inputs("beam_postprocess")
    assert ZARR_SKIMS in _declared_outputs("beam_postprocess")
