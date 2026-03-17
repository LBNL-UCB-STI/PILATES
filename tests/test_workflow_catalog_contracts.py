from __future__ import annotations

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.workflows.artifact_keys import (
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_SHARROW_CACHE_DIR,
    BEAM_CONFIG_FILE,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_NEXT,
    ZARR_SKIMS,
)
from pilates.workflows import catalog


def test_selected_catalog_step_contract_metadata_matches_current_wiring():
    expected = {
        "activitysim_preprocess": {
            "input_keys": (USIM_H5_UPDATED,),
            "optional_input_keys": (
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
            "output_keys": (
                ASIM_MUTABLE_DATA_DIR,
                ASIM_LAND_USE_IN,
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
                ASIM_OMX_SKIMS,
            ),
            "dynamic_output_families": (),
            "holder_inputs": (),
            "upstream_step_inputs": (),
        },
        "activitysim_compile": {
            "input_keys": (),
            "optional_input_keys": (),
            "output_keys": (ZARR_SKIMS, ASIM_SHARROW_CACHE_DIR),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_preprocess",),
            "upstream_step_inputs": ("activitysim_preprocess",),
        },
        "activitysim_run": {
            "input_keys": (ZARR_SKIMS,),
            "optional_input_keys": (),
            "output_keys": (
                ASIM_OUTPUT_DIR,
                *ActivitySimRunOutputs.declared_output_keys(),
            ),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_preprocess",),
            "upstream_step_inputs": ("activitysim_preprocess",),
        },
        "activitysim_postprocess": {
            "input_keys": (
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
                ASIM_LAND_USE_IN,
                ASIM_OMX_SKIMS,
                ZARR_SKIMS,
                USIM_DATASTORE_H5,
                USIM_FORECAST_OUTPUT,
            ),
            "optional_input_keys": (),
            "output_keys": (
                ASIM_OUTPUT_DIR,
                *ActivitySimRunOutputs.declared_output_keys(),
                USIM_INPUT_NEXT,
                USIM_DATASTORE_H5,
            ),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_run",),
            "upstream_step_inputs": ("activitysim_run",),
        },
        "beam_preprocess": {
            "input_keys": (BEAM_CONFIG_FILE,),
            "optional_input_keys": (),
            "output_keys": (
                BEAM_MUTABLE_DATA_DIR,
                *BeamPreprocessOutputs.declared_output_keys(),
                LINKSTATS_WARMSTART,
            ),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_postprocess",),
            "upstream_step_inputs": ("activitysim_postprocess",),
        },
        "beam_run": {
            "input_keys": (BEAM_CONFIG_FILE,),
            "optional_input_keys": (
                LINKSTATS_WARMSTART,
                BEAM_OUTPUT_PLANS_XML,
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
                BEAM_EXPERIENCED_PLANS_XML,
            ),
            "output_keys": (
                BEAM_OUTPUT_DIR,
                *BeamRunOutputs.declared_output_keys(),
                BEAM_OUTPUT_PLANS_XML,
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
                BEAM_EXPERIENCED_PLANS_XML,
            ),
            "dynamic_output_families": (
                "linkstats_{year}_{iteration}",
                "linkstats_parquet_{year}_{iteration}",
                "linkstats_unmodified_{year}_{iteration}",
                "linkstats_unmodified_parquet_{year}_{iteration}",
                "events_{year}_{iteration}",
                "events_parquet_{year}_{iteration}",
                "raw_od_skims_{year}_{iteration}",
                "raw_od_skims_zarr_{year}_{iteration}",
                "beam_plans_{year}_{iteration}",
                "beam_experienced_plans_{year}_{iteration}",
                "beam_output_*",
            ),
            "holder_inputs": ("beam_preprocess",),
            "upstream_step_inputs": ("beam_preprocess",),
        },
        "beam_postprocess": {
            "input_keys": (),
            "optional_input_keys": (),
            "output_keys": (
                BEAM_OUTPUT_DIR,
                *BeamRunOutputs.declared_output_keys(),
                BEAM_OUTPUT_PLANS_XML,
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
                BEAM_EXPERIENCED_PLANS_XML,
                ZARR_SKIMS,
                FINAL_SKIMS_OMX,
            ),
            "dynamic_output_families": (
                "events_parquet_{year}_{iteration}",
                "path_traversal_links_{year}_{iteration}",
            ),
            "holder_inputs": ("beam_run",),
            "upstream_step_inputs": ("beam_run",),
        },
        "beam_full_skim": {
            "input_keys": (),
            "optional_input_keys": (),
            "output_keys": (BEAM_FULL_SKIMS,),
            "dynamic_output_families": (),
            "holder_inputs": ("beam_preprocess",),
            "upstream_step_inputs": ("beam_preprocess",),
        },
    }

    for step_name, contract in expected.items():
        spec = catalog.workflow_step_spec_for_step_name(step_name)
        assert spec is not None
        assert spec.input_keys == contract["input_keys"]
        assert spec.optional_input_keys == contract["optional_input_keys"]
        assert spec.output_keys == contract["output_keys"]
        assert spec.dynamic_output_families == contract["dynamic_output_families"]
        assert spec.holder_inputs == contract["holder_inputs"]
        assert spec.upstream_step_inputs == contract["upstream_step_inputs"]


def test_catalog_output_keys_cover_declared_step_output_contracts():
    covered_steps = {
        "activitysim_preprocess",
        "activitysim_compile",
        "activitysim_run",
        "activitysim_postprocess",
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
    }
    for spec in catalog.WORKFLOW_STEP_SPECS:
        if spec.step_name not in covered_steps:
            continue
        if spec.outputs_class is None:
            continue
        canonical = tuple(spec.outputs_class.declared_output_keys())
        assert set(canonical).issubset(set(spec.output_keys))


def test_beam_catalog_dynamic_families_capture_runtime_fan_out():
    beam_run = catalog.workflow_step_spec_for_step_name("beam_run")
    beam_postprocess = catalog.workflow_step_spec_for_step_name("beam_postprocess")
    assert beam_run is not None
    assert beam_postprocess is not None

    assert "beam_output_*" in beam_run.dynamic_output_families
    assert "events_parquet_{year}_{iteration}" in beam_postprocess.dynamic_output_families
    assert "path_traversal_links_{year}_{iteration}" in beam_postprocess.dynamic_output_families
