from __future__ import annotations

from types import SimpleNamespace

from pilates.activitysim.outputs import (
    ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
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
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_SHARROW_CACHE_DIR,
    BEAM_CONFIG_FILE,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    OMX_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_NEXT,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows import catalog
from pilates.workflows.orchestration import _declared_required_and_optional_output_keys


def test_selected_catalog_step_contract_metadata_matches_current_wiring():
    expected = {
        "urbansim_preprocess": {
            "input_keys": (USIM_DATASTORE_BASE_H5,),
            "optional_input_keys": (
                USIM_DATASTORE_CURRENT_H5,
                FINAL_SKIMS_OMX,
                OMX_SKIMS,
            ),
            "optional_output_keys": (
                "usim_skims_input_updated",
                USIM_DATASTORE_BASE_H5,
            ),
            "dynamic_input_families": (),
            "output_keys": (
                USIM_DATASTORE_H5,
                "omx_skims",
                "hh_size",
                "income_rates",
                "relmap",
                "geoid_to_zone",
                "schools",
                "school_districts",
            ),
            "dynamic_output_families": (),
            "holder_inputs": (),
            "upstream_step_inputs": (),
        },
        "urbansim_run": {
            "input_keys": (
                USIM_DATASTORE_H5,
                "omx_skims",
                "hh_size",
                "income_rates",
                "relmap",
                "geoid_to_zone",
                "schools",
                "school_districts",
            ),
            "optional_input_keys": ("usim_skims_input_updated", USIM_DATASTORE_BASE_H5),
            "optional_output_keys": (USIM_FORECAST_OUTPUT,),
            "dynamic_input_families": (),
            "output_keys": (USIM_DATASTORE_H5,),
            "dynamic_output_families": (),
            "holder_inputs": ("urbansim_preprocess",),
            "upstream_step_inputs": ("urbansim_preprocess",),
        },
        "urbansim_postprocess": {
            "input_keys": (USIM_DATASTORE_H5,),
            "optional_input_keys": (USIM_DATASTORE_BASE_H5,),
            "optional_output_keys": (),
            "dynamic_input_families": (),
            "output_keys": (USIM_DATASTORE_H5,),
            "dynamic_output_families": (
                "usim_input_archive_{year}",
                "usim_input_merged_{year}",
            ),
            "holder_inputs": ("urbansim_run",),
            "upstream_step_inputs": ("urbansim_run",),
        },
        "atlas_preprocess": {
            "input_keys": (
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
            "optional_input_keys": (FINAL_SKIMS_OMX, OMX_SKIMS),
            "optional_output_keys": (
                *catalog._ATLAS_PREPROCESS_OPTIONAL_OUTPUT_KEYS,
                *catalog._ATLAS_STATIC_INPUT_KEYS,
            ),
            "dynamic_input_families": (),
            "output_keys": (
                "atlas_households_csv",
                "atlas_blocks_csv",
                "atlas_persons_csv",
                "atlas_residential_csv",
                "atlas_jobs_csv",
            ),
            "dynamic_output_families": (),
            "holder_inputs": (),
            "upstream_step_inputs": (),
        },
        "atlas_run": {
            "input_keys": (
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
                "atlas_households_csv",
                "atlas_blocks_csv",
                "atlas_persons_csv",
                "atlas_residential_csv",
                "atlas_jobs_csv",
            ),
            "optional_input_keys": (
                *catalog._ATLAS_PREPROCESS_OPTIONAL_OUTPUT_KEYS,
                *catalog._ATLAS_STATIC_INPUT_KEYS,
            ),
            "optional_output_keys": (),
            "dynamic_input_families": (),
            "output_keys": (),
            "dynamic_output_families": (
                "householdv_{year}",
                "vehicles_{year}",
            ),
            "holder_inputs": ("atlas_preprocess",),
            "upstream_step_inputs": ("atlas_preprocess",),
        },
        "atlas_postprocess": {
            "input_keys": (USIM_DATASTORE_CURRENT_H5,),
            "optional_input_keys": (),
            "optional_output_keys": (),
            "dynamic_input_families": (
                "householdv_{year}",
                "vehicles_{year}",
            ),
            "output_keys": (
                USIM_POPULATION_SOURCE_H5,
                ATLAS_VEHICLES2_OUTPUT,
            ),
            "dynamic_output_families": (),
            "holder_inputs": ("atlas_run",),
            "upstream_step_inputs": ("atlas_run",),
        },
        "activitysim_preprocess": {
            "input_keys": (USIM_POPULATION_SOURCE_H5,),
            "optional_input_keys": (FINAL_SKIMS_OMX,),
            "optional_output_keys": (),
            "dynamic_input_families": (),
            "output_keys": (
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
            "input_keys": (ASIM_OMX_SKIMS,),
            "optional_input_keys": (),
            "output_keys": (ZARR_SKIMS,),
            "optional_output_keys": (ASIM_SHARROW_CACHE_DIR,),
            "dynamic_input_families": (),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_preprocess",),
            "upstream_step_inputs": ("activitysim_preprocess",),
        },
        "activitysim_run": {
            "input_keys": (
                ASIM_LAND_USE_IN,
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
                ZARR_SKIMS,
            ),
            "optional_input_keys": (ASIM_SHARROW_CACHE_DIR,),
            "optional_output_keys": ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
            "dynamic_input_families": (),
            "output_keys": (*ActivitySimRunOutputs.required_output_keys(),),
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
                *ActivitySimRunOutputs.required_output_keys(),
            ),
            "optional_input_keys": (
                *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ),
            "optional_output_keys": (
                *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
                "asim_input_households_csv_archived",
                "asim_input_persons_csv_archived",
                "asim_input_land_use_csv_archived",
                "asim_input_skims_omx_archived",
                "asim_input_skims_zarr_archived",
            ),
            "dynamic_input_families": (),
            "output_keys": (
                *ActivitySimRunOutputs.required_output_keys(),
                USIM_DATASTORE_H5,
            ),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_run",),
            "upstream_step_inputs": ("activitysim_run",),
        },
        "beam_preprocess": {
            "input_keys": (
                BEAM_CONFIG_FILE,
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
            ),
            "optional_input_keys": (LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT),
            "optional_output_keys": ("vehicles_beam_in", LINKSTATS_WARMSTART),
            "dynamic_input_families": (),
            "output_keys": (*BeamPreprocessOutputs.required_output_keys(),),
            "dynamic_output_families": (),
            "holder_inputs": ("activitysim_postprocess",),
            "upstream_step_inputs": ("activitysim_postprocess",),
        },
        "beam_run": {
            "input_keys": (
                BEAM_CONFIG_FILE,
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
            ),
            "optional_input_keys": (LINKSTATS_WARMSTART,),
            "optional_output_keys": (
                LINKSTATS_WARMSTART,
                BEAM_OUTPUT_PLANS_XML,
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
                BEAM_EXPERIENCED_PLANS_XML,
                *catalog._BEAM_RUN_ARCHIVE_OUTPUT_KEYS,
            ),
            "dynamic_input_families": (),
            "output_keys": (*BeamRunOutputs.declared_output_keys(),),
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
            "optional_input_keys": (ZARR_SKIMS,),
            "optional_output_keys": (FINAL_SKIMS_OMX,),
            "dynamic_input_families": (
                "events_parquet_{year}_{iteration}",
                "raw_od_skims_{year}_{iteration}",
                "raw_od_skims_zarr_{year}_{iteration}",
            ),
            "output_keys": (ZARR_SKIMS,),
            "dynamic_output_families": (
                "events_parquet_{year}_{iteration}",
                "path_traversal_links_{year}_{iteration}",
            ),
            "holder_inputs": ("beam_run",),
            "upstream_step_inputs": ("beam_run",),
        },
        "beam_full_skim": {
            "input_keys": (
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
                LINKSTATS_WARMSTART,
            ),
            "optional_input_keys": (),
            "optional_output_keys": (),
            "dynamic_input_families": (),
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
        assert spec.optional_output_keys == contract["optional_output_keys"]
        assert spec.dynamic_input_families == contract["dynamic_input_families"]
        assert spec.dynamic_output_families == contract["dynamic_output_families"]
        assert spec.holder_inputs == contract["holder_inputs"]
        assert spec.upstream_step_inputs == contract["upstream_step_inputs"]


def test_workflow_step_contract_export_is_serializable_and_aligned():
    contracts = catalog.workflow_step_contracts_by_name()

    assert contracts["activitysim_run"] == {
        "step_name": "activitysim_run",
        "stage_name": "activity_demand",
        "phase": "run",
        "depends_on": ["activitysim_preprocess"],
        "input_keys": [
            ASIM_LAND_USE_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
            ZARR_SKIMS,
        ],
        "optional_input_keys": [ASIM_SHARROW_CACHE_DIR],
        "optional_output_keys": list(ASIM_OPTIONAL_RUN_OUTPUT_KEYS),
        "dynamic_input_families": [],
        "upstream_step_inputs": ["activitysim_preprocess"],
        "output_keys": [
            *ActivitySimRunOutputs.required_output_keys(),
        ],
        "dynamic_output_families": [],
        "optional": False,
    }
    assert contracts["urbansim_run"] == {
        "step_name": "urbansim_run",
        "stage_name": "land_use",
        "phase": "run",
        "depends_on": ["urbansim_preprocess"],
        "input_keys": [
            USIM_DATASTORE_H5,
            "omx_skims",
            "hh_size",
            "income_rates",
            "relmap",
            "geoid_to_zone",
            "schools",
            "school_districts",
        ],
        "optional_input_keys": ["usim_skims_input_updated", USIM_DATASTORE_BASE_H5],
        "optional_output_keys": [USIM_FORECAST_OUTPUT],
        "dynamic_input_families": [],
        "upstream_step_inputs": ["urbansim_preprocess"],
        "output_keys": [USIM_DATASTORE_H5],
        "dynamic_output_families": [],
        "optional": False,
    }
    assert contracts["atlas_postprocess"] == {
        "step_name": "atlas_postprocess",
        "stage_name": "vehicle_ownership_model",
        "phase": "postprocess",
        "depends_on": ["atlas_run"],
        "input_keys": [USIM_DATASTORE_CURRENT_H5],
        "optional_input_keys": [],
        "optional_output_keys": [],
        "dynamic_input_families": [
            "householdv_{year}",
            "vehicles_{year}",
        ],
        "upstream_step_inputs": ["atlas_run"],
        "output_keys": [
            USIM_POPULATION_SOURCE_H5,
            ATLAS_VEHICLES2_OUTPUT,
        ],
        "dynamic_output_families": [],
        "optional": False,
    }
    assert contracts["activitysim_postprocess"] == {
        "step_name": "activitysim_postprocess",
        "stage_name": "activity_demand",
        "phase": "postprocess",
        "depends_on": ["activitysim_run"],
        "input_keys": [
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
            ASIM_LAND_USE_IN,
            ASIM_OMX_SKIMS,
            ZARR_SKIMS,
            *ActivitySimRunOutputs.required_output_keys(),
        ],
        "optional_input_keys": [
            *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
            USIM_DATASTORE_BASE_H5,
        ],
        "optional_output_keys": [
            *ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
            "asim_input_households_csv_archived",
            "asim_input_persons_csv_archived",
            "asim_input_land_use_csv_archived",
            "asim_input_skims_omx_archived",
            "asim_input_skims_zarr_archived",
        ],
        "dynamic_input_families": [],
        "upstream_step_inputs": ["activitysim_run"],
        "output_keys": [
            *ActivitySimRunOutputs.required_output_keys(),
            USIM_DATASTORE_H5,
        ],
        "dynamic_output_families": [],
        "optional": False,
    }
    assert contracts["beam_preprocess"] == {
        "step_name": "beam_preprocess",
        "stage_name": "traffic_assignment",
        "phase": "preprocess",
        "depends_on": ["activitysim_postprocess"],
        "input_keys": [
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ],
        "optional_input_keys": [LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT],
        "optional_output_keys": ["vehicles_beam_in", LINKSTATS_WARMSTART],
        "dynamic_input_families": [],
        "upstream_step_inputs": ["activitysim_postprocess"],
        "output_keys": [
            *BeamPreprocessOutputs.required_output_keys(),
        ],
        "dynamic_output_families": [],
        "optional": False,
    }
    assert contracts["beam_run"] == {
        "step_name": "beam_run",
        "stage_name": "traffic_assignment",
        "phase": "run",
        "depends_on": ["beam_preprocess"],
        "input_keys": [
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ],
        "optional_input_keys": [LINKSTATS_WARMSTART],
        "optional_output_keys": [
            LINKSTATS_WARMSTART,
            BEAM_OUTPUT_PLANS_XML,
            BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            BEAM_EXPERIENCED_PLANS_XML,
            *catalog._BEAM_RUN_ARCHIVE_OUTPUT_KEYS,
        ],
        "dynamic_input_families": [],
        "upstream_step_inputs": ["beam_preprocess"],
        "output_keys": [*BeamRunOutputs.declared_output_keys()],
        "dynamic_output_families": [
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
        ],
        "optional": False,
    }
    assert contracts["beam_postprocess"] == {
        "step_name": "beam_postprocess",
        "stage_name": "traffic_assignment",
        "phase": "postprocess",
        "depends_on": ["beam_run"],
        "input_keys": [],
        "optional_input_keys": [ZARR_SKIMS],
        "optional_output_keys": [FINAL_SKIMS_OMX],
        "dynamic_input_families": [
            "events_parquet_{year}_{iteration}",
            "raw_od_skims_{year}_{iteration}",
            "raw_od_skims_zarr_{year}_{iteration}",
        ],
        "upstream_step_inputs": ["beam_run"],
        "output_keys": [ZARR_SKIMS],
        "dynamic_output_families": [
            "events_parquet_{year}_{iteration}",
            "path_traversal_links_{year}_{iteration}",
        ],
        "optional": False,
    }


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
        declared_contract = set(spec.output_keys) | set(spec.optional_output_keys)
        assert set(canonical).issubset(declared_contract)


def test_beam_catalog_dynamic_families_capture_runtime_fan_out():
    beam_run = catalog.workflow_step_spec_for_step_name("beam_run")
    beam_postprocess = catalog.workflow_step_spec_for_step_name("beam_postprocess")
    assert beam_run is not None
    assert beam_postprocess is not None

    assert "beam_output_*" in beam_run.dynamic_output_families
    assert (
        "events_parquet_{year}_{iteration}" in beam_postprocess.dynamic_output_families
    )
    assert (
        "path_traversal_links_{year}_{iteration}"
        in beam_postprocess.dynamic_output_families
    )


def test_atlas_preprocess_audit_contract_uses_settings_specialized_optional_outputs():
    settings = SimpleNamespace(
        atlas=SimpleNamespace(adscen="zev_mandate", scenario="zev_mandate")
    )

    declared, required, optional = _declared_required_and_optional_output_keys(
        "atlas_preprocess",
        settings=settings,
    )

    assert set(declared) == set(catalog._ATLAS_PREPROCESS_CORE_OUTPUT_KEYS)
    assert set(required) == set(catalog._ATLAS_PREPROCESS_CORE_OUTPUT_KEYS)
    assert "vehicle_type_mapping_evMandForced2" in optional
    assert "vehicle_type_mapping_baseline" not in optional
    assert "vehicle_type_mapping_ESS_const_220_price" not in optional
    assert "adopt/zev_mandate/new_vehicles" in optional
    assert "adopt/baseline/new_vehicles" not in optional
    assert "adopt/ess_cons/new_vehicles" not in optional


def test_atlas_run_audit_contract_expands_dynamic_required_outputs():
    declared, required, optional = _declared_required_and_optional_output_keys(
        "atlas_run",
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2021, forecast_year=2023, iteration=0),
    )

    assert declared == []
    assert required == ["householdv_2023", "vehicles_2023"]
    assert optional == []
