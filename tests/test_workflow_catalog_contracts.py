from __future__ import annotations

from dataclasses import fields
from pilates.runtime.scenario_runtime import build_schema_steps
from pilates.workflows import catalog
from pilates.workflows.steps import STEP_DEFINITIONS
from pilates.workflows.steps.shared import validate_workflow_step_contracts


def _metadata_keys(values) -> tuple[str, ...]:
    values = values or ()
    return tuple(values) if not isinstance(values, dict) else tuple(values)


def test_catalog_retains_only_policy_and_references_committed_definitions():
    removed_dependency_aliases = (
        "_".join(("holder", "inputs")),
        "_".join(("upstream", "step", "inputs")),
    )
    policy_fields = {field.name for field in fields(catalog.WorkflowStepSpec)}
    assert policy_fields.isdisjoint(
        {
            "input_keys",
            "optional_input_keys",
            "output_keys",
            "optional_output_keys",
            "model_name",
            *removed_dependency_aliases,
        }
    )

    for spec in catalog.WORKFLOW_STEP_SPECS:
        definition = STEP_DEFINITIONS[spec.step_name]
        metadata = definition.function.__consist_step__
        assert spec.definition is definition
        assert spec.input_keys == _metadata_keys(metadata.inputs)
        assert spec.optional_input_keys == _metadata_keys(metadata.optional_input_keys)
        assert spec.schema_output_keys == _metadata_keys(metadata.schema_outputs)
        assert set(spec.output_keys) | set(spec.optional_output_keys) == set(
            spec.schema_output_keys
        )
        assert not hasattr(spec, "model_name")
        assert all(
            not hasattr(spec, field_name) for field_name in removed_dependency_aliases
        )


def test_native_schema_steps_allow_shared_consist_model_metadata():
    """Preprocess/run/postprocess may share a Consist model category."""
    validate_workflow_step_contracts(
        declared_steps=build_schema_steps(),
        require_all_tracked_declared=False,
    )


def test_workflow_step_contract_export_projects_native_metadata():
    contracts = catalog.workflow_step_contracts_by_name()

    for spec in catalog.WORKFLOW_STEP_SPECS:
        definition = STEP_DEFINITIONS[spec.step_name]
        metadata = definition.function.__consist_step__
        contract = contracts[spec.step_name]
        assert contract["input_keys"] == list(_metadata_keys(metadata.inputs))
        assert contract["optional_input_keys"] == list(
            _metadata_keys(metadata.optional_input_keys)
        )
        assert contract["output_keys"] == list(spec.output_keys)
        assert set(contract["output_keys"]) | set(
            contract["optional_output_keys"]
        ) == set(contract["schema_outputs"])
        assert contract["depends_on"] == list(spec.depends_on)


def test_catalog_dependency_exports_only_project_depends_on():
    for dependency_map in (
        catalog.step_dependencies_from_catalog(),
        catalog.runtime_step_dependencies_from_catalog(),
    ):
        assert all(
            set(projection) == {"depends_on"} for projection in dependency_map.values()
        )


def test_catalog_key_matching_uses_native_static_metadata_and_policy_families():
    assert catalog.workflow_step_key_is_declared(
        "activitysim_run", "land_use_asim_in", direction="input"
    )
    assert catalog.workflow_step_key_is_declared(
        "beam_run", "linkstats", direction="output"
    )
    assert catalog.workflow_step_key_is_declared(
        "beam_postprocess", "events_parquet_2030_1", direction="input"
    )
    assert catalog.workflow_step_key_is_declared(
        "beam_postprocess", "path_traversal_links_2030_1", direction="output"
    )


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


def test_catalog_contract_uses_native_schema_outputs():
    contract = catalog.workflow_step_contracts_by_name()["atlas_preprocess"]
    declared = contract["output_keys"]
    expected = list(
        STEP_DEFINITIONS["atlas_preprocess"].function.__consist_step__.schema_outputs
    )
    assert declared == expected
    assert contract["optional_output_keys"] == []


def test_catalog_contract_retains_atlas_dynamic_output_policy():
    spec = catalog.workflow_step_spec_for_step_name("atlas_run")
    assert spec is not None
    assert spec.dynamic_output_families == ("householdv_{year}", "vehicles_{year}")
