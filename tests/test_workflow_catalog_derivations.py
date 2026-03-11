from pilates.workflows import catalog
from pilates.workflows.steps import shared as step_shared
from run import _build_schema_steps
from run import _is_model_enabled
from types import SimpleNamespace


def test_step_outputs_classes_are_catalog_derived():
    expected = {
        spec.step_name: spec.outputs_class
        for spec in catalog.tracked_step_specs()
    }
    assert step_shared.STEP_OUTPUTS_CLASSES == expected


def test_step_dependencies_are_catalog_derived():
    expected = {
        spec.step_name: {
            "depends_on": list(spec.depends_on),
            "holder_inputs": list(spec.holder_inputs),
        }
        for spec in catalog.tracked_step_specs()
    }
    assert step_shared.STEP_DEPENDENCIES == expected


def test_runtime_step_dependencies_include_untracked_schema_steps():
    expected = {
        spec.step_name: {
            "depends_on": list(spec.depends_on),
            "holder_inputs": list(spec.holder_inputs),
        }
        for spec in catalog.WORKFLOW_STEP_SPECS
    }
    assert step_shared.STEP_RUNTIME_DEPENDENCIES == expected


def test_schema_steps_follow_catalog_order():
    schema_steps = _build_schema_steps()
    models = [step.__consist_step__.model for step in schema_steps]
    assert models == list(catalog.schema_step_names())


def test_tracked_catalog_steps_define_outputs_class():
    for spec in catalog.tracked_step_specs():
        assert spec.outputs_class is not None


def test_catalog_step_names_are_unique():
    names = [spec.step_name for spec in catalog.WORKFLOW_STEP_SPECS]
    assert len(names) == len(set(names))


def test_tracked_steps_define_provenance_builder_keys():
    for spec in catalog.tracked_step_specs():
        assert spec.provenance is not None
        assert spec.provenance.builder_key
        assert (
            catalog.provenance_builder_key_for_step_name(spec.step_name)
            == spec.provenance.builder_key
        )
        assert (
            catalog.provenance_builder_key_for_model_name(spec.model_name)
            == spec.provenance.builder_key
        )


def test_provenance_metadata_is_optional_for_untracked_steps():
    untracked_without_provenance = [
        spec.step_name
        for spec in catalog.WORKFLOW_STEP_SPECS
        if not spec.tracked and spec.provenance is None
    ]
    assert "activitysim_compile" in untracked_without_provenance
    assert "postprocessing" in untracked_without_provenance

    assert catalog.provenance_builder_key_for_model_name("activitysim_compile") is None
    assert catalog.provenance_builder_key_for_model_name("postprocessing") is None


def test_enabled_schema_step_models_honors_settings_flags():
    settings = SimpleNamespace(
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
    )
    enabled = catalog.enabled_schema_step_models(
        settings,
        is_model_enabled=_is_model_enabled,
        include_optional=False,
    )
    assert all(not model.startswith("urbansim_") for model in enabled)
    assert all(not model.startswith("atlas_") for model in enabled)
    assert any(model.startswith("activitysim_") for model in enabled)
    assert any(model.startswith("beam_") for model in enabled)
    assert "beam_full_skim" not in enabled
