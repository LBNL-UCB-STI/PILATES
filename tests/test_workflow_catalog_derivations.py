from pilates.workflows import catalog
from pilates.workflows.steps import STEP_DEFINITIONS
from pilates.workflows.steps import shared as step_shared
from pilates.runtime.launcher import _build_schema_steps
from types import SimpleNamespace
from types import MappingProxyType


def test_step_outputs_classes_are_an_immutable_legacy_holder_map():
    assert isinstance(step_shared.STEP_OUTPUTS_CLASSES, MappingProxyType)
    assert set(step_shared.STEP_OUTPUTS_CLASSES) == {
        spec.step_name for spec in catalog.tracked_step_specs()
    }

    try:
        step_shared.STEP_OUTPUTS_CLASSES["new_step"] = object  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover - MappingProxyType always rejects assignment.
        raise AssertionError("legacy holder output map must be immutable")


def test_step_dependencies_are_catalog_derived():
    expected = {
        spec.step_name: {
            "depends_on": list(spec.depends_on),
            "holder_inputs": list(spec.holder_inputs),
        }
        for spec in catalog.tracked_step_specs()
    }
    assert step_shared.STEP_DEPENDENCIES == expected


def test_runtime_step_dependencies_match_catalog_steps():
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
    expected = [
        STEP_DEFINITIONS[spec.step_name].function
        for spec in catalog.schema_step_specs()
    ]
    assert schema_steps == expected


def test_schema_steps_do_not_construct_legacy_factory_closures(monkeypatch):
    monkeypatch.setattr(
        "pilates.workflows.steps.schema_step_builder_registry",
        lambda: (_ for _ in ()).throw(AssertionError("legacy factory registry used")),
    )

    assert _build_schema_steps() == [
        STEP_DEFINITIONS[spec.step_name].function
        for spec in catalog.schema_step_specs()
    ]


def test_tracked_catalog_steps_do_not_own_typed_output_classes():
    for spec in catalog.tracked_step_specs():
        assert not hasattr(spec, "outputs_class")


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


def test_provenance_metadata_is_optional_for_postprocessing():
    untracked_without_provenance = [
        spec.step_name
        for spec in catalog.WORKFLOW_STEP_SPECS
        if not spec.tracked and spec.provenance is None
    ]
    assert "postprocessing" in untracked_without_provenance

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
        is_model_enabled=lambda current_settings, *, flag_attr, model_attr: (
            bool(getattr(current_settings, flag_attr, None))
            or bool(
                getattr(
                    getattr(getattr(current_settings, "run", None), "models", None),
                    model_attr,
                    None,
                )
            )
        ),
        include_optional=False,
    )
    assert all(not model.startswith("urbansim_") for model in enabled)
    assert all(not model.startswith("atlas_") for model in enabled)
    assert any(model.startswith("activitysim_") for model in enabled)
    assert any(model.startswith("beam_") for model in enabled)
    assert "beam_full_skim" not in enabled
