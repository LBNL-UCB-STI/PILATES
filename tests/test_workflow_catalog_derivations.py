from pilates.workflows import catalog
from pilates.workflows.steps import shared as step_shared
from run import _build_schema_steps


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
