from pilates.workflows import catalog
from pilates.workflows.steps import STEP_DEFINITIONS
from pilates.runtime.launcher import _build_schema_steps
from types import SimpleNamespace


def test_schema_steps_follow_catalog_order():
    schema_steps = _build_schema_steps()
    expected = [
        STEP_DEFINITIONS[spec.step_name].function
        for spec in catalog.schema_step_specs()
    ]
    assert schema_steps == expected


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


def test_provenance_metadata_is_optional_for_postprocessing():
    untracked_without_provenance = [
        spec.step_name
        for spec in catalog.WORKFLOW_STEP_SPECS
        if not spec.tracked and spec.provenance is None
    ]
    assert "postprocessing" in untracked_without_provenance

    assert catalog.provenance_builder_key_for_step_name("postprocessing") is None


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


def test_contracts_module_is_leaf_and_import_graph_is_acyclic():
    import ast
    from pathlib import Path
    from pilates.workflows import consist_metadata, contracts

    workflows_dir = Path(contracts.__file__).resolve().parent
    contracts_tree = ast.parse((workflows_dir / "contracts.py").read_text())
    consist_metadata_tree = ast.parse(
        (workflows_dir / "consist_metadata.py").read_text()
    )
    catalog_tree = ast.parse((workflows_dir / "catalog.py").read_text())

    def _imported_modules(tree: ast.AST) -> set[str]:
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                modules.add(node.args[0].value)
        return modules

    contracts_imports = _imported_modules(contracts_tree)
    consist_metadata_imports = _imported_modules(consist_metadata_tree)
    catalog_imports = _imported_modules(catalog_tree)

    assert not any(
        mod == "pilates.workflows.catalog"
        or mod == "pilates.workflows.steps"
        or mod.startswith("pilates.workflows.steps.")
        for mod in contracts_imports
    )
    assert not any(
        mod == "pilates.workflows.steps" or mod.startswith("pilates.workflows.steps.")
        for mod in consist_metadata_imports
    )
    assert "pilates.workflows.steps" in catalog_imports

    for step_file in (workflows_dir / "steps").glob("*.py"):
        step_imports = _imported_modules(ast.parse(step_file.read_text()))
        assert "pilates.workflows.catalog" not in step_imports

    for spec in catalog.WORKFLOW_STEP_SPECS:
        assert isinstance(spec, catalog.WorkflowStepSpec)
        assert isinstance(spec.definition, contracts.StepDefinition)
    assert callable(consist_metadata.consist_step_meta)
    assert callable(consist_metadata.build_step_consist_kwargs)
