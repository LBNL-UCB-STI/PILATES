from types import SimpleNamespace

from run import _build_schema_steps, _filter_schema_steps_for_enabled_models
from run import _is_model_enabled
from pilates.workflows.catalog import enabled_schema_step_models


def test_filter_schema_steps_for_activitysim_beam_only_excludes_urbansim_atlas():
    settings = SimpleNamespace(
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
    )

    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        settings,
        include_optional=False,
    )

    models = {step.__consist_step__.model for step in required_steps}

    assert any(model.startswith("activitysim_") for model in models)
    assert any(model.startswith("beam_") for model in models)
    assert all(not model.startswith("urbansim_") for model in models)
    assert all(not model.startswith("atlas_") for model in models)
    assert "beam_full_skim" not in models


def test_filter_schema_steps_matches_catalog_enablement_models():
    settings = SimpleNamespace(
        land_use_enabled=True,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
    )
    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        settings,
        include_optional=False,
    )
    models = {step.__consist_step__.model for step in required_steps}
    expected = enabled_schema_step_models(
        settings,
        is_model_enabled=_is_model_enabled,
        include_optional=False,
    )
    assert models == expected


def test_filter_schema_steps_all_disabled_returns_empty():
    settings = SimpleNamespace(
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=False,
        traffic_assignment_enabled=False,
    )
    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        settings,
        include_optional=False,
    )
    assert required_steps == []


def test_filter_schema_steps_optional_toggle_controls_beam_full_skim():
    settings = SimpleNamespace(
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
    )
    all_steps = _build_schema_steps()
    with_optional = _filter_schema_steps_for_enabled_models(
        all_steps,
        settings,
        include_optional=True,
    )
    without_optional = _filter_schema_steps_for_enabled_models(
        all_steps,
        settings,
        include_optional=False,
    )
    with_models = {step.__consist_step__.model for step in with_optional}
    without_models = {step.__consist_step__.model for step in without_optional}
    assert "beam_full_skim" in with_models
    assert "beam_full_skim" not in without_models
