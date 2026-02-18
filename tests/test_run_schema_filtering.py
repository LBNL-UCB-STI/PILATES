from types import SimpleNamespace

from run import _build_schema_steps, _filter_schema_steps_for_enabled_models


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
