from types import SimpleNamespace

import pytest

from pilates.runtime.launcher import (
    _build_schema_steps,
    _filter_schema_steps_for_enabled_models,
)
from pilates.workflows.catalog import enabled_schema_step_models
from pilates.workflows.surface import build_enabled_workflow_surface


def _settings(
    *,
    land_use: str | None = None,
    vehicle_ownership: str | None = None,
    activity_demand: str | None = None,
    traffic_assignment: str | None = None,
):
    return SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=land_use,
                vehicle_ownership=vehicle_ownership,
                activity_demand=activity_demand,
                traffic_assignment=traffic_assignment,
                travel=traffic_assignment,
            )
        ),
        activitysim=SimpleNamespace(replan_iters=0, main_configs_dir="configs"),
        atlas=SimpleNamespace(scenario="baseline", adscen="baseline"),
    )


def _surface(settings):
    return build_enabled_workflow_surface(settings)


def test_filter_schema_steps_for_activitysim_beam_only_excludes_urbansim_atlas():
    settings = _settings(
        activity_demand="activitysim",
        traffic_assignment="beam",
    )

    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=False,
        surface=_surface(settings),
    )

    models = {step.__consist_step__.model for step in required_steps}

    assert any(model.startswith("activitysim_") for model in models)
    assert any(model.startswith("beam_") for model in models)
    assert all(not model.startswith("urbansim_") for model in models)
    assert all(not model.startswith("atlas_") for model in models)
    assert "beam_full_skim" not in models


def test_filter_schema_steps_matches_catalog_enablement_models():
    settings = _settings(
        land_use="urbansim",
        activity_demand="activitysim",
        traffic_assignment="beam",
    )
    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=False,
        surface=_surface(settings),
    )
    models = {step.__consist_step__.model for step in required_steps}
    expected = enabled_schema_step_models(
        settings,
        is_model_enabled=lambda current_settings, *, flag_attr, model_attr: bool(
            getattr(current_settings, flag_attr, None)
        )
        or bool(
            getattr(
                getattr(getattr(current_settings, "run", None), "models", None),
                model_attr,
                None,
            )
        ),
        include_optional=False,
    )
    assert models == expected


def test_filter_schema_steps_all_disabled_returns_empty():
    settings = _settings()
    all_steps = _build_schema_steps()
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=False,
        surface=_surface(settings),
    )
    assert required_steps == []


def test_filter_schema_steps_optional_toggle_controls_beam_full_skim():
    settings = _settings(
        activity_demand="activitysim",
        traffic_assignment="beam",
    )
    all_steps = _build_schema_steps()
    with_optional = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=True,
        surface=_surface(settings),
    )
    without_optional = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=False,
        surface=_surface(settings),
    )
    with_models = {step.__consist_step__.model for step in with_optional}
    without_models = {step.__consist_step__.model for step in without_optional}
    assert "beam_full_skim" in with_models
    assert "beam_full_skim" not in without_models


@pytest.mark.parametrize(
    ("enabled_model_attr", "expected_prefix"),
    [
        ("land_use", "urbansim_"),
        ("vehicle_ownership", "atlas_"),
        ("activity_demand", "activitysim_"),
        ("travel", "beam_"),
    ],
)
def test_filter_schema_steps_run_models_shape_uses_catalog_enablement_mapping(
    enabled_model_attr: str,
    expected_prefix: str,
):
    models_cfg = {
        "land_use": None,
        "vehicle_ownership": None,
        "activity_demand": None,
        "traffic_assignment": None,
    }
    model_names = {
        "land_use": "urbansim",
        "vehicle_ownership": "atlas",
        "activity_demand": "activitysim",
        "travel": "beam",
    }
    if enabled_model_attr == "travel":
        models_cfg["traffic_assignment"] = model_names[enabled_model_attr]
    else:
        models_cfg[enabled_model_attr] = model_names[enabled_model_attr]

    settings = _settings(**models_cfg)
    all_steps = _build_schema_steps()
    surface = _surface(settings)
    required_steps = _filter_schema_steps_for_enabled_models(
        all_steps,
        include_optional=False,
        surface=surface,
    )
    models = {step.__consist_step__.model for step in required_steps}

    assert any(model.startswith(expected_prefix) for model in models)
    for other_prefix in ("urbansim_", "atlas_", "activitysim_", "beam_"):
        if other_prefix == expected_prefix:
            continue
        assert all(not model.startswith(other_prefix) for model in models)
