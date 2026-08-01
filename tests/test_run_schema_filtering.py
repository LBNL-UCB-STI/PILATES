from types import SimpleNamespace

import pytest
from consist import define_step

from pilates.runtime.launcher import (
    _build_schema_steps,
    _filter_schema_steps_for_enabled_models,
)
from pilates.workflows.catalog import enabled_schema_step_models
from pilates.workflows.steps import STEP_DEFINITIONS
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


def _step_names(steps):
    functions_to_names = {
        definition.function: step_name
        for step_name, definition in STEP_DEFINITIONS.items()
    }
    return {functions_to_names[step] for step in steps}


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

    assert {
        STEP_DEFINITIONS[step_name].function
        for step_name in (
            "activitysim_preprocess",
            "activitysim_run",
            "activitysim_postprocess",
        )
    }.issubset(required_steps)
    assert all(
        step not in required_steps
        for step_name in (
            "urbansim_run",
            "urbansim_postprocess",
            "atlas_preprocess",
            "atlas_run",
            "atlas_postprocess",
            "beam_full_skim",
        )
        for step in (STEP_DEFINITIONS[step_name].function,)
    )


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
    step_names = _step_names(required_steps)
    expected = enabled_schema_step_models(
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
    assert step_names == expected


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
    assert "beam_full_skim" in _step_names(with_optional)
    assert "beam_full_skim" not in _step_names(without_optional)


def test_filter_schema_steps_rejects_nonregistry_callable_with_enabled_model_label():
    @define_step(model="beam_run")
    def unrelated_beam_run() -> None:
        pass

    settings = _settings(
        activity_demand="activitysim",
        traffic_assignment="beam",
    )
    filtered = _filter_schema_steps_for_enabled_models(
        [*_build_schema_steps(), unrelated_beam_run],
        include_optional=False,
        surface=_surface(settings),
    )

    assert unrelated_beam_run not in filtered
    assert STEP_DEFINITIONS["activitysim_run"].function in filtered


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
    step_names = _step_names(required_steps)

    assert any(step_name.startswith(expected_prefix) for step_name in step_names)
    for other_prefix in ("urbansim_", "atlas_", "activitysim_", "beam_"):
        if other_prefix == expected_prefix:
            continue
        assert all(not step_name.startswith(other_prefix) for step_name in step_names)
