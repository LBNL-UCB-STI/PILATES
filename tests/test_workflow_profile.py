from __future__ import annotations

from types import SimpleNamespace

from pilates.workflows._profile import (
    WorkflowProfile,
    ensure_runtime_flags_initialized,
    workflow_profile_from_flags,
)
from pilates.workflows.surface import build_enabled_workflow_surface


def _settings(
    *,
    land_use: str | None = "urbansim",
    vehicle_ownership: str | None = "atlas",
    activity_demand: str | None = "activitysim",
    traffic_assignment: str | None = "beam",
    warm_start_skims: bool = False,
    static_skims: bool = False,
    replan_iters: int = 0,
) -> SimpleNamespace:
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
        warm_start_skims=warm_start_skims,
        static_skims=static_skims,
        activitysim=SimpleNamespace(replan_iters=replan_iters),
    )


def _profile_from_settings(settings: SimpleNamespace) -> WorkflowProfile:
    return workflow_profile_from_flags(ensure_runtime_flags_initialized(settings))


def test_workflow_profile_derives_expected_flags():
    profile = _profile_from_settings(_settings(replan_iters=2))

    assert profile == WorkflowProfile(
        land_use_enabled=True,
        vehicle_ownership_model_enabled=True,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
        replanning_enabled=True,
        supply_demand_loop_enabled=True,
        activity_demand_direct_from_land_use=False,
    )


def test_workflow_profile_warm_start_disables_land_use():
    profile = _profile_from_settings(_settings(warm_start_skims=True))

    assert profile.land_use_enabled is False


def test_workflow_profile_static_skims_disables_traffic_assignment():
    profile = _profile_from_settings(_settings(static_skims=True))

    assert profile.traffic_assignment_enabled is False


def test_workflow_profile_polaris_disables_replanning():
    profile = _profile_from_settings(
        _settings(activity_demand="polaris", replan_iters=3)
    )

    assert profile.activity_demand_enabled is True
    assert profile.replanning_enabled is False


def test_workflow_profile_computes_direct_land_use_handoff_mode():
    profile = _profile_from_settings(
        _settings(activity_demand=None, traffic_assignment=None)
    )

    assert profile.land_use_enabled is True
    assert profile.activity_demand_enabled is False
    assert profile.activity_demand_direct_from_land_use is True
    assert profile.supply_demand_loop_enabled is False


def test_workflow_profile_respects_initialized_runtime_flags():
    settings = _settings()
    settings.runtime = SimpleNamespace(
        flags_initialized=True,
        flags=SimpleNamespace(
            land_use_enabled=False,
            vehicle_ownership_model_enabled=False,
            activity_demand_enabled=True,
            traffic_assignment_enabled=True,
            replanning_enabled=False,
        ),
    )

    profile = _profile_from_settings(settings)

    assert profile.land_use_enabled is False
    assert profile.vehicle_ownership_model_enabled is False
    assert profile.activity_demand_enabled is True


def test_surface_enabled_schema_step_names_filters_disabled_model_steps():
    surface = build_enabled_workflow_surface(
        _settings(land_use=None, vehicle_ownership=None)
    )

    enabled_steps = surface.enabled_schema_step_names(include_optional=True)

    assert "activitysim_preprocess" in enabled_steps
    assert "beam_preprocess" in enabled_steps
    assert "urbansim_preprocess" not in enabled_steps
    assert "atlas_preprocess" not in enabled_steps
