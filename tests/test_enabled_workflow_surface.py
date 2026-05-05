from __future__ import annotations

from types import SimpleNamespace

import pytest

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.surface import (
    EnabledWorkflowSurface,
    StepRuntimeSurface,
    RestartFrontierContract,
    RunMode,
    _validate_surface_invariants,
    build_enabled_workflow_surface,
)


def _settings(
    *,
    land_use: str | None = "urbansim",
    vehicle_ownership: str | None = "atlas",
    activity_demand: str | None = "activitysim",
    traffic_assignment: str | None = "beam",
    warm_start_skims: bool = False,
    static_skims: bool = False,
    replan_iters: int = 0,
    main_configs_dir: str = "configs",
    atlas_scenario: str = "baseline",
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
        warm_start_skims=warm_start_skims,
        static_skims=static_skims,
        activitysim=SimpleNamespace(
            replan_iters=replan_iters,
            main_configs_dir=main_configs_dir,
        ),
        atlas=SimpleNamespace(scenario=atlas_scenario, adscen=atlas_scenario),
    )


def _restart_state():
    stage = SimpleNamespace(
        supply_demand_loop=object(),
        traffic_assignment=object(),
    )
    return SimpleNamespace(
        is_restart_run=True,
        run_info_path="/tmp/run_state.yaml",
        current_major_stage=stage.supply_demand_loop,
        current_sub_stage=stage.traffic_assignment,
        Stage=stage,
    )


def test_surface_construction_for_beam_only_shape():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use=None,
            vehicle_ownership=None,
            activity_demand=None,
            traffic_assignment="beam",
        )
    )

    assert isinstance(surface, EnabledWorkflowSurface)
    assert surface.run_mode == RunMode.FRESH
    assert surface.enabled_model_names == frozenset({"beam"})
    assert surface.enabled_stage_names == frozenset(
        {"traffic_assignment", "supply_demand_loop"}
    )
    assert "beam_preprocess" in surface.enabled_step_names
    assert "activitysim_preprocess" not in surface.enabled_step_names
    assert surface.restart_frontier_contract is None


def test_surface_construction_for_activitysim_beam_shape():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use=None,
            vehicle_ownership=None,
            activity_demand="activitysim",
            traffic_assignment="beam",
        ),
        state=_restart_state(),
    )

    assert surface.run_mode == RunMode.RESTART
    assert surface.enabled_model_names == frozenset({"activitysim", "beam"})
    assert "activitysim_preprocess" in surface.enabled_step_names
    assert "beam_preprocess" in surface.enabled_step_names
    assert surface.restart_frontier_contract == RestartFrontierContract(
        frontier_stage="traffic_assignment",
        frontier_step="beam_preprocess",
        required_keys=(
            "beam_plans_asim_out",
            "households_asim_out",
            "persons_asim_out",
            ZARR_SKIMS,
        ),
    )


def test_surface_construction_for_full_shape():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use="urbansim",
            vehicle_ownership="atlas",
            activity_demand="activitysim",
            traffic_assignment="beam",
        )
    )

    assert surface.enabled_model_names == frozenset(
        {"urbansim", "atlas", "activitysim", "beam"}
    )
    assert {
        "land_use",
        "vehicle_ownership_model",
        "activity_demand",
        "traffic_assignment",
        "supply_demand_loop",
    }.issubset(surface.enabled_stage_names)
    assert {
        "urbansim_preprocess",
        "atlas_preprocess",
        "activitysim_preprocess",
        "beam_preprocess",
    }.issubset(surface.enabled_step_names)


def test_surface_bootstrap_owned_and_deferred_keys_follow_run_shape():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use=None,
            vehicle_ownership=None,
            activity_demand="activitysim",
            traffic_assignment="beam",
        )
    )

    assert "beam_mutable_data_dir" in surface.bootstrap_owned_artifact_keys
    assert "beam_region_input_dir" in surface.restart_prebootstrap_deferred_artifact_keys
    assert USIM_DATASTORE_BASE_H5 in surface.bootstrap_owned_artifact_keys
    assert USIM_DATASTORE_BASE_H5 in surface.restart_prebootstrap_deferred_artifact_keys
    assert any(
        key.startswith("activitysim_config_settings_yaml_")
        for key in surface.bootstrap_owned_artifact_keys
    )


def test_surface_projects_activitysim_restart_role_policy():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use="urbansim",
            vehicle_ownership=None,
            activity_demand="activitysim",
            traffic_assignment="beam",
        ),
        state=_restart_state(),
    )

    preprocess = surface.step_surface("activitysim_preprocess")
    assert preprocess is not None
    for key in (USIM_POPULATION_SOURCE_H5, USIM_DATASTORE_CURRENT_H5):
        policy = preprocess.input_role_policies[key]
        assert policy.restart_requires_explicit_before_execution is True
        assert policy.workspace_archive_fallback_allowed is False
        assert policy.restart_may_restore_synthetically is False


def test_surface_projects_activitysim_postprocess_required_keys_when_land_use_enabled():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use="urbansim",
            vehicle_ownership=None,
            activity_demand="activitysim",
            traffic_assignment="beam",
        )
    )

    postprocess = surface.step_surface("activitysim_postprocess")
    assert postprocess is not None
    assert USIM_POPULATION_SOURCE_H5 in postprocess.required_input_keys
    assert USIM_DATASTORE_CURRENT_H5 in postprocess.required_input_keys


def test_surface_projects_activitysim_postprocess_optional_keys_when_land_use_disabled():
    surface = build_enabled_workflow_surface(
        _settings(
            land_use=None,
            vehicle_ownership=None,
            activity_demand="activitysim",
            traffic_assignment="beam",
        )
    )

    postprocess = surface.step_surface("activitysim_postprocess")
    assert postprocess is not None
    assert USIM_POPULATION_SOURCE_H5 not in postprocess.optional_input_keys
    assert USIM_DATASTORE_CURRENT_H5 not in postprocess.optional_input_keys
    assert USIM_DATASTORE_BASE_H5 not in postprocess.optional_input_keys


def test_surface_reuses_initialized_runtime_flags():
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

    surface = build_enabled_workflow_surface(settings)

    assert surface.profile.land_use_enabled is False
    assert surface.profile.vehicle_ownership_model_enabled is False
    assert surface.profile.activity_demand_enabled is True


def test_surface_to_dict_contains_expected_debug_shape():
    surface = build_enabled_workflow_surface(_settings())

    dumped = surface.to_dict()

    assert dumped["run_mode"] == "fresh"
    assert "profile" in dumped
    assert "step_surfaces" in dumped
    assert "activitysim_preprocess" in dumped["step_surfaces"]


def test_surface_invariant_rejects_unknown_policy_key(monkeypatch):
    from pilates.workflows import surface as surface_module

    original_builder = surface_module._build_step_runtime_surface

    def _broken_builder(**kwargs):
        step_surface = original_builder(**kwargs)
        if step_surface.step_name != "activitysim_preprocess":
            return step_surface
        return step_surface.__class__(
            step_name=step_surface.step_name,
            stage_name=step_surface.stage_name,
            phase=step_surface.phase,
            optional=step_surface.optional,
            enabled=step_surface.enabled,
            required_input_keys=step_surface.required_input_keys,
            optional_input_keys=step_surface.optional_input_keys,
            required_output_keys=step_surface.required_output_keys,
            optional_output_keys=step_surface.optional_output_keys,
            input_role_policies={
                **step_surface.input_role_policies,
                "unknown_input": next(iter(step_surface.input_role_policies.values())),
            },
        )

    monkeypatch.setattr(surface_module, "_build_step_runtime_surface", _broken_builder)

    with pytest.raises(RuntimeError, match="undeclared input keys"):
        build_enabled_workflow_surface(
            _settings(
                land_use=None,
                vehicle_ownership=None,
                activity_demand="activitysim",
                traffic_assignment="beam",
            )
        )


def test_surface_invariant_rejects_enabled_step_with_disabled_dependency():
    surface = StepRuntimeSurface(
        step_name="beam_run",
        stage_name="traffic_assignment",
        phase="run",
        optional=False,
        enabled=True,
        required_input_keys=(),
        optional_input_keys=(),
        required_output_keys=(),
        optional_output_keys=(),
        input_role_policies={},
    )

    with pytest.raises(RuntimeError, match="depends on disabled step"):
        _validate_surface_invariants(
            step_surfaces={"beam_run": surface},
            enabled_step_names=frozenset({"beam_run"}),
            profile=build_enabled_workflow_surface(
                _settings(
                    land_use=None,
                    vehicle_ownership=None,
                    activity_demand=None,
                    traffic_assignment="beam",
                )
            ).profile,
        )
