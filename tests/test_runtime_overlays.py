from __future__ import annotations

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.profile import WorkflowProfile
from pilates.workflows.runtime_overlays import (
    StepRuntimeOverlayRule,
    resolve_runtime_input_output_overrides,
    resolve_runtime_role_policy_overrides,
)
from pilates.workflows.surface import RunMode


def _profile(*, land_use_enabled: bool) -> WorkflowProfile:
    return WorkflowProfile(
        land_use_enabled=land_use_enabled,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=True,
        traffic_assignment_enabled=True,
        replanning_enabled=False,
        supply_demand_loop_enabled=True,
        activity_demand_direct_from_land_use=False,
    )


def test_overlay_rule_applies_only_for_matching_runtime_shape() -> None:
    rule = StepRuntimeOverlayRule(
        step_name="activitysim_preprocess",
        run_mode="restart",
        land_use_enabled=True,
    )

    assert rule.applies(
        step_name="activitysim_preprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.RESTART,
    )
    assert not rule.applies(
        step_name="activitysim_preprocess",
        profile=_profile(land_use_enabled=False),
        run_mode=RunMode.RESTART,
    )
    assert not rule.applies(
        step_name="activitysim_preprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.FRESH,
    )
    assert not rule.applies(
        step_name="activitysim_postprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.RESTART,
    )


def test_runtime_overlays_promote_activitysim_restart_inputs_and_role_guards() -> None:
    required_inputs, optional_inputs = resolve_runtime_input_output_overrides(
        step_name="activitysim_preprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.RESTART,
        required_inputs=(USIM_POPULATION_SOURCE_H5,),
        optional_inputs=(),
    )

    assert required_inputs == (USIM_POPULATION_SOURCE_H5,)
    assert optional_inputs == (USIM_DATASTORE_CURRENT_H5,)

    role_overrides = resolve_runtime_role_policy_overrides(
        step_name="activitysim_preprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.RESTART,
    )

    for key in (USIM_POPULATION_SOURCE_H5, USIM_DATASTORE_CURRENT_H5):
        assert role_overrides[key]["workspace_archive_fallback_allowed"] is False
        assert role_overrides[key]["restart_may_restore_synthetically"] is False
        assert role_overrides[key]["restart_requires_explicit_before_execution"] is True


def test_runtime_overlays_strip_postprocess_h5_inputs_without_land_use() -> None:
    required_inputs, optional_inputs = resolve_runtime_input_output_overrides(
        step_name="activitysim_postprocess",
        profile=_profile(land_use_enabled=False),
        run_mode=RunMode.FRESH,
        required_inputs=(),
        optional_inputs=(
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
            USIM_DATASTORE_BASE_H5,
            "other_key",
        ),
    )

    assert required_inputs == ()
    assert optional_inputs == ("other_key",)


def test_runtime_overlays_promote_postprocess_h5_inputs_with_land_use() -> None:
    required_inputs, optional_inputs = resolve_runtime_input_output_overrides(
        step_name="activitysim_postprocess",
        profile=_profile(land_use_enabled=True),
        run_mode=RunMode.FRESH,
        required_inputs=(),
        optional_inputs=("existing_optional",),
    )

    assert required_inputs == (
        USIM_POPULATION_SOURCE_H5,
        USIM_DATASTORE_CURRENT_H5,
    )
    assert optional_inputs == ("existing_optional", USIM_DATASTORE_BASE_H5)
