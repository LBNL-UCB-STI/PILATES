from __future__ import annotations

from types import SimpleNamespace

import pytest

from pilates.runtime import restart as restart_runtime
from pilates.workflows.catalog import restart_artifact_producers
from workflow_state import WorkflowState


def _settings(*, activity_demand="activitysim", traffic_assignment="beam"):
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test-region",
            models=SimpleNamespace(
                activity_demand=activity_demand,
                traffic_assignment=traffic_assignment,
            ),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="scenario",
        ),
    )


def _state(
    *,
    major_stage=WorkflowState.Stage.supply_demand_loop,
    sub_stage=WorkflowState.Stage.traffic_assignment,
    year=2018,
    iteration=1,
):
    return SimpleNamespace(
        current_major_stage=major_stage,
        current_sub_stage=sub_stage,
        current_year=year,
        current_inner_iter=iteration,
    )


def test_restart_frontier_contract_scopes_v1_to_traffic_assignment():
    contract = restart_runtime.restart_frontier_contract(
        settings=_settings(),
        state=_state(),
        workflow_stage=WorkflowState.Stage,
    )

    assert contract is not None
    assert contract.frontier_stage == "traffic_assignment"
    assert contract.frontier_step == "beam_preprocess"
    assert contract.required_keys == (
        "beam_plans_asim_out",
        "households_asim_out",
        "persons_asim_out",
        "zarr_skims",
    )


def test_restart_frontier_contract_prefers_surface_projection():
    surface_contract = SimpleNamespace(
        frontier_stage="traffic_assignment",
        frontier_step="beam_preprocess",
        required_keys=("surface_only_key",),
    )
    surface = SimpleNamespace(restart_frontier=lambda: surface_contract)

    contract = restart_runtime.restart_frontier_contract(
        settings=_settings(activity_demand=None, traffic_assignment=None),
        state=_state(major_stage=WorkflowState.Stage.vehicle_ownership_model),
        workflow_stage=WorkflowState.Stage,
        surface=surface,
    )

    assert contract is not None
    assert contract.frontier_stage == "traffic_assignment"
    assert contract.frontier_step == "beam_preprocess"
    assert contract.required_keys == ("surface_only_key",)


def test_prebootstrap_missing_artifacts_are_split_by_surface():
    surface = SimpleNamespace(
        is_restart_prebootstrap_deferred_artifact_key=lambda key: (
            key == "bootstrap_owned"
        )
    )
    artifacts = [
        {"key": "runtime_owned", "path": "/tmp/runtime", "reason": "runtime"},
        {"key": "bootstrap_owned", "path": "/tmp/bootstrap", "reason": "bootstrap"},
    ]

    blocking, deferred = restart_runtime.split_prebootstrap_missing_artifacts(
        artifacts,
        surface=surface,
    )

    assert blocking == [artifacts[0]]
    assert deferred == [artifacts[1]]


def test_enforce_postbootstrap_missing_artifacts_raises_when_strict():
    artifacts = [
        {"key": "runtime_owned", "path": "/tmp/runtime", "reason": "runtime"},
    ]
    settings = SimpleNamespace(run=SimpleNamespace(restart_strict=True))

    with pytest.raises(RuntimeError, match="Strict restart preflight failed"):
        restart_runtime.enforce_postbootstrap_missing_artifacts(
            artifacts,
            settings=settings,
        )


def test_restart_artifact_producers_applies_traffic_assignment_overrides():
    producers = restart_artifact_producers(
        frontier_stage="traffic_assignment",
        enabled_models=("activitysim", "beam"),
    )

    assert producers["zarr_skims"][0].step_name == "activitysim_compile"
    assert producers["beam_plans_asim_out"][0].step_name == "activitysim_postprocess"
    assert producers["households_asim_out"][0].step_name == "activitysim_postprocess"
    assert producers["persons_asim_out"][0].step_name == "activitysim_postprocess"
