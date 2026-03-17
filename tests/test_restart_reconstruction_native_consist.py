from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import pytest
from consist import MaterializationResult
from pydantic import ValidationError

from pilates.config.models import RunConfig
from pilates.runtime import restart as restart_runtime
from workflow_state import WorkflowState


def _write_manifest(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(content), encoding="utf-8")


def _restart_state(
    *,
    iteration: int,
    sub_stage,
    major_stage=None,
    forecast_year: int = 2018,
    enabled_stages: Optional[Sequence] = None,
):
    return SimpleNamespace(
        current_year=2018,
        forecast_year=forecast_year,
        current_inner_iter=iteration,
        current_major_stage=major_stage or WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=sub_stage,
        _settings={"supply_demand_iters": 3},
        enabled_stages=set(enabled_stages) if enabled_stages is not None else None,
    )


def test_collect_restart_completed_run_ids_for_supply_demand_resume(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir / ".workflow" / "year_2018_iteration_0.yaml",
        {
            "activitysim_preprocess": {"run_id": "asim-pre-0"},
            "activitysim_run": {"run_id": "asim-run-0"},
            "activitysim_postprocess": {"run_id": "asim-post-0"},
        },
    )
    _write_manifest(
        archive_run_dir / ".workflow" / "year_2018_iteration_1.yaml",
        {
            "activitysim_preprocess": {"run_id": "asim-pre-1"},
            "activitysim_run": {"run_id": "asim-run-1"},
            "activitysim_postprocess": {"run_id": "asim-post-1"},
            "beam_run": {"run_id": "beam-run-1"},
        },
    )
    state = _restart_state(
        iteration=1,
        sub_stage=WorkflowState.Stage.traffic_assignment,
    )

    discovery = restart_runtime.collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    assert discovery["issues"] == []
    assert set(discovery["run_ids"]) == {
        "asim-pre-0",
        "asim-run-0",
        "asim-post-0",
        "asim-pre-1",
        "asim-run-1",
        "asim-post-1",
    }
    assert len(discovery["run_ids"]) == 6
    assert len(discovery["manifest_paths"]) == 2


def test_collect_restart_completed_run_ids_for_vehicle_ownership_resume_discovers_land_use_and_atlas_manifests(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir / ".workflow" / "land_use_year_2018.yaml",
        {
            "urbansim_preprocess": {"run_id": "usim-pre-2018"},
            "urbansim_run": {"run_id": "usim-run-2018"},
            "urbansim_postprocess": {"run_id": "usim-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2022_subyear_2018.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2018"},
            "atlas_run": {"run_id": "atlas-run-2018"},
            "atlas_postprocess": {"run_id": "atlas-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2022_subyear_2020.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2020"},
            "atlas_run": {"run_id": "atlas-run-2020"},
            "atlas_postprocess": {"run_id": "atlas-post-2020"},
        },
    )
    state = _restart_state(
        iteration=0,
        sub_stage=None,
        major_stage=WorkflowState.Stage.vehicle_ownership_model,
        forecast_year=2022,
        enabled_stages={
            WorkflowState.Stage.land_use,
            WorkflowState.Stage.vehicle_ownership_model,
        },
    )

    discovery = restart_runtime.collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    assert discovery["issues"] == []
    assert set(discovery["run_ids"]) == {
        "usim-pre-2018",
        "usim-run-2018",
        "usim-post-2018",
        "atlas-pre-2018",
        "atlas-run-2018",
        "atlas-post-2018",
        "atlas-pre-2020",
        "atlas-run-2020",
        "atlas-post-2020",
    }
    assert len(discovery["manifest_paths"]) == 3


def test_collect_restart_completed_run_ids_for_supply_demand_resume_merges_land_use_and_atlas(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir / ".workflow" / "land_use_year_2018.yaml",
        {
            "urbansim_preprocess": {"run_id": "usim-pre-2018"},
            "urbansim_run": {"run_id": "usim-run-2018"},
            "urbansim_postprocess": {"run_id": "usim-post-2018"},
        },
    )
    for sub_year in (2018, 2020, 2022):
        _write_manifest(
            archive_run_dir
            / ".workflow"
            / "vehicle_ownership"
            / f"forecast_year_2022_subyear_{sub_year}.yaml",
            {
                "atlas_preprocess": {"run_id": f"atlas-pre-{sub_year}"},
                "atlas_run": {"run_id": f"atlas-run-{sub_year}"},
                "atlas_postprocess": {"run_id": f"atlas-post-{sub_year}"},
            },
        )
    _write_manifest(
        archive_run_dir / ".workflow" / "year_2018_iteration_0.yaml",
        {
            "activitysim_preprocess": {"run_id": "asim-pre-0"},
            "activitysim_run": {"run_id": "asim-run-0"},
            "activitysim_postprocess": {"run_id": "asim-post-0"},
        },
    )
    _write_manifest(
        archive_run_dir / ".workflow" / "year_2018_iteration_1.yaml",
        {
            "activitysim_preprocess": {"run_id": "asim-pre-1"},
            "activitysim_run": {"run_id": "asim-run-1"},
            "activitysim_postprocess": {"run_id": "asim-post-1"},
            "beam_run": {"run_id": "beam-run-1"},
        },
    )
    state = _restart_state(
        iteration=1,
        sub_stage=WorkflowState.Stage.traffic_assignment,
        forecast_year=2022,
        enabled_stages={
            WorkflowState.Stage.land_use,
            WorkflowState.Stage.vehicle_ownership_model,
            WorkflowState.Stage.supply_demand_loop,
        },
    )

    discovery = restart_runtime.collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    assert discovery["issues"] == []
    assert set(discovery["run_ids"]) == {
        "usim-pre-2018",
        "usim-run-2018",
        "usim-post-2018",
        "atlas-pre-2018",
        "atlas-run-2018",
        "atlas-post-2018",
        "atlas-pre-2020",
        "atlas-run-2020",
        "atlas-post-2020",
        "atlas-pre-2022",
        "atlas-run-2022",
        "atlas-post-2022",
        "asim-pre-0",
        "asim-run-0",
        "asim-post-0",
        "asim-pre-1",
        "asim-run-1",
        "asim-post-1",
    }
    assert "beam-run-1" not in discovery["run_ids"]
    assert len(discovery["manifest_paths"]) == 6


def test_collect_restart_completed_run_ids_for_postprocessing_resume_discovers_completed_run_ids(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir / ".workflow" / "land_use_year_2018.yaml",
        {
            "urbansim_preprocess": {"run_id": "usim-pre-2018"},
            "urbansim_run": {"run_id": "usim-run-2018"},
            "urbansim_postprocess": {"run_id": "usim-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2018_subyear_2018.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2018"},
            "atlas_run": {"run_id": "atlas-run-2018"},
            "atlas_postprocess": {"run_id": "atlas-post-2018"},
        },
    )
    for iteration in range(0, 3):
        _write_manifest(
            archive_run_dir / ".workflow" / f"year_2018_iteration_{iteration}.yaml",
            {"activitysim_run": {"run_id": f"asim-run-{iteration}"}},
        )
    _write_manifest(
        archive_run_dir / ".workflow" / "postprocessing_year_2018.yaml",
        {"postprocessing": {"run_id": "post-2018"}},
    )
    state = _restart_state(
        iteration=0,
        sub_stage=None,
        major_stage=WorkflowState.Stage.postprocessing,
        enabled_stages={
            WorkflowState.Stage.land_use,
            WorkflowState.Stage.vehicle_ownership_model,
            WorkflowState.Stage.supply_demand_loop,
            WorkflowState.Stage.postprocessing,
        },
    )

    discovery = restart_runtime.collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    assert discovery["issues"] == []
    assert set(discovery["run_ids"]) == {
        "usim-pre-2018",
        "usim-run-2018",
        "usim-post-2018",
        "atlas-pre-2018",
        "atlas-run-2018",
        "atlas-post-2018",
        "asim-run-0",
        "asim-run-1",
        "asim-run-2",
        "post-2018",
    }
    assert len(discovery["manifest_paths"]) == 6


def test_reconstruct_restart_completed_run_outputs_uses_native_materialization(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir / ".workflow" / "land_use_year_2018.yaml",
        {
            "urbansim_preprocess": {"run_id": "usim-pre-2018"},
            "urbansim_run": {"run_id": "usim-run-2018"},
            "urbansim_postprocess": {"run_id": "usim-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2018_subyear_2018.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2018"},
            "atlas_run": {"run_id": "atlas-run-2018"},
            "atlas_postprocess": {"run_id": "atlas-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir / ".workflow" / "year_2018_iteration_0.yaml",
        {
            "activitysim_preprocess": {"run_id": "asim-pre-0"},
            "activitysim_run": {"run_id": "asim-run-0"},
            "activitysim_postprocess": {"run_id": "asim-post-0"},
        },
    )
    state = _restart_state(
        iteration=0,
        sub_stage=WorkflowState.Stage.traffic_assignment,
        enabled_stages={
            WorkflowState.Stage.land_use,
            WorkflowState.Stage.vehicle_ownership_model,
            WorkflowState.Stage.supply_demand_loop,
        },
    )

    class TrackerStub:
        def __init__(self):
            self.calls = []

        def materialize_run_outputs(self, **kwargs):
            self.calls.append(kwargs)
            run_id = kwargs["run_id"]
            return MaterializationResult(
                materialized_from_filesystem={run_id: f"/restored/{run_id}"}
            )

    tracker = TrackerStub()
    reconstruction = restart_runtime.reconstruct_restart_completed_run_outputs(
        tracker=tracker,
        state=state,
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    result = reconstruction["materialization_result"]
    assert result.complete is True
    assert len(tracker.calls) == 9
    assert {call["run_id"] for call in tracker.calls} == {
        "usim-pre-2018",
        "usim-run-2018",
        "usim-post-2018",
        "atlas-pre-2018",
        "atlas-run-2018",
        "atlas-post-2018",
        "asim-pre-0",
        "asim-run-0",
        "asim-post-0",
    }
    for call in tracker.calls:
        assert call["target_root"] == str(local_run_dir.resolve())
        assert call["source_root"] == str(archive_run_dir.resolve())
        assert call["preserve_existing"] is True


def test_reconstruct_restart_completed_run_outputs_reports_missing_manifest(tmp_path):
    state = _restart_state(
        iteration=0,
        sub_stage=WorkflowState.Stage.traffic_assignment,
    )

    reconstruction = restart_runtime.reconstruct_restart_completed_run_outputs(
        tracker=SimpleNamespace(),
        state=state,
        local_run_dir=str(tmp_path / "local-run"),
        archive_run_dir=str(tmp_path / "archive-run"),
        workflow_stage=WorkflowState.Stage,
    )

    result = reconstruction["materialization_result"]
    assert result.complete is False
    assert result.failed
    assert "workflow manifest is missing" in result.failed[0][1]


def test_collect_restart_completed_run_ids_for_vehicle_ownership_resume_uses_contiguous_atlas_prefix(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2022_subyear_2018.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2018"},
            "atlas_run": {"run_id": "atlas-run-2018"},
            "atlas_postprocess": {"run_id": "atlas-post-2018"},
        },
    )
    _write_manifest(
        archive_run_dir
        / ".workflow"
        / "vehicle_ownership"
        / "forecast_year_2022_subyear_2022.yaml",
        {
            "atlas_preprocess": {"run_id": "atlas-pre-2022"},
            "atlas_run": {"run_id": "atlas-run-2022"},
            "atlas_postprocess": {"run_id": "atlas-post-2022"},
        },
    )
    state = _restart_state(
        iteration=0,
        sub_stage=None,
        major_stage=WorkflowState.Stage.vehicle_ownership_model,
        forecast_year=2022,
        enabled_stages={WorkflowState.Stage.vehicle_ownership_model},
    )

    discovery = restart_runtime.collect_restart_completed_run_ids(
        state=state,
        archive_run_dir=str(archive_run_dir),
        workflow_stage=WorkflowState.Stage,
    )

    assert discovery["issues"] == []
    assert set(discovery["run_ids"]) == {
        "atlas-pre-2018",
        "atlas-run-2018",
        "atlas-post-2018",
    }
    assert discovery["manifest_paths"] == [
        str(
            archive_run_dir
            / ".workflow"
            / "vehicle_ownership"
            / "forecast_year_2022_subyear_2018.yaml"
        )
    ]


def test_resolve_restart_rehydrate_mode_prefers_native_off_and_defaults_unknown():
    assert (
        restart_runtime.resolve_restart_rehydrate_mode(
            SimpleNamespace(run=SimpleNamespace(restart_rehydrate_mode="native"))
        )
        == "native"
    )
    assert (
        restart_runtime.resolve_restart_rehydrate_mode(
            SimpleNamespace(run=SimpleNamespace(restart_rehydrate_mode="off"))
        )
        == "off"
    )
    assert (
        restart_runtime.resolve_restart_rehydrate_mode(
            SimpleNamespace(run=SimpleNamespace(restart_rehydrate_mode="unexpected"))
        )
        == "native"
    )


def test_run_config_rejects_legacy_restart_rehydrate_mode_aliases():
    base_kwargs = dict(
        start_year=2018,
        end_year=2018,
        travel_model_freq=1,
        region="test-region",
        warm_start_activities=False,
        output_directory="/tmp/output",
        models={
            "activity_demand": "activitysim",
            "traffic_assignment": "beam",
            "land_use": "urbansim",
            "vehicle_ownership": "atlas",
        },
    )

    with pytest.raises(ValidationError):
        RunConfig(restart_rehydrate_mode="bundle", **base_kwargs)

    with pytest.raises(ValidationError):
        RunConfig(restart_rehydrate_mode="full", **base_kwargs)
