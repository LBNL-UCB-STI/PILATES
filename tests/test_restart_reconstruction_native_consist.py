from pathlib import Path
from types import SimpleNamespace

from consist import MaterializationResult

from pilates.runtime import restart as restart_runtime
from workflow_state import WorkflowState


def _write_manifest(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(content), encoding="utf-8")


def _restart_state(*, iteration: int, sub_stage):
    return SimpleNamespace(
        current_year=2018,
        current_inner_iter=iteration,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=sub_stage,
        _settings={"supply_demand_iters": 3},
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


def test_reconstruct_restart_completed_run_outputs_uses_native_materialization(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
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
    assert len(tracker.calls) == 3
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
