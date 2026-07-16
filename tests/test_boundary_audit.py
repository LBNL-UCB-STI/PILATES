from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pilates.workflows.binding import BindingPlan
from pilates.workflows.boundary_audit import (
    emit_recovery_boundary_audit,
    preflight_recovery_boundary_audit,
)


class _Surface:
    def to_dict(self) -> dict[str, object]:
        return {"profile": {"land_use_enabled": True}, "run_mode": "fresh"}


def _runtime(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace, Path]:
    workspace_root = tmp_path / "local" / "run"
    workspace_root.mkdir(parents=True)
    archive_root = tmp_path / "archive" / "run"
    archive_root.mkdir(parents=True)
    state = SimpleNamespace(
        year=2035,
        forecast_year=2035,
        iteration=2,
        file_loc=str(archive_root / "run_state.yaml"),
    )
    workspace = SimpleNamespace(full_path=str(workspace_root))
    return state, workspace, archive_root


def test_boundary_audit_disabled_has_no_filesystem_side_effects(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PILATES_RECOVERY_BOUNDARY_AUDIT", raising=False)
    state, workspace, archive_root = _runtime(tmp_path)

    result = emit_recovery_boundary_audit(
        boundary="activitysim_run_completed",
        successor_step="activitysim_postprocess",
        binding=BindingPlan(step_name="activitysim_postprocess"),
        state=state,
        workspace=workspace,
        surface=_Surface(),
    )

    assert result is None
    assert not (archive_root / ".workflow").exists()


def test_boundary_audit_preflight_creates_archive_jsonl_before_model_execution(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PILATES_RECOVERY_BOUNDARY_AUDIT", "1")
    state, workspace, archive_root = _runtime(tmp_path)

    result = preflight_recovery_boundary_audit(state=state, workspace=workspace)

    expected_path = (
        archive_root / ".workflow" / "diagnostics" / "recovery_boundary_audit.jsonl"
    )
    assert result == expected_path
    assert expected_path.read_bytes() == b""


def test_boundary_audit_writes_resolved_binding_to_archive_jsonl(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PILATES_RECOVERY_BOUNDARY_AUDIT", "1")
    state, workspace, archive_root = _runtime(tmp_path)
    input_path = Path(workspace.full_path) / "activitysim" / "output" / "plans.csv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("person_id,plan\n1,work\n", encoding="utf-8")
    binding = BindingPlan(
        step_name="activitysim_postprocess",
        inputs={"beam_plans_asim_out": str(input_path)},
        input_keys=["beam_plans_asim_out"],
        optional_input_keys=["optional_output"],
        source_by_key={"beam_plans_asim_out": "explicit"},
        coupler_key_by_key={"beam_plans_asim_out": "beam_plans_asim_out"},
    )

    result = emit_recovery_boundary_audit(
        boundary="activitysim_run_completed",
        successor_step="activitysim_postprocess",
        binding=binding,
        state=state,
        workspace=workspace,
        surface=_Surface(),
    )

    expected_path = (
        archive_root / ".workflow" / "diagnostics" / "recovery_boundary_audit.jsonl"
    )
    assert result == expected_path
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert payload["boundary"] == "activitysim_run_completed"
    assert payload["successor_step"] == "activitysim_postprocess"
    assert payload["scope"] == {"year": 2035, "forecast_year": 2035, "iteration": 2}
    assert payload["binding"]["required_input_keys"] == ["beam_plans_asim_out"]
    assert payload["binding"]["optional_input_keys"] == ["optional_output"]
    assert payload["binding"]["source_by_key"] == {"beam_plans_asim_out": "explicit"}
    assert payload["artifacts"]["beam_plans_asim_out"]["existing_path"] == str(
        input_path.resolve()
    )
    assert (
        payload["artifacts"]["beam_plans_asim_out"]["workspace_relative_locator"]
        == "activitysim/output/plans.csv"
    )
    assert (
        payload["artifacts"]["beam_plans_asim_out"]["archive_relative_locator"] is None
    )
    assert payload["surface"]["run_mode"] == "fresh"


def test_boundary_audit_launcher_archive_root_preempts_state_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PILATES_RECOVERY_BOUNDARY_AUDIT", "1")
    state, workspace, state_archive_root = _runtime(tmp_path)
    launcher_archive_root = tmp_path / "launcher-archive"
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(launcher_archive_root))

    result = emit_recovery_boundary_audit(
        boundary="activitysim_postprocess_completed",
        successor_step="beam_preprocess",
        binding=BindingPlan(step_name="beam_preprocess"),
        state=state,
        workspace=workspace,
        surface=None,
    )

    assert result is not None
    assert result.is_relative_to(launcher_archive_root)
    assert not (state_archive_root / ".workflow").exists()


def test_boundary_audit_records_typed_predecessor_outputs_outside_binding(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PILATES_RECOVERY_BOUNDARY_AUDIT", "1")
    state, workspace, _ = _runtime(tmp_path)
    run_output = Path(workspace.full_path) / "activitysim" / "output" / "trips.csv"
    run_output.parent.mkdir(parents=True)
    run_output.write_text("trip_id\n1\n", encoding="utf-8")

    result = emit_recovery_boundary_audit(
        boundary="activitysim_run_completed",
        successor_step="activitysim_postprocess",
        binding=BindingPlan(step_name="activitysim_postprocess"),
        predecessor_outputs={"trips_asim_out": run_output},
        state=state,
        workspace=workspace,
        surface=_Surface(),
    )

    assert result is not None
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert (
        payload["predecessor_artifacts"]["trips_asim_out"]["workspace_relative_locator"]
        == "activitysim/output/trips.csv"
    )
