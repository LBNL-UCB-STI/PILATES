from __future__ import annotations

from workflow_state import WorkflowState


def test_asim_compiled_is_ignored_on_read_and_not_serialized_on_write(tmp_path) -> None:
    """Warmup state is node-local and cannot govern restart or cache behavior."""
    state_path = tmp_path / "run_state.yaml"
    state_path.write_text(
        "year: 2025\nstage: activity_demand\niteration: 1\nasim_compiled: true\n",
        encoding="utf-8",
    )

    loaded = WorkflowState.read_current_stage(str(state_path))

    assert loaded[3] is False
    WorkflowState.write_stage(
        2025,
        WorkflowState.Stage.activity_demand,
        str(state_path),
        1,
        True,
    )
    assert "asim_compiled" not in state_path.read_text(encoding="utf-8")
