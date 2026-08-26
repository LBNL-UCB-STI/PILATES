from __future__ import annotations

from pilates.config.models import load_config
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


def test_max_year_intervals_completes_after_first_interval_with_its_forecast():
    """A capped canary executes 2017 while retaining its 2019 forecast target."""

    settings = load_config(
        "scenarios/sfbay/settings-sfbay-consist-usim-hpc-2019-canary.yaml"
    )

    state = WorkflowState.from_settings(settings)

    assert state.current_year == 2017
    assert state.forecast_year == 2019
    assert state._year_schedule[:2] == (2017, 2019)

    state._advance_to_next_year()

    assert state.current_year == settings.run.end_year + 1
    assert state.current_major_stage is None
