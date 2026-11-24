from __future__ import annotations

from pilates.config.models import load_config
from pilates.workflows.state import WorkflowState


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
