from pilates.workflows.atlas_state import AtlasSubState


class DummyState:
    def __init__(self) -> None:
        self.year = 2023
        self.forecast_year = 2030
        self.start_year = 2017
        self.full_settings = "settings"
        self.sub_stage_progress = None

    def set_sub_stage_progress(self, value: str) -> None:
        self.sub_stage_progress = value


def test_atlas_substate_year_fields() -> None:
    parent = DummyState()
    atlas_state = AtlasSubState(parent, 2024)

    assert atlas_state.year == 2024
    assert atlas_state.current_year == 2024
    assert atlas_state.forecast_year == 2024
    assert atlas_state.main_forecast_year == 2030
    assert atlas_state.start_year == 2017
    assert atlas_state.atlas_interval_start_year == 2023
    assert atlas_state.full_settings == "settings"


def test_atlas_substate_is_start_year_true() -> None:
    parent = DummyState()
    atlas_state = AtlasSubState(parent, 2023)
    assert atlas_state.is_start_year()


def test_atlas_substate_is_start_year_false() -> None:
    parent = DummyState()
    atlas_state = AtlasSubState(parent, 2024)
    assert not atlas_state.is_start_year()


def test_atlas_substate_set_sub_stage_progress_updates_parent() -> None:
    parent = DummyState()
    atlas_state = AtlasSubState(parent, 2024)

    atlas_state.set_sub_stage_progress("atlas")

    assert parent.sub_stage_progress == "atlas"
