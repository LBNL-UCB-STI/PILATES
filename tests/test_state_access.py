from types import SimpleNamespace

from pilates.utils.state_access import current_year, iteration_index, uses_input_datastore


def test_current_year_prefers_current_year_alias():
    state = SimpleNamespace(current_year=2025, year=2023)

    assert current_year(state) == 2025


def test_iteration_index_falls_back_to_iteration():
    state = SimpleNamespace(iteration=3)

    assert iteration_index(state) == 3


def test_uses_input_datastore_prefers_is_start_year():
    state = SimpleNamespace(
        current_year=2025,
        forecast_year=2030,
        is_start_year=lambda: True,
    )

    assert uses_input_datastore(state) is True


def test_uses_input_datastore_falls_back_to_year_comparison():
    state = SimpleNamespace(current_year=2025, forecast_year=2025, start_year=2025)

    assert uses_input_datastore(state) is True


def test_uses_input_datastore_requires_start_year_on_fallback():
    state = SimpleNamespace(current_year=2025, forecast_year=2025, start_year=2023)

    assert uses_input_datastore(state) is False
