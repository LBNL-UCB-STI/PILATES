from types import SimpleNamespace

from pilates.atlas.inputs import (
    atlas_run_years,
    atlas_static_input_keys,
    atlas_static_input_keys_for_interval,
    atlas_static_input_relpaths,
)
from pilates.workflows.coupler_schema import build_coupler_schema


def _settings(*, scenario: str = "baseline", start: int = 2017, end: int = 2021, freq: int = 2):
    return SimpleNamespace(
        run=SimpleNamespace(
            start_year=start,
            end_year=end,
            vehicle_ownership_freq=freq,
        ),
        atlas=SimpleNamespace(scenario=scenario),
    )


def test_atlas_static_input_relpaths_are_deterministic_and_keep_adopt_series():
    relpaths = atlas_static_input_relpaths(_settings())

    assert "vehicle_type_mapping_baseline.csv" in relpaths
    assert "vehicle_type_mapping_ESS_const_220_price.csv" not in relpaths
    assert "vehicle_type_mapping_evMandForced2.csv" not in relpaths

    assert "adopt/baseline/new_vehicles_biannual_values_2017.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2019.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2021.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2023.csv" in relpaths
    assert "accessbility_2015.RData" in relpaths


def test_atlas_static_input_keys_match_adopt_relpaths():
    keys = set(atlas_static_input_keys(_settings()))
    assert "adopt/baseline/new_vehicles_biannual_values_2021" in keys
    assert "adopt/baseline/new_vehicles_biannual_values_2023" in keys


def test_atlas_static_input_keys_for_interval_filters_adopt_year_suffixes_only():
    keys = set(
        atlas_static_input_keys_for_interval(
            _settings(start=2017, end=2023),
            interval_start_year=2017,
            interval_end_year=2019,
        )
    )

    assert "adopt/baseline/new_vehicles_biannual_values_2017" in keys
    assert "adopt/baseline/new_vehicles_biannual_values_2019" in keys
    assert "adopt/baseline/new_vehicles_biannual_values_2021" not in keys
    assert "adopt/baseline/used_vehicles_2017" in keys
    assert "adopt/baseline/used_vehicles_2019" in keys
    assert "adopt/baseline/used_vehicles_2021" not in keys
    # Non-ADOPT year-suffixed key is intentionally preserved.
    assert "accessbility2017" in keys


def test_build_coupler_schema_uses_atlas_static_keys():
    schema = build_coupler_schema([], settings=_settings())
    assert "adopt/baseline/new_vehicles_biannual_values_2021" in schema
    assert "adopt/baseline/new_vehicles_biannual_values_2023" in schema


def test_build_coupler_schema_without_extras_is_empty_for_no_steps():
    schema = build_coupler_schema(
        [],
        settings=_settings(),
        include_extras=False,
    )
    assert schema == {}


def test_atlas_run_years_are_biannual_independent_of_vehicle_ownership_freq():
    years = atlas_run_years(_settings(start=2017, end=2023, freq=6))
    assert years == {2017, 2019, 2021, 2023}
