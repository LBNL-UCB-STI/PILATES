from types import SimpleNamespace

from pilates.atlas.inputs import (
    atlas_static_input_keys,
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


def test_atlas_static_input_relpaths_are_deterministic_and_year_filtered():
    relpaths = atlas_static_input_relpaths(_settings())

    assert "vehicle_type_mapping_baseline.csv" in relpaths
    assert "vehicle_type_mapping_ESS_const_220_price.csv" not in relpaths
    assert "vehicle_type_mapping_evMandForced2.csv" not in relpaths

    assert "adopt/baseline/new_vehicles_biannual_values_2017.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2019.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2021.csv" in relpaths
    assert "adopt/baseline/new_vehicles_biannual_values_2023.csv" not in relpaths


def test_atlas_static_input_keys_match_filtered_relpaths():
    keys = set(atlas_static_input_keys(_settings()))
    assert "adopt/baseline/new_vehicles_biannual_values_2021" in keys
    assert "adopt/baseline/new_vehicles_biannual_values_2023" not in keys


def test_build_coupler_schema_uses_filtered_atlas_static_keys():
    schema = build_coupler_schema([], settings=_settings())
    assert "adopt/baseline/new_vehicles_biannual_values_2021" in schema
    assert "adopt/baseline/new_vehicles_biannual_values_2023" not in schema
