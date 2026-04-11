import logging
from types import SimpleNamespace

import pandas as pd

from pilates.activitysim import preprocessor as asim_preprocessor
from pilates.utils.usim_h5 import (
    reconcile_usim_population_table_paths,
    resolve_usim_population_table_paths,
)
from pilates.workflows.artifact_keys import (
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
)


def test_activitysim_h5_table_path_normalizes_prefixes() -> None:
    assert asim_preprocessor._activitysim_h5_table_path("", "households") == "/households"
    assert (
        asim_preprocessor._activitysim_h5_table_path("2017", "households")
        == "/2017/households"
    )
    assert (
        asim_preprocessor._activitysim_h5_table_path("/2017/", "households")
        == "/2017/households"
    )


def test_activitysim_preprocess_h5_input_key_uses_start_year_suffix() -> None:
    assert (
        asim_preprocessor._activitysim_preprocess_h5_input_key(
            "households",
            table_path="/households",
            start_year=2017,
        )
        == "activitysim_preprocess_usim_households_table_input"
    )
    assert (
        asim_preprocessor._activitysim_preprocess_h5_input_key(
            "households",
            table_path="/2017/households",
            start_year=2017,
        )
        == "activitysim_preprocess_usim_households_table_start_year_input"
    )


def test_log_activitysim_usim_input_tables_logs_expected_keys(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        asim_preprocessor.cr,
        "log_h5_table",
        lambda path, key=None, table_path=None, direction="input", **meta: calls.append(
            (path, key, table_path, direction, meta)
        ),
    )

    asim_preprocessor._log_activitysim_usim_input_tables(
        h5_path="/tmp/model_data.h5",
        resolved_table_paths={
            "households": "/2017/households",
            "persons": "/2017/persons",
            "jobs": "/2017/jobs",
            "blocks": "/2017/blocks",
        },
        start_year=2017,
    )

    assert [call[1] for call in calls] == [
        "activitysim_preprocess_usim_households_table_start_year_input",
        "activitysim_preprocess_usim_persons_table_start_year_input",
        "activitysim_preprocess_usim_jobs_table_start_year_input",
        "activitysim_preprocess_usim_blocks_table_start_year_input",
    ]
    assert all(call[3] == "input" for call in calls)
    assert all(call[4]["profile_file_schema"] is True for call in calls)


def test_activitysim_h5_preferred_prefixes_prefers_forecast_year() -> None:
    state = SimpleNamespace(forecast_year=2019, year=2018)

    assert asim_preprocessor._activitysim_h5_preferred_prefixes(state) == (
        "2019",
        "2018",
        "",
    )


def test_detect_activitysim_h5_prefix_prefers_forecast_year_tables(tmp_path) -> None:
    h5_path = tmp_path / "model_data_2019.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for prefix in ("2018", "2019"):
            for table_name in ("households", "persons", "jobs", "blocks"):
                store.put(f"/{prefix}/{table_name}", pd.DataFrame({"value": [1]}))

    with pd.HDFStore(h5_path, mode="r") as store:
        prefix = asim_preprocessor._detect_activitysim_h5_prefix(
            store,
            required_tables=("households", "persons", "jobs", "blocks"),
            preferred_prefixes=asim_preprocessor._activitysim_h5_preferred_prefixes(
                SimpleNamespace(forecast_year=2019, year=2018)
            ),
        )

    assert prefix == "2019"


def test_resolve_usim_population_table_paths_prefers_target_year(tmp_path) -> None:
    h5_path = tmp_path / "model_data_2019.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for prefix in ("2018", "2019"):
            for table_name in ("households", "persons", "jobs", "blocks"):
                store.put(f"/{prefix}/{table_name}", pd.DataFrame({"value": [1]}))

    resolved = resolve_usim_population_table_paths(
        h5_path=str(h5_path),
        year=2019,
    )

    assert resolved == {
        USIM_POPULATION_HOUSEHOLDS_TABLE: "/2019/households",
        USIM_POPULATION_PERSONS_TABLE: "/2019/persons",
        USIM_POPULATION_JOBS_TABLE: "/2019/jobs",
        USIM_POPULATION_BLOCKS_TABLE: "/2019/blocks",
    }


def test_resolve_usim_population_table_paths_falls_back_to_root_tables(tmp_path) -> None:
    h5_path = tmp_path / "model_data_input.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))

    resolved = resolve_usim_population_table_paths(
        h5_path=str(h5_path),
        year=2019,
    )

    assert resolved == {
        USIM_POPULATION_HOUSEHOLDS_TABLE: "/households",
        USIM_POPULATION_PERSONS_TABLE: "/persons",
        USIM_POPULATION_JOBS_TABLE: "/jobs",
        USIM_POPULATION_BLOCKS_TABLE: "/blocks",
    }


def test_reconcile_usim_population_table_paths_replaces_stale_root_metadata(tmp_path) -> None:
    h5_path = tmp_path / "model_data_2019.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/2019/{table_name}", pd.DataFrame({"value": [1]}))

    resolved = reconcile_usim_population_table_paths(
        h5_path=str(h5_path),
        year=2019,
        provided_paths={
            USIM_POPULATION_HOUSEHOLDS_TABLE: "/households",
            USIM_POPULATION_PERSONS_TABLE: "/persons",
            USIM_POPULATION_JOBS_TABLE: "/jobs",
            USIM_POPULATION_BLOCKS_TABLE: "/blocks",
        },
    )

    assert resolved == {
        USIM_POPULATION_HOUSEHOLDS_TABLE: "/2019/households",
        USIM_POPULATION_PERSONS_TABLE: "/2019/persons",
        USIM_POPULATION_JOBS_TABLE: "/2019/jobs",
        USIM_POPULATION_BLOCKS_TABLE: "/2019/blocks",
    }


def test_coerce_integer_like_columns_converts_only_whole_number_floats() -> None:
    df = pd.DataFrame(
        {
            "whole_numbers": [1.0, 2.0, 3.0],
            "mixed_floats": [1.25, 2.0, 3.5],
            "already_int": [1, 2, 3],
        }
    )

    coerced = asim_preprocessor._coerce_integer_like_columns(df)

    assert coerced == ["whole_numbers"]
    assert str(df["whole_numbers"].dtype) == "int64"
    assert str(df["mixed_floats"].dtype) == "float64"
    assert str(df["already_int"].dtype) == "int64"


def test_log_land_use_table_schema_reports_column_positions_and_float_flags(
    caplog,
) -> None:
    df = pd.DataFrame(
        {
            "whole_numbers": [1.0, 2.0, 3.0],
            "mixed_floats": [1.25, 2.0, 3.5],
            "with_nulls": [1, None, 3],
        }
    )

    caplog.set_level(logging.DEBUG, logger=asim_preprocessor.logger.name)
    asim_preprocessor._log_land_use_table_schema(df)

    text = caplog.text
    assert "Land use table schema before CSV write (3 columns):" in text
    assert "1:whole_numbers(float64, nulls=0, integer_like_float)" in text
    assert "2:mixed_floats(float64, nulls=0)" in text
    assert "3:with_nulls(float64, nulls=1, integer_like_float)" in text
