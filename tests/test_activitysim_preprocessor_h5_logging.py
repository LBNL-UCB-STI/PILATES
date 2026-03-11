from pilates.activitysim import preprocessor as asim_preprocessor


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
