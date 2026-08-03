from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt
import pytest

from pilates.activitysim.postprocessor import (
    ActivitysimPostprocessor,
    _next_iter_usim_input_store_path,
    _prepare_updated_tables,
    create_usim_input_data,
    restore_atlas_household_cars,
)
from pilates.activitysim.outputs import ActivitySimRunOutputs


def _settings(vehicle_ownership=None):
    return SimpleNamespace(
        vehicle_ownership_model_enabled=vehicle_ownership == "atlas",
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(vehicle_ownership=vehicle_ownership),
        ),
        urbansim=SimpleNamespace(
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
    )


def test_next_iter_usim_input_store_path_uses_mutable_input_not_forecast_output(
    tmp_path,
):
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data")
    )
    settings = _settings()
    state = SimpleNamespace(forecast_year=2021)

    path = _next_iter_usim_input_store_path(settings, workspace, state)

    assert path == str(tmp_path / "urbansim" / "data" / "custom_mpo_001_model_data.h5")
    assert not path.endswith("model_data_2021.h5")


def test_restore_atlas_household_cars_preserves_stale_auto_ownership(tmp_path):
    households_input = tmp_path / "households.csv"
    households_output = tmp_path / "households.parquet"
    pd.DataFrame({"household_id": [1, 2], "cars": [1, 2]}).to_csv(
        households_input, index=False
    )
    pd.DataFrame(
        {"auto_ownership": [3, 1]},
        index=pd.Index([1, 2], name="household_id"),
    ).to_parquet(households_output)

    content_hash = restore_atlas_household_cars(
        households_output_path=households_output,
        households_input_path=households_input,
    )

    restored = pd.read_parquet(households_output)
    assert restored["cars"].tolist() == [1, 2]
    assert restored["auto_ownership"].tolist() == [3, 1]
    assert len(content_hash) == 64


def test_restore_atlas_household_cars_requires_exact_household_coverage(tmp_path):
    households_input = tmp_path / "households.csv"
    households_output = tmp_path / "households.parquet"
    pd.DataFrame({"household_id": [1, 2], "cars": [1, 2]}).to_csv(
        households_input, index=False
    )
    pd.DataFrame(
        {"auto_ownership": [3]},
        index=pd.Index([1], name="household_id"),
    ).to_parquet(households_output)

    with pytest.raises(ValueError, match="exact household-id coverage"):
        restore_atlas_household_cars(
            households_output_path=households_output,
            households_input_path=households_input,
        )


def test_atlas_postprocess_derives_households_output_without_mutating_run_input(
    tmp_path,
):
    asim_input_dir = tmp_path / "activitysim" / "data"
    asim_output_dir = tmp_path / "activitysim" / "output"
    asim_input_dir.mkdir(parents=True)
    asim_output_dir.mkdir(parents=True)
    households_input = asim_input_dir / "households.csv"
    raw_households_output = tmp_path / "resolved-inputs" / "households.parquet"
    raw_households_output.parent.mkdir()
    pd.DataFrame({"household_id": [1, 2], "cars": [1, 2]}).to_csv(
        households_input, index=False
    )
    pd.DataFrame(
        {"auto_ownership": [3, 1]},
        index=pd.Index([1, 2], name="household_id"),
    ).to_parquet(raw_households_output)

    settings = _settings(vehicle_ownership="atlas")
    settings.activitysim = SimpleNamespace(output_tables={"prefix": "", "tables": []})
    state = SimpleNamespace(
        full_settings=settings,
        year=2019,
        current_year=2019,
        forecast_year=2021,
        current_inner_iter=0,
        is_enabled=lambda _stage: False,
        set_sub_stage_progress=lambda _value: None,
    )
    workspace = SimpleNamespace(
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_asim_mutable_data_dir=lambda: str(asim_input_dir),
    )
    raw_outputs = ActivitySimRunOutputs(
        output_dir=asim_output_dir,
        raw_outputs={"households_asim_out": raw_households_output},
        raw_output_hashes={"households_asim_out": "upstream-hash"},
    )
    archived_households = (
        asim_output_dir / "year-2019-iteration-0" / "households.parquet"
    )
    archived_households.parent.mkdir()
    pd.DataFrame(
        {"auto_ownership": [9, 9]},
        index=pd.Index([1, 2], name="household_id"),
    ).to_parquet(archived_households)

    outputs = ActivitysimPostprocessor("activitysim", state).postprocess(
        raw_outputs,
        workspace,
        households_asim_input_path=str(households_input),
    )

    assert "cars" not in pd.read_parquet(raw_households_output).columns
    archived = pd.read_parquet(archived_households)
    assert archived["cars"].tolist() == [1, 2]
    assert archived["auto_ownership"].tolist() == [3, 1]
    assert raw_outputs.raw_output_hashes["households_asim_out"] == "upstream-hash"
    assert outputs.processed_output_hashes["households_asim_out"] != "upstream-hash"


def test_disabled_atlas_ownership_does_not_require_household_input(tmp_path):
    asim_input_dir = tmp_path / "activitysim" / "data"
    asim_output_dir = tmp_path / "activitysim" / "output"
    asim_input_dir.mkdir(parents=True)
    asim_output_dir.mkdir(parents=True)
    raw_households_output = tmp_path / "resolved-inputs" / "households.parquet"
    raw_households_output.parent.mkdir()
    pd.DataFrame(
        {"auto_ownership": [3]},
        index=pd.Index([1], name="household_id"),
    ).to_parquet(raw_households_output)

    settings = _settings(vehicle_ownership="atlas")
    settings.vehicle_ownership_model_enabled = False
    settings.activitysim = SimpleNamespace(output_tables={"prefix": "", "tables": []})
    state = SimpleNamespace(
        full_settings=settings,
        year=2019,
        current_year=2019,
        forecast_year=2021,
        current_inner_iter=0,
        is_enabled=lambda _stage: False,
        set_sub_stage_progress=lambda _value: None,
    )
    workspace = SimpleNamespace(
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_asim_mutable_data_dir=lambda: str(asim_input_dir),
    )

    outputs = ActivitysimPostprocessor("activitysim", state).postprocess(
        ActivitySimRunOutputs(
            output_dir=asim_output_dir,
            raw_outputs={"households_asim_out": raw_households_output},
        ),
        workspace,
    )

    archived = pd.read_parquet(outputs.processed_outputs["households_asim_out"])
    assert archived["auto_ownership"].tolist() == [3]
    assert "cars" not in archived.columns


def test_prepare_updated_tables_preserves_usim_person_household_ids_when_asim_ids_are_invalid(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    settings = _settings()
    state = SimpleNamespace(forecast_year=2023)
    asim_output_dict = {
        "households": pd.DataFrame(
            {"auto_ownership": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        ),
        "persons": pd.DataFrame(
            {"household_id": [1, None], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        ),
    }

    prepared = _prepare_updated_tables(
        settings=settings,
        state=state,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
        population_source_store_path=str(h5_path),
        prefix=None,
    )

    persons = prepared["persons"]
    assert persons.index.tolist() == [11, 21]
    assert persons["household_id"].dtype == "int64"
    assert persons["household_id"].tolist() == [1, 2]


def test_prepare_updated_tables_warns_on_household_member_person_alignment_fallback(
    tmp_path, caplog
):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {
                "household_id": [1, 2],
                "member_id": [1, 1],
                "work_zone_id": [-1, -1],
                "school_zone_id": [-1, -1],
            },
            index=pd.Index([11, 21], name="person_id"),
        )

    with caplog.at_level("WARNING"):
        _prepare_updated_tables(
            settings=_settings(vehicle_ownership="atlas"),
            state=SimpleNamespace(forecast_year=2023),
            asim_output_dict={
                "households": pd.DataFrame(
                    {"auto_ownership": [1, 2], "block_id": ["0001", "0002"]},
                    index=pd.Index([1, 2], name="household_id"),
                ),
                "persons": pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "member_id": [1, 1],
                        "workplace_taz": [555, 666],
                        "school_taz": [777, 888],
                    }
                ),
            },
            tables_updated_by_asim=["households", "persons"],
            population_source_store_path=str(h5_path),
            prefix=None,
        )

    assert (
        "missing person_id; falling back to household_id/member_id alignment"
        in caplog.text
    )


def test_prepare_updated_tables_falls_back_to_current_input_store(tmp_path):
    h5_path = tmp_path / "custom_mpo_001_model_data.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    settings = _settings()
    state = SimpleNamespace(forecast_year=2023)
    asim_output_dict = {
        "households": pd.DataFrame(
            {"auto_ownership": [3, 4], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        ),
        "persons": pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        ),
    }

    prepared = _prepare_updated_tables(
        settings=settings,
        state=state,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
        population_source_store_path=str(h5_path),
        prefix=2023,
    )

    assert prepared["households"]["cars"].tolist() == [3, 4]
    assert prepared["persons"]["household_id"].tolist() == [1, 2]


def test_create_usim_input_data_falls_back_to_current_input_store(tmp_path):
    input_h5 = tmp_path / "custom_mpo_001_model_data.h5"
    with pd.HDFStore(str(input_h5), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )
        store["jobs"] = pd.DataFrame(
            {"block_id": ["0001", "0002"]},
            index=pd.Index([101, 102], name="job_id"),
        )

    settings = _settings()
    state = SimpleNamespace(forecast_year=2023)
    asim_output_dict = {
        "households": pd.DataFrame(
            {"cars": [3, 4], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        ),
        "persons": pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        ),
    }

    new_input_path, output_record = create_usim_input_data(
        settings=settings,
        state=state,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
        asim_source_paths=[],
        current_input_store_path=str(input_h5),
        population_source_store_path=None,
    )

    assert new_input_path == str(input_h5)
    assert output_record is not None

    archive_path = tmp_path / "input_data_for_2023_outputs.h5"
    assert archive_path.exists()

    with pd.HDFStore(str(input_h5), mode="r") as store:
        pdt.assert_frame_equal(store["households"], asim_output_dict["households"])
        pdt.assert_frame_equal(store["persons"], asim_output_dict["persons"])
        pdt.assert_frame_equal(
            store["jobs"],
            pd.DataFrame(
                {"block_id": ["0001", "0002"]},
                index=pd.Index([101, 102], name="job_id"),
            ),
        )


def test_create_usim_input_data_uses_population_source_when_it_matches_current_input(
    tmp_path,
):
    input_h5 = tmp_path / "custom_mpo_001_model_data.h5"
    with pd.HDFStore(str(input_h5), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [7, 8], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )
        store["jobs"] = pd.DataFrame(
            {"block_id": ["0001", "0002"]},
            index=pd.Index([101, 102], name="job_id"),
        )

    asim_output_dict = {
        "households": pd.DataFrame(
            {"cars": [3, 4], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        ),
        "persons": pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        ),
    }

    new_input_path, output_record = create_usim_input_data(
        settings=_settings(),
        state=SimpleNamespace(forecast_year=2023),
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
        asim_source_paths=[],
        current_input_store_path=str(input_h5),
        population_source_store_path=str(input_h5),
    )

    assert new_input_path == str(input_h5)
    assert output_record is not None

    archive_path = tmp_path / "input_data_for_2023_outputs.h5"
    with pd.HDFStore(str(archive_path), mode="r") as store:
        assert store["households"]["cars"].tolist() == [7, 8]

    with pd.HDFStore(str(input_h5), mode="r") as store:
        assert store["households"]["cars"].tolist() == [3, 4]
        assert store["jobs"].index.tolist() == [101, 102]


def test_prepare_updated_tables_preserves_usim_owned_household_fields_when_atlas_enabled(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {
                "persons": [2, 4],
                "workers": [1, 3],
                "cars": [7, 8],
                "block_id": ["0001", "0002"],
            },
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    prepared = _prepare_updated_tables(
        settings=_settings(vehicle_ownership="atlas"),
        state=SimpleNamespace(forecast_year=2023),
        asim_output_dict={
            "households": pd.DataFrame(
                {"hhsize": [9, 10], "num_workers": [5, 6], "auto_ownership": [1, 2]},
                index=pd.Index([1, 2], name="household_id"),
            ),
            "persons": pd.DataFrame(
                {"household_id": [1, 2], "member_id": [1, 1]},
                index=pd.Index([11, 21], name="person_id"),
            ),
        },
        tables_updated_by_asim=["households", "persons"],
        population_source_store_path=str(h5_path),
        prefix=None,
    )

    households = prepared["households"]
    assert households["persons"].tolist() == [2, 4]
    assert households["workers"].tolist() == [1, 3]
    assert households["cars"].tolist() == [7, 8]


def test_prepare_updated_tables_updates_cars_from_asim_when_atlas_disabled(tmp_path):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {
                "persons": [2, 4],
                "workers": [1, 3],
                "cars": [7, 8],
                "block_id": ["0001", "0002"],
            },
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    prepared = _prepare_updated_tables(
        settings=_settings(vehicle_ownership=None),
        state=SimpleNamespace(forecast_year=2023),
        asim_output_dict={
            "households": pd.DataFrame(
                {"hhsize": [9, 10], "num_workers": [5, 6], "auto_ownership": [1, 2]},
                index=pd.Index([1, 2], name="household_id"),
            ),
            "persons": pd.DataFrame(
                {"household_id": [1, 2], "member_id": [1, 1]},
                index=pd.Index([11, 21], name="person_id"),
            ),
        },
        tables_updated_by_asim=["households", "persons"],
        population_source_store_path=str(h5_path),
        prefix=None,
    )

    households = prepared["households"]
    assert households["persons"].tolist() == [2, 4]
    assert households["workers"].tolist() == [1, 3]
    assert households["cars"].tolist() == [1, 2]


def test_prepare_updated_tables_preserves_usim_person_fields_but_updates_zone_ids(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"cars": [1, 2], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {
                "household_id": [1, 2],
                "member_id": [1, 1],
                "worker": [0, 1],
                "student": [1, 0],
                "work_zone_id": [-1, 200],
                "school_zone_id": [100, -1],
            },
            index=pd.Index([11, 21], name="person_id"),
        )

    prepared = _prepare_updated_tables(
        settings=_settings(vehicle_ownership="atlas"),
        state=SimpleNamespace(forecast_year=2023),
        asim_output_dict={
            "households": pd.DataFrame(
                {"auto_ownership": [3, 4], "block_id": ["0001", "0002"]},
                index=pd.Index([1, 2], name="household_id"),
            ),
            "persons": pd.DataFrame(
                {
                    "person_id": [11, 21],
                    "household_id": [99, 98],
                    "member_id": [7, 8],
                    "worker": [1, 0],
                    "student": [0, 1],
                    "workplace_taz": [555, 666],
                    "school_taz": [777, 888],
                }
            ),
        },
        tables_updated_by_asim=["households", "persons"],
        population_source_store_path=str(h5_path),
        prefix=None,
    )

    persons = prepared["persons"]
    assert persons["household_id"].tolist() == [1, 2]
    assert persons["member_id"].tolist() == [1, 1]
    assert persons["worker"].tolist() == [0, 1]
    assert persons["student"].tolist() == [1, 0]
    assert persons["work_zone_id"].tolist() == [555, 666]
    assert persons["school_zone_id"].tolist() == [777, 888]
