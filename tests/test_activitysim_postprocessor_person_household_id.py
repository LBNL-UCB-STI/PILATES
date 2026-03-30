from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt

from pilates.activitysim.postprocessor import (
    _prepare_updated_tables,
    create_usim_input_data,
)


def _settings(vehicle_ownership=None):
    return SimpleNamespace(
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
    workspace = SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path))

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
        workspace=workspace,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
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
            workspace=SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path)),
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
            prefix=None,
        )

    assert "missing person_id; falling back to household_id/member_id alignment" in caplog.text


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
    workspace = SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path))

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
        workspace=workspace,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
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
    workspace = SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path))

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
        workspace=workspace,
        asim_output_dict=asim_output_dict,
        tables_updated_by_asim=["households", "persons"],
        asim_source_paths=[],
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


def test_prepare_updated_tables_preserves_usim_owned_household_fields_when_atlas_enabled(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2023.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store["households"] = pd.DataFrame(
            {"persons": [2, 4], "workers": [1, 3], "cars": [7, 8], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    prepared = _prepare_updated_tables(
        settings=_settings(vehicle_ownership="atlas"),
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path)),
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
            {"persons": [2, 4], "workers": [1, 3], "cars": [7, 8], "block_id": ["0001", "0002"]},
            index=pd.Index([1, 2], name="household_id"),
        )
        store["persons"] = pd.DataFrame(
            {"household_id": [1, 2], "member_id": [1, 1]},
            index=pd.Index([11, 21], name="person_id"),
        )

    prepared = _prepare_updated_tables(
        settings=_settings(vehicle_ownership=None),
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path)),
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
        prefix=None,
    )

    households = prepared["households"]
    assert households["persons"].tolist() == [2, 4]
    assert households["workers"].tolist() == [1, 3]
    assert households["cars"].tolist() == [1, 2]


def test_prepare_updated_tables_preserves_usim_person_fields_but_updates_zone_ids(tmp_path):
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
        workspace=SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path)),
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
        prefix=None,
    )

    persons = prepared["persons"]
    assert persons["household_id"].tolist() == [1, 2]
    assert persons["member_id"].tolist() == [1, 1]
    assert persons["worker"].tolist() == [0, 1]
    assert persons["student"].tolist() == [1, 0]
    assert persons["work_zone_id"].tolist() == [555, 666]
    assert persons["school_zone_id"].tolist() == [777, 888]
