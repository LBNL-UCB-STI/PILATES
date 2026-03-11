from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt

from pilates.activitysim.postprocessor import (
    _prepare_updated_tables,
    create_usim_input_data,
)


def test_prepare_updated_tables_drops_persons_with_missing_household_id(tmp_path, caplog):
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

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
    )
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

    with caplog.at_level("WARNING"):
        prepared = _prepare_updated_tables(
            settings=settings,
            state=state,
            workspace=workspace,
            asim_output_dict=asim_output_dict,
            tables_updated_by_asim=["households", "persons"],
            prefix=None,
        )

    persons = prepared["persons"]
    assert persons.index.tolist() == [11]
    assert persons["household_id"].dtype == "int64"
    assert persons["household_id"].tolist() == [1]
    assert "Dropping 1 ActivitySim persons rows with missing/invalid household_id" in caplog.text


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

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
    )
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

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
    )
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
