from types import SimpleNamespace

import pandas as pd

from pilates.activitysim.postprocessor import _prepare_updated_tables


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
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
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
