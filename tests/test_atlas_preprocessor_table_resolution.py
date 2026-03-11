import pandas as pd

from pilates.atlas.preprocessor import _resolve_atlas_h5_table_key


def test_resolve_atlas_h5_table_key_prefers_year_scoped_when_available(tmp_path):
    path = tmp_path / "data.h5"
    df = pd.DataFrame({"x": [1]})
    df.to_hdf(path, key="households", mode="w")
    df.to_hdf(path, key="/2019/households", mode="a")

    with pd.HDFStore(path, mode="r") as store:
        resolved = _resolve_atlas_h5_table_key(
            store, year=2019, table="households", is_start_year=False
        )

    assert resolved == "/2019/households"


def test_resolve_atlas_h5_table_key_falls_back_to_root_for_non_start_year(tmp_path):
    path = tmp_path / "data.h5"
    pd.DataFrame({"x": [1]}).to_hdf(path, key="households", mode="w")

    with pd.HDFStore(path, mode="r") as store:
        resolved = _resolve_atlas_h5_table_key(
            store, year=2019, table="households", is_start_year=False
        )

    assert resolved == "households"


def test_resolve_atlas_h5_table_key_uses_root_for_start_year(tmp_path):
    path = tmp_path / "data.h5"
    pd.DataFrame({"x": [1]}).to_hdf(path, key="households", mode="w")

    with pd.HDFStore(path, mode="r") as store:
        resolved = _resolve_atlas_h5_table_key(
            store, year=2017, table="households", is_start_year=True
        )

    assert resolved == "households"


def test_resolve_atlas_h5_table_key_falls_back_to_prior_year_scoped_table(tmp_path):
    path = tmp_path / "data.h5"
    pd.DataFrame({"x": [1]}).to_hdf(path, key="/2023/households", mode="w")

    with pd.HDFStore(path, mode="r") as store:
        resolved = _resolve_atlas_h5_table_key(
            store, year=2025, table="households", is_start_year=False
        )

    assert resolved == "/2023/households"


def test_resolve_atlas_h5_table_key_falls_back_to_earliest_future_year_scoped_table(
    tmp_path,
):
    path = tmp_path / "data.h5"
    pd.DataFrame({"x": [1]}).to_hdf(path, key="/2027/households", mode="w")

    with pd.HDFStore(path, mode="r") as store:
        resolved = _resolve_atlas_h5_table_key(
            store, year=2025, table="households", is_start_year=False
        )

    assert resolved == "/2027/households"
