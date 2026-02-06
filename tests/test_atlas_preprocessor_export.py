import pandas as pd
import pytest

from pilates.atlas.preprocessor import (
    _export_atlas_table_to_csv,
    _prepare_atlas_table_for_export,
)


def test_prepare_atlas_table_for_export_keeps_expected_index() -> None:
    df = pd.DataFrame(
        {"income": [100, 200]},
        index=pd.Index([1, 2], name="household_id"),
    )

    prepared = _prepare_atlas_table_for_export(
        df,
        table_name_in_h5="/households",
        expected_index_name="household_id",
    )

    assert prepared.index.name == "household_id"
    assert "household_id" not in prepared.columns


def test_prepare_atlas_table_for_export_promotes_column_to_index() -> None:
    df = pd.DataFrame(
        {
            "household_id": [1, 2],
            "income": [100, 200],
        }
    )

    prepared = _prepare_atlas_table_for_export(
        df,
        table_name_in_h5="/households",
        expected_index_name="household_id",
    )

    assert prepared.index.name == "household_id"
    assert "household_id" not in prepared.columns
    assert list(prepared.index) == [1, 2]


def test_prepare_atlas_table_for_export_raises_when_key_missing() -> None:
    df = pd.DataFrame({"income": [100, 200]})

    with pytest.raises(ValueError, match="missing expected logical key"):
        _prepare_atlas_table_for_export(
            df,
            table_name_in_h5="/households",
            expected_index_name="household_id",
        )


def test_export_atlas_table_to_csv_writes_explicit_index_label(tmp_path) -> None:
    df = pd.DataFrame(
        {"income": [100, 200]},
        index=pd.Index([1, 2], name="household_id"),
    )
    output_csv = tmp_path / "households.csv"

    _export_atlas_table_to_csv(
        df,
        table_name_in_h5="/households",
        expected_index_name="household_id",
        output_csv_path=str(output_csv),
    )

    lines = output_csv.read_text().splitlines()
    assert lines
    assert lines[0].startswith("household_id,")

    loaded = pd.read_csv(output_csv)
    assert "household_id" in loaded.columns
    assert loaded["household_id"].tolist() == [1, 2]
