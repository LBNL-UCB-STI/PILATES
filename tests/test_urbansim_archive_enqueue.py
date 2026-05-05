from types import SimpleNamespace

import pandas as pd

from pilates.urbansim import postprocessor as usim_postprocessor


def _write_h5(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(str(path), "w") as store:
        for key, frame in tables.items():
            store[key] = frame


def test_create_next_iter_usim_data_returns_restart_h5s_without_archive_side_effects(
    monkeypatch, tmp_path
):
    mutable_dir = tmp_path / "urbansim" / "data"
    mutable_dir.mkdir(parents=True, exist_ok=True)

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="model_data_{region_id}.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "000"}},
        ),
    )
    forecast_year = 2023
    input_path = mutable_dir / "model_data_000.h5"
    output_path = mutable_dir / "model_data_2023.h5"

    _write_h5(
        input_path,
        {
            "/households": pd.DataFrame({"household_id": [1], "cars": [0]}),
            "/jobs": pd.DataFrame({"job_id": [10]}),
        },
    )
    _write_h5(
        output_path,
        {
            "/2023/households": pd.DataFrame({"household_id": [1], "cars": [1]}),
        },
    )

    monkeypatch.setattr(
        usim_postprocessor,
        "read_datastore",
        lambda *_args, **_kwargs: (pd.HDFStore(str(output_path), "r"), "2023"),
    )
    records = usim_postprocessor.create_next_iter_usim_data(
        settings=settings,
        forecast_year=forecast_year,
        mutable_data_dir=str(mutable_dir),
    )

    assert records is not None
    assert f"usim_input_archive_{forecast_year}" in records
    assert f"usim_input_merged_{forecast_year}" in records
    assert records[f"usim_input_archive_{forecast_year}"].name == (
        f"input_data_for_{forecast_year}_outputs.h5"
    )
    assert records[f"usim_input_merged_{forecast_year}"] == input_path
