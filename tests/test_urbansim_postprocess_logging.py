from types import SimpleNamespace

import pandas as pd

from pilates.urbansim.outputs import UrbanSimPostprocessOutputs
from pilates.workflows import steps
from pilates.workflows.steps import urbansim_atlas as steps_urbansim_atlas


def _write_h5_tables(path, tables):
    for table_name, df in tables.items():
        df.to_hdf(path, key=table_name, mode="a")


def test_urbansim_postprocess_logs_merged_h5_tables(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["output_logger"] = spec.output_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_urbansim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    set_output_keys = []
    publish_calls = []

    monkeypatch.setattr(steps_urbansim_atlas, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_and_set_output",
        lambda *, key, path, description, coupler, **meta: (
            set_output_keys.append(key),
            publish_calls.append(meta),
        ),
    )

    merged_h5 = tmp_path / "model_data_2023.h5"
    _write_h5_tables(
        merged_h5,
        {
            "households": pd.DataFrame({"cars": [1]}, index=pd.Index([1], name="household_id")),
            "persons": pd.DataFrame({"age": [40]}, index=pd.Index([1], name="person_id")),
            "land_use": pd.DataFrame({"TOTEMP": [10]}, index=pd.Index(["1"], name="TAZ")),
        },
    )

    outputs = UrbanSimPostprocessOutputs(
        usim_datastore_h5=merged_h5,
        processed_outputs={},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert set_output_keys == ["usim_datastore_h5"]
    assert len(publish_calls) == 1
    assert publish_calls[0]["child_selection"] == "include_only"
    assert publish_calls[0]["facet"]["artifact_family"] == "usim_datastore_h5"
    assert publish_calls[0]["facet"]["source_role"] == "usim_input_merged_2023"
    assert publish_calls[0]["facet"]["snapshot_role"] == "usim_input_merged"
    assert publish_calls[0]["facet"]["snapshot_reason"] == "post_merge_handoff"
    assert publish_calls[0]["facet"]["storage_event"] == "merged_h5_output"
    assert {
        path: spec.key for path, spec in publish_calls[0]["child_specs"].items()
    } == {
        "/households": "urbansim_postprocess_usim_households_table_updated",
        "/land_use": "urbansim_postprocess_usim_land_use_table_updated",
        "/persons": "urbansim_postprocess_usim_persons_table_updated",
    }


def test_urbansim_postprocess_logs_archived_h5_tables(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["output_logger"] = spec.output_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_urbansim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    output_only_keys = []
    output_only_meta = []

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_output_only",
        lambda *, key, path, description, **meta: (
            output_only_keys.append(key),
            output_only_meta.append(meta),
        ),
    )
    monkeypatch.setattr(steps_urbansim_atlas, "log_and_set_output", lambda **_kwargs: None)

    archive_h5 = tmp_path / "input_data_for_2023_outputs.h5"
    _write_h5_tables(
        archive_h5,
        {
            "households": pd.DataFrame({"cars": [1]}, index=pd.Index([1], name="household_id")),
            "parcels": pd.DataFrame({"acres": [0.5]}, index=pd.Index([10], name="parcel_id")),
        },
    )

    outputs = UrbanSimPostprocessOutputs(
        usim_datastore_h5=tmp_path / "model_data_2023.h5",
        processed_outputs={
            "usim_input_archive_2023": archive_h5,
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert output_only_keys == ["usim_input_archive_2023"]
    assert len(output_only_meta) == 1
    assert output_only_meta[0]["child_selection"] == "include_only"
    assert output_only_meta[0]["facet"]["artifact_family"] == "usim_input_archive"
    assert output_only_meta[0]["facet"]["source_role"] == "usim_datastore_h5"
    assert output_only_meta[0]["facet"]["snapshot_role"] == "usim_input_archive"
    assert output_only_meta[0]["facet"]["snapshot_reason"] == "pre_merge_input"
    assert output_only_meta[0]["facet"]["storage_event"] == "snapshot_move"
    assert {
        path: spec.key for path, spec in output_only_meta[0]["child_specs"].items()
    } == {
        "/households": "urbansim_postprocess_usim_households_table_archived",
        "/parcels": "urbansim_postprocess_usim_parcels_table_archived",
    }
