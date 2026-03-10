from types import SimpleNamespace

import pytest

from pilates.atlas import postprocessor as atlas_postprocessor
from pilates.atlas.outputs import AtlasPostprocessOutputs, AtlasRunOutputs
from pilates.workflows import steps
from pilates.workflows.steps import urbansim_atlas as steps_urbansim_atlas


def test_atlas_postprocess_logs_only_canonical_usim_h5_output(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_typed_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_make_typed_step_function",
        _fake_make_typed_step_function,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    output_only_keys = []
    set_output_keys = []
    h5_table_calls = []

    def _log_output_only(*, key, path, description, **meta):
        output_only_keys.append(key)

    def _log_and_set_output(*, key, path, description, coupler, **meta):
        set_output_keys.append(key)

    monkeypatch.setattr(steps_urbansim_atlas, "log_output_only", _log_output_only)
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_and_set_output",
        _log_and_set_output,
    )
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_log_named_h5_tables",
        lambda **kwargs: h5_table_calls.append(kwargs),
    )

    h5_path = tmp_path / "model_data_2023.h5"
    h5_path.write_text("x")
    vehicles2_path = tmp_path / "vehicles2_2023.csv"
    vehicles2_path.write_text("x")

    outputs = AtlasPostprocessOutputs(
        atlas_output_dir=tmp_path,
        usim_datastore_h5=h5_path,
        processed_outputs={
            "usim_h5_updated": h5_path,
            "atlas_vehicles2_output": vehicles2_path,
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2023, is_start_year=lambda: False),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert output_only_keys == ["atlas_vehicles2_output"]
    assert set_output_keys == ["usim_datastore_h5"]
    assert len(h5_table_calls) == 1
    assert h5_table_calls[0]["direction"] == "output"
    assert h5_table_calls[0]["table_keys"] == {
        "/2023/households": "atlas_postprocess_usim_households_table_updated"
    }


def test_atlas_postprocess_logs_usim_h5_as_input(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_typed_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_make_typed_step_function",
        _fake_make_typed_step_function,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []
    h5_table_calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path, meta))

    monkeypatch.setattr(steps_urbansim_atlas, "log_input_only", _log_input_only)
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_log_named_h5_tables",
        lambda **kwargs: h5_table_calls.append(kwargs),
    )

    usim_path = tmp_path / "model_data_2023.h5"
    usim_path.write_text("x")
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path),
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
    )
    state = SimpleNamespace(forecast_year=2023, is_start_year=lambda: False)

    input_logger(
        settings=settings,
        state=state,
        workspace=workspace,
        holder=SimpleNamespace(),
    )

    assert len(calls) == 1
    assert calls[0][0] == "usim_datastore_h5"
    assert calls[0][1] == str(usim_path)
    assert calls[0][2]["h5_container"] is True
    assert len(h5_table_calls) == 1
    assert h5_table_calls[0]["direction"] == "input"
    assert h5_table_calls[0]["table_keys"] == {
        "/2023/households": "atlas_postprocess_usim_households_table_input"
    }


def test_atlas_postprocess_enqueues_restart_critical_intermediates(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        atlas_postprocessor,
        "enqueue_archive_copy",
        lambda **kwargs: calls.append(kwargs),
    )

    usim_dir = tmp_path / "urbansim" / "data"
    atlas_output_dir = tmp_path / "atlas" / "atlas_output"
    atlas_input_dir = tmp_path / "atlas" / "atlas_input" / "year2023"
    usim_dir.mkdir(parents=True)
    atlas_output_dir.mkdir(parents=True)
    atlas_input_dir.mkdir(parents=True)

    usim_h5 = usim_dir / "model_data_2023.h5"
    usim_h5.write_text("h5")
    (atlas_output_dir / "householdv_2023.csv").write_text("household_id,nvehicles\n1,1\n")
    (atlas_output_dir / "vehicles_2023.csv").write_text(
        "bodytype,pred_power,modelyear\nsedan,gas,2020\n"
    )
    (atlas_input_dir / "vehicles_output.RData").write_text("rdata")

    state = SimpleNamespace(
        full_settings=SimpleNamespace(
            urbansim=SimpleNamespace(
                output_file_template="model_data_{year}.h5",
                input_file_template="model_data_{region_id}.h5",
                region_mappings={"region_to_region_id": {"test": "000"}},
            ),
            run=SimpleNamespace(region="test"),
        ),
        forecast_year=2023,
        current_year=2023,
        is_start_year=lambda: False,
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(usim_dir),
        get_atlas_output_dir=lambda: str(atlas_output_dir),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "atlas_input"),
    )

    postprocessor = atlas_postprocessor.AtlasPostprocessor("atlas", state)
    monkeypatch.setattr(
        postprocessor, "atlas_update_h5_vehicle", lambda *args, **kwargs: True
    )

    postprocessor._postprocess(
        AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs={
                "householdv_2023": atlas_output_dir / "householdv_2023.csv",
                "vehicles_2023": atlas_output_dir / "vehicles_2023.csv",
            },
        ),
        workspace,
    )

    assert any(
        call["key"] == "atlas_input_year_dir_2023"
        and str(call["path"]).endswith("year2023")
        for call in calls
    )
    assert any(
        call["key"] == "atlas_rdata_2023"
        and str(call["path"]).endswith("vehicles_output.RData")
        for call in calls
    )


def test_atlas_postprocess_raises_when_h5_update_fails(monkeypatch, tmp_path):
    usim_dir = tmp_path / "urbansim" / "data"
    atlas_output_dir = tmp_path / "atlas" / "atlas_output"
    usim_dir.mkdir(parents=True)
    atlas_output_dir.mkdir(parents=True)

    usim_h5 = usim_dir / "model_data_2023.h5"
    usim_h5.write_text("h5")
    hh_csv = atlas_output_dir / "householdv_2023.csv"
    veh_csv = atlas_output_dir / "vehicles_2023.csv"
    hh_csv.write_text("household_id,nvehicles\n1,1\n")
    veh_csv.write_text("bodytype,pred_power,modelyear\nsedan,gas,2020\n")

    state = SimpleNamespace(
        full_settings=SimpleNamespace(
            urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
        ),
        forecast_year=2023,
        current_year=2023,
        is_start_year=lambda: False,
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(usim_dir),
        get_atlas_output_dir=lambda: str(atlas_output_dir),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "atlas_input"),
    )
    postprocessor = atlas_postprocessor.AtlasPostprocessor("atlas", state)
    monkeypatch.setattr(
        postprocessor, "atlas_update_h5_vehicle", lambda *args, **kwargs: False
    )

    with pytest.raises(RuntimeError, match="failed to update UrbanSim HDF5"):
        postprocessor._postprocess(
            AtlasRunOutputs(
                atlas_output_dir=atlas_output_dir,
                raw_outputs={
                    "householdv_2023": hh_csv,
                    "vehicles_2023": veh_csv,
                },
            ),
            workspace,
        )
