from types import SimpleNamespace

import pandas as pd
import pytest

from pilates.atlas import postprocessor as atlas_postprocessor
from pilates.atlas.outputs import AtlasPostprocessOutputs, AtlasRunOutputs
from pilates.workflows.artifact_keys import USIM_POPULATION_SOURCE_H5
from pilates.workflows import steps
from pilates.workflows.steps import urbansim_atlas as steps_urbansim_atlas


def test_atlas_postprocess_logs_only_canonical_usim_h5_output(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["output_logger"] = spec.output_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    output_only_keys = []
    set_output_keys = []
    publish_meta = []

    def _log_output_only(*, key, path, description, **meta):
        output_only_keys.append(key)

    def _log_and_set_output(*, key, path, description, coupler, **meta):
        set_output_keys.append(key)
        publish_meta.append(meta)

    monkeypatch.setattr(steps_urbansim_atlas, "log_output_only", _log_output_only)
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_and_set_output",
        _log_and_set_output,
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
    assert set_output_keys == [USIM_POPULATION_SOURCE_H5]
    assert len(publish_meta) == 1
    assert publish_meta[0]["child_selection"] == "include_only"
    assert {
        path: spec.key for path, spec in publish_meta[0]["child_specs"].items()
    } == {
        "/2023/households": "atlas_postprocess_usim_households_table_updated"
    }


def test_atlas_postprocess_logs_usim_h5_as_input(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path, meta))

    monkeypatch.setattr(steps_urbansim_atlas, "log_input_only", _log_input_only)

    usim_path = tmp_path / "model_data_2023.h5"
    usim_path.write_text("x")
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas_output"),
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
    assert calls[0][2]["child_selection"] == "include_only"
    assert {
        path: spec.key for path, spec in calls[0][2]["child_specs"].items()
    } == {
        "/2023/households": "atlas_postprocess_usim_households_table_input"
    }


def test_atlas_postprocess_logs_selected_start_year_h5_as_input(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path, meta))

    monkeypatch.setattr(steps_urbansim_atlas, "log_input_only", _log_input_only)

    base_h5 = tmp_path / "model_data_000.h5"
    forecast_h5 = tmp_path / "model_data_2023.h5"
    base_h5.write_text("base")
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas_output"),
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(
            output_file_template="model_data_{year}.h5",
            input_file_template="model_data_{region_id}.h5",
            region_mappings={"region_to_region_id": {"test": "000"}},
        ),
        run=SimpleNamespace(region="test"),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        is_start_year=lambda: True,
        atlas_usim_datastore_h5=str(base_h5),
    )

    input_logger(
        settings=settings,
        state=state,
        workspace=workspace,
        holder=SimpleNamespace(),
    )

    assert not forecast_h5.exists()
    assert len(calls) == 1
    assert calls[0][0] == "usim_datastore_h5"
    assert calls[0][1] == str(base_h5)
    assert {
        path: spec.key for path, spec in calls[0][2]["child_specs"].items()
    } == {
        "/households": "atlas_postprocess_usim_households_table_input"
    }


def test_atlas_postprocess_logs_year_scoped_start_subyear_table(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path, meta))

    monkeypatch.setattr(steps_urbansim_atlas, "log_input_only", _log_input_only)

    year_scoped_h5 = tmp_path / "model_data_2023.h5"
    pd.DataFrame({"cars": [0]}, index=pd.Index([1], name="household_id")).to_hdf(
        year_scoped_h5, key="/2023/households", mode="w"
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas_output"),
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(
            output_file_template="model_data_{year}.h5",
            input_file_template="model_data_{region_id}.h5",
            region_mappings={"region_to_region_id": {"test": "000"}},
        ),
        run=SimpleNamespace(region="test"),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        is_start_year=lambda: True,
        atlas_usim_datastore_h5=str(year_scoped_h5),
    )

    input_logger(
        settings=settings,
        state=state,
        workspace=workspace,
        holder=SimpleNamespace(),
    )

    assert len(calls) == 1
    assert calls[0][0] == "usim_datastore_h5"
    assert calls[0][1] == str(year_scoped_h5)
    assert {
        path: spec.key for path, spec in calls[0][2]["child_specs"].items()
    } == {
        "/2023/households": "atlas_postprocess_usim_households_table_input"
    }


def test_atlas_postprocess_logs_resolved_fallback_households_table(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        captured["output_logger"] = spec.output_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    usim_path = tmp_path / "model_data_2023.h5"
    pd.DataFrame({"cars": [0]}, index=pd.Index([1], name="household_id")).to_hdf(
        usim_path, key="/2024/households", mode="w"
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas_output"),
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
    )
    state = SimpleNamespace(forecast_year=2023, is_start_year=lambda: False)

    input_calls = []
    output_calls = []

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_input_only",
        lambda **kwargs: input_calls.append(kwargs),
    )
    monkeypatch.setattr(
        steps_urbansim_atlas,
        "log_and_set_output",
        lambda **kwargs: output_calls.append(kwargs),
    )
    captured["input_logger"](
        settings=settings,
        state=state,
        workspace=workspace,
        holder=SimpleNamespace(),
    )
    captured["output_logger"](
        AtlasPostprocessOutputs(
            atlas_output_dir=tmp_path,
            usim_datastore_h5=usim_path,
            processed_outputs={"usim_h5_updated": usim_path},
        ),
        settings=settings,
        state=state,
        workspace=workspace,
        holder=SimpleNamespace(),
    )

    assert {
        path: spec.key for path, spec in input_calls[0]["child_specs"].items()
    } == {
        "/2024/households": "atlas_postprocess_usim_households_table_input"
    }
    assert {
        path: spec.key for path, spec in output_calls[0]["child_specs"].items()
    } == {
        "/2024/households": "atlas_postprocess_usim_households_table_updated"
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
        postprocessor,
        "atlas_update_h5_vehicle",
        lambda *args, **kwargs: "/2023/households",
    )
    monkeypatch.setattr(
        postprocessor,
        "_validate_updated_h5_table",
        lambda **_kwargs: None,
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


def test_atlas_postprocess_uses_selected_start_year_h5(monkeypatch, tmp_path):
    atlas_output_dir = tmp_path / "atlas" / "atlas_output"
    atlas_input_dir = tmp_path / "atlas" / "atlas_input" / "year2023"
    usim_dir = tmp_path / "urbansim" / "data"
    atlas_output_dir.mkdir(parents=True)
    atlas_input_dir.mkdir(parents=True)
    usim_dir.mkdir(parents=True)

    base_h5 = usim_dir / "model_data_000.h5"
    base_h5.write_text("h5")
    hh_csv = atlas_output_dir / "householdv_2023.csv"
    veh_csv = atlas_output_dir / "vehicles_2023.csv"
    hh_csv.write_text("household_id,nvehicles\n1,1\n")
    veh_csv.write_text("bodytype,pred_power,modelyear\nsedan,gas,2020\n")

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
        current_year=2017,
        atlas_usim_datastore_h5=str(base_h5),
        is_start_year=lambda: True,
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(usim_dir),
        get_atlas_output_dir=lambda: str(atlas_output_dir),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "atlas_input"),
    )

    postprocessor = atlas_postprocessor.AtlasPostprocessor("atlas", state)
    seen = {}

    def _fake_update(settings, output_year, h5_file_path, household_v_csv_path):
        seen["h5_file_path"] = h5_file_path
        seen["household_v_csv_path"] = household_v_csv_path
        return "/households"

    monkeypatch.setattr(postprocessor, "atlas_update_h5_vehicle", _fake_update)
    monkeypatch.setattr(
        postprocessor,
        "_validate_updated_h5_table",
        lambda **_kwargs: None,
    )

    outputs = postprocessor._postprocess(
        AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs={
                "householdv_2023": hh_csv,
                "vehicles_2023": veh_csv,
            },
        ),
        workspace,
    )

    assert seen["h5_file_path"] == str(base_h5)
    assert seen["household_v_csv_path"] == str(hh_csv)
    assert outputs.usim_datastore_h5 == base_h5


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
        postprocessor, "atlas_update_h5_vehicle", lambda *args, **kwargs: None
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


def test_atlas_update_h5_vehicle_rejects_household_id_set_mismatch(tmp_path, caplog):
    h5_path = tmp_path / "model_data_2023.h5"
    hh_csv = tmp_path / "householdv_2023.csv"

    pd.DataFrame(
        {"cars": [0, 1]},
        index=pd.Index([1, 2], name="household_id"),
    ).to_hdf(str(h5_path), key="/2023/households", mode="w")
    hh_csv.write_text("household_id,nvehicles\n1,1\n3,2\n", encoding="utf-8")

    state = SimpleNamespace(
        is_start_year=lambda: False,
    )
    postprocessor = atlas_postprocessor.AtlasPostprocessor("atlas", state)

    with caplog.at_level("ERROR"):
        updated = postprocessor.atlas_update_h5_vehicle(
            settings=SimpleNamespace(),
            output_year=2023,
            h5_file_path=str(h5_path),
            household_v_csv_path=str(hh_csv),
        )

    assert updated is None
    assert "missing_in_h5=1" in caplog.text
    assert "missing_in_atlas=1" in caplog.text


def test_atlas_update_h5_vehicle_returns_updated_table_path(tmp_path):
    h5_path = tmp_path / "model_data_2023.h5"
    hh_csv = tmp_path / "householdv_2023.csv"

    pd.DataFrame(
        {"cars": [0], "hh_cars": ["none"]},
        index=pd.Index([1], name="household_id"),
    ).to_hdf(str(h5_path), key="/2023/households", mode="w")
    hh_csv.write_text("household_id,nvehicles\n1,2\n", encoding="utf-8")

    postprocessor = atlas_postprocessor.AtlasPostprocessor(
        "atlas",
        SimpleNamespace(is_start_year=lambda: False),
    )

    updated = postprocessor.atlas_update_h5_vehicle(
        settings=SimpleNamespace(),
        output_year=2023,
        h5_file_path=str(h5_path),
        household_v_csv_path=str(hh_csv),
    )

    assert updated == "/2023/households"


def test_atlas_postprocess_validates_reported_updated_table_exists(tmp_path):
    h5_path = tmp_path / "model_data_2023.h5"
    pd.DataFrame({"cars": [0]}, index=pd.Index([1], name="household_id")).to_hdf(
        str(h5_path), key="/households", mode="w"
    )

    with pytest.raises(RuntimeError, match="not present after write"):
        atlas_postprocessor.AtlasPostprocessor._validate_updated_h5_table(
            h5_file_path=str(h5_path),
            table_path="/2023/households",
            output_year=2023,
        )
