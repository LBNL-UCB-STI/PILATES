from types import SimpleNamespace

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
)
from pilates.generic.records import RecordStore
from pilates.workflows import steps
from pilates.workflows.steps import activitysim as steps_activitysim


def _dummy_coupler():
    return SimpleNamespace(
        get=lambda *args, **kwargs: None,
        set=lambda *args, **kwargs: None,
        update=lambda *args, **kwargs: None,
        set_from_artifact=lambda *args, **kwargs: None,
    )


def test_activitysim_postprocess_logs_content_hash(monkeypatch, tmp_path) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.__pilates_output_replayer__
    calls = []
    h5_table_calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps_activitysim, "log_output_only", _log_output_only)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: h5_table_calls.append(kwargs),
    )

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={
            "asim_input_skims_zarr_archived": tmp_path / "skims.zarr"
        },
        processed_output_hashes={
            "asim_input_skims_zarr_archived": "abc123"
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=0, iteration=0),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert len(calls) == 1
    assert calls[0][0] == "asim_input_skims_zarr_archived"
    assert calls[0][1]["content_hash"] == "abc123"
    assert h5_table_calls == []


def test_activitysim_postprocess_logs_source_input_files(monkeypatch, tmp_path) -> None:
    fake_postprocessor = SimpleNamespace(
        postprocess=lambda _raw_outputs, _workspace: RecordStore()
    )
    monkeypatch.setattr(
        steps_activitysim.ModelFactory,
        "get_postprocessor",
        lambda self, *args, **kwargs: fake_postprocessor,
    )
    monkeypatch.setattr(
        steps_activitysim,
        "record_store_to_outputs",
        lambda **_kwargs: ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=tmp_path,
            processed_outputs={},
        ),
    )
    monkeypatch.setattr(steps_activitysim, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(steps_activitysim, "log_and_set_output", lambda **_kwargs: None)
    monkeypatch.setattr(steps_activitysim, "_log_named_h5_tables", lambda **_kwargs: None)
    calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path))

    monkeypatch.setattr(steps_activitysim, "log_input_only", _log_input_only)

    asim_input_dir = tmp_path / "asim" / "data"
    asim_output_dir = tmp_path / "asim" / "output"
    usim_data_dir = tmp_path / "urbansim" / "data"
    (asim_input_dir).mkdir(parents=True)
    (asim_output_dir / "cache").mkdir(parents=True)
    usim_data_dir.mkdir(parents=True)

    for rel in ("households.csv", "persons.csv", "land_use.csv", "skims.omx"):
        (asim_input_dir / rel).write_text("x")
    (asim_output_dir / "cache" / "skims.zarr").mkdir(parents=True, exist_ok=True)
    (usim_data_dir / "model_data_06197001.h5").write_text("x")
    (usim_data_dir / "model_data_2023.h5").write_text("x")

    workspace = SimpleNamespace(
        get_asim_mutable_data_dir=lambda: str(asim_input_dir),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_usim_mutable_data_dir=lambda: str(usim_data_dir),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        urbansim=SimpleNamespace(
            input_file_template="model_data_{region_id}.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"seattle": "06197001"}},
        ),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        is_enabled=lambda stage: True,
        Stage=SimpleNamespace(land_use="land_use"),
    )

    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(
            activitysim_run=SimpleNamespace(to_postprocess_record_store=RecordStore)
        ),
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    keys = {key for key, _path in calls}
    assert "households_asim_in" in keys
    assert "persons_asim_in" in keys
    assert "land_use_asim_in" in keys
    assert "omx_skims" in keys
    assert "zarr_skims" in keys
    assert "usim_datastore_h5" in keys
    assert "usim_forecast_output" in keys


def test_activitysim_preprocess_logs_selected_usim_h5_tables(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        steps_activitysim.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        steps_activitysim,
        "_execute_preprocess",
        lambda *_args, **_kwargs: RecordStore(),
    )
    table_calls = []

    monkeypatch.setattr(steps_activitysim, "resolve_preferred_step_input", lambda **_kwargs: {})
    monkeypatch.setattr(
        steps_activitysim,
        "first_resolved_key",
        lambda *_args, **_kwargs: "usim_datastore_base_h5",
    )
    monkeypatch.setattr(
        steps_activitysim,
        "resolved_value_for_key",
        lambda **_kwargs: str(tmp_path / "model_data.h5"),
    )
    monkeypatch.setattr(
        steps_activitysim,
        "resolve_artifact_from_value",
        lambda selected_value, **_kwargs: selected_value,
    )
    monkeypatch.setattr(
        steps_activitysim,
        "artifact_to_path",
        lambda value, _workspace=None: str(value),
    )
    monkeypatch.setattr(steps_activitysim, "log_and_set_input", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: table_calls.append(kwargs),
    )
    monkeypatch.setattr(steps_activitysim, "_log_step_records", lambda **_kwargs: None)

    h5_path = tmp_path / "model_data.h5"
    h5_path.write_text("x")
    asim_data_dir = tmp_path / "asim_data"
    asim_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("land_use.csv", "households.csv", "persons.csv"):
        (asim_data_dir / filename).write_text("x")
    monkeypatch.setattr(
        steps_activitysim,
        "record_store_to_outputs",
        lambda **_kwargs: ActivitySimPreprocessOutputs(
            mutable_data_dir=asim_data_dir,
            land_use_table=asim_data_dir / "land_use.csv",
            households_table=asim_data_dir / "households.csv",
            persons_table=asim_data_dir / "persons.csv",
            omx_skims=None,
        ),
    )

    step_fn = steps.make_activitysim_preprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )

    step_fn(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2023, start_year=2017),
        workspace=SimpleNamespace(get_asim_mutable_data_dir=lambda: str(asim_data_dir)),
    )

    assert len(table_calls) == 1
    assert table_calls[0]["direction"] == "input"
    assert table_calls[0]["path"] == str(h5_path)
    assert table_calls[0]["table_keys"]["/households"] == (
        "activitysim_preprocess_usim_households_table_input"
    )
    assert table_calls[0]["table_keys"]["/2017/households"] == (
        "activitysim_preprocess_usim_households_table_start_year_input"
    )


def test_activitysim_postprocess_logs_updated_usim_h5_tables(monkeypatch, tmp_path) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.__pilates_output_replayer__
    table_calls = []

    monkeypatch.setattr(steps_activitysim, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(steps_activitysim, "log_and_set_output", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: table_calls.append(kwargs),
    )

    h5_path = tmp_path / "next_iteration.h5"
    h5_path.write_text("x")
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=h5_path,
        asim_output_dir=tmp_path,
        processed_outputs={},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2030, iteration=2),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert len(table_calls) == 1
    assert table_calls[0]["direction"] == "output"
    assert table_calls[0]["table_keys"] == {
        "/households": "activitysim_postprocess_usim_households_table_updated",
        "/persons": "activitysim_postprocess_usim_persons_table_updated",
    }
