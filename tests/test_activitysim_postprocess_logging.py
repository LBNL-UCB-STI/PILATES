from types import SimpleNamespace

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.workflows import steps
from pilates.workflows.steps import activitysim as steps_activitysim


def test_activitysim_postprocess_logs_content_hash(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_activitysim,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
    )

    steps.make_activitysim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
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
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_activitysim,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
    )

    steps.make_activitysim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
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

    input_logger(settings=settings, state=state, workspace=workspace, holder=None)

    keys = {key for key, _path in calls}
    assert "households_asim_in" in keys
    assert "persons_asim_in" in keys
    assert "land_use_asim_in" in keys
    assert "omx_skims" in keys
    assert "zarr_skims" in keys
    assert "usim_datastore_h5" in keys
    assert "usim_forecast_output" in keys


def test_activitysim_preprocess_logs_selected_usim_h5_tables(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_activitysim,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
    )

    steps.make_activitysim_preprocess_step(
        coupler=SimpleNamespace(get=lambda *args, **kwargs: None, set=lambda *args, **kwargs: None),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
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

    h5_path = tmp_path / "model_data.h5"
    h5_path.write_text("x")

    input_logger(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2023, start_year=2017),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
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
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_activitysim,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
    )

    steps.make_activitysim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
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
