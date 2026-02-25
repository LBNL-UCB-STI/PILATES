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

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps_activitysim, "log_output_only", _log_output_only)

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
