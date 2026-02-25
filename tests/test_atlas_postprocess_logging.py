from types import SimpleNamespace

from pilates.atlas.outputs import AtlasPostprocessOutputs
from pilates.workflows import steps
from pilates.workflows.steps import urbansim_atlas as steps_urbansim_atlas


def test_atlas_postprocess_logs_only_canonical_usim_h5_output(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
    )

    steps.make_atlas_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    output_only_keys = []
    set_output_keys = []

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
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert output_only_keys == ["atlas_vehicles2_output"]
    assert set_output_keys == ["usim_datastore_h5"]


def test_atlas_postprocess_logs_usim_h5_as_input(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
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
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
    )
    state = SimpleNamespace(forecast_year=2023)

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
