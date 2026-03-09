from types import SimpleNamespace

from pilates.atlas.outputs import AtlasPreprocessOutputs
from pilates.workflows import steps
from pilates.workflows.steps import urbansim_atlas as steps_urbansim_atlas


def test_atlas_preprocess_output_logger_preserves_static_input_metadata(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_make_typed_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_urbansim_atlas,
        "_make_typed_step_function",
        _fake_make_typed_step_function,
    )

    steps.make_atlas_preprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, path, meta))

    monkeypatch.setattr(steps_urbansim_atlas, "log_output_only", _log_output_only)

    static_csv = tmp_path / "adopt" / "zev_mandate" / "new_vehicles_biannual_values_2021.csv"
    static_csv.parent.mkdir(parents=True, exist_ok=True)
    static_csv.write_text("year,value\n2021,1\n", encoding="utf-8")

    outputs = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=tmp_path,
        prepared_inputs={
            "adopt/zev_mandate/new_vehicles_biannual_values_2021": static_csv,
        },
        prepared_input_meta={
            "adopt/zev_mandate/new_vehicles_biannual_values_2021": {
                "atlas_static_input": True,
                "atlas_relpath": "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",
                "atlas_source_origin": "fallback",
                "atlas_input_group": "adopt",
                "atlas_scenario": "zev_mandate",
                "atlas_input_year": 2021,
                "profile_file_schema": True,
            }
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(atlas=SimpleNamespace(scenario="zev_mandate")),
        state=SimpleNamespace(forecast_year=2023),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert len(calls) == 1
    key, _path, meta = calls[0]
    assert key == "adopt/zev_mandate/new_vehicles_biannual_values_2021"
    assert meta["atlas_static_input"] is True
    assert meta["atlas_relpath"] == "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv"
    assert meta["atlas_scenario"] == "zev_mandate"
    assert meta["atlas_input_group"] == "adopt"
    assert meta["atlas_input_year"] == 2021
    assert meta["facet"]["artifact_family"] == "atlas_preprocess_output"
