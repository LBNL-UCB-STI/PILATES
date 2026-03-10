from pathlib import Path

import pytest

from pilates.atlas.outputs import AtlasPreprocessOutputs, AtlasRunOutputs
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.generic.records import RecordStore
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs, UrbanSimRunOutputs
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.urbansim.runner import UrbansimRunner
from pilates.workflows.steps.shared import StepOutputsHolder
from pilates.workflows.steps.urbansim_atlas import (
    _execute_atlas_postprocess_typed,
    _execute_atlas_run_typed,
    _execute_urbansim_postprocess_typed,
    _execute_urbansim_run_typed,
)


class _StubState:
    def __init__(self) -> None:
        self.sub_stage_progress = None

    def set_sub_stage_progress(self, progress: str) -> None:
        self.sub_stage_progress = progress


class _StubWorkspace:
    pass


def test_urbansim_run_executor_rejects_wrong_upstream_typed_output(tmp_path: Path) -> None:
    state = _StubState()
    runner = UrbansimRunner("urbansim", state)
    holder = StepOutputsHolder(
        urbansim_preprocess=AtlasPreprocessOutputs(
            atlas_mutable_input_dir=tmp_path / "atlas-input",
            prepared_inputs={},
        )
    )

    with pytest.raises(TypeError, match="UrbanSimPreprocessOutputs"):
        _execute_urbansim_run_typed(runner, _StubWorkspace(), holder)

    assert state.sub_stage_progress is None


def test_urbansim_postprocess_executor_rejects_wrong_upstream_typed_output(
    tmp_path: Path,
) -> None:
    state = _StubState()
    postprocessor = UrbansimPostprocessor("urbansim", state)
    holder = StepOutputsHolder(
        urbansim_run=AtlasRunOutputs(
            atlas_output_dir=tmp_path / "atlas-output",
            raw_outputs={},
        )
    )

    with pytest.raises(TypeError, match="UrbanSimRunOutputs"):
        _execute_urbansim_postprocess_typed(postprocessor, _StubWorkspace(), holder)

    assert state.sub_stage_progress is None


def test_atlas_run_executor_rejects_wrong_upstream_typed_output(tmp_path: Path) -> None:
    state = _StubState()
    runner = AtlasRunner("atlas", state)
    holder = StepOutputsHolder(
        atlas_preprocess=UrbanSimPreprocessOutputs(
            usim_mutable_data_dir=tmp_path / "usim-input",
            prepared_inputs={},
        )
    )

    with pytest.raises(TypeError, match="AtlasPreprocessOutputs"):
        _execute_atlas_run_typed(runner, _StubWorkspace(), holder)

    assert state.sub_stage_progress is None


def test_atlas_postprocess_executor_rejects_wrong_upstream_typed_output(
    tmp_path: Path,
) -> None:
    state = _StubState()
    postprocessor = AtlasPostprocessor("atlas", state)
    holder = StepOutputsHolder(
        atlas_run=UrbanSimRunOutputs(
            usim_datastore_h5=tmp_path / "model_data_2030.h5",
            raw_outputs={},
        )
    )

    with pytest.raises(TypeError, match="AtlasRunOutputs"):
        _execute_atlas_postprocess_typed(postprocessor, _StubWorkspace(), holder)

    assert state.sub_stage_progress is None


def test_urbansim_preprocess_returns_typed_outputs_and_sets_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    preprocessor = UrbansimPreprocessor("urbansim", state)
    expected = UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=tmp_path / "usim-input",
        prepared_inputs={"geoid_to_zone": tmp_path / "usim-input" / "geoid_to_zone.csv"},
    )
    seen = {}

    def _fake_preprocess(workspace, previous_records=None):
        seen["workspace"] = workspace
        seen["previous_records"] = previous_records
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace())

    assert outputs is expected
    assert isinstance(outputs, UrbanSimPreprocessOutputs)
    assert state.sub_stage_progress == "preprocessor"
    assert seen["previous_records"] is None


def test_atlas_preprocess_returns_typed_outputs_and_sets_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    preprocessor = AtlasPreprocessor("atlas", state)
    expected = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=tmp_path / "atlas-input",
        prepared_inputs={"jobs_csv": tmp_path / "atlas-input" / "jobs.csv"},
    )
    seen = {}

    def _fake_preprocess(workspace, previous_records=None):
        seen["workspace"] = workspace
        seen["previous_records"] = previous_records
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace())

    assert outputs is expected
    assert isinstance(outputs, AtlasPreprocessOutputs)
    assert state.sub_stage_progress == "preprocessor"
    assert seen["previous_records"] is None


def test_urbansim_preprocess_accepts_previous_records_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    preprocessor = UrbansimPreprocessor("urbansim", state)
    previous_records = RecordStore()
    expected = UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=tmp_path / "usim-input",
        prepared_inputs={},
    )
    seen = {}

    def _fake_preprocess(workspace, previous_records_arg=None):
        seen["previous_records"] = previous_records_arg
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace(), previous_records=previous_records)

    assert outputs is expected
    assert seen["previous_records"] is previous_records


def test_atlas_preprocess_accepts_previous_records_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    preprocessor = AtlasPreprocessor("atlas", state)
    previous_records = RecordStore()
    expected = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=tmp_path / "atlas-input",
        prepared_inputs={},
    )
    seen = {}

    def _fake_preprocess(workspace, previous_records_arg=None):
        seen["previous_records"] = previous_records_arg
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace(), previous_records=previous_records)

    assert outputs is expected
    assert seen["previous_records"] is previous_records


def test_atlas_postprocess_fails_closed_without_current_year_run_outputs(tmp_path: Path) -> None:
    state = _StubState()
    state.full_settings = type(
        "Settings",
        (),
        {
            "urbansim": type(
                "UrbansimCfg",
                (),
                {
                    "output_file_template": "model_data_{year}.h5",
                    "input_file_template": "model_data_{region_id}.h5",
                    "region_mappings": {"region_to_region_id": {"test": "000"}},
                },
            )(),
            "run": type("RunCfg", (), {"region": "test"})(),
        },
    )()
    state.forecast_year = 2030
    state.current_year = 2030
    postprocessor = AtlasPostprocessor("atlas", state)
    workspace = type(
        "Workspace",
        (),
        {
            "get_usim_mutable_data_dir": lambda self: str(tmp_path / "usim"),
            "get_atlas_output_dir": lambda self: str(tmp_path / "atlas-output"),
            "get_atlas_mutable_input_dir": lambda self: str(tmp_path / "atlas-input"),
        },
    )()

    with pytest.raises(RuntimeError, match="requires the current-year householdv CSV"):
        postprocessor.postprocess(
            AtlasRunOutputs(
                atlas_output_dir=tmp_path / "atlas-output",
                raw_outputs={},
            ),
            workspace,
        )


def test_urbansim_postprocess_uses_handed_off_run_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    state.full_settings = type(
        "Settings",
        (),
        {
            "run": type(
                "RunCfg",
                (),
                {"models": type("ModelsCfg", (), {"land_use": "urbansim"})()},
            )()
        },
    )()
    state.forecast_year = 2030
    state.current_year = 2030
    postprocessor = UrbansimPostprocessor("urbansim", state)
    workspace = type(
        "Workspace",
        (),
        {"get_usim_mutable_data_dir": lambda self: str(tmp_path / "usim")},
    )()
    output_h5 = tmp_path / "usim" / "model_data_2030.h5"
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_h5.write_text("h5", encoding="utf-8")
    captured = {}

    def _fake_create_next_iter_usim_data(
        settings,
        forecast_year,
        mutable_data_dir,
        *,
        output_store_path=None,
    ):
        captured["output_store_path"] = output_store_path
        return {f"usim_input_merged_{forecast_year}": tmp_path / "merged.h5"}

    monkeypatch.setattr(
        "pilates.urbansim.postprocessor.create_next_iter_usim_data",
        _fake_create_next_iter_usim_data,
    )

    outputs = postprocessor.postprocess(
        UrbanSimRunOutputs(
            usim_datastore_h5=output_h5,
            raw_outputs={},
        ),
        workspace,
    )

    assert captured["output_store_path"] == str(output_h5)
    assert outputs.usim_datastore_h5 == tmp_path / "merged.h5"
