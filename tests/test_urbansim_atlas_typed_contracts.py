from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pilates.atlas.outputs import AtlasPreprocessOutputs, AtlasRunOutputs
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.generic.records import RecordStore
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs, UrbanSimRunOutputs
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.outputs_base import step_output_mapping
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.urbansim.runner import UrbansimRunner
from pilates.workflows.steps.shared import StepOutputsHolder
from pilates.workflows.steps.urbansim_atlas import (
    _execute_atlas_postprocess_typed,
    _execute_atlas_run_typed,
    _execute_urbansim_postprocess_typed,
    _execute_urbansim_run_typed,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
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


def test_urbansim_run_outputs_publish_only_canonical_datastore_key(
    tmp_path: Path,
) -> None:
    forecast_h5 = tmp_path / "model_data_2030.h5"
    outputs = UrbanSimRunOutputs(
        usim_datastore_h5=forecast_h5,
        raw_outputs={USIM_FORECAST_OUTPUT: forecast_h5},
    )

    assert step_output_mapping(outputs, warn_lossy=False) == {
        USIM_DATASTORE_H5: str(forecast_h5),
    }


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

    def _fake_preprocess(workspace, previous_records=None, final_skims_omx=None):
        seen["workspace"] = workspace
        seen["previous_records"] = previous_records
        seen["final_skims_omx"] = final_skims_omx
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace())

    assert outputs is expected
    assert isinstance(outputs, UrbanSimPreprocessOutputs)
    assert state.sub_stage_progress == "preprocessor"
    assert seen["previous_records"] is None
    assert seen["final_skims_omx"] is None


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

    def _fake_preprocess(workspace, previous_records=None, final_skims_omx=None):
        seen["workspace"] = workspace
        seen["previous_records"] = previous_records
        seen["final_skims_omx"] = final_skims_omx
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace())

    assert outputs is expected
    assert isinstance(outputs, AtlasPreprocessOutputs)
    assert state.sub_stage_progress == "preprocessor"
    assert seen["previous_records"] is None
    assert seen["final_skims_omx"] is None


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

    def _fake_preprocess(workspace, previous_records_arg=None, final_skims_omx=None):
        seen["previous_records"] = previous_records_arg
        seen["final_skims_omx"] = final_skims_omx
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace(), previous_records=previous_records)

    assert outputs is expected
    assert seen["previous_records"] is previous_records
    assert seen["final_skims_omx"] is None


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

    def _fake_preprocess(workspace, previous_records_arg=None, final_skims_omx=None):
        seen["previous_records"] = previous_records_arg
        seen["final_skims_omx"] = final_skims_omx
        return expected

    monkeypatch.setattr(preprocessor, "_preprocess", _fake_preprocess)

    outputs = preprocessor.preprocess(_StubWorkspace(), previous_records=previous_records)

    assert outputs is expected
    assert seen["previous_records"] is previous_records
    assert seen["final_skims_omx"] is None


@pytest.mark.parametrize(
    ("factory", "expected_outputs_fn"),
    [
        (
            make_urbansim_preprocess_step,
            UrbansimPreprocessor.expected_outputs,
        ),
        (
            make_urbansim_run_step,
            UrbansimRunner.expected_outputs,
        ),
        (
            make_urbansim_postprocess_step,
            UrbansimPostprocessor.expected_outputs,
        ),
        (
            make_atlas_preprocess_step,
            AtlasPreprocessor.expected_outputs,
        ),
        (
            make_atlas_run_step,
            AtlasRunner.expected_outputs,
        ),
        (
            make_atlas_postprocess_step,
            AtlasPostprocessor.expected_outputs,
        ),
    ],
)
def test_urbansim_atlas_steps_publish_replay_metadata(
    factory, expected_outputs_fn, tmp_path: Path
) -> None:
    coupler = object()
    holder = StepOutputsHolder()
    step = factory(coupler=coupler, outputs_holder=holder)
    meta = step.__consist_step__

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "123"}},
        ),
        atlas=SimpleNamespace(),
    )
    state = SimpleNamespace(
        year=2023,
        forecast_year=2023,
        is_start_year=lambda: True,
    )
    workspace = SimpleNamespace(
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
    )

    assert meta.load_inputs is None
    assert meta.input_binding == "none"
    assert meta.cache_hydration == "metadata"
    assert meta.output_paths is expected_outputs_fn
    assert meta.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    ) == expected_outputs_fn(settings, state, workspace)


def test_atlas_runner_expected_outputs_only_include_runner_artifacts(tmp_path: Path) -> None:
    workspace = SimpleNamespace(
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "123"}},
        ),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        is_start_year=lambda: False,
    )

    outputs = AtlasRunner.expected_outputs(settings, state, workspace)

    assert outputs == {"atlas_output_dir": str(tmp_path / "atlas" / "output")}


def test_atlas_preprocess_outputs_require_grave_csv_for_non_start_year(
    tmp_path: Path,
) -> None:
    prepared_inputs = {}
    for key, filename in (
        ("atlas_households_csv", "households.csv"),
        ("atlas_blocks_csv", "blocks.csv"),
        ("atlas_persons_csv", "persons.csv"),
        ("atlas_residential_csv", "residential.csv"),
        ("atlas_jobs_csv", "jobs.csv"),
    ):
        path = tmp_path / "atlas-input" / "year2023" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
        prepared_inputs[key] = path

    outputs = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=tmp_path / "atlas-input",
        prepared_inputs=prepared_inputs,
    )

    with pytest.raises(AssertionError, match="atlas_grave_csv"):
        outputs.validate(
            context=ValidationContext(
                step_name="atlas_preprocess",
                state=SimpleNamespace(
                    start_year=2017,
                    year=2023,
                    current_year=2023,
                    is_start_year=lambda: True,
                ),
            )
        )


def test_atlas_runner_retries_container_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _StubState()
    state.current_year = 2030
    state.forecast_year = 2030
    state.full_settings = SimpleNamespace(
        atlas=SimpleNamespace(
            max_retries=2,
            num_processes=1,
            sample_size=0,
            beamac=0,
            mod=2,
            adscen="baseline",
            rebfactor=1,
            taxfactor=1,
            discIncent=0,
        ),
        run=SimpleNamespace(vehicle_ownership_freq=1),
    )
    runner = AtlasRunner("atlas", state)
    atlas_output_dir = tmp_path / "atlas-output"
    prepared_input = tmp_path / "atlas-input" / "jobs.csv"
    prepared_input.parent.mkdir(parents=True, exist_ok=True)
    prepared_input.write_text("household_id\n1\n", encoding="utf-8")
    inputs = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=prepared_input.parent,
        prepared_inputs={"jobs_csv": prepared_input},
    )
    workspace = type(
        "Workspace",
        (),
        {"get_atlas_output_dir": lambda self: str(atlas_output_dir)},
    )()
    attempts = []

    monkeypatch.setattr(
        AtlasRunner,
        "get_model_and_image",
        staticmethod(lambda settings, model_type: ("atlas", "atlas-image")),
    )
    monkeypatch.setattr("pilates.atlas.runner.get_atlas_docker_vols", lambda *_: {})
    monkeypatch.setattr("pilates.atlas.runner.get_atlas_cmd", lambda *_: "atlas-cmd")

    def _fake_run_container(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("Container execution failed for atlas_container")
        atlas_output_dir.mkdir(parents=True, exist_ok=True)
        (atlas_output_dir / "householdv_2030.csv").write_text("hh\n1\n", encoding="utf-8")
        (atlas_output_dir / "vehicles_2030.csv").write_text("veh\n1\n", encoding="utf-8")
        return True

    monkeypatch.setattr(AtlasRunner, "run_container", staticmethod(_fake_run_container))

    outputs = runner._run(inputs, workspace)

    assert len(attempts) == 2
    assert outputs.raw_outputs["householdv_2030"] == atlas_output_dir / "householdv_2030.csv"
    assert outputs.raw_outputs["vehicles_2030"] == atlas_output_dir / "vehicles_2030.csv"


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


def test_atlas_update_h5_vehicle_updates_nearest_year_scoped_households_table(
    tmp_path: Path,
) -> None:
    state = _StubState()
    state.forecast_year = 2029
    state.current_year = 2023
    state.is_start_year = lambda: False
    postprocessor = AtlasPostprocessor("atlas", state)

    h5_path = tmp_path / "model_data_2029.h5"
    original = pd.DataFrame(
        {"cars": [0, 1], "hh_cars": ["none", "one"]},
        index=pd.Index([10, 20], name="household_id"),
    )
    original.to_hdf(h5_path, key="/2024/households", mode="w")

    household_v_csv = tmp_path / "householdv_2029.csv"
    pd.DataFrame(
        {"household_id": [10, 20], "nvehicles": [2, 0]}
    ).to_csv(household_v_csv, index=False)

    updated = postprocessor.atlas_update_h5_vehicle(
        settings=None,
        output_year=2023,
        h5_file_path=str(h5_path),
        household_v_csv_path=str(household_v_csv),
    )

    assert updated == "/2024/households"
    with pd.HDFStore(h5_path, mode="r") as store:
        assert "/2024/households" in store
        households = store["/2024/households"]

    assert households["cars"].tolist() == [2, 0]
    assert households["hh_cars"].tolist() == ["two or more", "none"]


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
