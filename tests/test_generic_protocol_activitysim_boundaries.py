from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.generic.model_factory import ModelFactory
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from pilates.generic.runner import GenericRunner
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ZARR_SKIMS,
)
from pilates.workflows.step_exec import warm_start_activities
from pilates.workflows.steps import activitysim as activitysim_steps


class _StageTrackingState:
    def __init__(self) -> None:
        self.sub_stages = []

    def set_sub_stage_progress(self, progress: str) -> None:
        self.sub_stages.append(progress)


class _DummyCoupler:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value) -> None:
        return None


class _DummyWorkspace:
    def __init__(self, root: Path, asim_output_dir: Path) -> None:
        self.full_path = str(root)
        self._asim_output_dir = asim_output_dir

    def get_asim_output_dir(self) -> str:
        return str(self._asim_output_dir)


def test_generic_public_methods_now_exchange_model_specific_payloads() -> None:
    state = _StageTrackingState()
    workspace = object()
    previous_records = {"generic_input": "relative/input.txt"}
    captured = {}

    class _Preprocessor(GenericPreprocessor):
        def copy_data_to_mutable_location(self, settings, output_dir):
            raise AssertionError("not used by this contract test")

        def _preprocess(self, workspace_arg, previous_records_arg):
            captured["preprocess"] = (workspace_arg, previous_records_arg)
            return previous_records_arg

    class _Runner(GenericRunner):
        def _run(self, store_arg, workspace_arg):
            captured["run"] = (store_arg, workspace_arg)
            return store_arg

    class _Postprocessor(GenericPostprocessor):
        def _postprocess(self, raw_outputs_arg, workspace_arg, model_run_hash=None):
            captured["postprocess"] = (
                raw_outputs_arg,
                workspace_arg,
                model_run_hash,
            )
            return raw_outputs_arg

    preprocessor = _Preprocessor("generic_preprocess", state)
    runner = _Runner("generic_run", state)
    postprocessor = _Postprocessor("generic_postprocess", state)

    preprocessed = preprocessor.preprocess(workspace, previous_records)
    raw_outputs = runner.run(preprocessed, workspace)
    postprocessed = postprocessor.postprocess(
        raw_outputs, workspace, model_run_hash="run-hash"
    )

    assert preprocessed is previous_records
    assert raw_outputs is previous_records
    assert postprocessed is previous_records
    assert captured == {
        "preprocess": (workspace, previous_records),
        "run": (previous_records, workspace),
        "postprocess": (previous_records, workspace, "run-hash"),
    }
    assert state.sub_stages == ["preprocessor", "runner", "postprocessor"]


def test_warm_start_activities_forwards_model_specific_payload(monkeypatch) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = object()
    preprocess_outputs = object()
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace_arg):
            captured["preprocess_workspace"] = workspace_arg
            return preprocess_outputs

    class _Runner:
        def run(self, input_data_arg, workspace_arg):
            captured["runner_input_data"] = input_data_arg
            captured["runner_workspace"] = workspace_arg
            return object()

    def _capture_update(settings_arg, state_arg, workspace_arg, model_run_hash=None):
        captured["update_call"] = (
            settings_arg,
            state_arg,
            workspace_arg,
            model_run_hash,
        )

    monkeypatch.setattr(
        GenericRunner,
        "get_model_and_image",
        staticmethod(lambda settings, model_type: ("activitysim", "fake-image")),
    )
    monkeypatch.setattr(
        ModelFactory,
        "get_preprocessor",
        lambda self, model_name, workflow_state: _Preprocessor(),
    )
    monkeypatch.setattr(
        ModelFactory,
        "get_runner",
        lambda self, model_name, workflow_state: _Runner(),
    )
    monkeypatch.setattr(
        "pilates.workflows.step_exec.asim_post.update_usim_inputs_after_warm_start",
        _capture_update,
    )

    warm_start_activities(settings, state, workspace)

    assert captured["preprocess_workspace"] is workspace
    assert captured["runner_input_data"] is preprocess_outputs
    assert captured["runner_workspace"] is workspace
    assert captured["update_call"] == (settings, state, workspace, None)


def test_activitysim_compile_step_does_not_fallback_to_workspace_cache_path(
    monkeypatch, tmp_path: Path
) -> None:
    workspace_output_dir = tmp_path / "workspace-output"
    workspace_cache_zarr = workspace_output_dir / "cache" / "skims.zarr"
    workspace_cache_zarr.parent.mkdir(parents=True)
    workspace_cache_zarr.write_text("workspace-cache", encoding="utf-8")

    omx_skims = tmp_path / "asim-data" / "skims.omx"
    omx_skims.parent.mkdir(parents=True)
    omx_skims.write_text("omx", encoding="utf-8")

    expected_zarr_path = tmp_path / "expected-output" / "cache" / "skims.zarr"
    workspace = _DummyWorkspace(tmp_path, workspace_output_dir)
    outputs_holder = SimpleNamespace(
        activitysim_preprocess=ActivitySimPreprocessOutputs(
            mutable_data_dir=omx_skims.parent,
            land_use_table=omx_skims.parent / "land_use.csv",
            households_table=omx_skims.parent / "households.csv",
            persons_table=omx_skims.parent / "persons.csv",
            omx_skims=omx_skims,
        )
    )
    for path in (
        outputs_holder.activitysim_preprocess.land_use_table,
        outputs_holder.activitysim_preprocess.households_table,
        outputs_holder.activitysim_preprocess.persons_table,
    ):
        path.write_text("csv", encoding="utf-8")
    settings = SimpleNamespace(
        activitysim=SimpleNamespace(
            file_format="parquet",
            persist_sharrow_cache=False,
        )
    )
    state = SimpleNamespace(forecast_year=2020, iteration=0)
    published = []
    captured = {}

    class _CompileRunner:
        def run(self, inputs, workspace_arg):
            captured["runner_workspace"] = workspace_arg
            captured["runner_inputs"] = inputs
            return ActivitySimCompileOutputs()

    monkeypatch.setattr(
        activitysim_steps.ModelFactory,
        "get_runner",
        lambda self, model_name, workflow_state: _CompileRunner(),
    )
    monkeypatch.setattr(activitysim_steps.cr, "log_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        activitysim_steps,
        "log_and_set_output",
        lambda **kwargs: published.append(kwargs),
    )

    step_fn = activitysim_steps.make_activitysim_compile_step(
        coupler=_DummyCoupler(),
        outputs_holder=outputs_holder,
    )
    step_fn(
        settings=settings,
        state=state,
        workspace=workspace,
        expected_outputs={ZARR_SKIMS: str(expected_zarr_path)},
    )

    assert captured["runner_workspace"] is workspace
    assert captured["runner_inputs"] is outputs_holder.activitysim_preprocess
    assert captured["runner_inputs"].omx_skims == omx_skims
    assert published == []


def test_activitysim_compile_step_rejects_wrong_upstream_type(tmp_path: Path) -> None:
    workspace_output_dir = tmp_path / "workspace-output"
    workspace_output_dir.mkdir(parents=True)

    step_fn = activitysim_steps.make_activitysim_compile_step(
        coupler=_DummyCoupler(),
        outputs_holder=SimpleNamespace(activitysim_preprocess=RecordStore()),
    )

    with pytest.raises(
        TypeError,
        match=(
            "activitysim_compile requires ActivitySimPreprocessOutputs "
            "from activitysim_preprocess"
        ),
    ):
        step_fn(
            settings=SimpleNamespace(
                activitysim=SimpleNamespace(
                    file_format="parquet",
                    persist_sharrow_cache=False,
                )
            ),
            state=SimpleNamespace(forecast_year=2020, iteration=0),
            workspace=_DummyWorkspace(tmp_path, workspace_output_dir),
            expected_outputs={ZARR_SKIMS: str(tmp_path / "expected" / "skims.zarr")},
        )


def test_activitysim_postprocess_forwards_typed_run_outputs(
    tmp_path: Path,
) -> None:
    raw_output_path = tmp_path / "output" / "households.parquet"
    raw_output_path.parent.mkdir(parents=True)
    raw_output_path.write_text("raw", encoding="utf-8")

    households_input = tmp_path / "input" / "households.csv"
    households_input.parent.mkdir(parents=True)
    households_input.write_text("households", encoding="utf-8")

    zarr_skims = tmp_path / "output" / "cache" / "skims.zarr"
    zarr_skims.parent.mkdir(parents=True)
    zarr_skims.write_text("zarr", encoding="utf-8")

    run_outputs = ActivitySimRunOutputs(
        output_dir=tmp_path / "output",
        raw_outputs={"households_asim_out_temp": raw_output_path},
        raw_output_hashes={"households_asim_out_temp": "raw-hash"},
        source_input_paths={
            ASIM_HOUSEHOLDS_IN: households_input,
            ZARR_SKIMS: zarr_skims,
        },
        source_input_hashes={
            ASIM_HOUSEHOLDS_IN: "households-hash",
            ZARR_SKIMS: "zarr-hash",
        },
    )
    state = _StageTrackingState()
    workspace = _DummyWorkspace(tmp_path, tmp_path / "output")
    captured = {}

    class _InspectingPostprocessor(ActivitysimPostprocessor):
        def _postprocess(self, raw_outputs, workspace_arg, model_run_hash=None):
            captured["raw_outputs"] = raw_outputs
            captured["workspace"] = workspace_arg
            captured["model_run_hash"] = model_run_hash
            return ActivitySimPostprocessOutputs(
                usim_datastore_h5=None,
                asim_output_dir=tmp_path / "output",
            )

    postprocessor = _InspectingPostprocessor("activitysim_postprocess", state)
    outputs = postprocessor.postprocess(
        run_outputs,
        workspace,
        model_run_hash="run-hash",
    )

    assert captured["workspace"] is workspace
    assert captured["model_run_hash"] == "run-hash"
    assert captured["raw_outputs"] is run_outputs
    assert captured["raw_outputs"].raw_output_hashes == {
        "households_asim_out_temp": "raw-hash",
    }
    assert captured["raw_outputs"].source_input_paths == {
        ASIM_HOUSEHOLDS_IN: households_input,
        ZARR_SKIMS: zarr_skims,
    }
    assert captured["raw_outputs"].source_input_hashes == {
        ASIM_HOUSEHOLDS_IN: "households-hash",
        ZARR_SKIMS: "zarr-hash",
    }
    assert outputs.asim_output_dir == tmp_path / "output"
    assert state.sub_stages == ["postprocessor"]
