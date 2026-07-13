from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.activitysim.runner import ActivitysimNumbaWarmup
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import FileRecord, RecordStore
from pilates.generic.runner import GenericRunner
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ZARR_SKIMS,
)
from pilates.workflows.step_exec import warm_start_activities


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


def test_warm_start_activities_is_deprecated() -> None:
    with pytest.raises(
        RuntimeError,
        match="ActivitySim warm-start is deprecated and no longer supported",
    ):
        warm_start_activities(SimpleNamespace(), SimpleNamespace(), object())


def test_activitysim_preprocess_public_method_constructs_expected_outputs(
    tmp_path: Path,
) -> None:
    state = _StageTrackingState()
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_mutable_data_dir=lambda: str(tmp_path / "asim-data"),
    )
    previous_records = RecordStore()

    land_use = tmp_path / "asim-data" / "land_use.csv"
    households = tmp_path / "asim-data" / "households.csv"
    persons = tmp_path / "asim-data" / "persons.csv"
    land_use.parent.mkdir(parents=True)
    for path in (land_use, households, persons):
        path.write_text("csv", encoding="utf-8")

    record_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=os.path.relpath(land_use, tmp_path),
                short_name="land_use_asim_in",
                content_hash="land-use-hash",
            ),
            FileRecord(
                file_path=os.path.relpath(households, tmp_path),
                short_name="households_asim_in",
                content_hash="households-hash",
            ),
            FileRecord(
                file_path=os.path.relpath(persons, tmp_path),
                short_name="persons_asim_in",
            ),
        ]
    )

    class _Preprocessor:
        def __init__(self, model_name, workflow_state) -> None:
            self.model_name = model_name
            self.state = workflow_state

        preprocess = ActivitysimPreprocessor.preprocess

        def _preprocess(
            self,
            workspace_arg,
            previous_records_arg,
            *,
            final_skims_omx=None,
        ):
            assert workspace_arg is workspace
            assert previous_records_arg is previous_records
            assert final_skims_omx is None
            return record_store

    outputs = _Preprocessor("activitysim_preprocess", state).preprocess(
        workspace, previous_records
    )

    assert outputs == ActivitySimPreprocessOutputs(
        mutable_data_dir=land_use.parent,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
        input_hashes={
            "land_use_asim_in": "land-use-hash",
            "households_asim_in": "households-hash",
        },
    )
    assert state.sub_stages == ["preprocessor"]


def test_activitysim_numba_warmup_has_no_workflow_output_contract() -> None:
    warmup = ActivitysimNumbaWarmup(
        "activitysim_numba_warmup", SimpleNamespace(full_settings=SimpleNamespace())
    )

    assert warmup.model_name == "activitysim_numba_warmup"
    assert not hasattr(warmup, "expected_outputs")


def test_activitysim_numba_warmup_rejects_wrong_upstream_type(tmp_path: Path) -> None:
    warmup = ActivitysimNumbaWarmup(
        "activitysim_numba_warmup", SimpleNamespace(full_settings=SimpleNamespace())
    )
    with pytest.raises(
        TypeError,
        match="ActivitysimNumbaWarmup.run expects ActivitySimPreprocessOutputs",
    ):
        warmup.run(RecordStore(), _DummyWorkspace(tmp_path, tmp_path / "output"))


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
