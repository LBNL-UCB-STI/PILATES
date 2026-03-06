from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.workflows.orchestration import _update_coupler_from_outputs
from pilates.workflows.stages import land_use as land_use_stage
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps import shared as shared_steps
from pilates.workflows.steps.shared import _build_required_input_store, _execute_run


class _TrackingOutputs:
    def __init__(self, store: RecordStore) -> None:
        self._store = store
        self.to_record_store_calls = 0

    def to_record_store(self) -> RecordStore:
        self.to_record_store_calls += 1
        return self._store


class _ResolvedInputs:
    def __init__(self, *, inputs=None, input_keys=None, missing_required=None) -> None:
        self._inputs = inputs
        self._input_keys = input_keys
        self.missing_required = list(missing_required or [])

    def stepref_inputs(self):
        return self._inputs

    def stepref_input_keys(self):
        return self._input_keys


def _store_with_record(path: Path, key: str) -> RecordStore:
    path.write_text(key, encoding="utf-8")
    return RecordStore(
        recordList=[
            FileRecord(
                file_path=str(path),
                short_name=key,
                description=f"record for {key}",
            )
        ]
    )


def test_update_coupler_from_outputs_uses_record_store_bridge(
    monkeypatch, tmp_path: Path
) -> None:
    store = _store_with_record(tmp_path / "artifact.txt", "artifact_a")
    outputs = _TrackingOutputs(store)
    captured = {}

    def _fake_bridge(record_store, *, coupler, workspace) -> None:
        captured["record_store"] = record_store
        captured["coupler"] = coupler
        captured["workspace"] = workspace

    monkeypatch.setattr(
        "pilates.workflows.orchestration._update_coupler_from_record_store",
        _fake_bridge,
    )

    coupler = object()
    workspace = object()
    _update_coupler_from_outputs(outputs, coupler=coupler, workspace=workspace)

    assert outputs.to_record_store_calls == 1
    assert captured == {
        "record_store": store,
        "coupler": coupler,
        "workspace": workspace,
    }


def test_build_required_input_store_consumes_typed_outputs_via_to_record_store(
    tmp_path: Path,
) -> None:
    holder = StepOutputsHolder()
    store = _store_with_record(tmp_path / "activitysim-preprocess.txt", "input_a")
    upstream = _TrackingOutputs(store)
    holder.activitysim_preprocess = upstream

    input_store = _build_required_input_store(
        outputs_holder=holder,
        upstream_attr="activitysim_preprocess",
        missing_message="ActivitySim preprocess must complete first",
        context="activitysim_run",
        warn_missing_coupler_inputs=False,
    )

    assert upstream.to_record_store_calls == 1
    assert input_store is store


def test_execute_run_routes_upstream_typed_outputs_through_to_record_store(
    monkeypatch, tmp_path: Path
) -> None:
    holder = StepOutputsHolder()
    store = _store_with_record(tmp_path / "runner-input.txt", "input_a")
    upstream = _TrackingOutputs(store)
    holder.activitysim_preprocess = upstream
    runner_outputs = RecordStore()
    captured = {}

    def _fake_run_runner(runner, input_store, workspace):
        captured["runner"] = runner
        captured["input_store"] = input_store
        captured["workspace"] = workspace
        return runner_outputs

    monkeypatch.setattr(shared_steps, "run_runner", _fake_run_runner)

    runner = object()
    workspace = object()
    result = _execute_run(runner, workspace, holder, context="activitysim_run")

    assert upstream.to_record_store_calls == 1
    assert captured == {
        "runner": runner,
        "input_store": store,
        "workspace": workspace,
    }
    assert result is runner_outputs


def test_land_use_stage_builds_run_inputs_from_upstream_record_store_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    preprocess_store = _store_with_record(tmp_path / "usim-preprocess.h5", "run_input")
    upstream = _TrackingOutputs(preprocess_store)
    resolution_calls = []
    workflow_call_count = {"count": 0}

    def _fake_resolve_step_inputs(
        *, keys, explicit_inputs=None, coupler=None, fallback_inputs=None, required_keys=None
    ):
        resolution_calls.append(
            {
                "keys": list(keys),
                "explicit_inputs": explicit_inputs,
                "fallback_inputs": fallback_inputs,
                "required_keys": list(required_keys or []),
            }
        )
        return _ResolvedInputs(inputs=explicit_inputs, input_keys=list(keys))

    def _fake_run_workflow(**kwargs) -> None:
        workflow_call_count["count"] += 1
        outputs_holder = kwargs["outputs_holder"]
        if workflow_call_count["count"] == 1:
            outputs_holder.urbansim_preprocess = upstream
        else:
            outputs_holder.urbansim_run = SimpleNamespace(usim_datastore_h5=None)
            outputs_holder.urbansim_postprocess = SimpleNamespace(
                usim_datastore_h5=None
            )

    usim_base = tmp_path / "base.h5"
    usim_current = tmp_path / "current.h5"
    usim_base.write_text("base", encoding="utf-8")
    usim_current.write_text("current", encoding="utf-8")

    monkeypatch.setattr(
        land_use_stage,
        "build_urbansim_inputs",
        lambda settings, state, workspace, year: (
            {
                USIM_DATASTORE_BASE_H5: str(usim_base),
                USIM_DATASTORE_CURRENT_H5: str(usim_current),
            },
            {},
        ),
    )
    monkeypatch.setattr(land_use_stage, "log_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        land_use_stage, "merge_model_expected_inputs", lambda *args: args[1]
    )
    monkeypatch.setattr(land_use_stage, "resolve_step_inputs", _fake_resolve_step_inputs)
    monkeypatch.setattr(land_use_stage, "run_workflow", _fake_run_workflow)
    monkeypatch.setattr(
        land_use_stage, "make_urbansim_preprocess_step", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        land_use_stage, "make_urbansim_run_step", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        land_use_stage, "make_urbansim_postprocess_step", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        land_use_stage, "enqueue_archive_copy", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        land_use_stage, "flush_archive_queue", lambda *args, **kwargs: None
    )

    outputs_holder = StepOutputsHolder()
    workspace = SimpleNamespace(get_usim_mutable_data_dir=lambda: str(tmp_path))
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="forecast_{year}.h5")
    )
    state = SimpleNamespace(forecast_year=2035)

    result = land_use_stage.run_land_use_stage(
        scenario=object(),
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=object(),
        year=2035,
        outputs_holder_year=outputs_holder,
    )

    assert upstream.to_record_store_calls == 1
    assert workflow_call_count["count"] == 2
    assert resolution_calls[1]["explicit_inputs"] == preprocess_store.to_mapping()
    assert result[USIM_DATASTORE_CURRENT_H5] == str(usim_current)
