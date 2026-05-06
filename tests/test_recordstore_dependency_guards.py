from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
import pilates.generic.model as generic_model
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.workflows.orchestration import _update_coupler_from_outputs
from pilates.workflows.outputs_base import iter_step_output_items, step_output_mapping
from pilates.workflows.stages import land_use as land_use_stage
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps.activitysim import _execute_activitysim_run


class _TrackingOutputs:
    def __init__(self, store: RecordStore) -> None:
        self._store = store
        self.iter_record_item_calls = 0

    def _iter_record_items(self):
        self.iter_record_item_calls += 1
        for record in self._store.all_records():
            yield record.short_name, Path(record.file_path), record.description


class _LegacyOnlyOutputs:
    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def to_record_store(self) -> RecordStore:
        return self._store


class _DuplicateKeyOutputs:
    def __init__(self, first: Path, second: Path) -> None:
        self.first = first
        self.second = second

    def _iter_record_items(self):
        yield "linkstats", self.first, "canonical linkstats"
        yield "linkstats", self.second, "duplicate linkstats"


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


def test_update_coupler_from_outputs_uses_direct_typed_output_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    store = _store_with_record(tmp_path / "artifact.txt", "artifact_a")
    outputs = _TrackingOutputs(store)
    captured = {}

    def _fake_publish(mapping, *, coupler, workspace) -> None:
        captured["mapping"] = mapping
        captured["coupler"] = coupler
        captured["workspace"] = workspace

    monkeypatch.setattr(
        "pilates.workflows.orchestration._update_coupler_from_mapping",
        _fake_publish,
    )

    coupler = object()
    workspace = object()
    _update_coupler_from_outputs(outputs, coupler=coupler, workspace=workspace)

    assert outputs.iter_record_item_calls == 1
    assert captured == {
        "mapping": store.to_mapping(),
        "coupler": coupler,
        "workspace": workspace,
    }


def test_iter_step_output_items_materializes_direct_typed_output_items(
    tmp_path: Path,
) -> None:
    store = _store_with_record(tmp_path / "activitysim-preprocess.txt", "input_a")
    outputs = _TrackingOutputs(store)

    items = iter_step_output_items(outputs)

    assert outputs.iter_record_item_calls == 1
    assert items == (
        (
            "input_a",
            tmp_path / "activitysim-preprocess.txt",
            "record for input_a",
        ),
    )


def test_iter_step_output_items_rejects_outputs_without_iter_record_items(
    tmp_path: Path,
) -> None:
    outputs = _LegacyOnlyOutputs(
        _store_with_record(tmp_path / "legacy-input.txt", "input_a")
    )

    with pytest.raises(TypeError, match="_iter_record_items"):
        iter_step_output_items(outputs)


def test_execute_activitysim_run_forwards_typed_preprocess_outputs(
    tmp_path: Path,
) -> None:
    holder = StepOutputsHolder()
    land_use = tmp_path / "land_use.csv"
    households = tmp_path / "households.csv"
    persons = tmp_path / "persons.csv"
    for path in (land_use, households, persons):
        path.write_text(path.stem, encoding="utf-8")
    upstream = ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )
    holder.activitysim_preprocess = upstream
    runner_outputs = object()
    captured = {}

    class _Runner:
        def run(self, input_outputs, workspace, *, extra_inputs=None):
            captured["runner"] = self
            captured["input_outputs"] = input_outputs
            captured["extra_inputs"] = extra_inputs
            captured["workspace"] = workspace
            return runner_outputs

    runner = _Runner()
    workspace = object()
    result = _execute_activitysim_run(runner, workspace, holder)

    assert captured["runner"] is runner
    assert captured["workspace"] is workspace
    assert captured["input_outputs"] is upstream
    assert captured["extra_inputs"] is None
    assert result is runner_outputs


def test_land_use_stage_builds_run_inputs_from_upstream_record_store_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    preprocess_store = _store_with_record(tmp_path / "usim-preprocess.h5", "run_input")
    upstream = _TrackingOutputs(preprocess_store)
    resolution_calls = []
    workflow_call_count = {"count": 0}

    original_build_binding_plan = land_use_stage.build_binding_plan

    def _capturing_build_binding_plan(**kwargs):
        resolution_calls.append(
            {
                "step_name": kwargs["step_name"],
                "explicit_inputs": kwargs.get("explicit_inputs"),
                "fallback_inputs": kwargs.get("fallback_inputs"),
                "required_keys": list(kwargs.get("required_keys") or []),
            }
        )
        return original_build_binding_plan(**kwargs)

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
        lambda settings, state, workspace, year, **_kwargs: (
            {
                USIM_DATASTORE_BASE_H5: str(usim_base),
                USIM_DATASTORE_CURRENT_H5: str(usim_current),
            },
            {},
        ),
    )
    monkeypatch.setattr(land_use_stage, "log_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        land_use_stage, "build_binding_plan", _capturing_build_binding_plan
    )
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
    monkeypatch.setattr(land_use_stage, "archive_copy_now", lambda **kwargs: None)
    monkeypatch.setattr(
        land_use_stage, "flush_archive_queue", lambda *args, **kwargs: None
    )

    outputs_holder = StepOutputsHolder()
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_usim_mutable_data_dir=lambda: str(tmp_path),
    )
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="forecast_{year}.h5")
    )
    state = SimpleNamespace(forecast_year=2035)
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=SimpleNamespace(
            profile=SimpleNamespace(),
            step_surface=lambda *_args, **_kwargs: None,
        ),
    )

    result = land_use_stage.run_land_use_stage(
        scenario=object(),
        coupler=object(),
        year=2035,
        outputs_holder_year=outputs_holder,
        context=context,
    )

    assert upstream.iter_record_item_calls == 1
    assert workflow_call_count["count"] == 2
    run_binding_call = next(
        call for call in resolution_calls if call["step_name"] == "urbansim_run"
    )
    assert run_binding_call["explicit_inputs"] == {
        **preprocess_store.to_mapping(),
        USIM_DATASTORE_BASE_H5: str(usim_base),
        USIM_DATASTORE_CURRENT_H5: str(usim_current),
    }
    assert result[USIM_DATASTORE_CURRENT_H5] == str(usim_current)


def test_step_output_mapping_matches_real_output_record_store_mapping(
    tmp_path: Path,
) -> None:
    beam_plans = tmp_path / "beam_plans.parquet"
    households = tmp_path / "households.parquet"
    persons = tmp_path / "persons.parquet"
    for path in (beam_plans, households, persons):
        path.write_text(path.name, encoding="utf-8")

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={
            "beam_plans_asim_out": beam_plans,
            "households_asim_out": households,
            "persons_asim_out": persons,
        },
    )

    assert (
        step_output_mapping(outputs, warn_lossy=False)
        == outputs.to_record_store().to_mapping()
    )


def test_step_output_mapping_keeps_first_duplicate_key(tmp_path: Path, caplog) -> None:
    first = tmp_path / "linkstats-first.parquet"
    second = tmp_path / "linkstats-second.parquet"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    with caplog.at_level("WARNING"):
        mapping = step_output_mapping(
            _DuplicateKeyOutputs(first, second),
            warn_lossy=False,
        )

    assert mapping == {"linkstats": str(first)}
    assert "Duplicate typed-output artifact key 'linkstats'" in caplog.text


def test_generic_model_boundary_drops_decorator_compat_surface() -> None:
    assert not hasattr(generic_model, "provenance_logging")
    assert not hasattr(generic_model.Model, "update_state")
