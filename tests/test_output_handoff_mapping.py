from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.workflows.outputs_base import (
    step_output_handoff_mapping,
    step_output_mapping,
)
from pilates.workflows.stages import land_use as land_use_stage
from pilates.workflows.steps import StepOutputsHolder


class _TrackingOutputs:
    def __init__(self, path: Path, key: str) -> None:
        self.path = path
        self.key = key
        self.iter_record_item_calls = 0

    def _iter_record_items(self):
        self.iter_record_item_calls += 1
        yield self.key, self.path, f"record for {self.key}"


class _ResolvedInputs:
    def __init__(self, *, inputs=None, input_keys=None, missing_required=None) -> None:
        self._inputs = inputs
        self._input_keys = input_keys
        self.missing_required = list(missing_required or [])

    def stepref_inputs(self):
        return self._inputs

    def stepref_input_keys(self):
        return self._input_keys


def test_step_output_handoff_mapping_prefers_coupler_artifacts(
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
    artifact = SimpleNamespace(
        key="beam_plans_asim_out",
        path=str(beam_plans),
        container_uri="workspace://beam_plans.parquet",
    )
    coupler = SimpleNamespace(
        get=lambda key, default=None: artifact if key == "beam_plans_asim_out" else default
    )

    mapping = step_output_handoff_mapping(outputs, coupler=coupler)

    assert mapping["beam_plans_asim_out"] is artifact
    assert mapping["households_asim_out"] == str(households)
    assert mapping["persons_asim_out"] == str(persons)


def test_step_output_handoff_mapping_warns_without_coupler(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output_path = tmp_path / "beam_plans.parquet"
    output_path.write_text("x", encoding="utf-8")
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={"beam_plans_asim_out": output_path},
    )

    with caplog.at_level("WARNING"):
        mapping = step_output_handoff_mapping(outputs, coupler=None)

    assert mapping["beam_plans_asim_out"] == str(output_path)
    assert "called without a readable coupler" in caplog.text


def test_step_output_mapping_warns_that_it_is_lossy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output_path = tmp_path / "beam_plans.parquet"
    output_path.write_text("x", encoding="utf-8")
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={"beam_plans_asim_out": output_path},
    )

    with caplog.at_level("WARNING"):
        mapping = step_output_mapping(outputs)

    assert mapping["beam_plans_asim_out"] == str(output_path)
    assert "is lossy and should not be used for runtime workflow handoffs" in caplog.text


def test_land_use_stage_prefers_coupler_artifacts_for_runtime_handoffs(
    monkeypatch, tmp_path: Path
) -> None:
    preprocess_path = tmp_path / "usim-preprocess.h5"
    preprocess_path.write_text("preprocess", encoding="utf-8")
    upstream = _TrackingOutputs(preprocess_path, "run_input")
    resolution_calls = []
    workflow_call_count = {"count": 0}
    artifact = SimpleNamespace(
        key="run_input",
        path=str(preprocess_path),
        container_uri="workspace://usim-preprocess.h5",
    )

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

    coupler = SimpleNamespace(
        get=lambda key, default=None: artifact if key == "run_input" else default
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

    land_use_stage.run_land_use_stage(
        scenario=object(),
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=2035,
        outputs_holder_year=outputs_holder,
    )

    assert resolution_calls[1]["explicit_inputs"]["run_input"] is artifact
