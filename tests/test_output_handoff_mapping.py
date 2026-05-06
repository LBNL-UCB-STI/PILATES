from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import consist
import pytest

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.runtime.context import WorkflowRuntimeContext
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
        get=lambda key, default=None: (
            artifact if key == "beam_plans_asim_out" else default
        )
    )

    mapping = step_output_handoff_mapping(outputs, coupler=coupler)

    assert mapping["beam_plans_asim_out"] is artifact
    assert mapping["households_asim_out"] == str(households)
    assert mapping["persons_asim_out"] == str(persons)


def test_step_output_handoff_mapping_uses_namespace_aware_coupler_lookup(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "usim-preprocess.h5"
    output_path.write_text("x", encoding="utf-8")
    outputs = _TrackingOutputs(output_path, "usim_datastore_h5")
    artifact = SimpleNamespace(
        key="urbansim/usim_datastore_h5",
        path=str(output_path),
        container_uri="workspace://usim-preprocess.h5",
    )

    class _Coupler:
        def get(self, key, default=None):
            return default

        def view(self, namespace):
            class _View:
                def get(self, key, default=None):
                    if namespace == "urbansim" and key == "usim_datastore_h5":
                        return artifact
                    return default

            return _View()

    mapping = step_output_handoff_mapping(outputs, coupler=_Coupler())

    assert mapping["usim_datastore_h5"] is artifact


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


def test_step_output_handoff_mapping_ignores_noop_artifact_placeholders(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output_path = tmp_path / "beam_plans.parquet"
    output_path.write_text("x", encoding="utf-8")
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={"beam_plans_asim_out": output_path},
    )
    noop_artifact = consist.NoopArtifact(
        key="beam_plans_asim_out",
        path=output_path,
        container_uri=f"workspace://{output_path.name}",
    )
    coupler = SimpleNamespace(
        get=lambda key, default=None: (
            noop_artifact if key == "beam_plans_asim_out" else default
        )
    )

    with caplog.at_level("WARNING"):
        mapping = step_output_handoff_mapping(outputs, coupler=coupler)

    assert mapping["beam_plans_asim_out"] == str(output_path)
    assert (
        "Ignoring NoopArtifact placeholder for coupler key 'beam_plans_asim_out'"
        in caplog.text
    )


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
    assert (
        "is lossy and should not be used for runtime workflow handoffs" in caplog.text
    )


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
    base_artifact = SimpleNamespace(
        key=USIM_DATASTORE_BASE_H5,
        path=str(tmp_path / "base.h5"),
        container_uri="workspace://base.h5",
    )
    current_artifact = SimpleNamespace(
        key=USIM_DATASTORE_CURRENT_H5,
        path=str(tmp_path / "current.h5"),
        container_uri="workspace://current.h5",
    )

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

    class _Coupler:
        def get(self, key, default=None):
            values = {
                "run_input": artifact,
                USIM_DATASTORE_BASE_H5: base_artifact,
                USIM_DATASTORE_CURRENT_H5: current_artifact,
            }
            return values.get(key, default)

        def view(self, namespace):
            class _View:
                def get(self, key, default=None):
                    if namespace != "urbansim":
                        return default
                    if key == USIM_DATASTORE_BASE_H5:
                        return base_artifact
                    if key == USIM_DATASTORE_CURRENT_H5:
                        return current_artifact
                    return default

            return _View()

    coupler = _Coupler()
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

    land_use_stage.run_land_use_stage(
        scenario=object(),
        coupler=coupler,
        year=2035,
        outputs_holder_year=outputs_holder,
        context=context,
    )

    run_binding_call = next(
        call for call in resolution_calls if call["step_name"] == "urbansim_run"
    )
    assert run_binding_call["explicit_inputs"]["run_input"] is artifact
    assert run_binding_call["explicit_inputs"][USIM_DATASTORE_BASE_H5] is base_artifact
    assert (
        run_binding_call["explicit_inputs"][USIM_DATASTORE_CURRENT_H5]
        is current_artifact
    )


def test_land_use_stage_ignores_noop_datastore_placeholders_for_runtime_handoffs(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    preprocess_path = tmp_path / "usim-preprocess.h5"
    preprocess_path.write_text("preprocess", encoding="utf-8")
    upstream = _TrackingOutputs(preprocess_path, "run_input")
    resolution_calls = []
    workflow_call_count = {"count": 0}
    noop_base = consist.NoopArtifact(
        key=USIM_DATASTORE_BASE_H5,
        path=tmp_path / "noop-base.h5",
        container_uri="workspace://noop-base.h5",
    )
    noop_current = consist.NoopArtifact(
        key=USIM_DATASTORE_CURRENT_H5,
        path=tmp_path / "noop-current.h5",
        container_uri="workspace://noop-current.h5",
    )

    original_build_binding_plan = land_use_stage.build_binding_plan

    def _capturing_build_binding_plan(**kwargs):
        resolution_calls.append(
            {
                "step_name": kwargs["step_name"],
                "explicit_inputs": kwargs.get("explicit_inputs"),
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

    class _Coupler:
        def get(self, key, default=None):
            values = {
                USIM_DATASTORE_BASE_H5: noop_base,
                USIM_DATASTORE_CURRENT_H5: noop_current,
            }
            return values.get(key, default)

        def view(self, namespace):
            class _View:
                def get(self, key, default=None):
                    if namespace != "urbansim":
                        return default
                    if key == USIM_DATASTORE_BASE_H5:
                        return noop_base
                    if key == USIM_DATASTORE_CURRENT_H5:
                        return noop_current
                    return default

            return _View()

    with caplog.at_level("DEBUG"):
        context = WorkflowRuntimeContext.from_parts(
            settings=SimpleNamespace(
                urbansim=SimpleNamespace(output_file_template="forecast_{year}.h5")
            ),
            state=SimpleNamespace(forecast_year=2035),
            workspace=SimpleNamespace(
                full_path=str(tmp_path),
                get_usim_mutable_data_dir=lambda: str(tmp_path),
            ),
            surface=SimpleNamespace(
                profile=SimpleNamespace(),
                step_surface=lambda *_args, **_kwargs: None,
            ),
        )
        land_use_stage.run_land_use_stage(
            scenario=object(),
            coupler=_Coupler(),
            year=2035,
            outputs_holder_year=StepOutputsHolder(),
            context=context,
        )

    run_binding_call = next(
        call for call in resolution_calls if call["step_name"] == "urbansim_run"
    )
    assert run_binding_call["explicit_inputs"][USIM_DATASTORE_BASE_H5] == str(usim_base)
    assert run_binding_call["explicit_inputs"][USIM_DATASTORE_CURRENT_H5] == str(
        usim_current
    )
    assert (
        "Ignoring NoopArtifact placeholder for coupler key 'usim_datastore_base_h5'"
        in caplog.text
    )
    assert (
        "Ignoring NoopArtifact placeholder for coupler key 'usim_datastore_h5'"
        in caplog.text
    )
    assert (
        "[land_use] Runtime handoff for usim_datastore_base_h5 resolved via missing"
        in caplog.text
    )
