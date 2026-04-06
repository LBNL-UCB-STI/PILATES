from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Tuple

from pilates.workflows.outputs_base import StepOutputsBase
from pilates.workflows.steps.shared import (
    StandardStepSpec,
    StepOutputsHolder,
    build_standard_step,
)


@dataclass
class _BuilderOutputs(StepOutputsBase):
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_file",)
    output_file: Path


def test_build_standard_step_supports_non_logged_wrapper_and_explicit_step_name(
    tmp_path: Path,
    caplog,
) -> None:
    output_path = tmp_path / "beam_full_skims.omx"
    output_path.write_text("ok", encoding="utf-8")

    holder = StepOutputsHolder()
    step_logger = logging.getLogger("tests.standard_step_builder.non_logged")
    step = build_standard_step(
        coupler=SimpleNamespace(),
        outputs_holder=holder,
        spec=StandardStepSpec(
            step_name="beam_full_skim",
            model_name="beam_full",
            phase="skim",
            outputs_class=_BuilderOutputs,
            component_getter=lambda _factory, _state: object(),
            component_executor=lambda _component, _workspace, _holder, **_kwargs: (
                _BuilderOutputs(output_file=output_path)
            ),
            use_logged_wrapper=False,
            step_logger=step_logger,
        ),
    )

    with caplog.at_level(logging.DEBUG):
        step(
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=SimpleNamespace(full_path=str(tmp_path)),
        )

    assert step.__consist_step__.model == "beam_full_skim"
    assert holder.beam_full_skim == _BuilderOutputs(output_file=output_path)
    assert "Starting beam_full skim step" not in caplog.text
    assert "beam_full skim completed successfully" not in caplog.text


def test_build_standard_step_prefers_custom_output_replayer(tmp_path: Path) -> None:
    output_path = tmp_path / "output.txt"
    output_path.write_text("ok", encoding="utf-8")

    events = []
    step = build_standard_step(
        coupler=SimpleNamespace(),
        outputs_holder=StepOutputsHolder(),
        spec=StandardStepSpec(
            step_name="atlas_preprocess",
            model_name="atlas",
            phase="preprocess",
            outputs_class=_BuilderOutputs,
            component_getter=lambda _factory, _state: object(),
            component_executor=lambda _component, _workspace, _holder, **_kwargs: (
                _BuilderOutputs(output_file=output_path)
            ),
            output_logger=lambda *_args, **_kwargs: events.append("output_logger"),
            output_replayer=lambda *_args, **_kwargs: events.append("output_replayer"),
        ),
    )

    step.pilates_output_replayer(
        _BuilderOutputs(output_file=output_path),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        StepOutputsHolder(),
    )

    assert events == ["output_replayer"]


def test_build_standard_step_replayer_defaults_to_output_logger(tmp_path: Path) -> None:
    output_path = tmp_path / "output.txt"
    output_path.write_text("ok", encoding="utf-8")

    events = []
    step = build_standard_step(
        coupler=SimpleNamespace(),
        outputs_holder=StepOutputsHolder(),
        spec=StandardStepSpec(
            step_name="urbansim_run",
            model_name="urbansim",
            phase="run",
            outputs_class=_BuilderOutputs,
            component_getter=lambda _factory, _state: object(),
            component_executor=lambda _component, _workspace, _holder, **_kwargs: (
                _BuilderOutputs(output_file=output_path)
            ),
            output_logger=lambda *_args, **_kwargs: events.append("output_logger"),
        ),
    )

    step.pilates_output_replayer(
        _BuilderOutputs(output_file=output_path),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        StepOutputsHolder(),
    )

    assert events == ["output_logger"]
