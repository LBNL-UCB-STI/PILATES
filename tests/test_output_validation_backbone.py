from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Tuple

import pytest

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.outputs_base import (
    OutputValidator,
    StepOutputsBase,
    ValidationContext,
    ValidationResult,
)
from pilates.workflows.steps.shared import (
    StandardStepSpec,
    StepOutputsHolder,
    build_standard_step,
)


@dataclass
class _BaseTestOutputs(StepOutputsBase):
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_file",)
    output_file: Path


class _WarningValidator:
    name = "warning_validator"
    level = "warning"

    def validate(
        self, outputs: StepOutputsBase, context: ValidationContext
    ) -> list[ValidationResult]:
        return [
            ValidationResult(
                message="warning check",
                metadata={"step": context.step_name},
            )
        ]


class _ErrorValidator:
    name = "error_validator"
    level = "error"

    def validate(
        self, outputs: StepOutputsBase, context: ValidationContext
    ) -> list[ValidationResult]:
        return [
            ValidationResult(
                message="error check",
                metadata={"hint": "repair upstream mapping"},
            )
        ]


@dataclass
class _WarningOutputs(StepOutputsBase):
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_file",)
    validators: ClassVar[Tuple[OutputValidator, ...]] = (_WarningValidator(),)
    output_file: Path


@dataclass
class _ErrorOutputs(StepOutputsBase):
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_file",)
    validators: ClassVar[Tuple[OutputValidator, ...]] = (_ErrorValidator(),)
    output_file: Path


def test_empty_validator_parity_preserves_existing_path_checks(tmp_path: Path) -> None:
    output_path = tmp_path / "output.txt"
    output_path.write_text("ok")

    outputs = _BaseTestOutputs(output_file=output_path)
    outputs.validate()

    missing_outputs = _BaseTestOutputs(output_file=tmp_path / "missing.txt")
    with pytest.raises(AssertionError, match="output_file missing"):
        missing_outputs.validate()


def test_warning_validator_does_not_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output_path = tmp_path / "output.txt"
    output_path.write_text("ok")
    outputs = _WarningOutputs(output_file=output_path)

    with caplog.at_level("WARNING"):
        outputs.validate(context=ValidationContext(step_name="warning_step"))

    assert "OUTPUT VALIDATION WARNING" in caplog.text
    assert "warning_validator" in caplog.text
    assert "warning check" in caplog.text


def test_error_validator_fails_with_actionable_message(tmp_path: Path) -> None:
    output_path = tmp_path / "output.txt"
    output_path.write_text("ok")
    outputs = _ErrorOutputs(output_file=output_path)

    with pytest.raises(AssertionError) as excinfo:
        outputs.validate(context=ValidationContext(step_name="error_step"))

    message = str(excinfo.value)
    assert "error_validator" in message
    assert "error check" in message
    assert "repair upstream mapping" in message
    assert "Fix the flagged output contract issue(s)" in message


class _CrossStepContextValidator:
    name = "cross_step_context_validator"
    level = "error"

    def __init__(self) -> None:
        self.captured_step_name = None
        self.captured_token = None

    def validate(
        self, outputs: StepOutputsBase, context: ValidationContext
    ) -> list[ValidationResult]:
        self.captured_step_name = context.step_name
        upstream = context.upstream_outputs.get("urbansim_run")
        self.captured_token = getattr(upstream, "token", None)
        if self.captured_token != "upstream-ok":
            return [ValidationResult(message="missing upstream token")]
        return []


_cross_step_validator = _CrossStepContextValidator()


@dataclass
class _CrossStepOutputs(StepOutputsBase):
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_file",)
    record_keys: ClassVar[dict] = {"output_file": "dummy_output"}
    validators: ClassVar[Tuple[OutputValidator, ...]] = (_cross_step_validator,)
    output_file: Path


class _ValidationWorkspace:
    def __init__(self, root: Path) -> None:
        self.full_path = str(root)


def test_cross_step_validator_reads_upstream_outputs_from_context(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "generated.txt"
    output_path.write_text("ok")

    outputs_holder = StepOutputsHolder()
    outputs_holder.urbansim_run = SimpleNamespace(token="upstream-ok")

    step_func = build_standard_step(
        coupler=SimpleNamespace(),
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="dummy_run",
            model_name="dummy",
            phase="run",
            outputs_class=_CrossStepOutputs,
            component_getter=lambda _factory, _state: object(),
            component_executor=lambda _component, _workspace, _holder, **_kwargs: (
                RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(output_path),
                            short_name="dummy_output",
                            description="dummy output",
                        )
                    ]
                )
            ),
        ),
    )

    step_func(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=_ValidationWorkspace(tmp_path),
    )

    assert _cross_step_validator.captured_step_name == "dummy_run"
    assert _cross_step_validator.captured_token == "upstream-ok"


def test_activitysim_boundary_validator_warns_when_tables_leave_mutable_dir(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mutable_dir = tmp_path / "activitysim" / "data"
    mutable_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True)

    land_use = outside_dir / "land_use.csv"
    households = mutable_dir / "households.csv"
    persons = mutable_dir / "persons.csv"
    for path in (land_use, households, persons):
        path.write_text("x")

    outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )
    context = ValidationContext(
        step_name="activitysim_preprocess",
        upstream_outputs={"urbansim_postprocess": SimpleNamespace()},
    )

    with caplog.at_level("WARNING"):
        outputs.validate(context=context)

    assert "activitysim_preprocess_urbansim_boundary" in caplog.text
    assert "land_use_table should be written under mutable_data_dir" in caplog.text
