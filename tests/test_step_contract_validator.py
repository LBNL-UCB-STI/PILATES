from __future__ import annotations

import pytest
from consist import define_step

from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
)
from pilates.workflows.steps import shared as step_shared


class _DummyCoupler:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        return None

    def set_from_artifact(self, _key, _value):
        return None


def _declared_schema_steps():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()
    return [
        make_urbansim_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_urbansim_run_step(coupler=coupler, outputs_holder=holder),
        make_urbansim_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_atlas_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_atlas_run_step(coupler=coupler, outputs_holder=holder),
        make_atlas_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_compile_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_run_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_beam_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_beam_run_step(coupler=coupler, outputs_holder=holder),
        make_beam_postprocess_step(coupler=coupler, outputs_holder=holder),
    ]


def test_validate_workflow_step_contracts_passes_for_current_setup():
    step_shared.validate_workflow_step_contracts(
        declared_steps=_declared_schema_steps()
    )


def test_validate_workflow_step_contracts_detects_holder_output_drift(monkeypatch):
    patched_classes = dict(step_shared.STEP_OUTPUTS_CLASSES)
    patched_classes.pop("beam_run", None)
    monkeypatch.setattr(step_shared, "STEP_OUTPUTS_CLASSES", patched_classes)

    with pytest.raises(RuntimeError, match="Missing output classes"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps()
        )


def test_validate_workflow_step_contracts_detects_bad_dependency_reference(monkeypatch):
    patched_deps = {key: dict(value) for key, value in step_shared.STEP_DEPENDENCIES.items()}
    beam_run_spec = dict(patched_deps["beam_run"])
    beam_run_spec["depends_on"] = ["not_a_real_step"]
    patched_deps["beam_run"] = beam_run_spec
    monkeypatch.setattr(step_shared, "STEP_DEPENDENCIES", patched_deps)

    with pytest.raises(RuntimeError, match="depends_on unknown steps"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps()
        )


def test_validate_workflow_step_contracts_flags_untracked_declared_steps():
    @define_step(model="unexpected_new_step")
    def _extra_step(*args, **kwargs):
        return None

    with pytest.raises(RuntimeError, match="Declared step names are not tracked"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=[*_declared_schema_steps(), _extra_step]
        )


def test_validate_workflow_step_contracts_allows_explicit_untracked_allowlist():
    @define_step(model="intentional_untracked_step")
    def _extra_step(*args, **kwargs):
        return None

    step_shared.validate_workflow_step_contracts(
        declared_steps=[*_declared_schema_steps(), _extra_step],
        allow_untracked_declared={"intentional_untracked_step"},
    )
