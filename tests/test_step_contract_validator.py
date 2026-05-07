"""
Narrative tests for startup workflow step contract validation.

These tests document the invariant that step wiring must be internally
consistent before long runs begin.

The validator checks four layers together:

1. ``StepOutputsHolder`` fields (runtime in-memory handoff contract)
2. ``STEP_OUTPUTS_CLASSES`` (typed output reconstruction contract)
3. ``STEP_DEPENDENCIES`` (execution ordering contract)
4. Declared step models (Consist metadata contract)

This file intentionally mutates one layer at a time to show what class of
integration drift is caught and what the expected startup failure looks like.
"""

from __future__ import annotations

import re

import pytest
from consist import define_step

from pilates.workflows.artifact_keys import (
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    ZARR_SKIMS,
)
from pilates.workflows.orchestration import StepRef
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_impacts_postprocess_step,
    make_impacts_preprocess_step,
    make_impacts_run_step,
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
        make_beam_full_skim_step(coupler=coupler, outputs_holder=holder),
        make_impacts_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_impacts_run_step(coupler=coupler, outputs_holder=holder),
        make_impacts_postprocess_step(coupler=coupler, outputs_holder=holder),
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
    patched_deps = {
        key: dict(value) for key, value in step_shared.STEP_DEPENDENCIES.items()
    }
    beam_run_spec = dict(patched_deps["beam_run"])
    beam_run_spec["depends_on"] = ["not_a_real_step"]
    patched_deps["beam_run"] = beam_run_spec
    patched_runtime_deps = {
        key: dict(value) for key, value in step_shared.STEP_RUNTIME_DEPENDENCIES.items()
    }
    patched_runtime_beam_run = dict(patched_runtime_deps["beam_run"])
    patched_runtime_beam_run["depends_on"] = ["not_a_real_step"]
    patched_runtime_deps["beam_run"] = patched_runtime_beam_run
    monkeypatch.setattr(step_shared, "STEP_DEPENDENCIES", patched_deps)
    monkeypatch.setattr(step_shared, "STEP_RUNTIME_DEPENDENCIES", patched_runtime_deps)

    with pytest.raises(RuntimeError, match="depends_on unknown steps"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps()
        )


def test_validate_step_ready_enforces_untracked_activitysim_compile_inputs():
    holder = StepOutputsHolder()

    with pytest.raises(
        RuntimeError,
        match="activitysim_compile requires activitysim_preprocess to complete first",
    ):
        step_shared.validate_step_ready("activitysim_compile", holder)


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


def test_validate_workflow_step_contracts_reports_output_contract_conflicts():
    steps = _declared_schema_steps()

    @define_step(model="urbansim_run", outputs=["metadata_override"])
    def _conflicting_urbansim_run(*args, **kwargs):
        return None

    steps = [
        step for step in steps if step.__consist_step__.model != "urbansim_run"
    ] + [_conflicting_urbansim_run]

    expected = (
        "Step 'urbansim_run': canonical outputs ['usim_datastore_h5'] "
        "conflict with metadata outputs ['metadata_override']. "
        "Fix: remove metadata override or update declared_outputs in UrbanSimRunOutputs."
    )

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        step_shared.validate_workflow_step_contracts(declared_steps=steps)


def test_validate_workflow_step_contracts_flags_missing_canonical_outputs_when_metadata_declares_outputs():
    steps = _declared_schema_steps()

    @define_step(model="beam_run", outputs=["beam_linkstats"])
    def _beam_run_with_metadata_override(*args, **kwargs):
        return None

    steps = [step for step in steps if step.__consist_step__.model != "beam_run"] + [
        _beam_run_with_metadata_override
    ]

    expected = (
        "Step 'beam_run': canonical outputs ['linkstats', 'beam_plans_out'] conflict with metadata outputs "
        "['beam_linkstats']. Fix: remove metadata override or update declared_outputs "
        "in BeamRunOutputs."
    )

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        step_shared.validate_workflow_step_contracts(declared_steps=steps)


def test_tracked_beam_step_output_classes_define_explicit_canonical_outputs():
    allowed_empty = frozenset()
    expected = {
        "beam_preprocess": (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN),
        "beam_run": (LINKSTATS, BEAM_PLANS_OUT),
        "beam_postprocess": (ZARR_SKIMS,),
        "beam_full_skim": (BEAM_FULL_SKIMS,),
    }

    for step_name, expected_outputs in expected.items():
        outputs_class = step_shared.STEP_OUTPUTS_CLASSES[step_name]
        canonical = declared_outputs_for_step_outputs_class(outputs_class)
        if not canonical and step_name in allowed_empty:
            continue
        assert canonical == expected_outputs


def test_validate_workflow_step_contracts_requires_rationale_for_required_outputs_override():
    @define_step(model="dummy_step")
    def _dummy_step(*args, **kwargs):
        return None

    with pytest.raises(RuntimeError, match="requires StepRef.required_outputs_rationale"):
        step_shared.validate_workflow_step_contracts(
            step_refs=[
                StepRef(
                    name="dummy_step",
                    step_func=_dummy_step,
                    required_outputs=["override_key"],
                )
            ]
        )


def test_validate_workflow_step_contracts_accepts_rationalized_required_outputs_override():
    @define_step(model="dummy_step")
    def _dummy_step(*args, **kwargs):
        return None

    step_shared.validate_workflow_step_contracts(
        step_refs=[
            StepRef(
                name="dummy_step",
                step_func=_dummy_step,
                required_outputs=["override_key"],
                required_outputs_rationale="Temporary bridge during migration.",
            )
        ]
    )
