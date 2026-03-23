from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import ActivitySimRunOutputs
from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.activitysim.outputs import ASIM_OUTPUT_KEY_MAP
from pilates.activitysim.outputs import ASIM_OPTIONAL_RUN_OUTPUT_KEYS
from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.runtime.scenario_runtime import SchemaCoupler
from pilates.workflows.orchestration import StepRef
from pilates.workflows.orchestration import _build_step_run_kwargs
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps import make_activitysim_run_step


def test_activitysim_run_outputs_expose_canonical_declared_outputs():
    declared = declared_outputs_for_step_outputs_class(ActivitySimRunOutputs)
    expected = tuple(dict.fromkeys(ASIM_OUTPUT_KEY_MAP.values()))
    assert declared == expected
    assert ActivitySimRunOutputs.declared_output_keys() == expected


def test_activitysim_run_outputs_warn_on_unrecognized_output_keys(tmp_path, caplog):
    canonical_path = tmp_path / "households.parquet"
    canonical_path.write_text("ok", encoding="utf-8")
    extra_path = tmp_path / "mystery.parquet"
    extra_path.write_text("ok", encoding="utf-8")

    outputs = ActivitySimRunOutputs(
        output_dir=tmp_path,
        raw_outputs={
            "households_asim_out_temp": canonical_path,
            "mystery_output": extra_path,
        },
    )

    with caplog.at_level(logging.WARNING):
        outputs.validate(context=ValidationContext(step_name="activitysim_run"))

    assert "Unrecognized ActivitySim run output key 'mystery_output'" in caplog.text
    assert "households_asim_out_temp" not in caplog.text


def test_activitysim_run_outputs_expose_required_output_subset():
    required = ActivitySimRunOutputs.required_output_keys()

    assert required == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert set(ASIM_OPTIONAL_RUN_OUTPUT_KEYS).isdisjoint(required)
    assert set(required) | set(ASIM_OPTIONAL_RUN_OUTPUT_KEYS) == set(
        ActivitySimRunOutputs.declared_output_keys()
    )


def test_activitysim_run_stepref_uses_required_outputs_for_runtime_contract():
    step_func = make_activitysim_run_step(
        coupler=SchemaCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    step = StepRef(
        name="activitysim_run",
        step_func=step_func,
        year=2023,
        iteration=0,
    )

    run_kwargs = _build_step_run_kwargs(
        step=step,
        settings=SimpleNamespace(run=None),
        state=SimpleNamespace(),
        runtime_kwargs={},
        stage_name="activity_demand_run",
        default_iteration=0,
    )

    assert tuple(run_kwargs["outputs"]) == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert "school_shadow_prices_asim_out" not in run_kwargs["outputs"]
    assert "workplace_shadow_prices_asim_out" not in run_kwargs["outputs"]


def test_activitysim_postprocess_outputs_require_processed_asim_tables_but_not_usim_next():
    required = ActivitySimPostprocessOutputs.required_output_keys()

    assert required == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert "usim_input_next" not in required


def test_activitysim_postprocess_validation_requires_usim_output_when_land_use_enabled():
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=Path("/tmp/asim"),
        processed_outputs={},
    )

    with pytest.raises(
        AssertionError,
        match="usim_input_next/usim_datastore_h5 is required",
    ):
        outputs.validate(
            context=ValidationContext(
                step_name="activitysim_postprocess",
                settings=SimpleNamespace(land_use_enabled=True),
            )
        )


def test_activitysim_postprocess_validation_allows_missing_usim_output_when_land_use_disabled():
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=Path("/tmp/asim"),
        processed_outputs={},
    )

    outputs.validate(
        context=ValidationContext(
            step_name="activitysim_postprocess",
            settings=SimpleNamespace(land_use_enabled=False),
        )
    )
