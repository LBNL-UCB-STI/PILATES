from __future__ import annotations

import logging

from pilates.activitysim.outputs import ActivitySimRunOutputs
from pilates.activitysim.outputs import ASIM_OUTPUT_KEY_MAP
from pilates.activitysim.outputs import ASIM_OPTIONAL_RUN_OUTPUT_KEYS
from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class


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
