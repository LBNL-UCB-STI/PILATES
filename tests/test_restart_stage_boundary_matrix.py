"""Focused BEAM restart diagnostics and explicit canary policy.

Whole-stage and whole-year execution are exercised through the native golden
workflow.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import BindingResult

from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.stages import supply_demand_beam as beam_stage
from pilates.workflows.stages.supply_demand_beam import (
    _FAIL_AFTER_BEAM_RUN_ENV,
    _maybe_fail_after_beam_run_for_canary,
)


def test_beam_restart_canary_failpoint_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(_FAIL_AFTER_BEAM_RUN_ENV, raising=False)

    _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)

    monkeypatch.setenv(_FAIL_AFTER_BEAM_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="Injected failure after completed beam_run"):
        _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)


def test_beam_stage_passes_one_compiled_launch_config_to_binding_and_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The config Consist binds is the same config the runner receives at launch."""

    from pilates.beam.launch_config import BeamLaunchConfig

    launch_root = tmp_path / "launch"
    launch_primary = launch_root / "beam.conf"
    launch_root.mkdir()
    launch_primary.write_text("beam {}\n", encoding="utf-8")
    launch_config = BeamLaunchConfig(root=launch_root, primary_config=launch_primary)
    preprocess_outputs = SimpleNamespace(
        prepared_inputs={"plans_beam_in": tmp_path / "plans.csv"}
    )
    run_inputs = ResolvedStepInputs(
        step_name="beam_run",
        binding=BindingResult(inputs={"beam_config_file": launch_primary}),
    )
    postprocess_inputs = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
    )
    observed: dict[str, object] = {}

    class Coupler:
        def get(self, _key, default=None):
            return default

    scenario = SimpleNamespace(coupler=Coupler(), tracker=SimpleNamespace(run_dir=tmp_path))
    context = SimpleNamespace(
        settings=SimpleNamespace(beam=SimpleNamespace(full_skim=None)),
        state=SimpleNamespace(is_restart_run=False),
        workspace=SimpleNamespace(get_beam_output_dir=lambda: tmp_path),
    )

    def fake_execute_step(
        *, phase, runtime_kwargs=None, resolved_inputs=None, **_kwargs
    ):
        if phase == "preprocess":
            return SimpleNamespace(), preprocess_outputs
        if phase == "run":
            observed["runtime"] = runtime_kwargs
            observed["resolved"] = resolved_inputs
            return (
                SimpleNamespace(run=SimpleNamespace(id="run-1"), outputs={}),
                SimpleNamespace(),
            )
        if phase == "postprocess":
            return SimpleNamespace(), SimpleNamespace()
        pytest.fail(f"unexpected phase {phase}")

    def resolve_run_inputs(**kwargs):
        observed["resolver"] = kwargs
        return run_inputs

    monkeypatch.setattr(
        beam_stage, "_compile_beam_launch_config", lambda **_: launch_config
    )
    monkeypatch.setattr(
        beam_stage, "beam_run", SimpleNamespace(resolve_inputs=resolve_run_inputs)
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_postprocess",
        SimpleNamespace(resolve_inputs=lambda **_kwargs: postprocess_inputs),
    )
    monkeypatch.setattr(beam_stage, "execute_step", fake_execute_step)
    monkeypatch.setattr(
        beam_stage, "_archive_beam_run_outputs_native", lambda **_: None
    )
    monkeypatch.setattr(
        beam_stage, "_maybe_fail_after_beam_run_for_canary", lambda **_: None
    )
    monkeypatch.setattr(beam_stage, "_archive_run_dir_for_restart", lambda _state: None)
    monkeypatch.setattr(
        beam_stage, "step_output_handoff_mapping", lambda *_args, **_kwargs: {}
    )

    beam_stage._run_beam_steps(
        scenario=scenario,
        year=2030,
        iteration=1,
        context=context,
    )

    assert observed["resolver"]["launch_config"] is launch_config
    assert observed["resolved"] is run_inputs
    assert observed["runtime"] == {"beam_launch_config": launch_config}
