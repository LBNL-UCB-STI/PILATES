"""
Focused golden-workflow contract tests.

These tests are smaller companions to ``test_golden_stub_workflow``. They keep
the end-to-end narrative value of the golden harness while locking in a few
stable surfaces that are easy to regress during orchestration refactors:

1. which step publishes raw vs finalized outputs
2. which artifacts are promoted to the scenario surface
3. which temporary artifacts must remain step-local
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pilates.runtime import bootstrap as bootstrap_runtime
from pilates.urbansim.outputs import UrbanSimRunOutputs
from pilates.urbansim.runner import UrbansimRunner
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
)
from pilates.workflows.stages.land_use import run_land_use_stage as _run_land_use_stage
from tests.test_golden_stub_workflow import _write_file


def _initialize(env) -> None:
    settings = env["settings"]
    state = env["state"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = env["coupler"]
    source_path = Path(env["usim_input_path"])
    init_marker = Path(workspace.full_path) / ".golden_surface_init_marker.txt"
    with scenario.trace(
        "initialization",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init", "surface-contract"],
    ):
        cr.log_input(source_path, key="golden_surface_init_source")
        _write_file(init_marker, "initialized")
        cr.log_output(init_marker, key="golden_surface_init_marker")
    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )
    state.set_data_initialized(True)


@pytest.mark.slow_ci
def test_golden_workflow_preserves_current_stage_surfaces_on_scenario_outputs(
    golden_stub_env, monkeypatch
):
    settings = golden_stub_env["settings"]
    workspace = golden_stub_env["workspace"]
    state = golden_stub_env["state"]
    scenario = golden_stub_env["scenario"]
    coupler = golden_stub_env["coupler"]
    tracker = golden_stub_env["tracker"]

    def _stub_urbansim_runner(
        _self, _inputs, runner_workspace, _model_run_hash=None
    ) -> UrbanSimRunOutputs:
        return UrbanSimRunOutputs(
            usim_datastore_h5=runner_workspace.output_datastore,
            raw_outputs={},
        )

    monkeypatch.setattr(UrbansimRunner, "_run", _stub_urbansim_runner)
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    _initialize(golden_stub_env)

    usim_inputs = _run_land_use_stage(
        scenario=scenario,
        coupler=coupler,
        year=state.forecast_year,
        context=golden_stub_env["context"],
    )
    land_use_datastore = artifact_to_path(coupler.get(USIM_DATASTORE_H5), workspace)
    assert land_use_datastore is not None
    assert (
        Path(land_use_datastore).resolve()
        == Path(usim_inputs[USIM_DATASTORE_CURRENT_H5]).resolve()
    )
    base_usim_datastore = artifact_to_path(
        coupler.get(USIM_DATASTORE_BASE_H5),
        workspace,
    )
    assert base_usim_datastore is not None
    assert (
        Path(base_usim_datastore).resolve()
        == Path(usim_inputs[USIM_DATASTORE_BASE_H5]).resolve()
    )

    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs
    scenario_run = runs[0]
    steps = scenario_run.meta["steps"]
    assert [step["model"] for step in steps] == [
        "initialization",
        "urbansim_run",
        "urbansim_postprocess",
    ]

    scenario_outputs = tracker.get_artifacts_for_run(scenario_run.id).outputs
    assert USIM_DATASTORE_H5 in scenario_outputs
