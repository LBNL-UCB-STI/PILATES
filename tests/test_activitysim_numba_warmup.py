from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim import runner as activitysim_runner
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import ActivitysimRunner
from pilates.workflows.artifact_keys import ZARR_SKIMS


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.full_path = str(root)

    def get_asim_output_dir(self) -> str:
        return str(Path(self.full_path) / "activitysim" / "output")


def test_activitysim_run_outputs_round_trip_generated_zarr(tmp_path: Path) -> None:
    """OMX-mode primary output persists Zarr as a first-class record."""
    workspace = _Workspace(tmp_path)
    zarr_path = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    outputs = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        zarr_skims=zarr_path,
    )

    records = outputs.to_record_store()
    recovered = ActivitySimRunOutputs.from_record_store(records, workspace)

    assert recovered.zarr_skims == zarr_path
    assert [record.short_name for record in records.all_records()] == [ZARR_SKIMS]


def _activitysim_inputs(tmp_path: Path) -> ActivitySimPreprocessOutputs:
    return ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path / "mutable_data",
        land_use_table=tmp_path / "land_use.csv",
        households_table=tmp_path / "households.csv",
        persons_table=tmp_path / "persons.csv",
    )


def _runner_state(*, persist_cache: bool, num_processes: int) -> SimpleNamespace:
    return SimpleNamespace(
        full_settings=SimpleNamespace(
            activitysim=SimpleNamespace(
                persist_sharrow_cache=persist_cache,
                num_processes=num_processes,
            )
        ),
        set_sub_stage_progress=lambda _stage: None,
    )


@pytest.mark.parametrize(
    (
        "skip_numba_warmup",
        "persist_cache",
        "num_processes",
        "cache_present",
        "expected_decision",
        "expected_warmup_calls",
    ),
    [
        (False, True, 2, False, "ActivitySim Numba warmup: running", 1),
        (
            True,
            True,
            2,
            False,
            "ActivitySim Numba warmup: skipped (explicit rewind skip)",
            0,
        ),
        (
            False,
            False,
            2,
            False,
            "ActivitySim Numba warmup: skipped (persistent cache disabled)",
            0,
        ),
        (
            False,
            False,
            1,
            True,
            "ActivitySim Numba warmup: skipped (persistent cache disabled)",
            0,
        ),
        (
            False,
            True,
            1,
            False,
            "ActivitySim Numba warmup: skipped (single-process run)",
            0,
        ),
        (
            False,
            True,
            1,
            True,
            "ActivitySim Numba warmup: skipped (single-process run)",
            0,
        ),
        (
            False,
            True,
            2,
            True,
            "ActivitySim Numba warmup: skipped (node-local cache present)",
            0,
        ),
        (
            True,
            False,
            1,
            True,
            "ActivitySim Numba warmup: skipped (explicit rewind skip)",
            0,
        ),
    ],
)
def test_activitysim_runner_logs_one_numba_warmup_decision_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *,
    skip_numba_warmup: bool,
    persist_cache: bool,
    num_processes: int,
    cache_present: bool,
    expected_decision: str,
    expected_warmup_calls: int,
) -> None:
    workspace = _Workspace(tmp_path)
    cache_dir = tmp_path / "shared_cache" / "numba"
    if cache_present:
        cache_dir.mkdir(parents=True)
        (cache_dir / "numba.nbi").touch()

    runner = ActivitysimRunner(
        "activitysim",
        _runner_state(persist_cache=persist_cache, num_processes=num_processes),
    )
    inputs = _activitysim_inputs(tmp_path)
    warmup_calls: list[None] = []

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        _workspace: _Workspace,
    ) -> None:
        warmup_calls.append(None)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "numba.nbi").touch()

    def run_after_decision(*_args: object, **_kwargs: object) -> ActivitySimRunOutputs:
        decisions = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("ActivitySim Numba warmup:")
        ]
        assert decisions == [expected_decision]
        assert len(warmup_calls) == expected_warmup_calls
        return ActivitySimRunOutputs(output_dir=tmp_path / "activitysim" / "output")

    monkeypatch.setattr(activitysim_runner.ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(runner, "_run", run_after_decision)
    caplog.set_level(logging.INFO, logger="pilates.activitysim.runner")

    runner.run(inputs, workspace, skip_numba_warmup=skip_numba_warmup)

    assert expected_decision in caplog.text
    assert len(warmup_calls) == expected_warmup_calls
