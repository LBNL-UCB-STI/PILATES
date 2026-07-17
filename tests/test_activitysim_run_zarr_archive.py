"""Regression coverage ensuring OMX-mode ActivitySim runs archive skims.zarr.

When the ActivitySim compile step was folded into the normal run step
(c79c379e), the container invocation that actually produces ``skims.zarr``
in OMX mode stopped declaring that path in ``output_paths``. Consist only
archives what a ``run_container`` call declares, so the file was written to
local scratch but never copied to the archive root, and a later BEAM
checkpoint's pinned-closure verification failed with "Pinned closure
hydration did not verify zarr_skims."
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.activitysim.runner import ActivitysimRunner, asim_runtime_zarr_path


def _workspace(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
        get_asim_runtime_cache_dir=lambda: str(
            tmp_path / "activitysim" / "output" / "cache"
        ),
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_output_folder="activitysim/output",
            command_template="activitysim",
            household_sample_size=None,
            num_processes=1,
            chunk_size=None,
        ),
    )


def _run_runner(monkeypatch, tmp_path: Path, *, skim_mode: str):
    workspace = _workspace(tmp_path)
    settings = _settings()
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        current_inner_iter=0,
    )
    runner = ActivitysimRunner("activitysim", state)

    monkeypatch.setattr(runner, "get_asim_docker_vols", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runner, "get_model_and_image", lambda *_a: ("activitysim", "image")
    )
    monkeypatch.setattr(runner, "get_base_asim_cmd", lambda *_a, **_k: ["activitysim"])
    monkeypatch.setattr(runner, "get_asim_additional_args", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "pilates.activitysim.runner._log_activitysim_launch_context",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "pilates.activitysim.runner.clear_asim_run_marker", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "pilates.activitysim.runner.write_asim_run_marker", lambda *_a, **_k: None
    )

    captured_output_paths: list = []

    def _run_container(**kwargs):
        captured_output_paths.append(kwargs["output_paths"])
        return True

    monkeypatch.setattr(runner, "run_container", _run_container)

    inputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path,
        land_use_table=tmp_path / "land_use.csv",
        households_table=tmp_path / "households.csv",
        persons_table=tmp_path / "persons.csv",
    )
    extra_inputs = (
        {"zarr_skims": asim_runtime_zarr_path(workspace)}
        if skim_mode == "zarr"
        else None
    )
    if skim_mode == "zarr":
        zarr_path = Path(asim_runtime_zarr_path(workspace))
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        zarr_path.mkdir()

    outputs = runner._run(
        inputs, workspace, skim_mode=skim_mode, extra_inputs=extra_inputs
    )
    return outputs, captured_output_paths, workspace


def test_omx_mode_run_declares_zarr_skims_as_container_output(monkeypatch, tmp_path):
    outputs, captured_output_paths, workspace = _run_runner(
        monkeypatch, tmp_path, skim_mode="omx"
    )

    assert len(captured_output_paths) == 1
    output_paths = captured_output_paths[0]
    assert asim_runtime_zarr_path(workspace) in output_paths


def test_zarr_mode_run_does_not_redeclare_input_zarr_as_output(monkeypatch, tmp_path):
    outputs, captured_output_paths, workspace = _run_runner(
        monkeypatch, tmp_path, skim_mode="zarr"
    )

    assert len(captured_output_paths) == 1
    output_paths = captured_output_paths[0]
    assert asim_runtime_zarr_path(workspace) not in output_paths
