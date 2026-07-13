"""Regression coverage for the private ActivitySim Numba warmup handshake.

The filename is retained for test-discovery continuity.  The public workflow
compile step it formerly covered no longer exists.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.activitysim.runner import ActivitysimNumbaWarmup


def test_numba_warmup_is_private_and_returns_only_a_local_zarr_side_effect(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_output_folder="activitysim/output",
        ),
    )
    state = SimpleNamespace(full_settings=settings)
    warmup = ActivitysimNumbaWarmup("activitysim_numba_warmup", state)
    zarr_path = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"

    monkeypatch.setattr(warmup, "get_asim_docker_vols", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        warmup, "get_model_and_image", lambda *_args: ("activitysim", "image")
    )
    monkeypatch.setattr(
        warmup, "get_base_asim_cmd", lambda *_args, **_kwargs: ["activitysim"]
    )
    monkeypatch.setattr(warmup, "get_asim_additional_args", lambda *_args: [])
    monkeypatch.setattr(
        "pilates.activitysim.runner._log_activitysim_launch_context",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "pilates.activitysim.runner.finalize_activitysim_zarr_skims",
        lambda path, *_args: Path(path),
    )

    def _run_container(**kwargs):
        assert kwargs["model_name"] == "activitysim_numba_warmup"
        assert kwargs["lineage_mode"] == "none"
        zarr_path.mkdir(parents=True)
        return True

    monkeypatch.setattr(warmup, "run_container", _run_container)
    result = warmup.run(
        ActivitySimPreprocessOutputs(
            mutable_data_dir=tmp_path,
            land_use_table=tmp_path / "land_use.csv",
            households_table=tmp_path / "households.csv",
            persons_table=tmp_path / "persons.csv",
        ),
        workspace,
    )

    assert result == zarr_path
    assert not hasattr(warmup, "expected_outputs")
