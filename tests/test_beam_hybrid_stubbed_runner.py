"""
Hybrid BEAM postprocessor integration test:
- real BeamPostprocessor orchestration
- stubbed BEAM runner outputs
- patched heavy merge/writer helpers

This complements helper-level tests by validating that Beam postprocess wiring
selects the right runner artifacts and emits expected output records.
"""

from pathlib import Path

import pytest

from pilates.beam.outputs import BeamRunOutputs
from pilates.beam.postprocessor import BeamPostprocessor
from pilates.utils.settings_helper import get as real_get_setting
from pilates.workspace import Workspace
from tests.test_golden_stub_workflow import _build_settings
from workflow_state import WorkflowState


def _write_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_beam_postprocess_hybrid_orchestrates_stubbed_runner_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    """
    Exercise BeamPostprocessor.postprocess with runner-like raw outputs.

    The test verifies orchestration logic that helper-level tests do not cover:
    - chooses latest events parquet sub-iteration
    - accepts fallback skim short name (raw_od_skims_zarr_*)
    - calls merge helper with expected paths
    - emits split events, links table, and zarr output records
    """
    settings = _build_settings(tmp_path)
    settings.state_file_loc = str(tmp_path / "state.yaml")
    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    state.current_inner_iter = 0

    beam_output_dir = Path(workspace.get_beam_output_dir())
    beam_output_dir.mkdir(parents=True, exist_ok=True)
    asim_cache_dir = Path(workspace.get_asim_output_dir()) / "cache"
    asim_cache_dir.mkdir(parents=True, exist_ok=True)
    zarr_skims_target = asim_cache_dir / "skims.zarr"
    _write_file(zarr_skims_target, b"main-zarr")

    events_sub0 = beam_output_dir / "events_sub0.parquet"
    events_sub1 = beam_output_dir / "events_sub1.parquet"
    raw_skims = beam_output_dir / "raw_od_skims_iter0.zarr"
    _write_file(events_sub0)
    _write_file(events_sub1)
    _write_file(raw_skims)

    split_calls = []
    merge_calls = []

    def _fake_split_events_parquet_by_type(
        events_path, event_types=None, output_dir=None, output_prefix=None, create_path_traversal_links=False
    ):
        split_calls.append((events_path, create_path_traversal_links))
        output_root = Path(output_dir) if output_dir else Path(events_path).parent
        path_traversal = output_root / "split.PathTraversal.parquet"
        mode_choice = output_root / "split.ModeChoice.parquet"
        links = output_root / "split.PathTraversal.links.parquet"
        _write_file(path_traversal)
        _write_file(mode_choice)
        _write_file(links)
        return (
            {
                "PathTraversal": str(path_traversal),
                "ModeChoice": str(mode_choice),
            },
            str(links),
        )

    def _fake_merge_beam_skims_to_zarr(
        *,
        all_skims_path,
        iteration_skims_path,
        beam_output_dir,
        settings,
        workspace,
        override=None,
    ):
        merge_calls.append(
            {
                "all_skims_path": all_skims_path,
                "iteration_skims_path": iteration_skims_path,
                "beam_output_dir": beam_output_dir,
                "override": override,
            }
        )
        return all_skims_path

    def _fake_get_setting(settings_obj, key, default=None):
        if key == "write_skims_to_omx":
            return False
        if key == "run.models.land_use":
            return "not_urbansim"
        return real_get_setting(settings_obj, key, default)

    monkeypatch.setattr(
        "pilates.beam.postprocessor.split_events_parquet_by_type",
        _fake_split_events_parquet_by_type,
    )
    monkeypatch.setattr(
        "pilates.beam.postprocessor._merge_beam_skims_to_zarr",
        _fake_merge_beam_skims_to_zarr,
    )
    monkeypatch.setattr(
        "pilates.beam.postprocessor.get_setting",
        _fake_get_setting,
    )

    raw_outputs = BeamRunOutputs(
        beam_output_dir=beam_output_dir,
        raw_outputs={
            f"events_parquet_{state.forecast_year}_{state.iteration}_sub0": events_sub0,
            f"events_parquet_{state.forecast_year}_{state.iteration}_sub1": events_sub1,
            f"raw_od_skims_zarr_{state.forecast_year}_{state.iteration}": raw_skims,
        },
    )

    postprocessor = BeamPostprocessor("beam", state)
    outputs = postprocessor.postprocess(raw_outputs, workspace)
    output_keys = {
        *outputs.split_events.keys(),
        *outputs.split_event_links.keys(),
    }
    if outputs.zarr_skims is not None:
        output_keys.add("zarr_skims")
    if outputs.final_skims_omx is not None:
        output_keys.add("final_skims_omx")

    assert split_calls == [(str(events_sub1), True)]
    assert len(merge_calls) == 1
    assert merge_calls[0]["all_skims_path"] == str(zarr_skims_target)
    assert merge_calls[0]["iteration_skims_path"] == str(raw_skims)
    assert merge_calls[0]["beam_output_dir"] == str(beam_output_dir)
    assert merge_calls[0]["override"] == str(raw_skims)

    assert f"events_parquet_{state.forecast_year}_{state.iteration}_type_PathTraversal" in output_keys
    assert f"events_parquet_{state.forecast_year}_{state.iteration}_type_ModeChoice" in output_keys
    assert f"path_traversal_links_{state.forecast_year}_{state.iteration}" in output_keys
    assert "zarr_skims" in output_keys
    assert "final_skims_omx" not in output_keys
    assert state.sub_stage_progress == "postprocessor"


def test_beam_postprocess_rejects_non_typed_run_outputs(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    postprocessor = BeamPostprocessor("beam", state)

    with pytest.raises(TypeError, match="BeamRunOutputs"):
        postprocessor.postprocess(object(), workspace)


def test_beam_postprocess_prefers_explicit_zarr_skims_input_when_local_target_missing(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _build_settings(tmp_path)
    settings.state_file_loc = str(tmp_path / "state.yaml")
    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    state.current_inner_iter = 0

    beam_output_dir = Path(workspace.get_beam_output_dir())
    beam_output_dir.mkdir(parents=True, exist_ok=True)
    restored_zarr = tmp_path / "restored" / "skims.zarr"
    _write_file(restored_zarr, b"restored-zarr")

    events_sub0 = beam_output_dir / "events_sub0.parquet"
    raw_skims = beam_output_dir / "raw_od_skims_iter0.zarr"
    _write_file(events_sub0)
    _write_file(raw_skims)

    merge_calls = []

    def _fake_split_events_parquet_by_type(
        events_path,
        event_types=None,
        output_dir=None,
        output_prefix=None,
        create_path_traversal_links=False,
    ):
        output_root = Path(output_dir) if output_dir else Path(events_path).parent
        path_traversal = output_root / "split.PathTraversal.parquet"
        links = output_root / "split.PathTraversal.links.parquet"
        _write_file(path_traversal)
        _write_file(links)
        return ({"PathTraversal": str(path_traversal)}, str(links))

    def _fake_merge_beam_skims_to_zarr(
        *,
        all_skims_path,
        iteration_skims_path,
        beam_output_dir,
        settings,
        workspace,
        override=None,
    ):
        merge_calls.append(
            {
                "all_skims_path": all_skims_path,
                "iteration_skims_path": iteration_skims_path,
            }
        )
        return all_skims_path

    def _fake_get_setting(settings_obj, key, default=None):
        if key == "write_skims_to_omx":
            return False
        if key == "run.models.land_use":
            return "not_urbansim"
        return real_get_setting(settings_obj, key, default)

    monkeypatch.setattr(
        "pilates.beam.postprocessor.split_events_parquet_by_type",
        _fake_split_events_parquet_by_type,
    )
    monkeypatch.setattr(
        "pilates.beam.postprocessor._merge_beam_skims_to_zarr",
        _fake_merge_beam_skims_to_zarr,
    )
    monkeypatch.setattr(
        "pilates.beam.postprocessor.get_setting",
        _fake_get_setting,
    )

    raw_outputs = BeamRunOutputs(
        beam_output_dir=beam_output_dir,
        raw_outputs={
            f"events_parquet_{state.forecast_year}_{state.iteration}_sub0": events_sub0,
            f"raw_od_skims_zarr_{state.forecast_year}_{state.iteration}": raw_skims,
        },
    )

    postprocessor = BeamPostprocessor("beam", state)
    outputs = postprocessor.postprocess(
        raw_outputs,
        workspace,
        zarr_skims=str(restored_zarr),
    )

    assert len(merge_calls) == 1
    assert merge_calls[0]["all_skims_path"] == str(restored_zarr)
    assert merge_calls[0]["iteration_skims_path"] == str(raw_skims)
    assert outputs.zarr_skims == restored_zarr
