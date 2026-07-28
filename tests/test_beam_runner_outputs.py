from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.beam.launch_config import BeamLaunchConfig
from pilates.beam.runner import BeamRunner
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.outputs_base import ValidationContext


class _Workspace:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.full_path = tmp_path

    def get_beam_output_dir(self) -> str:
        return str(self._tmp_path / "beam-output")


class _StubState:
    forecast_year = 2030
    iteration = 2

    def __init__(self) -> None:
        self.sub_stage_progress = None

    def set_sub_stage_progress(self, progress: str) -> None:
        self.sub_stage_progress = progress


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub")


def test_gather_outputs_logs_phys_sim_linkstats_parquet_files(tmp_path):
    beam_output_dir = tmp_path / "beam-output"
    it0 = beam_output_dir / "ITERS" / "it.0"
    it1 = beam_output_dir / "ITERS" / "it.1"

    _touch(it0 / "0.linkstats_unmodified_physSimIter1.parquet")
    _touch(it0 / "0.linkstats_unmodified_physSimIter3.parquet")
    _touch(it1 / "1.linkstats_unmodified_physSimIter2.parquet")

    runner = BeamRunner("beam_runner", _StubState())
    outputs = runner.gather_outputs(str(beam_output_dir))
    short_names = {record.short_name for record in outputs}

    assert (
        "linkstats_unmodified_parquet__y2030__i2__phys_sim_iter1__beam_sub_iter0"
        in short_names
    )
    assert (
        "linkstats_unmodified_parquet__y2030__i2__phys_sim_iter3__beam_sub_iter0"
        in short_names
    )
    assert "linkstats_unmodified_parquet__y2030__i2__phys_sim_iter2" in short_names

    by_key = {record.short_name: record for record in outputs}
    promoted = by_key["linkstats_unmodified_parquet__y2030__i2__phys_sim_iter2"]
    facet = (promoted.metadata or {}).get("facet", {})
    assert facet.get("artifact_family") == "linkstats_unmodified_phys_sim_iter_parquet"
    assert facet.get("year") == 2030
    assert facet.get("iteration") == 2
    assert facet.get("phys_sim_iteration") == 2
    assert facet.get("beam_sub_iteration") == 1


def test_gather_outputs_discovers_activitysim_omx_basename(tmp_path):
    beam_output_dir = tmp_path / "beam-output"
    skim_path = (
        beam_output_dir
        / "seattle"
        / "run"
        / "ITERS"
        / "it.2"
        / "2.activitySimODSkims_current.omx"
    )
    _touch(skim_path)

    runner = BeamRunner("beam_runner", _StubState())
    outputs = runner.gather_outputs(str(beam_output_dir))

    by_name = {record.short_name: record for record in outputs}
    assert by_name["raw_od_skims_2030_2"].file_path == str(skim_path)


def test_beam_runner_run_returns_typed_outputs(tmp_path, monkeypatch) -> None:
    state = _StubState()
    runner = BeamRunner("beam_runner", state)
    workspace = _Workspace(tmp_path)
    captured = {}

    def _fake_run(store: RecordStore, _workspace, _launch_config) -> RecordStore:
        captured["store"] = store
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(tmp_path / "beam-output" / "plans.csv.gz"),
                    short_name="beam_plans_out_2030_2",
                )
            ]
        )

    monkeypatch.setattr(runner, "_run", _fake_run)
    preprocess_outputs = BeamPreprocessOutputs(
        beam_mutable_data_dir=tmp_path / "beam-input",
        prepared_inputs={"beam_plans": tmp_path / "beam-input" / "plans.csv.gz"},
    )

    outputs = runner.run(
        preprocess_outputs,
        workspace,
        launch_config=BeamLaunchConfig(
            root=tmp_path / "beam-launch",
            primary_config=tmp_path / "beam-launch" / "beam.conf",
        ),
        extra_inputs={"zarr_skims": tmp_path / "asim-output" / "cache" / "skims.zarr"},
    )

    assert isinstance(outputs, BeamRunOutputs)
    assert captured["store"].to_mapping()["beam_plans"] == str(
        tmp_path / "beam-input" / "plans.csv.gz"
    )
    assert captured["store"].to_mapping()["zarr_skims"] == str(
        tmp_path / "asim-output" / "cache" / "skims.zarr"
    )
    assert outputs.raw_outputs["beam_plans_out_2030_2"] == (
        tmp_path / "beam-output" / "plans.csv.gz"
    )
    assert state.sub_stage_progress == "runner"


def test_beam_runner_run_rejects_non_typed_inputs(tmp_path) -> None:
    runner = BeamRunner("beam_runner", _StubState())

    with pytest.raises(TypeError, match="BeamPreprocessOutputs"):
        runner.run(
            object(),
            _Workspace(tmp_path),
            launch_config=BeamLaunchConfig(
                root=tmp_path / "beam-launch",
                primary_config=tmp_path / "beam-launch" / "beam.conf",
            ),
        )


def test_beam_runner_mounts_the_bound_launch_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _StubState()
    state.current_year = 2030
    state.current_inner_iter = 2
    state.full_settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(config="beam.conf", memory="4g"),
        shared=SimpleNamespace(skims=SimpleNamespace(fname="skims.omx")),
    )
    runner = BeamRunner("beam_runner", state)
    workspace = _Workspace(tmp_path)
    launch_root = tmp_path / "launch" / "seattle"
    launch_primary = launch_root / "configs" / "beam.conf"
    launch_primary.parent.mkdir(parents=True)
    launch_primary.write_text("beam {}\n", encoding="utf-8")
    launch_config = BeamLaunchConfig(root=launch_root, primary_config=launch_primary)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "get_model_and_image", lambda *_args: ("beam", "image"))
    monkeypatch.setattr(
        runner,
        "run_container",
        lambda **kwargs: captured.update(kwargs) or True,
    )
    monkeypatch.setattr(
        "pilates.beam.runner.rename_beam_output_directory",
        lambda *_args: ("old", str(tmp_path / "beam-output")),
    )
    monkeypatch.setattr(runner, "gather_outputs", lambda *_args: [])

    runner._run(RecordStore(), workspace, launch_config)

    assert captured["command"] == "--config=/app/input/configs/beam.conf"
    assert captured["volumes"] == {
        str(launch_root): {"bind": "/app/input", "mode": "rw"},
        str(launch_root.parent): {"bind": str(launch_root.parent), "mode": "rw"},
        str(tmp_path / "beam-output"): {"bind": "/app/output", "mode": "rw"},
    }


def test_beam_postprocess_outputs_preserve_all_paths_when_omx_exists(
    tmp_path: Path,
) -> None:
    zarr_skims = tmp_path / "skims.zarr"
    final_skims_omx = tmp_path / "final-skims.omx"
    split_event = tmp_path / "events.parquet"
    split_links = tmp_path / "links.parquet"

    outputs = BeamPostprocessOutputs(
        zarr_skims=zarr_skims,
        final_skims_omx=final_skims_omx,
        split_events={"events_parquet_2030_2_type_PathTraversal": split_event},
        split_event_links={"path_traversal_links_2030_2": split_links},
    )

    mapping = outputs.to_record_store().to_mapping()

    assert mapping["final_skims_omx"] == str(final_skims_omx)
    assert mapping["zarr_skims"] == str(zarr_skims)
    assert mapping["events_parquet_2030_2_type_PathTraversal"] == str(split_event)
    assert mapping["path_traversal_links_2030_2"] == str(split_links)


def test_beam_postprocess_outputs_validate_allows_beam_only_without_skims(
    tmp_path: Path,
) -> None:
    split_event = tmp_path / "events.parquet"
    split_links = tmp_path / "links.parquet"
    _touch(split_event)
    _touch(split_links)

    outputs = BeamPostprocessOutputs(
        split_events={"events_parquet_2030_2_type_PathTraversal": split_event},
        split_event_links={"path_traversal_links_2030_2": split_links},
    )

    outputs.validate(
        context=ValidationContext(
            settings=SimpleNamespace(
                run=SimpleNamespace(
                    models=SimpleNamespace(activity_demand=None, land_use=None)
                ),
                write_skims_to_omx=False,
            ),
            step_name="beam_postprocess",
        )
    )

    assert BeamPostprocessOutputs.declared_output_keys() == ("zarr_skims",)
    assert BeamPostprocessOutputs.required_output_keys() == ()
    assert BeamPostprocessOutputs.optional_output_keys() == (
        "zarr_skims",
        "final_skims_omx",
    )


def test_beam_postprocess_outputs_validate_requires_zarr_when_activitysim_enabled(
    tmp_path: Path,
) -> None:
    split_event = tmp_path / "events.parquet"
    _touch(split_event)

    outputs = BeamPostprocessOutputs(
        split_events={"events_parquet_2030_2_type_PathTraversal": split_event},
    )

    with pytest.raises(AssertionError, match="zarr_skims is required"):
        outputs.validate(
            context=ValidationContext(
                settings=SimpleNamespace(
                    run=SimpleNamespace(
                        models=SimpleNamespace(
                            activity_demand="activitysim",
                            land_use=None,
                        )
                    ),
                    write_skims_to_omx=False,
                ),
                step_name="beam_postprocess",
            )
        )


def test_beam_postprocess_outputs_validate_requires_omx_when_atlas_enabled(
    tmp_path: Path,
) -> None:
    split_event = tmp_path / "events.parquet"
    zarr_skims = tmp_path / "skims.zarr"
    _touch(split_event)
    zarr_skims.mkdir()

    outputs = BeamPostprocessOutputs(
        zarr_skims=zarr_skims,
        split_events={"events_parquet_2030_2_type_PathTraversal": split_event},
    )

    with pytest.raises(AssertionError, match="final_skims_omx is required"):
        outputs.validate(
            context=ValidationContext(
                settings=SimpleNamespace(
                    run=SimpleNamespace(
                        models=SimpleNamespace(
                            activity_demand="activitysim",
                            land_use=None,
                            vehicle_ownership="atlas",
                        )
                    ),
                    write_skims_to_omx=False,
                ),
                step_name="beam_postprocess",
            )
        )
