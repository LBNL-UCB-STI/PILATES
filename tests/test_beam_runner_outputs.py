from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.beam.runner import BeamRunner
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.outputs_base import ValidationContext


class _Workspace:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path

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


def test_gather_outputs_includes_beam_log_out(tmp_path):
    beam_output_dir = tmp_path / "beam-output"
    _touch(beam_output_dir / "beamLog.out")

    runner = BeamRunner("beam_runner", _StubState())
    outputs = runner.gather_outputs(str(beam_output_dir))
    by_key = {record.short_name: record for record in outputs}

    record = by_key["beam_log_out_2030_2"]
    assert record.file_path == str(beam_output_dir / "beamLog.out")


def test_gather_outputs_includes_emissions_skims_parquet_fallback(tmp_path):
    beam_output_dir = tmp_path / "beam-output"
    _touch(beam_output_dir / "ITERS" / "it.0" / "0.skimsEmissions.parquet")

    runner = BeamRunner("beam_runner", _StubState())
    outputs = runner.gather_outputs(str(beam_output_dir))
    by_key = {record.short_name: record for record in outputs}

    record = by_key["skims_emissions_2030_2"]
    assert record.file_path == str(
        beam_output_dir / "ITERS" / "it.0" / "0.skimsEmissions.parquet"
    )


def test_gather_outputs_prefers_emissions_skims_csv_over_parquet(tmp_path):
    beam_output_dir = tmp_path / "beam-output"
    csv_path = beam_output_dir / "ITERS" / "it.0" / "0.skimsEmissions.csv.gz"
    parquet_path = beam_output_dir / "ITERS" / "it.0" / "0.skimsEmissions.parquet"
    _touch(csv_path)
    _touch(parquet_path)

    runner = BeamRunner("beam_runner", _StubState())
    outputs = runner.gather_outputs(str(beam_output_dir))
    by_key = {record.short_name: record for record in outputs}

    record = by_key["skims_emissions_2030_2"]
    assert record.file_path == str(csv_path)


def test_beam_runner_run_returns_typed_outputs(tmp_path, monkeypatch) -> None:
    state = _StubState()
    runner = BeamRunner("beam_runner", state)
    workspace = _Workspace(tmp_path)
    captured = {}

    def _fake_run(store: RecordStore, _workspace) -> RecordStore:
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
        runner.run(object(), _Workspace(tmp_path))


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
    assert (
        mapping["events_parquet_2030_2_type_PathTraversal"] == str(split_event)
    )
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
