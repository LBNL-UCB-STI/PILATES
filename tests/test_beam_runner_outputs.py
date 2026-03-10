from pathlib import Path

import pytest

from pilates.beam.outputs import BeamPreprocessOutputs, BeamRunOutputs
from pilates.beam.runner import BeamRunner
from pilates.generic.records import FileRecord, RecordStore


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
