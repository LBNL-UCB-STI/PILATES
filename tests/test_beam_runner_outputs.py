from pathlib import Path

from pilates.beam.runner import BeamRunner


class _StubState:
    forecast_year = 2030
    iteration = 2


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
