from types import SimpleNamespace

from pilates.beam.preprocessor import BeamPreprocessor


def _make_preprocessor(sample: float = 0.25) -> BeamPreprocessor:
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(
            local_mutable_data_folder="beam/input",
            config="beam.conf",
            sample=sample,
            replanning_portion=0.2,
            max_plans_memory=5,
        ),
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=0,
    )
    return BeamPreprocessor("beam", state)


def test_update_beam_config_skips_rewrite_when_value_unchanged(tmp_path):
    preprocessor = _make_preprocessor(sample=0.25)
    config_path = (
        tmp_path / "beam" / "input" / "seattle" / "beam.conf"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 0.25\n",
        encoding="utf-8",
    )
    before_stat = config_path.stat()

    preprocessor._update_beam_config(
        "beam_sample",
        value_override=0.25,
        base_path=str(tmp_path),
    )

    after_stat = config_path.stat()
    assert config_path.read_text(encoding="utf-8").strip().endswith("= 0.25")
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_update_beam_config_rewrites_when_value_changes(tmp_path):
    preprocessor = _make_preprocessor(sample=0.50)
    config_path = (
        tmp_path / "beam" / "input" / "seattle" / "beam.conf"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 0.25\n",
        encoding="utf-8",
    )

    preprocessor._update_beam_config(
        "beam_sample",
        value_override=0.5,
        base_path=str(tmp_path),
    )

    contents = config_path.read_text(encoding="utf-8")
    assert "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 0.5" in contents
