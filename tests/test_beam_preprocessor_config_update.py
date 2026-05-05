from types import SimpleNamespace

import pytest

import pilates.beam.config_hocon as beam_config_hocon
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
    assert "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 0.25" in config_path.read_text(
        encoding="utf-8"
    )
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
    assert "# BEGIN PILATES managed overrides" in contents


def test_update_beam_config_updates_zone_settings_semantically(tmp_path):
    preprocessor = _make_preprocessor(sample=0.50)
    config_path = tmp_path / "beam" / "input" / "seattle" / "beam.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        'beam.inputDirectory = "beam/input/seattle"\n',
        encoding="utf-8",
    )

    preprocessor._update_beam_config(
        "beam.agentsim.taz.filePath",
        value_override='${beam.inputDirectory}/shape/canonical_zones_sorted.geojson',
        base_path=str(tmp_path),
    )
    preprocessor._update_beam_config(
        "beam.agentsim.taz.tazIdFieldName",
        value_override="zone_id",
        base_path=str(tmp_path),
    )

    resolved_file_path = beam_config_hocon.resolve_beam_config_value(
        config_path,
        key="beam.agentsim.taz.filePath",
        env_overrides=beam_config_hocon.beam_config_env_overrides(
            preprocessor.settings,
            workspace_path=str(tmp_path),
        ),
    )
    resolved_id_field = beam_config_hocon.resolve_beam_config_value(
        config_path,
        key="beam.agentsim.taz.tazIdFieldName",
        env_overrides=beam_config_hocon.beam_config_env_overrides(
            preprocessor.settings,
            workspace_path=str(tmp_path),
        ),
    )

    assert str(resolved_file_path).endswith("/beam/input/seattle/shape/canonical_zones_sorted.geojson")
    assert resolved_id_field == "zone_id"


def test_update_beam_config_raises_clearly_when_pyhocon_missing(tmp_path, monkeypatch):
    preprocessor = _make_preprocessor(sample=0.50)
    config_path = tmp_path / "beam" / "input" / "seattle" / "beam.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 0.25\n",
        encoding="utf-8",
    )

    original = beam_config_hocon._require_pyhocon

    def _raise():
        raise beam_config_hocon.BeamConfigHoconError(
            "pyhocon is required for staged BEAM config mutation and resolution."
        )

    monkeypatch.setattr(beam_config_hocon, "_require_pyhocon", _raise)

    with pytest.raises(beam_config_hocon.BeamConfigHoconError, match="pyhocon is required"):
        preprocessor._update_beam_config(
            "beam_sample",
            value_override=0.5,
            base_path=str(tmp_path),
        )

    monkeypatch.setattr(beam_config_hocon, "_require_pyhocon", original)
