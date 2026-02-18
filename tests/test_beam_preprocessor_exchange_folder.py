from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pilates.beam.preprocessor import BeamPreprocessor


def _make_preprocessor(
    region: str = "sfbay",
    scenario_folder: str = "urbansim",
    config: str = "sfbay-pilates-base-omx.conf",
):
    settings = SimpleNamespace(
        run=SimpleNamespace(region=region),
        beam=SimpleNamespace(
            scenario_folder=scenario_folder,
            config=config,
        ),
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=0,
    )
    return BeamPreprocessor("beam", state)


def _make_workspace(tmp_path: Path):
    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")
    return workspace


def test_resolve_beam_exchange_scenario_folder_reads_config_folder(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    config_path.write_text(
        'beam.exchange.scenario {\n'
        '  source = "urbansim_v2"\n'
        '  folder = ${beam.inputDirectory}"/urbansim/2018"\n'
        '}\n',
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim" / "2018")


def test_resolve_beam_exchange_scenario_folder_falls_back_on_unparseable_folder(tmp_path):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    config_path.write_text(
        'beam.exchange.scenario {\n'
        '  # missing ${beam.inputDirectory} placeholder on purpose\n'
        '  folder = "/app/input/sfbay/urbansim/2018"\n'
        '}\n',
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim")


def test_resolve_beam_exchange_scenario_folder_falls_back_when_config_missing(tmp_path):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim")
