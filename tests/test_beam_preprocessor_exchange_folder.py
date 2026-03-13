from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.generic.records import RecordStore
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)


def _make_preprocessor(
    region: str = "sfbay",
    scenario_folder: str = "urbansim",
    config: str = "sfbay-pilates-base-omx.conf",
    *,
    activity_demand_enabled: bool = False,
):
    settings = SimpleNamespace(
        run=SimpleNamespace(region=region),
        beam=SimpleNamespace(
            scenario_folder=scenario_folder,
            config=config,
            discard_plans_every_year=False,
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=activity_demand_enabled,
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=0,
        run_info_path=None,
    )
    state.set_sub_stage_progress = lambda _value: None
    return BeamPreprocessor("beam", state)


def _make_workspace(tmp_path: Path):
    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")
    workspace.full_path = str(tmp_path)
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


def test_beam_preprocess_registers_existing_default_scenario_inputs(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.parquet").write_text(stem, encoding="utf-8")

    monkeypatch.setattr(preprocessor, "_update_beam_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    outputs = preprocessor.preprocess(workspace)

    assert outputs.prepared_inputs == {
        BEAM_PLANS_IN: scenario_dir / "plans.parquet",
        BEAM_HOUSEHOLDS_IN: scenario_dir / "households.parquet",
        BEAM_PERSONS_IN: scenario_dir / "persons.parquet",
    }


def test_beam_preprocess_fails_early_when_default_scenario_inputs_missing(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "plans.parquet").write_text("plans", encoding="utf-8")
    (scenario_dir / "households.parquet").write_text("households", encoding="utf-8")

    monkeypatch.setattr(preprocessor, "_update_beam_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    with pytest.raises(FileNotFoundError, match="persons.parquet"):
        preprocessor.preprocess(workspace)


def test_beam_preprocess_does_not_fallback_to_defaults_when_activitysim_enabled(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(
        scenario_folder="urbansim",
        activity_demand_enabled=True,
    )
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.parquet").write_text(stem, encoding="utf-8")

    monkeypatch.setattr(preprocessor, "_update_beam_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    with pytest.raises(RuntimeError, match="expected ActivitySim to stage the canonical"):
        preprocessor.preprocess(workspace)
