import pytest
from unittest.mock import MagicMock

from pilates.utils.consist_config import (
    build_activitysim_identity_inputs,
    build_beam_identity_inputs,
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)


def test_build_scenario_consist_kwargs_includes_run_facet():
    settings = MagicMock()
    settings.get_initialization_signature.return_value = {"hello": "world"}
    settings.run = MagicMock()
    settings.run.to_consist_facet.return_value = {"region": "test", "start_year": 2010}

    kwargs = build_scenario_consist_kwargs(settings)
    assert kwargs["config"] == {"hello": "world"}
    assert kwargs["facet"]["run"] == {"region": "test", "start_year": 2010}
    assert kwargs["facet_schema_version"] == "pilates_scenario_v1"


def test_build_activitysim_identity_inputs_requires_dir(tmp_path):
    settings = MagicMock()
    settings.activitysim = MagicMock()
    settings.activitysim.local_mutable_configs_folder = "activitysim/configs"

    (tmp_path / "activitysim" / "configs").mkdir(parents=True)
    identity_inputs = build_activitysim_identity_inputs(settings, str(tmp_path))
    assert len(identity_inputs) == 1
    assert identity_inputs[0][0] == "asim_mutable_configs"
    assert identity_inputs[0][1] == (tmp_path / "activitysim" / "configs")


def test_build_activitysim_identity_inputs_missing_dir_raises(tmp_path):
    settings = MagicMock()
    settings.activitysim = MagicMock()
    settings.activitysim.local_mutable_configs_folder = "activitysim/configs"

    with pytest.raises(FileNotFoundError):
        build_activitysim_identity_inputs(settings, str(tmp_path))


def test_build_beam_identity_inputs_discovers_conf_files(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"

    root = tmp_path / "beam" / "input"
    (root / "sub").mkdir(parents=True)
    (root / "a.conf").write_text("a=1")
    (root / "sub" / "b.conf").write_text("b=2")
    (root / "sub" / "ignore.txt").write_text("nope")

    identity_inputs = build_beam_identity_inputs(settings, str(tmp_path))
    labels = [lbl for (lbl, _) in identity_inputs]
    paths = [p for (_, p) in identity_inputs]

    assert labels == ["beam_conf/a.conf", "beam_conf/sub/b.conf"]
    assert paths == [root / "a.conf", root / "sub" / "b.conf"]


def test_build_step_consist_kwargs_beam_includes_identity_inputs(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"
    settings.beam.config = "main.conf"
    settings.beam.sample = 1.0
    settings.beam.replanning_portion = 0.4
    settings.beam.memory = "180g"
    settings.beam.discard_plans_every_year = False
    settings.beam.max_plans_memory = 5
    settings.beam.router_directory = "r5"
    settings.beam.scenario_folder = "scenario"
    settings.beam.to_consist_facet.return_value = {"sample": 1.0}

    root = tmp_path / "beam" / "input"
    root.mkdir(parents=True)
    (root / "main.conf").write_text("x=1")

    kwargs = build_step_consist_kwargs("beam", settings, workspace_path=str(tmp_path))
    assert kwargs["facet_schema_version"] == "beam_v1"
    assert kwargs["facet"] == {"sample": 1.0}
    assert kwargs["identity_inputs"][0][0] == "beam_conf/main.conf"


def test_build_beam_identity_inputs_missing_dir_returns_empty(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"

    assert build_beam_identity_inputs(settings, str(tmp_path)) == []
