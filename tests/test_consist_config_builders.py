"""Tests for Consist config builder registry and step kwargs."""

from unittest.mock import MagicMock

import pytest

from pilates.utils.consist_config import _CONFIG_BUILDERS, build_step_consist_kwargs


def _make_settings():
    settings = MagicMock()
    settings.get_initialization_signature.return_value = {"init": "yes"}

    run_cfg = MagicMock()
    run_cfg.to_consist_facet.return_value = {"region": "test", "start_year": 2020}
    settings.run = run_cfg

    activitysim = MagicMock()
    activitysim.household_sample_size = 1
    activitysim.chunk_size = 100
    activitysim.num_processes = 2
    activitysim.file_format = "csv"
    activitysim.warm_start_activities = False
    activitysim.replan_iters = 0
    activitysim.replan_hh_samp_size = 0
    activitysim.replan_after = 0
    activitysim.random_seed = 0
    activitysim.database = None
    activitysim.local_mutable_configs_folder = "activitysim/configs"
    activitysim.to_consist_facet.return_value = {"household_sample_size": 1}
    settings.activitysim = activitysim

    beam = MagicMock()
    beam.config = "main.conf"
    beam.sample = 1.0
    beam.replanning_portion = 0.4
    beam.memory = "16g"
    beam.discard_plans_every_year = False
    beam.max_plans_memory = 5
    beam.router_directory = "r5"
    beam.scenario_folder = "scenario"
    beam.local_mutable_data_folder = "beam/input"
    beam.to_consist_facet.return_value = {"sample": 1.0}
    settings.beam = beam

    urbansim = MagicMock()
    urbansim.command_template = "run"
    urbansim.input_file_template = "input_{region_id}.h5"
    urbansim.input_file_template_year = "input_{year}.h5"
    urbansim.output_file_template = "output_{year}.h5"
    urbansim.region_id = "test"
    urbansim.model_dump.return_value = {"region_id": "test"}
    settings.urbansim = urbansim

    atlas = MagicMock()
    atlas.model_dump.return_value = {"max_retries": 1}
    settings.atlas = atlas

    postprocessing = MagicMock()
    postprocessing.model_dump.return_value = {"enabled": True}
    settings.postprocessing = postprocessing

    return settings


def test_all_models_registered():
    expected = {"activitysim", "beam", "urbansim", "atlas", "postprocessing"}
    assert set(_CONFIG_BUILDERS.keys()) == expected


def test_builders_expose_required_methods():
    required = {
        "build_identity_config",
        "build_facet",
        "build_hash_inputs",
        "get_facet_schema_version",
        "requires_workspace_path",
    }
    for name, builder in _CONFIG_BUILDERS.items():
        missing = [attr for attr in required if not hasattr(builder, attr)]
        assert not missing, f"Builder {name} missing: {missing}"


def test_workspace_path_requirement_flags():
    assert _CONFIG_BUILDERS["activitysim"].requires_workspace_path is True
    assert _CONFIG_BUILDERS["beam"].requires_workspace_path is True
    assert _CONFIG_BUILDERS["urbansim"].requires_workspace_path is False
    assert _CONFIG_BUILDERS["atlas"].requires_workspace_path is False
    assert _CONFIG_BUILDERS["postprocessing"].requires_workspace_path is False


def test_activitysim_requires_workspace_path():
    settings = _make_settings()
    with pytest.raises(ValueError, match="workspace_path is required"):
        build_step_consist_kwargs("activitysim_run", settings, workspace_path=None)


def test_beam_requires_workspace_path():
    settings = _make_settings()
    with pytest.raises(ValueError, match="workspace_path is required"):
        build_step_consist_kwargs("beam_run", settings, workspace_path=None)


def test_urbansim_no_workspace_path_needed():
    settings = _make_settings()
    result = build_step_consist_kwargs("urbansim_run", settings)
    assert "config" in result
    assert "facet" in result


def test_hash_inputs_included_for_activitysim(tmp_path):
    settings = _make_settings()
    asim_dir = tmp_path / "activitysim" / "configs"
    asim_dir.mkdir(parents=True)

    result = build_step_consist_kwargs(
        "activitysim_run",
        settings,
        workspace_path=str(tmp_path),
    )
    assert "hash_inputs" in result
    assert len(result["hash_inputs"]) > 0


def test_no_hash_inputs_for_urbansim():
    settings = _make_settings()
    result = build_step_consist_kwargs("urbansim_run", settings)
    assert "hash_inputs" not in result


def test_initialization_model_special_case():
    settings = _make_settings()
    result = build_step_consist_kwargs("initialization", settings)
    assert "config" in result
    assert result["facet_schema_version"] == "initialization_v1"


def test_facet_schema_version_routing(tmp_path):
    settings = _make_settings()

    asim_dir = tmp_path / "activitysim" / "configs"
    asim_dir.mkdir(parents=True)
    beam_root = tmp_path / "beam" / "input"
    beam_root.mkdir(parents=True)
    (beam_root / "main.conf").write_text("x=1")

    result = build_step_consist_kwargs(
        "activitysim_preprocess",
        settings,
        workspace_path=str(tmp_path),
    )
    assert result["facet_schema_version"] == "activitysim_preprocess_v1"

    result = build_step_consist_kwargs(
        "beam_run",
        settings,
        workspace_path=str(tmp_path),
    )
    assert result["facet_schema_version"] == "beam_run_v1"

    result = build_step_consist_kwargs("urbansim_run", settings)
    assert result["facet_schema_version"] == "urbansim_run_v1"
