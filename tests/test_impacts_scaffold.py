from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.config.models import ImpactsConfig
from pilates.impacts import ImpactsPostsimRunner, ImpactsPresimRunner
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
    _deep_merge,
    _to_container_path,
    build_pilates_bridge_settings,
    resolve_population_override,
)
from pilates.workflows.stages.impacts_presim import run_impacts_presim_stage
from pilates.workflows.state import WorkflowState
from pilates.workspace import Workspace


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value):
    if isinstance(value, dict):
        return AttrDict({key: _wrap(val) for key, val in value.items()})
    return value


def _make_settings(**impacts_kwargs) -> AttrDict:
    settings = _wrap(
        {
            "infrastructure": {
                "container_manager": "docker",
                "docker_images": {"impacts": "impacts:latest"},
            },
            "run": {
                "region": "seattle",
                "scenario": "baseline",
                "start_year": 2020,
                "output_run_name": "testrun",
                "models": {
                    "land_use": None,
                    "travel": None,
                    "activity_demand": None,
                    "vehicle_ownership": None,
                    "impacts": "impacts",
                },
            },
            "shared": {
                "geography": {
                    "FIPS": {"state": "53", "county": "033"},
                    "local_crs": "EPSG:32148",
                },
            },
            "beam": {
                "local_mutable_data_folder": "beam/input",
                "local_output_folder": "beam/output",
                "router_directory": "beam/router",
            },
        }
    )
    impacts_kwargs.setdefault("local_input_folder", "impacts/input")
    impacts_kwargs.setdefault("local_output_folder", "impacts/output")
    impacts_kwargs.setdefault("seed", 42)
    settings["impacts"] = ImpactsConfig(**impacts_kwargs)
    return settings


def _make_state(settings: AttrDict, forecast_year: int = 2020) -> AttrDict:
    return AttrDict(
        {
            "full_settings": settings,
            "forecast_year": forecast_year,
            "iteration": 0,
            "set_sub_stage_progress": lambda *_args, **_kwargs: None,
        }
    )


def _make_workspace(tmp_path: Path, settings: AttrDict) -> Workspace:
    return Workspace(settings, str(tmp_path), "run")


def test_deep_merge_recurses_nested_dicts() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 30, "e": 40}, "f": 50}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"c": 2, "d": 30, "e": 40}, "f": 50}
    # base is left untouched
    assert base == {"a": 1, "b": {"c": 2, "d": 3}}


def test_to_container_path_translates_relative_to_workspace_root(tmp_path: Path) -> None:
    host_path = str(tmp_path / "impacts" / "output")
    result = _to_container_path(WORKSPACE_CONTAINER_DIR, host_path, str(tmp_path))
    assert result == "/app/workspace/impacts/output"


def test_build_pilates_bridge_settings_presim(tmp_path: Path) -> None:
    settings = _make_settings(overrides={"scenario": "custom-scenario"})
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    bridge = build_pilates_bridge_settings(settings, state, workspace, stage="presim")

    assert bridge["run"] == {
        "region": "seattle",
        "scenario": "baseline",
        "start_year": 2025,
        "output_run_name": "testrun_y2025",
    }
    assert bridge["shared"]["geography"] == {
        "FIPS": {"state": "53", "county": "033"},
        "local_crs": "EPSG:32148",
    }
    assert bridge["beam"]["local_input_folder"] == BEAM_INPUT_CONTAINER_DIR
    assert bridge["beam"]["local_output_folder"] == "/app/workspace/beam/output"
    assert bridge["beam"]["router_directory"] == "beam/router"

    impacts_section = bridge["impacts"]
    assert impacts_section["local_input_folder"] == "/app/workspace/impacts/input"
    assert impacts_section["local_output_folder"] == "/app/workspace/impacts/output"
    assert impacts_section["seed"] == 42
    assert impacts_section["pipeline"]["presim"] == {"activities": True, "fleet": True}
    assert impacts_section["pipeline"]["postsim"] == {
        "emissions": True,
        "inmap": True,
        "aermod": True,
        "exposure": True,
    }
    # overrides are deep-merged into the impacts section
    assert impacts_section["scenario"] == "custom-scenario"
    # presim does not touch population
    assert "population" not in impacts_section


def test_build_pilates_bridge_settings_postsim_without_population_data(
    tmp_path: Path,
) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    bridge = build_pilates_bridge_settings(settings, state, workspace, stage="postsim")

    assert bridge["run"]["output_run_name"] == "testrun_y2025"
    assert "population" not in bridge["impacts"]


def test_build_pilates_bridge_settings_postsim_with_population_fallback(
    tmp_path: Path,
) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_mutable_dir.mkdir(parents=True, exist_ok=True)
    (beam_mutable_dir / "households.csv.gz").write_text("hh", encoding="utf-8")
    (beam_mutable_dir / "persons.csv.gz").write_text("pp", encoding="utf-8")

    bridge = build_pilates_bridge_settings(settings, state, workspace, stage="postsim")

    assert bridge["impacts"]["population"] == {
        "passenger_folder": "/app/workspace/beam/input"
    }


def test_build_pilates_bridge_settings_postsim_population_fallback_disabled(
    tmp_path: Path,
) -> None:
    settings = _make_settings(population_fallback_to_beam_inputs=False)
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_mutable_dir.mkdir(parents=True, exist_ok=True)
    (beam_mutable_dir / "households.csv.gz").write_text("hh", encoding="utf-8")
    (beam_mutable_dir / "persons.csv.gz").write_text("pp", encoding="utf-8")

    bridge = build_pilates_bridge_settings(settings, state, workspace, stage="postsim")

    assert "population" not in bridge["impacts"]


def test_resolve_population_override_returns_empty_when_no_exchange_files(
    tmp_path: Path,
) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    assert resolve_population_override(settings, workspace) == {}


def test_resolve_population_override_returns_container_path_when_present(
    tmp_path: Path,
) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_mutable_dir.mkdir(parents=True, exist_ok=True)
    (beam_mutable_dir / "households.csv.gz").write_text("hh", encoding="utf-8")
    (beam_mutable_dir / "persons.csv.gz").write_text("pp", encoding="utf-8")

    assert resolve_population_override(settings, workspace) == {
        "population": {"passenger_folder": "/app/workspace/beam/input"}
    }


def test_impacts_presim_runner_run(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    calls = []

    def fake_run_container(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        ImpactsPresimRunner, "run_container", staticmethod(fake_run_container)
    )

    runner = ImpactsPresimRunner("impacts_presim", state)
    outputs = runner.run(None, workspace)

    assert len(calls) == 2
    assert "derive_settings_from_pilates" in calls[0]["command"]
    assert calls[1]["command"].startswith("presim --config")

    impacts_output_dir = Path(workspace.get_impacts_output_dir())
    assert outputs.output_dir == impacts_output_dir / "impacts_presim--seattle--testrun_y2025"
    assert outputs.settings_path == Path(workspace.get_impacts_settings_path("presim"))
    assert outputs.bridge_settings_path == Path(
        workspace.get_impacts_bridge_settings_path("presim")
    )
    assert outputs.bridge_settings_path.exists()

    bridge = yaml.safe_load(outputs.bridge_settings_path.read_text(encoding="utf-8"))
    assert bridge["run"]["output_run_name"] == "testrun_y2025"


def test_impacts_postsim_runner_run(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_mutable_dir.mkdir(parents=True, exist_ok=True)
    (beam_mutable_dir / "households.csv.gz").write_text("hh", encoding="utf-8")
    (beam_mutable_dir / "persons.csv.gz").write_text("pp", encoding="utf-8")

    calls = []

    def fake_run_container(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        ImpactsPostsimRunner, "run_container", staticmethod(fake_run_container)
    )

    runner = ImpactsPostsimRunner("impacts_postsim", state)
    outputs = runner.run(None, workspace)

    assert len(calls) == 2
    assert "derive_settings_from_pilates" in calls[0]["command"]
    assert calls[1]["command"].startswith("postsim --config")

    impacts_output_dir = Path(workspace.get_impacts_output_dir())
    expected_output_dir = impacts_output_dir / "impacts-postsim--seattle--testrun_y2025"
    assert outputs.output_dir == expected_output_dir
    assert outputs.pipeline_manifest == expected_output_dir / "pipeline_manifest.yaml"
    assert outputs.postprocess_manifest == expected_output_dir / "postprocess_manifest.yaml"
    assert (
        outputs.exposure_table
        == expected_output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )

    expected_env_path = _to_container_path(
        WORKSPACE_CONTAINER_DIR, str(expected_output_dir), workspace.full_path
    )
    assert calls[1]["environment"] == {"IMPACTS_POSTSIM_OUTPUT_DIR": expected_env_path}

    bridge = yaml.safe_load(outputs.bridge_settings_path.read_text(encoding="utf-8"))
    assert bridge["impacts"]["population"] == {
        "passenger_folder": "/app/workspace/beam/input"
    }


def test_impacts_runner_raises_when_container_fails(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    monkeypatch.setattr(
        ImpactsPresimRunner, "run_container", staticmethod(lambda **_kwargs: False)
    )

    runner = ImpactsPresimRunner("impacts_presim", state)
    try:
        runner.run(None, workspace)
    except RuntimeError as exc:
        assert "derive_settings_from_pilates" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when container run fails")


def test_run_impacts_presim_stage_returns_early_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _make_settings()
    settings["run"]["models"]["impacts"] = None
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    def _fail_execute_step(*_args, **_kwargs):
        raise AssertionError("execute_step should not run when IMPACTS is disabled")

    monkeypatch.setattr(
        "pilates.workflows.stages.impacts_presim.execute_step", _fail_execute_step
    )

    context = SimpleNamespace(settings=settings, state=state, workspace=workspace)

    run_impacts_presim_stage(
        scenario=None,
        coupler=None,
        year=2025,
        context=context,
    )


def test_run_impacts_presim_stage_uses_native_step(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)
    captured = {}
    monkeypatch.setattr(
        "pilates.workflows.stages.impacts_presim.execute_step",
        lambda **kwargs: (None, captured.update(kwargs)),
    )

    run_impacts_presim_stage(
        scenario="scenario",
        coupler=None,
        year=2025,
        context=SimpleNamespace(settings=settings, state=state, workspace=workspace),
    )

    assert captured["definition"].name == "impacts_presim"
    assert captured["stage"] == "impacts_presim"


def test_workflow_state_enables_terminal_postprocessing_for_impacts() -> None:
    state = WorkflowState(
        start_year=2020,
        end_year=2021,
        travel_model_freq=1,
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=False,
        traffic_assignment_enabled=False,
        year=2020,
        major_stage=None,
        inner_iter=0,
        sub_stage=None,
        file_loc=None,
        asim_compiled=False,
        full_settings=_make_settings(),
    )
    assert WorkflowState.Stage.postprocessing in state.enabled_stages
