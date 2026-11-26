from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd
import yaml
from consist import Artifact, Tracker

from pilates.config.models import ImpactsConfig, load_config
from pilates.impacts import ImpactsPostsimRunner, ImpactsPresimRunner
from pilates.impacts.population import write_exposure_population
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    INPUT_CACHE_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
    _deep_merge,
    _to_container_path,
    build_pilates_bridge_settings,
    presim_fleet_manifest_path,
    resolve_shared_input_cache_folder,
)
from pilates.workflows.artifact_keys import (
    BEAM_EMISSIONS_SKIMS,
    BEAM_NETWORK_FINAL,
    BEAM_R5_OSM_FILE,
    IMPACTS_FLEET_MANIFEST,
    IMPACTS_EXPOSURE_PERSONS,
    IMPACTS_EXPOSURE_SOURCE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows import orchestrator
from pilates.workflows.dry_run import plan_pilates
from pilates.workflows.sim_state import LandUseState, SimState
from pilates.workflows.steps import beam_run, impacts_postsim, impacts_presim
from pilates.workflows.steps.impacts import exposure_population_paths
from pilates.workspace import Workspace
from scripts.run_impacts_container_smoke import FIXTURE


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
                "output_directory": "/tmp/pilates-test-output",
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
    impacts_kwargs.setdefault("pipeline_postsim", {"exposure": False})
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


def test_generated_fleet_postsim_uses_current_year_beam_and_forecast_fleet(
    tmp_path: Path,
) -> None:
    settings = _make_settings(use_generated_fleet=True, pipeline_presim={"fleet": True})
    state = _make_state(settings, forecast_year=2021)
    state.current_year = 2020
    workspace = _make_workspace(tmp_path, settings)
    manifest = presim_fleet_manifest_path(settings, workspace, 2021)
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    beam_root = Path(workspace.get_impacts_input_dir()) / "beam-output" / "2021"
    presim = build_pilates_bridge_settings(settings, state, workspace, stage="presim")
    postsim = build_pilates_bridge_settings(
        settings, state, workspace, stage="postsim", beam_output_dir=beam_root
    )
    assert presim["impacts"]["local_input_folder"].endswith("/presim/testrun_y2021")
    assert postsim["impacts"]["presim_fleet_manifest"].endswith(
        "/impacts_presim--seattle--testrun_y2021/activities/fleet_manifest.yaml"
    )
    assert postsim["beam"]["local_output_folder"] == (
        "/app/workspace/impacts/input/beam-output/2021"
    )


def test_to_container_path_translates_relative_to_workspace_root(
    tmp_path: Path,
) -> None:
    host_path = str(tmp_path / "impacts" / "output")
    result = _to_container_path(WORKSPACE_CONTAINER_DIR, host_path, str(tmp_path))
    assert result == "/app/workspace/impacts/output"


def test_build_pilates_bridge_settings_presim(tmp_path: Path) -> None:
    settings = _make_settings(
        overrides={"scenario": "custom-scenario"}, pipeline_postsim={"exposure": True}
    )
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


def test_exposure_binds_baseline_population_before_beam(tmp_path, monkeypatch):
    settings = _make_settings(
        pipeline_postsim={"exposure": True},
        overrides={
            "population": {"passenger_folder": "urbansim/atlas-2019"},
            "exposure": {
                "persons_file": "/app/beam_input/urbansim/atlas-2019/persons.parquet",
                "primary_pm25_integration_strategy": "impute_inmap_primary_in_aermod_domain",
            },
        },
    )
    settings.run.models["travel"] = "beam"
    settings.run.supply_demand_iters = 1
    workspace = _make_workspace(tmp_path / "workspace", settings)
    state = _make_state(settings, forecast_year=2018)
    state.is_restart_run = False
    beam = Path(workspace.get_beam_mutable_data_dir())
    beam.mkdir(parents=True)
    for stem in ("persons", "households"):
        pd.DataFrame({f"{stem[:-1]}_id": [999]}).to_parquet(beam / f"{stem}.parquet")

    class ReachedPresim(Exception):
        pass

    real_execute = orchestrator.execute_step
    post_inputs = {}

    def execute_until_presim(**kwargs):
        if kwargs["definition"] is impacts_presim:
            post_inputs.update(kwargs["sim_state"].extras)
            raise ReachedPresim
        return real_execute(**kwargs)

    monkeypatch.setattr(orchestrator, "execute_step", execute_until_presim)
    tracker = Tracker(
        run_dir=Path(workspace.full_path), db_path=str(tmp_path / "provenance.duckdb")
    )
    try:
        with tracker.scenario("exposure-flow") as scenario:
            run = dict(
                scenario=scenario,
                settings=settings,
                state=state,
                workspace=workspace,
                sim_state=SimState(),
                year=2018,
            )
            with pytest.raises(
                ValueError, match="requires the baseline UrbanSim HDF5"
            ):
                orchestrator._run_supply_demand_loop(**run)
            assert not post_inputs
            with pytest.raises(ValueError, match="prepared UrbanSim population"):
                build_pilates_bridge_settings(
                    settings, state, workspace, stage="postsim"
                )

            baseline_h5 = FIXTURE / "population/model_data_2018.h5"
            baseline_artifact = scenario.add_input(
                baseline_h5, key=USIM_POPULATION_SOURCE_H5
            )
            run["sim_state"] = SimState(
                land_use=LandUseState(population_source=baseline_artifact)
            )
            state.forecast_year = 2017
            with pytest.raises(ValueError, match="missing exact-year tables"):
                orchestrator._run_supply_demand_loop(**run)
            assert not post_inputs
            state.forecast_year = 2018
            with pytest.raises(ReachedPresim):
                orchestrator._run_supply_demand_loop(**run)

            files = exposure_population_paths(
                settings=settings, state=state, workspace=workspace
            )
            manifest = yaml.safe_load(files[IMPACTS_EXPOSURE_SOURCE].read_text())
            assert manifest["forecast_year"] == 2018
            assert manifest["source_artifact_key"] == USIM_POPULATION_SOURCE_H5
            assert manifest["source_type"] == "baseline_urbansim_input"
            people = pd.read_parquet(files[IMPACTS_EXPOSURE_PERSONS])
            assert len(people) == manifest["person_count"] == 37
            assert 37 in set(people.person_id)
            assert list(pd.read_parquet(beam / "persons.parquet").person_id) == [999]
            lineage = tracker.get_run_inputs(
                post_inputs[IMPACTS_EXPOSURE_PERSONS].run_id
            )
            assert (
                lineage[USIM_POPULATION_SOURCE_H5].meta["source_artifact_id"]
                == manifest["source_artifact_id"]
            )

            post_inputs.update(
                {
                    key: Artifact(key=key, uri=f"{key}.parquet", driver="parquet")
                    for key in (BEAM_EMISSIONS_SKIMS, BEAM_NETWORK_FINAL)
                }
            )
            bound = impacts_postsim.resolve_inputs(
                settings=settings,
                state=state,
                workspace=workspace,
                sim_state=SimState(extras=post_inputs),
            )
            assert (
                bound.inputs[IMPACTS_EXPOSURE_PERSONS]
                == post_inputs[IMPACTS_EXPOSURE_PERSONS]
            )
            bridge = build_pilates_bridge_settings(
                settings,
                state,
                workspace,
                stage="postsim",
                exposure_population=tuple(files.values()),
            )
            assert bridge["impacts"]["population"]["passenger_folder"] == (
                "urbansim/atlas-2019"
            )
            root = "/app/workspace/impacts/exposure-population/urbansim-forecast-2018"
            assert bridge["impacts"]["exposure"] == {
                "persons_file": f"{root}/persons.parquet",
                "households_file": f"{root}/households.parquet",
                "source_manifest": f"{root}/population_source.yaml",
                "primary_pm25_integration_strategy": "impute_inmap_primary_in_aermod_domain",
            }
            files[IMPACTS_EXPOSURE_SOURCE].unlink()
            with pytest.raises(ValueError, match="prepared UrbanSim population"):
                build_pilates_bridge_settings(
                    settings,
                    state,
                    workspace,
                    stage="postsim",
                    exposure_population=tuple(files.values()),
                )
            post_inputs[IMPACTS_EXPOSURE_PERSONS] = Artifact(
                key=IMPACTS_EXPOSURE_PERSONS,
                uri="other/persons.parquet",
                driver="parquet",
                run_id="other",
            )
            with pytest.raises(ValueError, match="same forecast preparation"):
                impacts_postsim.resolve_inputs(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    sim_state=SimState(extras=post_inputs),
                )
    finally:
        tracker.close()


def test_exposure_export_retains_people_and_households_without_home_points(tmp_path):
    forecast = tmp_path / "model_data_2018.h5"
    forecast.write_bytes((FIXTURE / "population/model_data_2018.h5").read_bytes())
    with pd.HDFStore(forecast, mode="a") as store:
        persons = store["/2018/persons"]
        extra_household = persons.loc[
            persons["person_id"].eq(37), "household_id"
        ].item()
        households = store["/2018/households"]
        households.loc[households["household_id"].eq(extra_household), "block_id"] = (
            "missing"
        )
        store.put("/2018/households", households, format="table")
    destination = tmp_path / "exposure"
    write_exposure_population(
        forecast,
        destination,
        year=2018,
        source={
            "source_run_id": "fixture-run",
            "source_artifact_id": "fixture-artifact",
            "source_artifact_key": USIM_POPULATION_SOURCE_H5,
            "source_type": "baseline_urbansim_input",
        },
    )
    people = pd.read_parquet(destination / "persons.parquet")
    manifest = yaml.safe_load((destination / "population_source.yaml").read_text())
    assert len(people) == manifest["person_count"] == 37
    assert len(pd.read_parquet(destination / "households.parquet")) == 36
    assert (
        people["home_x"].isna().sum()
        == manifest["persons_missing_home_coordinates"]
        == 2
    )
    assert manifest["households_missing_home_coordinates"] == 1


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
    cache_folder = resolve_shared_input_cache_folder(settings)
    assert calls[0]["volumes"][str(cache_folder)] == {
        "bind": INPUT_CACHE_CONTAINER_DIR,
        "mode": "rw",
    }
    assert calls[1]["volumes"][str(cache_folder)] == {
        "bind": INPUT_CACHE_CONTAINER_DIR,
        "mode": "rw",
    }

    impacts_output_dir = Path(workspace.get_impacts_output_dir())
    assert (
        outputs.output_dir
        == impacts_output_dir / "impacts_presim--seattle--testrun_y2025"
    )
    assert outputs.settings_path == Path(workspace.get_impacts_settings_path("presim"))
    assert outputs.bridge_settings_path == Path(
        workspace.get_impacts_bridge_settings_path("presim")
    )
    assert outputs.bridge_settings_path.exists()

    bridge = yaml.safe_load(outputs.bridge_settings_path.read_text(encoding="utf-8"))
    assert bridge["run"]["output_run_name"] == "testrun_y2025"
    assert bridge["impacts"]["shared_input_cache_dir"] == INPUT_CACHE_CONTAINER_DIR


def test_impacts_postsim_runner_run(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings()
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

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
    assert (
        outputs.postprocess_manifest
        == expected_output_dir / "postprocess_manifest.yaml"
    )
    assert (
        outputs.exposure_table
        == expected_output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )

    expected_env_path = _to_container_path(
        WORKSPACE_CONTAINER_DIR, str(expected_output_dir), workspace.full_path
    )
    assert calls[1]["environment"] == {"IMPACTS_POSTSIM_OUTPUT_DIR": expected_env_path}

    bridge = yaml.safe_load(outputs.bridge_settings_path.read_text(encoding="utf-8"))
    assert "population" not in bridge["impacts"]


def test_impacts_postsim_mounts_isrm_dataset_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    inmap_data = tmp_path / "inmap"
    inmap_data.mkdir()
    (inmap_data / "isrm.zarr").mkdir()
    cache_folder = tmp_path / "shared-isrm-cache"
    settings = _make_settings(
        shared_input_cache_folder=str(cache_folder),
        local_inmap_data_folder=str(inmap_data),
        overrides={"dispersions": {"inmap": {"isrm_zarr": "/app/inmap/isrm.zarr"}}},
    )
    workspace = _make_workspace(tmp_path / "workspace", settings)
    state = _make_state(settings, forecast_year=2025)
    calls = []
    monkeypatch.setattr(
        ImpactsPostsimRunner,
        "run_container",
        staticmethod(lambda **kwargs: calls.append(kwargs) or True),
    )

    ImpactsPostsimRunner("impacts_postsim", state).run(None, workspace)

    expected_mount = {
        "bind": "/app/inmap",
        "mode": "ro",
    }
    assert calls[0]["volumes"][str(inmap_data.resolve())] == expected_mount
    assert calls[1]["volumes"][str(inmap_data.resolve())] == expected_mount
    assert calls[0]["volumes"][str(cache_folder.resolve())] == {
        "bind": INPUT_CACHE_CONTAINER_DIR,
        "mode": "rw",
    }
    assert calls[1]["volumes"][str(cache_folder.resolve())] == {
        "bind": INPUT_CACHE_CONTAINER_DIR,
        "mode": "rw",
    }
    bridge = yaml.safe_load(
        Path(workspace.get_impacts_bridge_settings_path("postsim")).read_text(
            encoding="utf-8"
        )
    )
    assert (
        bridge["impacts"]["dispersions"]["inmap"]["isrm_zarr"] == "/app/inmap/isrm.zarr"
    )
    assert bridge["impacts"]["shared_input_cache_dir"] == INPUT_CACHE_CONTAINER_DIR


def test_impacts_inmap_host_folder_expands_environment_variables(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TEST_INMAP_DATA_DIR", str(tmp_path / "inmap"))
    settings = _make_settings(local_inmap_data_folder="$TEST_INMAP_DATA_DIR")

    assert settings.impacts.local_inmap_data_folder == str(tmp_path / "inmap")


def test_impacts_postsim_requires_host_folder_for_container_inmap_path(
    tmp_path: Path,
) -> None:
    settings = _make_settings(
        overrides={"dispersions": {"inmap": {"isrm_zarr": "/app/inmap/isrm.zarr"}}}
    )
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings, forecast_year=2025)

    try:
        ImpactsPostsimRunner("impacts_postsim", state).run(None, workspace)
    except ValueError as exc:
        assert "local_inmap_data_folder" in str(exc)
    else:
        raise AssertionError("Expected an error without the host InMAP directory")


def test_default_input_cache_is_shared_outside_run_workspace(tmp_path: Path) -> None:
    settings = _make_settings()
    settings.run.output_directory = str(tmp_path / "archive")

    assert resolve_shared_input_cache_folder(settings) == (
        tmp_path / "archive" / ".pilates-cache" / "impacts"
    )


def test_impacts_runner_raises_when_container_fails(
    tmp_path: Path, monkeypatch
) -> None:
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


def test_impacts_dry_run_tracks_fleet_and_final_beam_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACTS_INMAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BEAM_MEMORY", "6g")
    monkeypatch.setenv("BEAM_EXTRA_JVM_ARGS", "")
    settings = load_config(
        Path(__file__).resolve().parents[1]
        / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml"
    )
    settings.impacts.use_generated_fleet = True
    settings.impacts.use_beam_osm_artifact = True
    settings.postprocessing = None
    plan = plan_pilates(settings)

    assert [step.step_name for step in plan.steps] == [
        "impacts_exposure_population",
        "impacts_presim",
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "impacts_postsim",
    ]
    population, presim, _, beam, _, postsim = plan.steps
    assert postsim.inputs[IMPACTS_EXPOSURE_PERSONS] == population.step_id
    assert beam.inputs[IMPACTS_FLEET_MANIFEST] == presim.step_id
    assert postsim.inputs[IMPACTS_FLEET_MANIFEST] == presim.step_id
    assert postsim.inputs[BEAM_EMISSIONS_SKIMS] == beam.step_id
    assert postsim.inputs[BEAM_NETWORK_FINAL] == beam.step_id
    assert postsim.inputs[BEAM_R5_OSM_FILE] == beam.step_id
    assert "impacts_postsim_dir" in postsim.outputs

    settings.impacts.pipeline_postsim.exposure = False
    with pytest.raises(RuntimeError, match=BEAM_EMISSIONS_SKIMS):
        inputs = impacts_postsim.resolve_inputs(
            settings=settings,
            state=_make_state(settings, forecast_year=2018),
            workspace=_make_workspace(tmp_path, settings),
            sim_state=SimState(),
        )
        inputs.require_complete(impacts_postsim.name)


def test_impacts_binds_selected_osm_by_beam_run_and_iteration(tmp_path):
    settings = _make_settings(use_beam_osm_artifact=True)
    settings.beam.artifact_formats = AttrDict(activitysim_skims="zarr")
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings)
    roles = (BEAM_EMISSIONS_SKIMS, BEAM_NETWORK_FINAL, BEAM_R5_OSM_FILE)
    runs = []
    for iteration in range(2):
        state.iteration = iteration
        paths = beam_run.output_paths(
            settings=settings, state=state, workspace=workspace
        )
        artifacts = {
            role: Artifact(
                key=f"{role}_2020_{iteration}",
                uri=str(paths[f"{role}_2020_{iteration}"]),
                driver="file",
                run_id=f"beam-{iteration}",
            )
            for role in roles
        }
        runs.append(artifacts)
        resolve = dict(settings=settings, state=state, workspace=workspace)
        step_inputs = impacts_postsim.resolve_inputs(
            **resolve, sim_state=SimState(extras=artifacts)
        )
        osm = step_inputs.inputs[BEAM_R5_OSM_FILE]
        path = step_inputs.logical_destinations[BEAM_R5_OSM_FILE]
        assert str(path) == osm.uri
        path.parent.mkdir(parents=True)
        path.write_text(f"osm for iteration {iteration}")
        options = impacts_postsim.execution_options(step_inputs=step_inputs)
        bridge = build_pilates_bridge_settings(
            settings,
            state,
            workspace,
            stage="postsim",
            beam_osm_file=path,
            **options.runtime_kwargs,
        )
        assert bridge["impacts"]["emissions"] == {
            "osm_network_file": (
                f"/app/workspace/beam/output/seattle/year-2020-iteration-{iteration}/"
                f".pilates-consist-outputs/beam_run/beam_r5_osm_file_2020_{iteration}.osm.pbf"
            ),
            "osm_network_artifact_id": str(osm.id),
        }

    for wrong_osm in (
        runs[0][BEAM_R5_OSM_FILE],
        Artifact(
            key=f"{BEAM_R5_OSM_FILE}_2020_1",
            uri=runs[0][BEAM_R5_OSM_FILE].uri,
            driver="file",
            run_id="beam-0",
        ),
    ):
        with pytest.raises(ValueError, match="same BEAM run"):
            impacts_postsim.resolve_inputs(
                **resolve,
                sim_state=SimState(extras={**runs[1], BEAM_R5_OSM_FILE: wrong_osm}),
            )

    with pytest.raises(RuntimeError, match=BEAM_R5_OSM_FILE):
        inputs = impacts_postsim.resolve_inputs(
            **resolve,
            sim_state=SimState(extras={key: runs[1][key] for key in roles[:2]}),
        )
        inputs.require_complete(impacts_postsim.name)

    settings.beam.local_output_folder = str(tmp_path / "external-beam-output")
    inputs = impacts_postsim.resolve_inputs(
        **resolve, sim_state=SimState(extras=runs[1])
    )
    assert inputs.logical_destinations[BEAM_R5_OSM_FILE] == (
        Path(workspace.get_impacts_input_dir())
        / "beam-output/2020/iteration-1/beam_r5_osm_file_2020_1.osm.pbf"
    )

    settings.impacts.use_beam_osm_artifact = False
    inputs = impacts_postsim.resolve_inputs(
        **resolve, sim_state=SimState(extras=runs[1])
    )
    inputs.require_complete(impacts_postsim.name)
    assert BEAM_R5_OSM_FILE not in inputs.inputs


def test_impacts_resume_requests_archived_fleet_and_beam_outputs(tmp_path):
    settings = _make_settings(use_generated_fleet=True)
    workspace = _make_workspace(tmp_path, settings)
    state = _make_state(settings)
    state.is_restart_run = False
    for step in (impacts_presim, beam_run, impacts_postsim):
        options = step.cache_options(
            settings=settings, state=state, workspace=workspace
        )
        assert options.cache_mode == "overwrite"
    state.is_restart_run = True
    for step in (impacts_presim, beam_run, impacts_postsim):
        options = step.cache_options(
            settings=settings, state=state, workspace=workspace
        )
        assert options.cache_mode != "overwrite"
        assert options.cache_hydration == "outputs-requested"
        assert options.cache_hydration_failure == "miss"
