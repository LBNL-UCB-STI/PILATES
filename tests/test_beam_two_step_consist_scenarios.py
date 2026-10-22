"""Test suite exercising BEAM two-step native Consist restart scenarios.

Per the BEAM Two-Step Spike v2 Brief:
- Real stage code (_run_beam_steps) is driven with real step definitions.
- Only the container runner (BeamRunner.run_container) is stubbed.
- Crashes are simulated by raising from the container or a single named fault hook.
- At most two monkeypatches per test.
- Covers Scenarios 1 through 7.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest
import xarray as xr

from consist import Tracker
from consist.types import CacheOptions
import dataclasses
from pilates.beam.postprocessor import (
    BeamPostprocessor,
    beam_postprocess_zarr_skims_output_path,
)
from pilates.beam.runner import BeamRunner
from pilates.config import PilatesConfig
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    ZARR_SKIMS,
)
from pilates.workflows.stages import supply_demand_beam
from pilates.workflows.stages.supply_demand_beam import (
    _run_beam_steps,
)
from pilates.workflows.steps.beam import beam_run
from pilates.workspace import Workspace
from workflow_state import WorkflowState

_TEST_ZONE_IDS = ["101", "102"]
_TEST_TIME_PERIODS = ["EA", "AM", "MD", "PM", "EV"]


def _create_test_environment(
    base_dir: Path,
    *,
    enable_activitysim_skims: bool = False,
) -> dict[str, Any]:
    """Build a minimal, real PILATES BEAM execution environment."""
    ws_root = base_dir / "workspace"
    sim_name = "sim_output"
    db_path = base_dir / "provenance.duckdb"
    shared_db = base_dir / "shared.duckdb"

    zones_cfg = None
    if enable_activitysim_skims:
        zones_geojson = base_dir / "canonical_zones.geojson"
        zones_geojson.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"zone_key": geoid},
                            "geometry": {"type": "Point", "coordinates": [0, 0]},
                        }
                        for geoid in _TEST_ZONE_IDS
                    ],
                }
            ),
            encoding="utf-8",
        )
        zones_cfg = {
            "source_file": str(zones_geojson),
            "activitysim_index_col": "TAZ",
            "zone_type": "taz",
            "canonical_id_col": "zone_key",
        }

    data: dict[str, Any] = {
        "run": {
            "region": "test-region",
            "scenario": "test-scenario",
            "start_year": 2020,
            "end_year": 2021,
            "output_directory": str(ws_root),
            "output_run_name": sim_name,
            "models": {
                "land_use": None,
                "travel": "beam",
                "activity_demand": "activitysim" if enable_activitysim_skims else None,
                "vehicle_ownership": None,
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["06001"]},
                "local_crs": "EPSG:32048",
                "zones": zones_cfg,
            },
            "skims": {
                "zone_type": "taz",
                "fname": "skimsActivitySimOD_current",
                "geoms_fname": "geoms.geojson",
                "geoms_index_col": "TAZ",
                "periods": _TEST_TIME_PERIODS,
            },
            "database": {
                "enabled": True,
                "type": "duckdb",
                "path": str(shared_db),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "docker_images": {"beam": "beam:latest"},
        },
        "beam": {
            "config": "beam.conf",
            "sample": 1.0,
            "replanning_portion": 0.4,
            "memory": "4g",
            "local_input_folder": "beam_input",
            "local_mutable_data_folder": "beam_mutable",
            "local_output_folder": "beam_output",
            "scenario_folder": "scenario",
            "router_directory": "r5",
            "skims_shapefile": "shape.shp",
            "skim_zone_source_id_col": "TAZ",
            "skim_zone_geoid_col": "TAZ",
            "ridehail_path_map": {},
        },
    }
    if enable_activitysim_skims:
        data["beam"]["artifact_formats"] = {"activitysim_skims": "omx"}
        data["activitysim"] = {
            "local_input_folder": "asim_input",
            "local_mutable_data_folder": "asim_mutable",
            "local_output_folder": "asim_output",
            "local_configs_folder": "asim_configs",
            "local_mutable_configs_folder": "asim_mutable_configs",
            "validation_folder": "asim_val",
            "command_template": "echo asim",
            "final_plans_folder": "asim_final_plans",
            "file_format": "csv",
            "region_mappings": {"region_to_subdir": {"test-region": "test"}},
        }
    cfg = PilatesConfig(**data)
    ws = Workspace(cfg, str(ws_root), sim_name)

    # Set up mutable inputs
    region_mutable = Path(ws.get_beam_mutable_data_dir()) / cfg.run.region
    region_mutable.mkdir(parents=True, exist_ok=True)
    beam_conf_path = region_mutable / "beam.conf"
    beam_conf_path.write_text(
        """
beam {
  inputDirectory = ${?inputDirectory}
  outputs.baseOutputDirectory = "/app/output"
  replanning.maxAgentPlanMemorySize = 4
  warmStart.initialLinkstatsFilePath = ""
  agentsim.taz.filePath = ""
  agentsim.taz.tazIdFieldName = "taz"
  exchange.scenario.folder = ${beam.inputDirectory}"/scenario"
  exchange.scenario.fileFormat = "csv"
  router.skim.activity-sim-skimmer.fileBaseName = "skimsActivitySimOD"
  router.skim.activity-sim-skimmer.fileOutputFormat = "omx"
  outputs.events.fileOutputFormats = "csv.gz"
  physsim.linkStatsOutputFileType = "csv.gz"
  routing.r5.directory = ${beam.inputDirectory}"/r5"
  physsim.inputNetworkFilePath = ""
}
matsim.modules.network.inputNetworkFile = ""
""",
        encoding="utf-8",
    )
    scenario_dir = region_mutable / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plans_file = scenario_dir / "plans.csv"
    plans_file.write_text("person_id,trip_id\n1,1\n", encoding="utf-8")
    hh_file = scenario_dir / "households.csv"
    hh_file.write_text("household_id\n1\n", encoding="utf-8")
    persons_file = scenario_dir / "persons.csv"
    persons_file.write_text("person_id,household_id\n1,1\n", encoding="utf-8")

    r5_dir = region_mutable / "r5"
    r5_dir.mkdir(parents=True, exist_ok=True)
    (r5_dir / "network.osm.pbf").write_bytes(b"dummy_osm_pbf")

    state = WorkflowState(
        start_year=2020,
        end_year=2021,
        travel_model_freq=1,
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=enable_activitysim_skims,
        traffic_assignment_enabled=True,
        year=2020,
        major_stage=WorkflowState.Stage.traffic_assignment,
        inner_iter=0,
        sub_stage=None,
        file_loc=None,
        asim_compiled=False,
        full_settings=cfg,
    )
    state.forecast_year = 2020
    context = WorkflowRuntimeContext.from_parts(settings=cfg, state=state, workspace=ws)

    local_run_dir = Path(ws.full_path).resolve()
    tracker = Tracker(
        run_dir=local_run_dir,
        db_path=str(db_path),
        hashing_strategy="full",
    )

    initial_skims_path: Path | None = None
    if enable_activitysim_skims:
        initial_skims_path = Path(ws.get_asim_output_dir()) / "cache" / "skims.zarr"
        initial_skims_path.parent.mkdir(parents=True, exist_ok=True)
        n_zones = len(_TEST_ZONE_IDS)
        n_periods = len(_TEST_TIME_PERIODS)
        ds = xr.Dataset(
            {
                "SOV_TRIPS": (
                    ("otaz", "dtaz", "time_period"),
                    np.full((n_zones, n_zones, n_periods), 100.0, dtype=np.float32),
                ),
                "SOV_FAILURES": (
                    ("otaz", "dtaz", "time_period"),
                    np.zeros((n_zones, n_zones, n_periods), dtype=np.float32),
                ),
                "SOV_TIME": (
                    ("otaz", "dtaz", "time_period"),
                    np.full((n_zones, n_zones, n_periods), 20.0, dtype=np.float32),
                ),
            },
            coords={
                "otaz": np.arange(n_zones),
                "dtaz": np.arange(n_zones),
                "time_period": _TEST_TIME_PERIODS,
            },
            attrs={"original_zone_ids": list(_TEST_ZONE_IDS)},
        )
        ds.to_zarr(initial_skims_path, mode="w", consolidated=True, zarr_format=2)

    seed_artifacts = {}
    with tracker.start_run("seed", "test"):
        seed_artifacts[BEAM_PLANS_IN] = tracker.log_artifact(
            plans_file, key=BEAM_PLANS_IN, direction="input"
        )
        seed_artifacts[BEAM_HOUSEHOLDS_IN] = tracker.log_artifact(
            hh_file, key=BEAM_HOUSEHOLDS_IN, direction="input"
        )
        seed_artifacts[BEAM_PERSONS_IN] = tracker.log_artifact(
            persons_file, key=BEAM_PERSONS_IN, direction="input"
        )
        if initial_skims_path is not None:
            seed_artifacts[ZARR_SKIMS] = tracker.log_artifact(
                initial_skims_path,
                key=ZARR_SKIMS,
                direction="input",
            )

    return {
        "cfg": cfg,
        "ws": ws,
        "context": context,
        "tracker": tracker,
        "seed_artifacts": seed_artifacts,
        "beam_conf_path": beam_conf_path,
        "local_run_dir": local_run_dir,
        "initial_skims_path": initial_skims_path,
    }


def _make_container_stub(
    events_content: str = "PathTraversal",
    linkstats_content: str = "link,time\n1,10\n",
    *,
    write_real_omx_skims: bool = False,
) -> Callable[..., bool]:
    """Factory creating a realistic stub for BeamRunner.run_container."""

    def stub(self: Any, **kwargs: Any) -> bool:
        output_dir = Path(kwargs["output_paths"][0])
        sim_out = output_dir / "raw_sim" / "ITERS" / "it.0"
        sim_out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "type": [events_content],
                "time": [100.0],
                "links": ["101,102"],
                "linkTravelTime": ["15.0,25.0"],
            }
        )
        df.to_parquet(sim_out / "0.events.parquet", index=False)
        with gzip.open(sim_out / "0.linkstats.csv.gz", "wt", encoding="utf-8") as f:
            f.write(linkstats_content)
        with gzip.open(sim_out / "0.plans.csv.gz", "wt", encoding="utf-8") as f:
            f.write("person_id,trip_id\n1,1\n")
        omx_path = sim_out / "0.skimsActivitySimOD_current.omx"
        if write_real_omx_skims:
            n_zones = len(_TEST_ZONE_IDS)
            trips_mat = np.zeros((n_zones, n_zones), dtype=np.float32)
            trips_mat[0, 0] = 5.0
            with omx.open_file(str(omx_path), mode="w") as f_omx:
                f_omx.create_mapping("TAZ", list(_TEST_ZONE_IDS))
                for period in _TEST_TIME_PERIODS:
                    f_omx[f"SOV_TRIPS__{period}"] = trips_mat
                    f_omx[f"SOV_FAILURES__{period}"] = np.zeros(
                        (n_zones, n_zones), dtype=np.float32
                    )
                    f_omx[f"SOV_TIME__{period}"] = np.full(
                        (n_zones, n_zones), 30.0, dtype=np.float32
                    )
        else:
            omx_path.write_bytes(b"omx_content")
        return True

    return stub


# ==============================================================================
# Scenario 1: Crash during beam_run
# ==============================================================================
def test_scenario_1_crash_during_beam_run(tmp_path: Path) -> None:
    """Scenario 1: Crash during beam_run reruns container on restart without corrupt state."""
    env = _create_test_environment(tmp_path)
    run_counts = {"container": 0}
    normal_container = _make_container_stub()

    def crashing_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        if run_counts["container"] == 1:
            raise RuntimeError("Simulated crash during container execution")
        return normal_container(self, **kwargs)

    # Monkeypatch 1: Container stub that fails on first invocation
    with patch.object(BeamRunner, "run_container", crashing_container):
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            # Attempt 1: Crashes during container execution
            with pytest.raises(
                RuntimeError, match="Simulated crash during container execution"
            ):
                _run_beam_steps(
                    scenario=scenario, year=2020, iteration=0, context=env["context"]
                )

        assert run_counts["container"] == 1

        # Attempt 2: Restart reruns beam_run and completes
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            res = _run_beam_steps(
                scenario=scenario, year=2020, iteration=0, context=env["context"]
            )
            assert res is None
            assert scenario.coupler.require(LINKSTATS) is not None
            assert run_counts["container"] == 2


# ==============================================================================
# Scenario 2: Crash after beam_run commits, before postprocess
# ==============================================================================
def test_scenario_2_crash_after_beam_run_commits_cache_hits_and_runs_postprocess(
    tmp_path: Path,
) -> None:
    """Scenario 2: Crash after beam_run commits produces cache hit on restart and runs postprocess."""
    env = _create_test_environment(tmp_path)
    run_counts = {"container": 0}
    container_stub = _make_container_stub()

    def tracked_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        return container_stub(self, **kwargs)

    def fail_after_beam_run(**kwargs: Any) -> None:
        raise RuntimeError("Simulated crash after beam_run commits")

    # Monkeypatch 1: Container stub
    # Monkeypatch 2: Canary fault injection hook after beam_run commits
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch(
            "pilates.workflows.stages.supply_demand_beam._maybe_fail_after_beam_run_for_canary",
            side_effect=fail_after_beam_run,
        ):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                with pytest.raises(
                    RuntimeError, match="Simulated crash after beam_run commits"
                ):
                    _run_beam_steps(
                        scenario=scenario,
                        year=2020,
                        iteration=0,
                        context=env["context"],
                    )

        assert run_counts["container"] == 1

        # Attempt 2: Restart! beam_run must cache hit (no container rerun)
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            res = _run_beam_steps(
                scenario=scenario, year=2020, iteration=0, context=env["context"]
            )
            assert res is None
            assert scenario.coupler.require(LINKSTATS) is not None
            # Container was NOT rerun; cache hit on beam_run!
            assert run_counts["container"] == 1


# ==============================================================================
# Scenario 3: Crash partway through postprocess restarts idempotently
# ==============================================================================
def test_scenario_3_crash_partway_through_postprocess_restarts_idempotently(
    tmp_path: Path,
) -> None:
    """Scenario 3: Interrupted postprocess restarts cleanly without double-counting skims."""
    env = _create_test_environment(tmp_path, enable_activitysim_skims=True)
    run_counts = {"container": 0}
    container_stub = _make_container_stub(write_real_omx_skims=True)
    real_postprocess = BeamPostprocessor.postprocess
    postprocess_calls = {"count": 0}

    def tracked_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        return container_stub(self, **kwargs)

    def fail_after_writing_output_skims(
        self: BeamPostprocessor, *args: Any, **kwargs: Any
    ) -> Any:
        postprocess_calls["count"] += 1
        result = real_postprocess(self, *args, **kwargs)
        if postprocess_calls["count"] == 1:
            raise RuntimeError("Simulated mid-postprocess crash")
        return result

    # Monkeypatch 1: Container stub
    # Monkeypatch 2: Postprocess hook crashes after writing the output Zarr store on Attempt 1
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch.object(
            BeamPostprocessor,
            "postprocess",
            fail_after_writing_output_skims,
        ):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                with pytest.raises(
                    RuntimeError, match="Simulated mid-postprocess crash"
                ):
                    _run_beam_steps(
                        scenario=scenario,
                        year=2020,
                        iteration=0,
                        context=env["context"],
                    )

            assert run_counts["container"] == 1
            assert postprocess_calls["count"] == 1

            # Verify no legacy in-progress marker exists and input Zarr skims remain untouched (100.0)
            marker = (
                Path(env["ws"].get_beam_output_dir()) / ".beam_postprocess_in_progress"
            )
            assert not marker.exists()
            initial_skims_path = env["initial_skims_path"]
            assert initial_skims_path is not None
            with xr.open_zarr(initial_skims_path) as input_ds:
                np.testing.assert_allclose(input_ds["SOV_TRIPS"].values, 100.0)

            # Attempt 2: Direct restart with zero manual intervention succeeds idempotently
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                res = _run_beam_steps(
                    scenario=scenario, year=2020, iteration=0, context=env["context"]
                )
                assert res is None
                coupler_skims_art = scenario.coupler.require(ZARR_SKIMS)
                coupler_skims_path = scenario.coupler.path(ZARR_SKIMS)

    # Container was NOT rerun; beam_run hit cache while beam_postprocess reran cleanly
    assert run_counts["container"] == 1
    assert postprocess_calls["count"] == 2

    expected_output_skims = Path(
        beam_postprocess_zarr_skims_output_path(env["ws"], env["context"].state)
    )
    assert Path(coupler_skims_art.path).resolve() == expected_output_skims.resolve()
    assert coupler_skims_path.resolve() == expected_output_skims.resolve()
    assert expected_output_skims.resolve() != initial_skims_path.resolve()

    # Input store remains strictly read-only (100.0)
    with xr.open_zarr(initial_skims_path) as input_ds:
        np.testing.assert_allclose(input_ds["SOV_TRIPS"].values, 100.0)

    # Output store has single-merged trips:
    # - observed OD cell (0, 0) has 5.0
    # - unobserved OD cell (0, 1) decayed once: ceil(0.9 * 100.0 + 0.0) = 90.0 (NOT double-merged 81.0)
    with xr.open_zarr(expected_output_skims) as output_ds:
        np.testing.assert_allclose(output_ds["SOV_TRIPS"].values[0, 0, :], 5.0)
        np.testing.assert_allclose(output_ds["SOV_TRIPS"].values[0, 1, :], 90.0)


def test_scenario_3_scratch_wiped_after_interrupted_postprocess_restarts_idempotently(
    tmp_path: Path,
) -> None:
    """Scenario 3b: If scratch is wiped after an interrupted postprocess, restart hydrates beam_run and succeeds idempotently."""
    env = _create_test_environment(tmp_path, enable_activitysim_skims=True)
    archive_dir = env["local_run_dir"] / "archive_run"
    archive_dir.mkdir(parents=True, exist_ok=True)
    env["context"].state.archive_run_dir = str(archive_dir)

    run_counts = {"container": 0}
    container_stub = _make_container_stub(write_real_omx_skims=True)
    real_postprocess = BeamPostprocessor.postprocess
    postprocess_calls = {"count": 0}

    def tracked_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        return container_stub(self, **kwargs)

    def fail_after_writing_output_skims(
        self: BeamPostprocessor, *args: Any, **kwargs: Any
    ) -> Any:
        postprocess_calls["count"] += 1
        result = real_postprocess(self, *args, **kwargs)
        if postprocess_calls["count"] == 1:
            raise RuntimeError("Simulated mid-postprocess crash")
        return result

    # Monkeypatch 1: Container stub
    # Monkeypatch 2: Postprocess fails mid-run after writing output Zarr skims
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch.object(
            BeamPostprocessor,
            "postprocess",
            fail_after_writing_output_skims,
        ):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                with pytest.raises(
                    RuntimeError, match="Simulated mid-postprocess crash"
                ):
                    _run_beam_steps(
                        scenario=scenario,
                        year=2020,
                        iteration=0,
                        context=env["context"],
                    )

            assert run_counts["container"] == 1
            assert postprocess_calls["count"] == 1

            # Neither scratch nor archive recovery root should have an in-progress marker
            scratch_marker = (
                Path(env["ws"].get_beam_output_dir()) / ".beam_postprocess_in_progress"
            )
            archive_marker = archive_dir / ".beam_postprocess_in_progress"
            assert not scratch_marker.exists()
            assert not archive_marker.exists()

            # Simulate a scratch purge / wipe after the interrupted postprocess
            beam_out = Path(env["ws"].get_beam_output_dir())
            shutil.rmtree(beam_out)
            assert not beam_out.exists()

            # Attempt 2: Restart after scratch wipe hydrates beam_run outputs from recovery root
            # and completes beam_postprocess cleanly without double-counting
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                res = _run_beam_steps(
                    scenario=scenario, year=2020, iteration=0, context=env["context"]
                )
                assert res is None
                assert scenario.coupler.require(ZARR_SKIMS) is not None

    assert run_counts["container"] == 1
    assert postprocess_calls["count"] == 2
    expected_output_skims = Path(
        beam_postprocess_zarr_skims_output_path(env["ws"], env["context"].state)
    )
    with xr.open_zarr(expected_output_skims) as output_ds:
        np.testing.assert_allclose(output_ds["SOV_TRIPS"].values[0, 0, :], 5.0)
        np.testing.assert_allclose(output_ds["SOV_TRIPS"].values[0, 1, :], 90.0)


# ==============================================================================
# Scenario 4: Scratch wiped after beam_run committed
# ==============================================================================
def test_scenario_4_scratch_wiped_after_beam_run_committed_hydrates_from_recovery_root(
    tmp_path: Path,
) -> None:
    """Scenario 4: Scratch wiped after beam_run hydrates declared destinations from recovery root.

    With cache_hydration='outputs-requested', Consist checks registered artifact recovery roots
    and hydrates cached outputs directly into their declared destination paths in scratch.
    """
    env = _create_test_environment(tmp_path)
    run_counts = {"container": 0}
    container_stub = _make_container_stub()

    def tracked_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        return container_stub(self, **kwargs)

    # Monkeypatch 1: Container stub
    # Monkeypatch 2: Hook crashing after beam_run commits and archives
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch(
            "pilates.workflows.stages.supply_demand_beam._maybe_fail_after_beam_run_for_canary",
            side_effect=RuntimeError("Simulated crash after beam_run committed"),
        ):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                with pytest.raises(
                    RuntimeError, match="Simulated crash after beam_run committed"
                ):
                    _run_beam_steps(
                        scenario=scenario,
                        year=2020,
                        iteration=0,
                        context=env["context"],
                    )

    assert run_counts["container"] == 1

    # Wipe scratch beam_output directory
    beam_out = Path(env["ws"].get_beam_output_dir())
    assert beam_out.exists()
    shutil.rmtree(beam_out)
    assert not beam_out.exists()

    # Attempt 2: Restart. Consist cache_hydration='outputs-requested' probes
    # recovery roots and restores declared destination paths in scratch.
    with patch.object(BeamRunner, "run_container", tracked_container):
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            res = _run_beam_steps(
                scenario=scenario, year=2020, iteration=0, context=env["context"]
            )
            assert res is None
            assert scenario.coupler.require(LINKSTATS) is not None

    # Verify no second container run (cache hit) and outputs restored to scratch
    assert run_counts["container"] == 1
    assert beam_out.exists()
    restored_parquet = (
        beam_out
        / "test-region/year-2020-iteration-0/.pilates-consist-outputs/beam_run/events_parquet_2020_0.parquet"
    )
    assert restored_parquet.exists()


def test_scenario_4b_scratch_wiped_after_iteration_0_postprocess_hydrates_zarr_skims_for_iteration_1(
    tmp_path: Path,
) -> None:
    """Scenario 4b: Scratch wiped after Iteration 0 beam_postprocess hydrates zarr_skims for Iteration 1.

    Verifies that beam_postprocess archives its declared zarr_skims directory artifact
    and registers recovery_roots so restarting after a scratch wipe cache-hydrates
    Iteration 0's zarr_skims from the archive (without re-running Iteration 0) and
    Iteration 1 merges cleanly on top of it.
    """
    env = _create_test_environment(tmp_path, enable_activitysim_skims=True)
    archive_dir = tmp_path / "archive_root"
    archive_dir.mkdir(parents=True, exist_ok=True)
    env["context"].state.archive_run_dir = str(archive_dir)
    run_counts = {"container": 0, "postprocess": 0}
    container_stub = _make_container_stub(write_real_omx_skims=True)

    def tracked_container(self_runner: Any, **kwargs: Any) -> None:
        run_counts["container"] += 1
        return container_stub(self_runner, **kwargs)

    orig_postprocess = BeamPostprocessor.postprocess

    def tracked_postprocess(self_pp: Any, *args: Any, **kwargs: Any) -> Any:
        run_counts["postprocess"] += 1
        return orig_postprocess(self_pp, *args, **kwargs)

    # Attempt 1: Complete Iteration 0, then crash before Iteration 1
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch.object(BeamPostprocessor, "postprocess", tracked_postprocess):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                env["context"].state.inner_iter = 0
                res0 = _run_beam_steps(
                    scenario=scenario, year=2020, iteration=0, context=env["context"]
                )
                assert res0 is None
                iter0_zarr_art = scenario.coupler.require(ZARR_SKIMS)
                assert isinstance(iter0_zarr_art.meta, dict)
                assert iter0_zarr_art.meta.get("recovery_roots")

    assert run_counts["container"] == 1
    assert run_counts["postprocess"] == 1

    iter0_output_skims = Path(
        beam_postprocess_zarr_skims_output_path(env["ws"], env["context"].state)
    )
    assert iter0_output_skims.exists()

    # Wipe scratch beam_output directory after Iteration 0 completes
    beam_out = Path(env["ws"].get_beam_output_dir())
    shutil.rmtree(beam_out)
    assert not beam_out.exists()
    assert not iter0_output_skims.exists()

    # Attempt 2: Restart scenario. Iteration 0 hits cache and hydrates zarr_skims
    # from archive_dir; Iteration 1 runs and merges on top of hydrated zarr_skims.
    with patch.object(BeamRunner, "run_container", tracked_container):
        with patch.object(BeamPostprocessor, "postprocess", tracked_postprocess):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                env["context"].state.inner_iter = 0
                res0_restart = _run_beam_steps(
                    scenario=scenario, year=2020, iteration=0, context=env["context"]
                )
                assert res0_restart is None
                assert run_counts["container"] == 1
                assert run_counts["postprocess"] == 1
                assert iter0_output_skims.exists()

                # Simulate ActivitySim Iteration 1 publishing fresh plans/hh/persons
                # while keeping ZARR_SKIMS from Iteration 0's hydrated postprocess.
                for k in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
                    scenario.coupler.set_from_artifact(k, env["seed_artifacts"][k])

                env["context"].state.inner_iter = 1
                res1 = _run_beam_steps(
                    scenario=scenario, year=2020, iteration=1, context=env["context"]
                )
                assert res1 is None
                assert scenario.coupler.require(ZARR_SKIMS) is not None
                assert run_counts["container"] == 2
                assert run_counts["postprocess"] == 2

    iter1_output_skims = Path(
        beam_postprocess_zarr_skims_output_path(env["ws"], env["context"].state)
    )
    assert iter1_output_skims.exists()
    with xr.open_zarr(iter1_output_skims) as output_ds:
        # Iter 0 merged 100.0 -> 90.0; Iter 1 merges ceil(0.9 * 90.0) -> 81.0
        np.testing.assert_allclose(output_ds["SOV_TRIPS"].values[0, 1, :], 81.0)


# ==============================================================================
# Scenario 5: BEAM config changed between attempts
# ==============================================================================
def test_scenario_5_beam_config_changed_causes_cache_miss(tmp_path: Path) -> None:
    """Scenario 5: Changing BEAM config between attempts causes cache miss and reruns container."""
    env = _create_test_environment(tmp_path)
    run_counts = {"container": 0}
    container_stub = _make_container_stub()

    def tracked_container(self: Any, **kwargs: Any) -> bool:
        run_counts["container"] += 1
        return container_stub(self, **kwargs)

    # Monkeypatch 1: Container stub
    with patch.object(BeamRunner, "run_container", tracked_container):
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            res1 = _run_beam_steps(
                scenario=scenario, year=2020, iteration=0, context=env["context"]
            )
            assert res1 is None
            assert scenario.coupler.require(LINKSTATS) is not None

        assert run_counts["container"] == 1

        # Modify beam.conf to alter config hash
        conf_text = env["beam_conf_path"].read_text(encoding="utf-8")
        env["beam_conf_path"].write_text(
            conf_text.replace(
                "replanning.maxAgentPlanMemorySize = 4",
                "replanning.maxAgentPlanMemorySize = 8",
            ),
            encoding="utf-8",
        )

        # Attempt 2: Rerun. Must be a cache miss on beam_run!
        with env["tracker"].scenario("beam_scenario") as scenario:
            for k, art in env["seed_artifacts"].items():
                scenario.coupler.set_from_artifact(k, art)

            res2 = _run_beam_steps(
                scenario=scenario, year=2020, iteration=0, context=env["context"]
            )
            assert res2 is None
            assert scenario.coupler.require(LINKSTATS) is not None
            assert run_counts["container"] == 2


# ==============================================================================
# Scenario 6: Two committed runs with same identity, different bytes (stochastic BEAM)
# ==============================================================================
def test_scenario_6_two_committed_runs_same_identity_exposes_recency_gap(
    tmp_path: Path,
) -> None:
    """Scenario 6: Two committed beam_runs with same identity but different bytes.

    Exposes the Consist gap: find_matching_run orders by epoch_us(created_at) DESC.
    A cache hit unconditionally selects the latest committed run, which breaks
    provenance if downstream steps already consumed outputs from the earlier run.
    """
    env = _create_test_environment(tmp_path)
    container_v1 = _make_container_stub(linkstats_content="link,time\n1,100\n")
    container_v2 = _make_container_stub(linkstats_content="link,time\n1,200\n")

    # Run 1: commits with container_v1
    with patch.object(BeamRunner, "run_container", container_v1):
        with env["tracker"].scenario("beam_scenario_1") as s1:
            for k, art in env["seed_artifacts"].items():
                s1.coupler.set_from_artifact(k, art)
            _run_beam_steps(scenario=s1, year=2020, iteration=0, context=env["context"])

    runs_after_1 = env["tracker"].find_runs(model="beam_run")
    assert len(runs_after_1) == 1
    run1 = runs_after_1[0]
    run1_id = run1.id

    # Run 2: committed with container_v2 using forced cache bypass (cache_mode="overwrite")
    # in a new scenario to simulate another stochastic run under the same inputs/config.
    # Uses EXACTLY two monkeypatches: BeamRunner.run_container and supply_demand_beam.beam_run
    overwrite_cache = lambda **k: CacheOptions(
        cache_hydration="outputs-all",
        cache_hydration_failure="warn",
        cache_version=1,
        cache_mode="overwrite",
    )
    patched_beam_run = dataclasses.replace(beam_run, cache_options=overwrite_cache)

    with patch.object(BeamRunner, "run_container", container_v2), patch.object(
        supply_demand_beam, "beam_run", patched_beam_run
    ):
        with env["tracker"].scenario("beam_scenario_2") as s2:
            for k, art in env["seed_artifacts"].items():
                s2.coupler.set_from_artifact(k, art)
            _run_beam_steps(scenario=s2, year=2020, iteration=0, context=env["context"])

    runs_after_2 = env["tracker"].find_runs(model="beam_run")
    assert len(runs_after_2) == 2
    run2 = runs_after_2[0] if runs_after_2[0].id != run1_id else runs_after_2[1]
    run2_id = run2.id
    assert run2_id != run1_id

    # Verify identical cache identities
    assert run1.input_hash == run2.input_hash
    assert run1.config_hash == run2.config_hash
    assert run1.git_hash == run2.git_hash

    # Now inspect cache lookup: find_matching_run
    matching = env["tracker"].find_matching_run(
        input_hash=run1.input_hash,
        config_hash=run1.config_hash,
        git_hash=run1.git_hash,
    )
    assert matching is not None

    # Consist's find_matching_run orders by epoch_us(created_at) DESC, selecting Run 2
    assert matching.id == run2_id

    # Verify that the two runs actually produced different byte outputs
    # Run 1 linkstats had "1,100" while Run 2 linkstats had "1,200"
    out1 = env["tracker"].get_run_outputs(run1.id)["linkstats"]
    out2 = env["tracker"].get_run_outputs(run2.id)["linkstats"]
    assert out1.hash != out2.hash


# ==============================================================================
# Scenario 7: Archive copy failure raises loudly
# ==============================================================================
def test_scenario_7_archive_copy_failure_raises(tmp_path: Path) -> None:
    """Scenario 7: Archive failure after beam_run raises loudly; archive not treated as present."""
    env = _create_test_environment(tmp_path)
    container_stub = _make_container_stub()

    def failing_archive(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated disk full during archive copy")

    # Monkeypatch 1: Container stub
    # Monkeypatch 2: Archive fault injection hook
    with patch.object(BeamRunner, "run_container", container_stub):
        with patch(
            "pilates.workflows.stages.supply_demand_beam._archive_beam_run_outputs_native",
            side_effect=failing_archive,
        ):
            with env["tracker"].scenario("beam_scenario") as scenario:
                for k, art in env["seed_artifacts"].items():
                    scenario.coupler.set_from_artifact(k, art)

                # Must fail loudly; no swallowed exception
                with pytest.raises(
                    RuntimeError, match="Simulated disk full during archive copy"
                ):
                    _run_beam_steps(
                        scenario=scenario,
                        year=2020,
                        iteration=0,
                        context=env["context"],
                    )

    # Recovery root must NOT have been recorded on artifacts in duckdb
    import duckdb

    con = duckdb.connect(str(tmp_path / "provenance.duckdb"))
    rows = con.execute("SELECT meta FROM artifact WHERE key = 'linkstats'").fetchall()
    for row in rows:
        meta = row[0]
        if meta and isinstance(meta, str):
            import json

            meta = json.loads(meta)
        if meta:
            assert "recovery_roots" not in meta or not meta["recovery_roots"]
