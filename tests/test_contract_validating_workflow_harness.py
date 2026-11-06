"""
Layer 3: Contract-Validating Workflow Test Harness (Happy Path + Boundary & Chaos Suite).

Exercises real Preprocessor, Runner, and Postprocessor implementations and step definitions
(`execute_step` + `STEP_DEFINITIONS`) against the 3-zone, 5-household, 8-person micro-world
fixture and `ContractValidatingContainerBackend`.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import pytest

from pilates.activitysim.runner import validate_activitysim_zarr_skims
from pilates.utils import consist_runtime as cr
from pilates.utils.archive_paths import resolve_archive_run_dir
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.stages.supply_demand_beam import _compile_beam_launch_config
from pilates.workflows.stages.vehicle_ownership import _publish_current_h5_alias
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import (
    activitysim_postprocess,
    activitysim_preprocess,
    activitysim_run,
    atlas_postprocess,
    atlas_preprocess,
    atlas_run,
    beam_postprocess,
    beam_preprocess,
    beam_run,
    urbansim_postprocess,
    urbansim_run,
)
from tests.helpers.contract_container_backend import ContractValidatingContainerBackend
from tests.helpers.micro_world_fixture import (
    MICRO_HOUSEHOLD_IDS,
    MicroWorldEnv,
    build_micro_world,
    write_micro_skims_omx,
)

pytestmark = pytest.mark.contract_harness


def _run_activitysim_and_beam_preprocess(
    env: MicroWorldEnv,
    sc,
    *,
    year: int = 2015,
    iteration: int = 0,
    bypass_atlas_restore_suffix_bug: bool = False,
):
    """Run ActivitySim (preprocess -> run -> postprocess) and BEAM preprocess via execute_step."""
    env.state.current_year = year
    env.state.forecast_year = year
    env.state.current_inner_iter = iteration

    _, asim_pre = execute_step(
        scenario=sc,
        definition=activitysim_preprocess,
        settings=env.settings,
        state=env.state,
        workspace=env.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="preprocess",
    )
    assert asim_pre is not None

    _, asim_run_out = execute_step(
        scenario=sc,
        definition=activitysim_run,
        settings=env.settings,
        state=env.state,
        workspace=env.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="run",
    )
    assert asim_run_out is not None

    prev_vo_enabled = env.settings.vehicle_ownership_model_enabled
    if bypass_atlas_restore_suffix_bug:
        env.settings.vehicle_ownership_model_enabled = False
    try:
        _, asim_post = execute_step(
            scenario=sc,
            definition=activitysim_postprocess,
            settings=env.settings,
            state=env.state,
            workspace=env.workspace,
            stage="supply_demand",
            year=year,
            iteration=iteration,
            phase="postprocess",
        )
    finally:
        env.settings.vehicle_ownership_model_enabled = prev_vo_enabled
    assert asim_post is not None

    _, beam_pre = execute_step(
        scenario=sc,
        definition=beam_preprocess,
        settings=env.settings,
        state=env.state,
        workspace=env.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="preprocess",
    )
    assert beam_pre is not None

    archive_run_dir = resolve_archive_run_dir(workspace=env.workspace, state=env.state)
    launch_config = _compile_beam_launch_config(
        scenario=sc,
        settings=env.settings,
        workspace=env.workspace,
        year=year,
        iteration=iteration,
        preprocess_outputs=beam_pre,
        archive_run_dir=archive_run_dir,
    )
    return beam_pre, launch_config


def test_happy_path_full_micro_world_workflow_contracts(tmp_path: Path) -> None:
    """
    End-to-end happy-path contract verification across UrbanSim, ATLAS, ActivitySim, and BEAM.
    Proves that `build_micro_world` and `ContractValidatingContainerBackend` work cleanly with
    real Preprocessor, Runner, and Postprocessor implementations.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2010,
        end_year=2015,
        travel_model_freq=5,
        activitysim_skims_format="zarr",
        sharrow_enabled=True,
        persist_sharrow_cache=True,
        split_events_by_type=True,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend() as backend:
        with env.scenario_context("happy_path_full_chain") as sc:
            # 1. UrbanSim Run & Postprocess (2010 -> 2015)
            _, usim_run_out = execute_step(
                scenario=sc,
                definition=urbansim_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="run",
            )
            assert usim_run_out is not None
            assert Path(usim_run_out.usim_datastore_h5).exists()

            _, usim_post_out = execute_step(
                scenario=sc,
                definition=urbansim_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="postprocess",
            )
            assert usim_post_out is not None
            assert Path(usim_post_out.usim_datastore_h5).exists()
            assert USIM_POPULATION_SOURCE_H5 in usim_post_out.processed_outputs

            # Advance state to forecast year 2015 for ATLAS + ActivitySim + BEAM
            env.state.current_year = 2015
            env.state.forecast_year = 2015
            atlas_state = AtlasSubState(env.state, 2015)

            # 2. ATLAS Preprocess, Run & Postprocess
            _, atlas_pre_out = execute_step(
                scenario=sc,
                definition=atlas_preprocess,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="preprocess",
            )
            assert atlas_pre_out is not None

            _, atlas_run_out = execute_step(
                scenario=sc,
                definition=atlas_run,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="run",
            )
            assert atlas_run_out is not None

            _, atlas_post_out = execute_step(
                scenario=sc,
                definition=atlas_postprocess,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="postprocess",
            )
            assert atlas_post_out is not None
            assert atlas_post_out.usim_datastore_h5 is not None
            _publish_current_h5_alias(
                coupler=sc.coupler,
                fallback_path=atlas_post_out.usim_datastore_h5,
            )
            with pd.HDFStore(str(atlas_post_out.usim_datastore_h5), mode="r") as store:
                assert "/households" in store.keys()
                hh_df = store["/households"]
                assert "cars" in hh_df.columns
                assert "hh_cars" in hh_df.columns
                assert set(hh_df.index).issubset(set(MICRO_HOUSEHOLD_IDS))

            # 3. ActivitySim Preprocess, Run & Postprocess + BEAM Preprocess
            _, launch_cfg = _run_activitysim_and_beam_preprocess(
                env, sc, year=2015, iteration=0
            )

            # 4. BEAM Run & Postprocess
            beam_run_inputs = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
            )
            beam_run_result, beam_run_out = execute_step(
                scenario=sc,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg},
                resolved_inputs=beam_run_inputs,
            )
            assert beam_run_out is not None
            assert Path(beam_run_out.raw_outputs["raw_od_skims_zarr_2015_0"]).is_dir()

            beam_post_inputs = beam_postprocess.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
                beam_run_outputs=beam_run_result.outputs,
            )
            _, beam_post_out = execute_step(
                scenario=sc,
                definition=beam_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="postprocess",
                resolved_inputs=beam_post_inputs,
            )
            assert beam_post_out is not None
            assert beam_post_out.zarr_skims is not None
            assert Path(beam_post_out.zarr_skims).is_dir()
            assert "events_parquet_2015_0_type_ModeChoice" in beam_post_out.split_events
            assert (
                "events_parquet_2015_0_type_PathTraversal" in beam_post_out.split_events
            )
            assert "path_traversal_links_2015_0" in beam_post_out.split_event_links
            assert Path(
                beam_post_out.split_events["events_parquet_2015_0_type_ModeChoice"]
            ).exists()
            assert Path(
                beam_post_out.split_events["events_parquet_2015_0_type_PathTraversal"]
            ).exists()
            assert Path(
                beam_post_out.split_event_links["path_traversal_links_2015_0"]
            ).exists()

    executed_models = [inv.model_name for inv in backend.invocations]
    assert executed_models == [
        "urbansim",
        "atlas",
        "activitysim_numba_warmup",
        "activitysim",
        "beam_run",
    ]


def test_beam_run_direct_zarr_cold_scratch_cache_hit_and_missing_archive_recovery(
    tmp_path: Path,
) -> None:
    """
    Boundary Scenario 1:
    - Run `beam_preprocess` -> `beam_run` with `activitysim_skims="zarr"` and `PILATES_ARCHIVE_ROOT` enabled.
    - Phase A (cold scratch + intact archive): Wipe `PILATES_SCRATCH_ROOT` (`beam/output`) while keeping
      `PILATES_ARCHIVE_ROOT` intact. Re-run `beam_run` and verify it takes a cache hit (`backend_executed=False`)
      and hydrates `raw_od_skims_zarr_2015_0` back to disk with valid Zarr arrays.
    - Phase B (cold scratch + missing archive): Wipe `PILATES_SCRATCH_ROOT` (`beam/output`) AND remove the
      archived Zarr directory in `PILATES_ARCHIVE_ROOT`. Re-run `beam_run` and verify it falls back cleanly
      via `cache_hydration_failure="miss"` to re-execution (`backend_executed=True`) and regenerates the Zarr store.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2015,
        activitysim_skims_format="zarr",
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend() as backend:
        with env.scenario_context("beam_zarr_warm") as sc1:
            _, launch_cfg = _run_activitysim_and_beam_preprocess(
                env, sc1, year=2015, iteration=0
            )
            run_inputs_1 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc1.coupler,
            )
            exec_before_warm = len(backend.invocations)
            _, first_run = execute_step(
                scenario=sc1,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg},
                resolved_inputs=run_inputs_1,
            )
            assert len(backend.invocations) == exec_before_warm + 1
            assert first_run is not None
            raw_zarr_path = Path(first_run.raw_outputs["raw_od_skims_zarr_2015_0"])
            assert raw_zarr_path.is_dir()
            assert validate_activitysim_zarr_skims(raw_zarr_path) is None

        # Phase A: Wipe scratch BEAM output directory while keeping archive_root intact
        beam_output_dir = Path(env.workspace.get_beam_output_dir())
        shutil.rmtree(beam_output_dir)
        beam_output_dir.mkdir(parents=True, exist_ok=True)
        assert not raw_zarr_path.exists()

        with env.scenario_context("beam_zarr_cold_hit_hydration") as sc2:
            _, launch_cfg_2 = _run_activitysim_and_beam_preprocess(
                env, sc2, year=2015, iteration=0
            )
            run_inputs_2 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc2.coupler,
            )
            exec_before_phase_a = len(backend.invocations)
            _, hydrated_run = execute_step(
                scenario=sc2,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg_2},
                resolved_inputs=run_inputs_2,
            )
            # Cache hit: container backend was NOT re-executed
            assert len(backend.invocations) == exec_before_phase_a
            assert hydrated_run is not None
            hydrated_zarr = Path(hydrated_run.raw_outputs["raw_od_skims_zarr_2015_0"])
            assert hydrated_zarr.is_dir()
            assert validate_activitysim_zarr_skims(hydrated_zarr) is None

        # Phase B: Wipe scratch BEAM output directory AND delete archived Zarr copies
        shutil.rmtree(beam_output_dir)
        beam_output_dir.mkdir(parents=True, exist_ok=True)
        for archived_zarr in env.paths.archive_root.rglob("*.zarr"):
            if "skimsActivitySimOD" in archived_zarr.name:
                shutil.rmtree(archived_zarr)

        with env.scenario_context("beam_zarr_cold_missing_archive_recovery") as sc3:
            _, launch_cfg_3 = _run_activitysim_and_beam_preprocess(
                env, sc3, year=2015, iteration=0
            )
            run_inputs_3 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc3.coupler,
            )
            exec_before_phase_b = len(backend.invocations)
            _, recovered_run = execute_step(
                scenario=sc3,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg_3},
                resolved_inputs=run_inputs_3,
            )
            # Cache hydration failure -> clean fallback to re-execution
            assert len(backend.invocations) == exec_before_phase_b + 1
            assert recovered_run is not None
            recovered_zarr = Path(recovered_run.raw_outputs["raw_od_skims_zarr_2015_0"])
            assert recovered_zarr.is_dir()
            assert validate_activitysim_zarr_skims(recovered_zarr) is None


def test_beam_postprocess_cold_scratch_events_parquet_materialization(
    tmp_path: Path,
) -> None:
    """
    Boundary Scenario 2:
    - Run `beam_run` with `split_events_by_type=True` and `PILATES_ARCHIVE_ROOT` enabled.
    - Wipe `PILATES_SCRATCH_ROOT` (`beam/output`) so `0.events.parquet` only exists in `PILATES_ARCHIVE_ROOT`
      (`consist-recovery`).
    - Run `beam_postprocess` via `execute_step`, asserting `_resolve_beam_postprocess_inputs` and `scenario.run`
      materialize `events_parquet` from archive and succeed instead of crashing in
      `_beam_postprocess_split_output_paths` before `scenario.run`.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2015,
        activitysim_skims_format="zarr",
        split_events_by_type=True,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend():
        with env.scenario_context("beam_postprocess_cold_events") as sc:
            _, launch_cfg = _run_activitysim_and_beam_preprocess(
                env, sc, year=2015, iteration=0
            )
            run_inputs = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
            )
            beam_run_res, beam_run_out = execute_step(
                scenario=sc,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg},
                resolved_inputs=run_inputs,
            )
            assert beam_run_out is not None
            scratch_events = Path(beam_run_out.raw_outputs["events_parquet_2015_0"])
            assert scratch_events.exists()

            # Verify archive has events_parquet_2015_0.parquet in consist-recovery, then wipe scratch beam/output
            archived_events = list(
                env.paths.archive_root.rglob("events_parquet_*.parquet")
            )
            assert archived_events, (
                "Expected events_parquet_*.parquet to be archived under archive_root"
            )

            beam_output_dir = Path(env.workspace.get_beam_output_dir())
            shutil.rmtree(beam_output_dir)
            beam_output_dir.mkdir(parents=True, exist_ok=True)
            assert not scratch_events.exists()

            # Now run beam_postprocess on cold scratch
            _, beam_post_out = execute_step(
                scenario=sc,
                definition=beam_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="postprocess",
            )
            assert beam_post_out is not None
            assert "events_parquet_2015_0_type_ModeChoice" in beam_post_out.split_events
            assert Path(
                beam_post_out.split_events["events_parquet_2015_0_type_ModeChoice"]
            ).exists()


def test_urbansim_run_step_lineage_mode_and_cold_scratch_execution(
    tmp_path: Path,
) -> None:
    """
    Boundary Scenario 3:
    - Run `urbansim_run` through `execute_step` using Consist's `run_container` with `ContractValidatingContainerBackend`.
    - Verify `UrbansimRunner` passes `lineage_mode="none"` (like `AtlasRunner`, `ActivitysimRunner`, `BeamRunner`)
      and does not trigger a false nested container cache hit with `output_paths=None` when scratch is wiped
      and `urbansim_run` is forced to re-execute.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2010,
        end_year=2015,
        travel_model_freq=5,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend(delegate_through_consist=True) as backend:
        with env.scenario_context("urbansim_lineage_warm") as sc1:
            _, res1 = execute_step(
                scenario=sc1,
                definition=urbansim_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="run",
            )
            assert res1 is not None
            out_h5 = Path(res1.usim_datastore_h5)
            assert out_h5.exists()

        # Wipe model_data_2015.h5 from both scratch and archive, then force outer step cache miss
        out_h5.unlink()
        for archived_h5 in env.paths.archive_root.rglob("model_data_2015.h5"):
            archived_h5.unlink()

        # Touch base input HDF5 so outer step `urbansim_run` misses cache and re-executes `UrbansimRunner.run`
        region_id = env.settings.urbansim.region_mappings["region_to_region_id"][
            env.settings.run.region
        ]
        usim_base_h5 = (
            Path(env.workspace.get_usim_mutable_data_dir())
            / f"custom_mpo_{region_id}_model_data.h5"
        )
        with pd.HDFStore(str(usim_base_h5), mode="a") as store:
            hh_df = store["/households"].copy()
            hh_df["income"] = hh_df["income"] + 1
            store.put("/households", hh_df, format="fixed")

        with env.scenario_context("urbansim_lineage_cold_rerun") as sc2:
            _, res2 = execute_step(
                scenario=sc2,
                definition=urbansim_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="run",
            )
            assert res2 is not None
            assert Path(res2.usim_datastore_h5).exists()

        usim_invocations = [
            inv for inv in backend.invocations if inv.model_name == "urbansim"
        ]
        assert len(usim_invocations) == 2
        for inv in usim_invocations:
            assert inv.lineage_mode == "none"
            assert inv.backend_executed is True


def test_activitysim_omx_mode_with_sharrow_disabled_does_not_require_zarr_cache(
    tmp_path: Path,
) -> None:
    """
    Boundary Scenario 4:
    - Run `ActivitysimPreprocessor` + `ActivitysimRunner` in `skim_mode="omx"` with `activitysim.sharrow=False`.
    - Because `ContractValidatingContainerBackend` only writes `output/cache/skims.zarr` when
      `--persist-sharrow-cache` is passed, `ActivitysimRunner.run` must not unconditionally call
      `finalize_activitysim_zarr_skims` when `sharrow=False` / `produces_zarr=False`.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2015,
        activitysim_skims_format="omx",
        sharrow_enabled=False,
        persist_sharrow_cache=False,
        enable_archive_copy=True,
    )

    # Remove pre-seeded skims.zarr in runtime cache so only OMX skims exist
    cache_zarr = Path(env.workspace.get_asim_runtime_cache_dir()) / "skims.zarr"
    if cache_zarr.exists():
        shutil.rmtree(cache_zarr)

    with ContractValidatingContainerBackend() as backend:
        with env.scenario_context("activitysim_omx_no_sharrow") as sc:
            sc.coupler.pop(ZARR_SKIMS, None)
            _, asim_pre = execute_step(
                scenario=sc,
                definition=activitysim_preprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="preprocess",
            )
            assert asim_pre is not None

            _, asim_run_out = execute_step(
                scenario=sc,
                definition=activitysim_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
            )
            assert asim_run_out is not None
            assert Path(asim_run_out.raw_outputs["households_asim_out"]).exists()

    asim_inv = [inv for inv in backend.invocations if inv.model_name == "activitysim"]
    assert len(asim_inv) == 1
    assert "--persist-sharrow-cache" not in asim_inv[0].command


def test_atlas_preprocess_beamac_uses_resolved_beam_skims_omx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Boundary Scenario 5:
    - Run `AtlasPreprocessor` via `execute_step` from a non-repo working directory with `atlas.beamac=1`
      and `final_skims_omx` provided via coupler/inputs, WITHOUT pre-populating `asim_mutable_data_dir/skims.omx`.
    - `AtlasPreprocessor` must compute accessibility from the resolved `final_skims_omx` rather than
      hardcoding `os.path.join(workspace.get_asim_mutable_data_dir(), "skims.omx")`.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2015,
        atlas_beamac=1,
        enable_archive_copy=True,
    )

    # Remove `asim_mutable_data_dir/skims.omx` so only the coupler-provided `final_skims_omx` exists
    mutable_omx = Path(env.workspace.get_asim_mutable_data_dir()) / "skims.omx"
    if mutable_omx.exists():
        mutable_omx.unlink()

    external_beam_omx = tmp_path / "coupler_beam_skims" / "beam_skims_2015.omx"
    write_micro_skims_omx(external_beam_omx)

    non_repo_cwd = tmp_path / "external_cwd"
    non_repo_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(non_repo_cwd)

    with ContractValidatingContainerBackend():
        with env.scenario_context("atlas_beamac_coupler_omx") as sc:
            with sc.trace(
                "seed_external_beam_omx",
                model="initialization",
                year=2015,
                iteration=0,
            ):
                omx_art = cr.log_input(external_beam_omx, key=FINAL_SKIMS_OMX)
                sc.coupler.set_from_artifact(FINAL_SKIMS_OMX, omx_art)

            atlas_state = AtlasSubState(env.state, 2015)
            _, atlas_pre = execute_step(
                scenario=sc,
                definition=atlas_preprocess,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="preprocess",
            )
            assert atlas_pre is not None
            assert (
                Path(env.workspace.get_atlas_mutable_input_dir())
                / "year2015"
                / "accessibility_2015_tract.csv"
            ).exists()


def test_multi_year_hdf5_datastore_preservation_and_multi_iteration_beam_replan(
    tmp_path: Path,
) -> None:
    """
    Boundary Scenario 6:
    - Part (a): Run `UrbansimPostprocessor` -> `AtlasPostprocessor` -> `ActivitysimPostprocessor`
      (with `update_usim_datastore=True`) in the same year, asserting `input_data_for_2015_outputs.h5`
      (the pre-UrbanSim input archive) is NOT overwritten/clobbered by `ActivitysimPostprocessor.create_usim_input_data`,
      and all unprefixed + year-prefixed tables (`/blocks`, `/jobs`, `/households`, `/persons`, `/cars`, `/2015/cars`)
      are preserved in `custom_mpo_{region_id}_model_data.h5`.
    - Part (b): Run supply-demand loop `iteration=0 -> iteration=1` with `replan_hh_samp_size=2` through
      real `ActivitysimPreprocessor` -> `ActivitysimRunner` -> `ActivitysimPostprocessor` -> `BeamPreprocessor`
      (`copy_plans_from_asim`) -> `BeamRunner` -> `BeamPostprocessor`.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2010,
        end_year=2015,
        travel_model_freq=5,
        replan_hh_samp_size=2,
        enable_archive_copy=True,
    )
    region_id = env.settings.urbansim.region_mappings["region_to_region_id"][
        env.settings.run.region
    ]
    usim_mutable = Path(env.workspace.get_usim_mutable_data_dir())
    base_h5_path = usim_mutable / f"custom_mpo_{region_id}_model_data.h5"

    with ContractValidatingContainerBackend():
        with env.scenario_context("multi_year_datastore_and_replan") as sc:
            # 1. UrbanSim Run & Postprocess (2010 -> 2015)
            execute_step(
                scenario=sc,
                definition=urbansim_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="run",
            )
            execute_step(
                scenario=sc,
                definition=urbansim_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="land_use",
                year=2015,
                iteration=0,
                phase="postprocess",
            )

            archive_h5_path = usim_mutable / "input_data_for_2015_outputs.h5"
            assert archive_h5_path.exists()
            with pd.HDFStore(str(archive_h5_path), mode="a") as store:
                store.put(
                    "/pre_urbansim_marker",
                    pd.DataFrame({"stage": ["pre_urbansim_2010"]}),
                    format="fixed",
                )

            env.state.current_year = 2015
            env.state.forecast_year = 2015
            atlas_state = AtlasSubState(env.state, 2015)

            # 2. ATLAS Preprocess, Run & Postprocess (writes /2015/cars and /cars to datastore)
            execute_step(
                scenario=sc,
                definition=atlas_preprocess,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="preprocess",
            )
            execute_step(
                scenario=sc,
                definition=atlas_run,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="run",
            )
            _, atlas_post_out = execute_step(
                scenario=sc,
                definition=atlas_postprocess,
                settings=env.settings,
                state=atlas_state,
                workspace=env.workspace,
                stage="vehicle_ownership",
                year=2015,
                iteration=0,
                phase="postprocess",
            )
            assert atlas_post_out.usim_datastore_h5 is not None
            _publish_current_h5_alias(
                coupler=sc.coupler,
                fallback_path=atlas_post_out.usim_datastore_h5,
            )

            # 3. Iteration 0: ActivitySim + BEAM
            _, launch_cfg_0 = _run_activitysim_and_beam_preprocess(
                env, sc, year=2015, iteration=0
            )
            run_inputs_0 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
            )
            beam_res_0, _ = execute_step(
                scenario=sc,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg_0},
                resolved_inputs=run_inputs_0,
            )
            post_inputs_0 = beam_postprocess.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
                beam_run_outputs=beam_res_0.outputs,
            )
            execute_step(
                scenario=sc,
                definition=beam_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=0,
                phase="postprocess",
                resolved_inputs=post_inputs_0,
            )

            # Verify (a) `input_data_for_2015_outputs.h5` was NOT clobbered by ActivitysimPostprocessor
            with pd.HDFStore(str(archive_h5_path), mode="r") as archive_store:
                assert "/pre_urbansim_marker" in archive_store.keys(), (
                    "ActivitysimPostprocessor.create_usim_input_data overwrote "
                    "input_data_for_2015_outputs.h5 and destroyed the pre-UrbanSim archive!"
                )

            # Verify (b) `custom_mpo_{region_id}_model_data.h5` still contains unprefixed + ATLAS `/cars`
            with pd.HDFStore(str(base_h5_path), mode="r") as active_store:
                active_keys = set(active_store.keys())
                assert "/blocks" in active_keys
                assert "/jobs" in active_keys
                assert "/households" in active_keys
                assert "/persons" in active_keys
                assert "/cars" in active_keys or "/2015/cars" in active_keys

            # 4. Iteration 1 (replan loop): ActivitySim + ActivitySim Postprocess + BEAM Preprocess/Run/Postprocess
            _, launch_cfg_1 = _run_activitysim_and_beam_preprocess(
                env, sc, year=2015, iteration=1
            )
            assert launch_cfg_1 is not None
            run_inputs_1 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
            )
            beam_res_1, _ = execute_step(
                scenario=sc,
                definition=beam_run,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=1,
                phase="run",
                runtime_kwargs={"beam_launch_config": launch_cfg_1},
                resolved_inputs=run_inputs_1,
            )
            post_inputs_1 = beam_postprocess.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc.coupler,
                beam_run_outputs=beam_res_1.outputs,
            )
            execute_step(
                scenario=sc,
                definition=beam_postprocess,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                stage="supply_demand",
                year=2015,
                iteration=1,
                phase="postprocess",
                resolved_inputs=post_inputs_1,
            )


def _latest_run_dir(root: Path) -> Path:
    matches = sorted(root.glob("pilates-run--*"))
    assert matches, f"Expected a timestamped run directory under {root}"
    return matches[-1]


def _extract_atlas_outyear(command: tuple[str, ...]) -> int:
    for idx, token in enumerate(command):
        if token == "--outyear" and idx + 1 < len(command):
            return int(command[idx + 1])
        if token.startswith("--outyear="):
            return int(token.split("=", 1)[1])
    raise AssertionError(f"Expected --outyear in ATLAS command: {command}")


def test_stage_orchestrators_and_launcher_happy_path_workflow(tmp_path: Path) -> None:
    """
    Scenario 1 (Main Happy-Path Orchestration via Launcher & Stage Runners):
    Drives the 4-model micro-world through `pilates.runtime.launcher.main`, exercising
    `seed_bootstrap_artifacts_to_coupler`, `run_land_use_stage`,
    `run_vehicle_ownership_stage` (both warmstart `2015` and forecast `2017`),
    `run_supply_demand_stage` (`ActivitySim -> BEAM`), `run_postprocessing_stage`,
    `ConsistDbSnapshotManager` archive DB snapshots, and `run_state.yaml` updates.
    """
    from pilates.runtime import launcher

    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        database_enabled=True,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend() as backend:
        launcher.main(settings=env.settings)

    invoked_models = [inv.model_name for inv in backend.invocations]
    assert invoked_models == [
        "urbansim",
        "atlas",
        "atlas",
        "activitysim_numba_warmup",
        "activitysim",
        "beam_run",
    ]
    assert [
        _extract_atlas_outyear(inv.command)
        for inv in backend.invocations
        if inv.model_name == "atlas"
    ] == [2015, 2017]

    archive_run_dir = _latest_run_dir(env.paths.archive_root)
    scratch_run_dir = _latest_run_dir(env.paths.scratch_root)
    snapshot_db = (
        archive_run_dir / ".consist" / "snapshots" / "latest" / "provenance.duckdb"
    )
    if not snapshot_db.is_file():
        snapshot_db = archive_run_dir / ".consist" / "provenance.duckdb"
    assert snapshot_db.is_file()

    import duckdb

    with duckdb.connect(str(snapshot_db), read_only=True) as conn:
        init_snapshot_rows = conn.execute(
            "SELECT role_key, artifact_id FROM run_coupler_snapshot WHERE phase = 'init'"
        ).fetchall()
    init_keys = {row[0] for row in init_snapshot_rows}
    assert USIM_DATASTORE_BASE_H5 in init_keys
    assert USIM_DATASTORE_H5 in init_keys
    assert ASIM_OMX_SKIMS in init_keys
    assert all(row[1] is not None for row in init_snapshot_rows)

    usim_mutable = scratch_run_dir / "urbansim" / "data"
    base_store_path = (
        usim_mutable / f"custom_mpo_{env.settings.urbansim.region_id}_model_data.h5"
    )
    forecast_store_path = usim_mutable / "model_data_2017.h5"
    assert base_store_path.is_file()
    assert forecast_store_path.is_file()
    with pd.HDFStore(str(base_store_path), mode="r") as base_store:
        assert "/households" in base_store.keys()
    with pd.HDFStore(str(forecast_store_path), mode="r") as forecast_store:
        assert "/2017/households" in forecast_store.keys()
        assert "/households" not in forecast_store.keys()
    assert (
        archive_run_dir
        / "consist-recovery"
        / "activitysim_postprocess__supply_demand__y2017__i0__postprocess"
        / "activitysim"
        / "output"
        / "year-2015-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (
        archive_run_dir
        / "consist-recovery"
        / "beam_postprocess__supply_demand__y2015__i0__postprocess"
        / "sfbay"
        / "skims.omx"
    ).is_file()
    assert (
        scratch_run_dir
        / "activitysim"
        / "output"
        / "year-2015-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (scratch_run_dir / "beam" / "input" / "sfbay" / "skims.omx").is_file()


def test_full_workflow_cache_hit_replay_skips_all_containers_on_cold_scratch(
    tmp_path: Path,
) -> None:
    """
    Scenario 2 (Full-Workflow Cache-Hit Replay on Cold Scratch):
    Runs the full 4-model stage-orchestrated workflow (`UrbanSim -> ATLAS -> ActivitySim -> BEAM`),
    promotes the completed run's Consist DB snapshot into `shared.database.path`, wipes the
    mutable node-local scratch workspace (`local_workspace_root`), and runs the full workflow
    a second time. Verifies zero new container executions on the second pass while hydrating
    required outputs from the archive.
    """
    from pilates.runtime import launcher
    from pilates.runtime.promote_run_archive import promote_run_to_recovery_roots

    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        database_enabled=True,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend() as backend:
        launcher.main(settings=env.settings)
        first_run_count = len(backend.invocations)
        assert first_run_count == 6

        first_archive_run_dir = _latest_run_dir(env.paths.archive_root)
        promo = promote_run_to_recovery_roots(
            settings=env.settings,
            archive_run_dir=first_archive_run_dir,
            merge_db_only=True,
            merge_main_db=env.paths.db_path,
        )
        assert promo.success

        # Wipe node-local scratch workspace completely (simulating a cold compute node)
        shutil.rmtree(env.paths.scratch_root)
        env.paths.scratch_root.mkdir(parents=True, exist_ok=True)

        # Second run seeds its local Consist DB from shared.database.path,
        # hits cache on bootstrap + all 4 model stages, and invokes 0 containers
        second_settings = env.settings.model_copy(deep=True)
        launcher.main(settings=second_settings)

        assert len(backend.invocations) == first_run_count

    second_scratch_run_dir = _latest_run_dir(env.paths.scratch_root)
    assert (
        second_scratch_run_dir
        / "activitysim"
        / "output"
        / "year-2015-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (second_scratch_run_dir / "beam" / "input" / "sfbay" / "skims.omx").is_file()


def test_mid_run_interruption_and_restart_from_run_state_on_fresh_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Scenario 3 (Mid-Run Interruption & Restart on a Fresh Node):
    - Case A (Committed Stage Boundary Restart at `supply_demand_loop`):
      Runs `launcher.main` until `supply_demand_loop` (after `land_use` and `vehicle_ownership`
      have completed, archived their outputs, and snapshotted `run_state.yaml` + DuckDB),
      simulates a node interruption, wipes `local_workspace_root`, and restarts via `launcher.main`.
      Verifies `UrbanSim` and `ATLAS` are NOT re-executed while `ActivitySim -> BEAM`
      run to completion using restored coupler artifacts.
    - Case B (Uncommitted Mid-ActivitySim Rewind Restart):
      Simulates an interruption mid-`supply_demand_loop` (`current_sub_stage="activity_demand"`)
      before BEAM commits a checkpoint on a wiped scratch workspace. Verifies
      `_rewind_uncommitted_activitysim_restart_to_provider_boundary` rewinds to `land_use`,
      replays `UrbanSim` and `ATLAS` via Consist cache hits (0 container re-executions),
      and finishes `ActivitySim -> BEAM`.
    """
    from pilates.runtime import launcher

    real_run_supply_demand_stage = launcher.run_supply_demand_stage

    # --- Case A: Committed boundary restart at supply_demand_loop ---
    case_a_dir = tmp_path / "case_a"
    env_a = build_micro_world(
        case_a_dir,
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        database_enabled=True,
        enable_archive_copy=True,
    )

    def _interrupt_at_supply_demand_boundary(**_kwargs):
        raise RuntimeError("Simulated node preemption at supply_demand_loop boundary")

    with ContractValidatingContainerBackend() as backend_a:
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", _interrupt_at_supply_demand_boundary
        )
        with pytest.raises(
            RuntimeError,
            match="Simulated node preemption at supply_demand_loop boundary",
        ):
            launcher.main(settings=env_a.settings)

        assert [inv.model_name for inv in backend_a.invocations] == [
            "urbansim",
            "atlas",
            "atlas",
        ]
        archive_run_dir_a = _latest_run_dir(env_a.paths.archive_root)
        archived_state_a = archive_run_dir_a / "run_state.yaml"
        assert archived_state_a.is_file()

        # Restore real supply_demand stage, wipe node-local scratch, and restart
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", real_run_supply_demand_stage
        )
        shutil.rmtree(env_a.paths.scratch_root)
        env_a.paths.scratch_root.mkdir(parents=True, exist_ok=True)

        restart_settings_a = env_a.settings.model_copy(deep=True)
        restart_settings_a.runtime.options.state_file_loc = str(archived_state_a)
        launcher.main(settings=restart_settings_a)

        # Only ActivitySim (warmup + main) and BEAM should have executed after restart!
        assert [inv.model_name for inv in backend_a.invocations[3:]] == [
            "activitysim_numba_warmup",
            "activitysim",
            "beam_run",
        ]

    # --- Case B: Uncommitted mid-ActivitySim rewind restart ---
    case_b_dir = tmp_path / "case_b"
    env_b = build_micro_world(
        case_b_dir,
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        database_enabled=True,
        enable_archive_copy=True,
    )

    def _interrupt_mid_activitysim(**kwargs):
        ctx = kwargs["context"]
        ctx.state.current_major_stage = ctx.state.Stage.supply_demand_loop
        ctx.state.current_sub_stage = ctx.state.Stage.activity_demand
        ctx.state.sub_stage_progress = "runner"
        ctx.state.write_state()
        raise RuntimeError("Simulated crash mid-ActivitySim before BEAM checkpoint")

    with ContractValidatingContainerBackend() as backend_b:
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", _interrupt_mid_activitysim
        )
        with pytest.raises(
            RuntimeError,
            match="Simulated crash mid-ActivitySim before BEAM checkpoint",
        ):
            launcher.main(settings=env_b.settings)

        pre_restart_count = len(backend_b.invocations)
        assert pre_restart_count == 3
        archive_run_dir_b = _latest_run_dir(env_b.paths.archive_root)
        archived_state_b = archive_run_dir_b / "run_state.yaml"
        assert archived_state_b.is_file()

        # Restore real supply_demand stage, wipe scratch, and restart with allow_rewind_resume=True
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", real_run_supply_demand_stage
        )
        shutil.rmtree(env_b.paths.scratch_root)
        env_b.paths.scratch_root.mkdir(parents=True, exist_ok=True)

        restart_settings_b = env_b.settings.model_copy(deep=True)
        restart_settings_b.runtime.options.state_file_loc = str(archived_state_b)
        restart_settings_b.runtime.options.allow_rewind_resume = True
        launcher.main(settings=restart_settings_b)

        # UrbanSim and ATLAS replay from Consist cache (0 container re-runs);
        # only ActivitySim (warmup + main) and BEAM execute containers!
        assert [
            inv.model_name for inv in backend_b.invocations[pre_restart_count:]
        ] == [
            "activitysim_numba_warmup",
            "activitysim",
            "beam_run",
        ]


def test_stage_skipping_disabled_model_topologies(tmp_path: Path) -> None:
    """
    Scenario 4 (Stage Skipping / Disabled-Model Topologies):
    - Case A (`ATLAS` disabled: `UrbanSim -> ActivitySim -> BEAM`):
      Verifies `USIM_POPULATION_SOURCE_H5` and `USIM_DATASTORE_CURRENT_H5` hand off
      cleanly from `urbansim_postprocess` to `ActivitySim` and `BEAM` when
      `vehicle_ownership` is disabled.
    - Case B (`UrbanSim` + `ATLAS` disabled: standalone `ActivitySim -> BEAM`):
      Verifies bootstrap datastore and initial skims feed `ActivitySim` and `BEAM`
      directly when both `land_use` and `vehicle_ownership` are disabled.
    """
    from pilates.runtime import launcher

    # --- Case A: ATLAS disabled (UrbanSim -> ActivitySim -> BEAM) ---
    env_no_atlas = build_micro_world(
        tmp_path / "no_atlas",
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        vehicle_ownership_model=None,
        database_enabled=True,
        enable_archive_copy=True,
    )
    with ContractValidatingContainerBackend() as backend_no_atlas:
        launcher.main(settings=env_no_atlas.settings)

    assert [inv.model_name for inv in backend_no_atlas.invocations] == [
        "urbansim",
        "activitysim_numba_warmup",
        "activitysim",
        "beam_run",
    ]
    scratch_no_atlas = _latest_run_dir(env_no_atlas.paths.scratch_root)
    assert (
        scratch_no_atlas
        / "activitysim"
        / "output"
        / "year-2015-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (scratch_no_atlas / "beam" / "input" / "sfbay" / "skims.omx").is_file()

    # --- Case B: UrbanSim + ATLAS disabled (standalone ActivitySim -> BEAM) ---
    env_standalone = build_micro_world(
        tmp_path / "standalone_asim_beam",
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=1,
        land_use_model=None,
        vehicle_ownership_model=None,
        database_enabled=True,
        enable_archive_copy=True,
    )
    with ContractValidatingContainerBackend() as backend_standalone:
        launcher.main(settings=env_standalone.settings)

    assert [inv.model_name for inv in backend_standalone.invocations] == [
        "activitysim_numba_warmup",
        "activitysim",
        "beam_run",
    ]
    scratch_standalone = _latest_run_dir(env_standalone.paths.scratch_root)
    assert (
        scratch_standalone
        / "activitysim"
        / "output"
        / "year-2015-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (scratch_standalone / "beam" / "input" / "sfbay" / "skims.omx").is_file()


def test_standalone_beam_with_linkstats_admission_and_run_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise a standalone just-BEAM run with linkstats admission and run publishing."""
    import gzip
    import hashlib
    import json
    import requests
    from pilates.beam.admission import (
        BeamInputAdmissionError,
        LINKSTATS_PROVENANCE_SCHEMA_VERSION,
        write_linkstats_provenance_sidecar,
    )
    from pilates.config.models import (
        BeamAdmissionConfig,
        BeamLinkstatsAdmissionConfig,
        DeclaredDigestExpectation,
    )
    from pilates.runtime import launcher
    from pilates.runtime.promote_run_archive import promote_run_to_recovery_roots

    env = build_micro_world(
        tmp_path / "just_beam",
        start_year=2015,
        end_year=2017,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=2,  # Clamped to 1 outer iteration in BEAM-only mode
        max_year_intervals=1,
        land_use_model=None,
        vehicle_ownership_model=None,
        activity_demand_model=None,
        travel_model="beam",
        database_enabled=True,
        enable_archive_copy=True,
    )
    assert env.settings.write_skims_to_omx is False
    assert env.settings.beam is not None

    # Create a warmstart linkstats file + provenance sidecar in static BEAM inputs
    warmstart_rel = "warmstart_linkstats.csv.gz"
    warmstart_src = env.paths.beam_static_input_dir / "sfbay" / warmstart_rel
    with gzip.open(warmstart_src, "wt", encoding="utf-8") as gz:
        gz.write(
            "link,from,to,hour,length,freespeed,capacity,stat,volume,traveltime\n"
            "1,1,2,8.0,1000.0,13.8,1800.0,AVG,420.0,78.5\n"
        )
    digest_hex = hashlib.sha256(warmstart_src.read_bytes()).hexdigest()
    expected_identity = f"sha256:file:{digest_hex}"
    write_linkstats_provenance_sidecar(
        warmstart_src,
        {
            "schema_version": LINKSTATS_PROVENANCE_SCHEMA_VERSION,
            "identity": expected_identity,
            "source_run_id": "baseline_beam_2015",
            "forecast_year": 2015,
        },
    )
    env.settings.beam.warmstart_linkstats_path = warmstart_rel

    # --- Part A: Strict linkstats admission rejects a mismatched digest before BEAM runs ---
    rejected_scratch = tmp_path / "just_beam_rejected_scratch"
    rejected_archive = tmp_path / "just_beam_rejected_archive"
    rejected_scratch.mkdir(parents=True, exist_ok=True)
    rejected_archive.mkdir(parents=True, exist_ok=True)

    rejected_settings = env.settings.model_copy(deep=True)
    rejected_settings.run.local_workspace_root = str(rejected_scratch)
    rejected_settings.run.output_directory = str(rejected_archive)
    assert rejected_settings.beam is not None
    rejected_settings.beam.admission = BeamAdmissionConfig(
        linkstats=BeamLinkstatsAdmissionConfig(
            mode="strict",
            expectation=DeclaredDigestExpectation(
                identity="sha256:file:" + ("0" * 64),
                source_label="wrong-baseline-digest",
            ),
        )
    )

    with ContractValidatingContainerBackend() as backend_rejected:
        with pytest.raises(
            BeamInputAdmissionError,
            match="BEAM linkstats admission rejected before execution: mismatched",
        ):
            launcher.main(settings=rejected_settings)

    assert backend_rejected.invocations == []
    rejected_archive_run = _latest_run_dir(rejected_archive)
    rejected_report_path = (
        rejected_archive_run / "admission" / "beam-linkstats-warmstart.json"
    )
    assert rejected_report_path.is_file()
    rejected_report = json.loads(rejected_report_path.read_text(encoding="utf-8"))
    assert rejected_report["outcome"] == "mismatched"
    rejected_events_path = rejected_archive_run / ".pilates" / "run_events.jsonl"
    assert rejected_events_path.is_file()
    rejected_events = [
        json.loads(line)
        for line in rejected_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        ev["event_type"] == "failed"
        and ev["run_kind"] == "scenario"
        and "BEAM linkstats admission rejected before execution" in (ev["error"] or "")
        for ev in rejected_events
    )

    # --- Part B: Verified linkstats admission + standalone BEAM execution + publishing ---
    published_webhook_payloads: list[dict[str, object]] = []

    class _DummyResponse:
        status_code = 200
        text = "ok"

        def raise_for_status(self) -> None:
            return None

    def _capture_webhook_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> _DummyResponse:
        published_webhook_payloads.append(
            {"url": url, "json": json, "timeout": timeout}
        )
        return _DummyResponse()

    monkeypatch.setenv("PILATES_GSHEET_PUBLISH", "1")
    monkeypatch.setenv(
        "PILATES_GSHEET_WEBHOOK_URL", "https://example.invalid/pilates-webhook"
    )
    monkeypatch.setenv("PILATES_GSHEET_SECRET", "test-secret")
    monkeypatch.setattr(requests, "post", _capture_webhook_post)

    env.settings.beam.admission = BeamAdmissionConfig(
        linkstats=BeamLinkstatsAdmissionConfig(
            mode="strict",
            expectation=DeclaredDigestExpectation(
                identity=expected_identity,
                source_label="baseline_beam_2015",
            ),
        )
    )

    with ContractValidatingContainerBackend() as backend_verified:
        launcher.main(settings=env.settings)

    # Clamped to 1 outer iteration even though supply_demand_iters=2
    assert [inv.model_name for inv in backend_verified.invocations] == ["beam_run"]

    archive_run_dir = _latest_run_dir(env.paths.archive_root)
    scratch_run_dir = _latest_run_dir(env.paths.scratch_root)

    # Verify linkstats admission report & staged warmstart provenance sidecar
    verified_report_path = (
        archive_run_dir / "admission" / "beam-linkstats-warmstart.json"
    )
    assert verified_report_path.is_file()
    verified_report = json.loads(verified_report_path.read_text(encoding="utf-8"))
    assert verified_report["outcome"] == "verified"
    assert verified_report["observed_artifact_id"] == expected_identity
    assert (
        verified_report["sidecar_provenance"]["source_run_id"] == "baseline_beam_2015"
    )
    staged_warmstarts = list(
        (scratch_run_dir / ".pilates-beam-launch-config").rglob("*.provenance.json")
    )
    assert len(staged_warmstarts) >= 1

    # Verify local JSONL, HTML summary, and webhook run publishing
    events_jsonl = archive_run_dir / ".pilates" / "run_events.jsonl"
    summary_html = archive_run_dir / ".pilates" / "run_summary.html"
    assert events_jsonl.is_file()
    assert summary_html.is_file()
    events = [
        json.loads(line)
        for line in events_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed_models = [ev["model"] for ev in events if ev["event_type"] == "complete"]
    assert "beam_preprocess" in completed_models
    assert "beam_run" in completed_models
    assert "beam_postprocess" in completed_models
    assert any(
        ev["event_type"] == "complete" and ev["run_kind"] == "scenario" for ev in events
    )
    html_text = summary_html.read_text(encoding="utf-8")
    assert "beam_run" in html_text
    assert len(published_webhook_payloads) == len(events)
    assert all(
        item["url"] == "https://example.invalid/pilates-webhook"
        and isinstance(item["json"], dict)
        and item["json"].get("secret") == "test-secret"
        for item in published_webhook_payloads
    )

    # Verify archive promotion to long-term recovery roots
    recovery_root = tmp_path / "just_beam_recovery_root"
    promoted_db = tmp_path / "just_beam_promoted.duckdb"
    promotion_summary = promote_run_to_recovery_roots(
        env.settings,
        archive_run_dir=archive_run_dir,
        roots=[recovery_root],
        merge_main_db=promoted_db,
    )
    assert promotion_summary.success
    assert len(promotion_summary.roots) == 1
    promoted_run_dir = Path(promotion_summary.roots[0].destination_run_dir)
    assert (promoted_run_dir / ".pilates" / "run_events.jsonl").is_file()
    assert (promoted_run_dir / "admission" / "beam-linkstats-warmstart.json").is_file()
    assert promoted_db.is_file()


def test_multi_year_mid_run_restart_rewinds_coupler_to_year_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Scenario 6 (Multi-Year Mid-Run Interruption & Rewind Restart on a Fresh Node):
    Runs a 2-interval multi-year horizon (`2015 -> 2017 -> 2019`, `max_year_intervals=2`).
    Year 2015 completes all 4 stages (`UrbanSim`, `ATLAS`, `ActivitySim`, `BEAM`, `Postprocessing`).
    In Year 2017 (`2017 -> 2019`), `UrbanSim` and `ATLAS` complete and snapshot their
    updated 2019 datastore to DuckDB before an uncommitted crash occurs mid-`ActivitySim`
    in 2017. On restart on a wiped scratch workspace with `allow_rewind_resume=True`:
    - `_rewind_uncommitted_activitysim_restart_to_provider_boundary` rewinds `2017` to `land_use`,
    - `restore_restart_coupler_snapshot` uses `before_step_ordinal` to restore the end-of-2015
      coupler snapshot (not the post-crash 2017 `atlas_postprocess` snapshot),
    - `UrbanSim(2017)` and `ATLAS(2017/2019)` replay via Consist cache hits (0 container re-runs),
    - `ActivitySim(2017) -> BEAM(2017) -> Postprocessing(2017)` execute to completion.
    """
    from pilates.runtime import launcher

    real_run_supply_demand_stage = launcher.run_supply_demand_stage

    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2019,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=2,
        database_enabled=True,
        enable_archive_copy=True,
    )

    def _interrupt_mid_activitysim_in_2017(**kwargs):
        year = kwargs["year"]
        if year == 2015:
            return real_run_supply_demand_stage(**kwargs)
        ctx = kwargs["context"]
        ctx.state.current_major_stage = ctx.state.Stage.supply_demand_loop
        ctx.state.current_sub_stage = ctx.state.Stage.activity_demand
        ctx.state.sub_stage_progress = "runner"
        ctx.state.write_state()
        raise RuntimeError(
            "Simulated crash mid-ActivitySim in year 2017 after 2017 UrbanSim+ATLAS"
        )

    with ContractValidatingContainerBackend() as backend:
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", _interrupt_mid_activitysim_in_2017
        )
        with pytest.raises(
            RuntimeError,
            match="Simulated crash mid-ActivitySim in year 2017 after 2017 UrbanSim\\+ATLAS",
        ):
            launcher.main(settings=env.settings)

        pre_restart_models = [inv.model_name for inv in backend.invocations]
        # Year 2015: urbansim, atlas(2015), atlas(2017), activitysim_numba_warmup, activitysim, beam_run (6)
        # Year 2017 before crash: urbansim, atlas(2017), atlas(2019) (3) -> 9 total
        assert pre_restart_models == [
            "urbansim",
            "atlas",
            "atlas",
            "activitysim_numba_warmup",
            "activitysim",
            "beam_run",
            "urbansim",
            "atlas",
            "atlas",
        ]
        pre_restart_count = len(backend.invocations)

        archive_run_dir = _latest_run_dir(env.paths.archive_root)
        archived_state = archive_run_dir / "run_state.yaml"
        assert archived_state.is_file()

        # Restore real supply_demand stage, wipe node-local scratch, and restart
        monkeypatch.setattr(
            launcher, "run_supply_demand_stage", real_run_supply_demand_stage
        )
        shutil.rmtree(env.paths.scratch_root)
        env.paths.scratch_root.mkdir(parents=True, exist_ok=True)

        restart_settings = env.settings.model_copy(deep=True)
        restart_settings.runtime.options.state_file_loc = str(archived_state)
        restart_settings.runtime.options.allow_rewind_resume = True
        launcher.main(settings=restart_settings)

        # Year 2017 UrbanSim and ATLAS replay from Consist cache (0 container re-runs);
        # only Year 2017 ActivitySim (warmup + main) and BEAM execute containers!
        post_restart_models = [
            inv.model_name for inv in backend.invocations[pre_restart_count:]
        ]
        assert post_restart_models == [
            "activitysim_numba_warmup",
            "activitysim",
            "beam_run",
        ]

    restarted_scratch_dir = _latest_run_dir(env.paths.scratch_root)
    assert (
        restarted_scratch_dir
        / "activitysim"
        / "output"
        / "year-2017-iteration-0"
        / "beam_plans.parquet"
    ).is_file()
    assert (restarted_scratch_dir / "beam" / "input" / "sfbay" / "skims.omx").is_file()


def test_forecast_year_scenario_fork_reuses_base_year_cache_and_branches_forecast_year(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Scenario 7 (Forecast-Year Scenario Forking from a Cached Base Year):
    1. Runs a 2-interval Baseline Scenario (`2015 -> 2017` and `2017 -> 2019`, `max_year_intervals=2`),
       executing all 12 containers across both intervals, and promotes the completed run's
       Consist DB into `shared.database.path`.
    2. Wipes node-local scratch (`local_workspace_root`) and launches a Forked Policy Scenario
       seeded from `shared.database.path` where Year 2015 is identical to baseline, but at the
       start of Year 2017 a policy parameter (`beam.replanning_portion`) is modified.
    3. Verifies:
       - All of Year 2015 (`UrbanSim`, `ATLAS`, `ActivitySim`, `BEAM`, `Postprocessing`) and
         Year 2017 `UrbanSim`, `ATLAS`, and `ActivitySim` hit the Consist cache on cold scratch
         (0 container re-executions) while hydrating the 2015 -> 2017 handoff artifacts,
       - Only Year 2017 `BEAM` (whose `replanning_portion` policy changed) misses the cache
         and executes its container in the forked scenario.
    """
    from pilates.runtime import launcher
    from pilates.runtime.promote_run_archive import promote_run_to_recovery_roots

    real_run_land_use_stage = launcher.run_land_use_stage

    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2019,
        land_use_freq=2,
        travel_model_freq=2,
        vehicle_ownership_freq=2,
        supply_demand_iters=1,
        max_year_intervals=2,
        database_enabled=True,
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend() as backend:
        # 1. Baseline 2-interval run (2015 and 2017)
        # Note: activitysim_numba_warmup only runs once per workspace (in 2015)
        launcher.main(settings=env.settings)
        baseline_models = [inv.model_name for inv in backend.invocations]
        assert baseline_models == [
            # Year 2015
            "urbansim",
            "atlas",
            "atlas",
            "activitysim_numba_warmup",
            "activitysim",
            "beam_run",
            # Year 2017
            "urbansim",
            "atlas",
            "atlas",
            "activitysim",
            "beam_run",
        ]
        baseline_count = len(backend.invocations)

        baseline_archive_dir = _latest_run_dir(env.paths.archive_root)
        promo = promote_run_to_recovery_roots(
            settings=env.settings,
            archive_run_dir=baseline_archive_dir,
            merge_db_only=True,
            merge_main_db=env.paths.db_path,
        )
        assert promo.success

        # 2. Wipe scratch completely and launch Forked Scenario with a Year-2017 BEAM policy tweak
        shutil.rmtree(env.paths.scratch_root)
        env.paths.scratch_root.mkdir(parents=True, exist_ok=True)

        fork_settings = env.settings.model_copy(deep=True)

        def _apply_2017_policy_fork(**kwargs):
            year = kwargs["year"]
            if year == 2017:
                kwargs["context"].settings.beam.replanning_portion = 0.75
            return real_run_land_use_stage(**kwargs)

        monkeypatch.setattr(launcher, "run_land_use_stage", _apply_2017_policy_fork)
        launcher.main(settings=fork_settings)

        # Year 2015 (all 4 models) + Year 2017 UrbanSim, ATLAS, and ActivitySim all hit cache;
        # only Year 2017 beam_run executes a container due to the 2017 BEAM policy change!
        fork_models = [inv.model_name for inv in backend.invocations[baseline_count:]]
        assert fork_models == ["beam_run"]
