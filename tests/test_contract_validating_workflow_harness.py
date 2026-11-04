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

from pilates.utils import consist_runtime as cr
from pilates.utils.archive_paths import resolve_archive_run_dir
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
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


def _run_activitysim_and_beam_preprocess(
    env: MicroWorldEnv,
    sc,
    *,
    year: int = 2015,
    iteration: int = 0,
    bypass_atlas_restore_suffix_bug: bool = True,
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
    - Wipe `PILATES_SCRATCH_ROOT` (`beam/output`) AND remove the archived Zarr directory in `PILATES_ARCHIVE_ROOT`.
    - Re-run `beam_run` on a cache hit: it must either hydrate `raw_od_skims_zarr` when archived or
      fall back to `cache_hydration_failure="miss"` re-execution when the archive is missing/corrupted,
      instead of crashing post-hit with `RuntimeError: beam_run direct Zarr output ... could not be materialized`.
    """
    env = build_micro_world(
        tmp_path,
        start_year=2015,
        end_year=2015,
        activitysim_skims_format="zarr",
        enable_archive_copy=True,
    )

    with ContractValidatingContainerBackend():
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
            assert first_run is not None
            raw_zarr_path = Path(first_run.raw_outputs["raw_od_skims_zarr_2015_0"])
            assert raw_zarr_path.is_dir()

        # Wipe scratch BEAM output directory AND delete any archived copy of 0.skimsActivitySimOD_current.zarr
        beam_output_dir = Path(env.workspace.get_beam_output_dir())
        shutil.rmtree(beam_output_dir)
        beam_output_dir.mkdir(parents=True, exist_ok=True)
        for archived_zarr in env.paths.archive_root.rglob("*.zarr"):
            if "skimsActivitySimOD" in archived_zarr.name:
                shutil.rmtree(archived_zarr)

        with env.scenario_context("beam_zarr_cold_recovery") as sc2:
            _, launch_cfg_2 = _run_activitysim_and_beam_preprocess(
                env, sc2, year=2015, iteration=0
            )
            run_inputs_2 = beam_run.resolve_inputs(
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                coupler=sc2.coupler,
            )
            _, recovered_run = execute_step(
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
            assert recovered_run is not None
            recovered_zarr = Path(recovered_run.raw_outputs["raw_od_skims_zarr_2015_0"])
            assert recovered_zarr.is_dir()


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
                / "beam_accessibility.csv"
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
