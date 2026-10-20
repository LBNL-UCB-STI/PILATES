"""Unit tests for PILATES BEAM full-skim OMX -> ActivitySim OMX converter."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from pilates.beam.skim_provenance import (
    read_skim_provenance_sidecar,
    skim_provenance_sidecar_path,
)
from pilates.beam.full_skim_converter import (
    FullSkimConversionConfig,
    convert_beam_full_skims_to_activitysim_omx,
    is_omx_already_converted,
    run_tracked_full_skim_conversion,
)


def _create_synthetic_raw_beam_omx(path: Path, n_zones: int = 4) -> Path:
    """Create a synthetic raw BEAM FullSkimsCreatorApp OMX file with zero diagonals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["OMX_VERSION"] = b"0.2"
        f.attrs["SHAPE"] = np.array([n_zones, n_zones], dtype=np.int32)
        data_grp = f.create_group("data")
        lookup_grp = f.create_group("lookup")
        lookup_grp.create_dataset("TAZ", data=np.arange(1, n_zones + 1, dtype=np.int64))

        # Base distance matrix with zero diagonal
        sov_dist = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.2, 0.0, 1.5, 2.5],
                [2.0, 1.5, 0.0, 1.8],
                [3.0, 2.5, 1.8, 0.0],
            ],
            dtype=np.float64,
        )[:n_zones, :n_zones]
        sov_time = sov_dist * 2.5  # 2.5 min per mile
        sovtoll_dist = sov_dist * 0.95
        sovtoll_time = sov_time * 0.85
        sovtoll_btoll = np.full((n_zones, n_zones), 700.0, dtype=np.float64)
        np.fill_diagonal(sovtoll_btoll, 0.0)
        sovtoll_vtoll = sovtoll_dist * 25.0

        walk_dist = sov_dist * 1.05
        walk_time = walk_dist * 20.0
        bike_dist = sov_dist * 1.02
        bike_time = bike_dist * 6.0

        for tp in ("EA", "AM", "MD", "PM", "EV"):
            data_grp.create_dataset(f"SOV_DIST__{tp}", data=sov_dist)
            data_grp.create_dataset(f"SOV_TIME__{tp}", data=sov_time)
            data_grp.create_dataset(
                f"SOV_BTOLL__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"SOV_VTOLL__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"SOV_TRIPS__{tp}",
                data=np.ones((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"SOV_FAILURES__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )

            data_grp.create_dataset(f"SOVTOLL_DIST__{tp}", data=sovtoll_dist)
            data_grp.create_dataset(f"SOVTOLL_TIME__{tp}", data=sovtoll_time)
            data_grp.create_dataset(f"SOVTOLL_BTOLL__{tp}", data=sovtoll_btoll)
            data_grp.create_dataset(f"SOVTOLL_VTOLL__{tp}", data=sovtoll_vtoll)
            data_grp.create_dataset(
                f"SOVTOLL_TRIPS__{tp}",
                data=np.ones((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"SOVTOLL_FAILURES__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )

            data_grp.create_dataset(f"WALK_DIST__{tp}", data=walk_dist)
            data_grp.create_dataset(f"WALK_TIME__{tp}", data=walk_time)
            data_grp.create_dataset(f"BIKE_DIST__{tp}", data=bike_dist)
            data_grp.create_dataset(f"BIKE_TIME__{tp}", data=bike_time)

            # Transit: WLK_LOC_WLK (20 min IVT everywhere off-diagonal)
            loc_ivt = np.where(sov_dist > 0, 20.0, 0.0)
            loc_iwait = np.where(sov_dist > 0, 5.0, 0.0)
            loc_xwait = np.where(sov_dist > 0, 3.0, 0.0)
            loc_wacc = np.where(sov_dist > 0, 4.0, 0.0)
            loc_waux = np.where(sov_dist > 0, 1.0, 0.0)
            loc_wegr = np.where(sov_dist > 0, 4.0, 0.0)
            loc_far = np.where(sov_dist > 0, 250.0, 0.0)
            loc_boards = np.where(sov_dist > 0, 1.0, 0.0)

            data_grp.create_dataset(f"WLK_LOC_WLK_TOTIVT__{tp}", data=loc_ivt)
            data_grp.create_dataset(f"WLK_LOC_WLK_IVT__{tp}", data=loc_ivt)
            data_grp.create_dataset(f"WLK_LOC_WLK_IWAIT__{tp}", data=loc_iwait)
            data_grp.create_dataset(f"WLK_LOC_WLK_XWAIT__{tp}", data=loc_xwait)
            data_grp.create_dataset(f"WLK_LOC_WLK_WACC__{tp}", data=loc_wacc)
            data_grp.create_dataset(f"WLK_LOC_WLK_WAUX__{tp}", data=loc_waux)
            data_grp.create_dataset(f"WLK_LOC_WLK_WEGR__{tp}", data=loc_wegr)
            data_grp.create_dataset(f"WLK_LOC_WLK_FAR__{tp}", data=loc_far)
            data_grp.create_dataset(f"WLK_LOC_WLK_BOARDS__{tp}", data=loc_boards)

            # WLK_HVY_WLK: faster (12 min IVT) on cell (0, 2)
            hvy_ivt = np.zeros((n_zones, n_zones), dtype=np.float64)
            hvy_ivt[0, 2] = 12.0
            hvy_far = np.zeros((n_zones, n_zones), dtype=np.float64)
            hvy_far[0, 2] = 450.0
            data_grp.create_dataset(f"WLK_HVY_WLK_TOTIVT__{tp}", data=hvy_ivt)
            data_grp.create_dataset(f"WLK_HVY_WLK_KEYIVT__{tp}", data=hvy_ivt)
            data_grp.create_dataset(
                f"WLK_HVY_WLK_IWAIT__{tp}",
                data=np.where(hvy_ivt > 0, 4.0, 0.0),
            )
            data_grp.create_dataset(
                f"WLK_HVY_WLK_XWAIT__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"WLK_HVY_WLK_WACC__{tp}",
                data=np.where(hvy_ivt > 0, 6.0, 0.0),
            )
            data_grp.create_dataset(
                f"WLK_HVY_WLK_WAUX__{tp}",
                data=np.zeros((n_zones, n_zones), dtype=np.float64),
            )
            data_grp.create_dataset(
                f"WLK_HVY_WLK_WEGR__{tp}",
                data=np.where(hvy_ivt > 0, 5.0, 0.0),
            )
            data_grp.create_dataset(f"WLK_HVY_WLK_FAR__{tp}", data=hvy_far)
            data_grp.create_dataset(
                f"WLK_HVY_WLK_BOARDS__{tp}",
                data=np.where(hvy_ivt > 0, 1.0, 0.0),
            )

            # Drive-transit DRV_HVY_WLK
            drv_ddist = np.where(sov_dist > 0, 1.5, 0.0)
            drv_dtim = np.where(sov_dist > 0, 4.0, 0.0)
            data_grp.create_dataset(f"DRV_HVY_WLK_TOTIVT__{tp}", data=loc_ivt)
            data_grp.create_dataset(f"DRV_HVY_WLK_KEYIVT__{tp}", data=loc_ivt)
            data_grp.create_dataset(f"DRV_HVY_WLK_DDIST__{tp}", data=drv_ddist)
            data_grp.create_dataset(f"DRV_HVY_WLK_DTIM__{tp}", data=drv_dtim)
            data_grp.create_dataset(f"DRV_HVY_WLK_IWAIT__{tp}", data=loc_iwait)
            data_grp.create_dataset(f"DRV_HVY_WLK_XWAIT__{tp}", data=loc_xwait)
            data_grp.create_dataset(f"DRV_HVY_WLK_WAUX__{tp}", data=loc_waux)
            data_grp.create_dataset(f"DRV_HVY_WLK_WEGR__{tp}", data=loc_wegr)
            data_grp.create_dataset(f"DRV_HVY_WLK_FAR__{tp}", data=loc_far)
            data_grp.create_dataset(f"DRV_HVY_WLK_BOARDS__{tp}", data=loc_boards)

            # Non-zero Express Bus in raw input (to verify disable_express_bus zeroes it)
            data_grp.create_dataset(f"WLK_EXP_WLK_TOTIVT__{tp}", data=loc_ivt)
            data_grp.create_dataset(f"WLK_EXP_WLK_FAR__{tp}", data=loc_far)

            # Inject NaNs in unrouted transit cells and WALK_DIST (> 3 miles) plus a negative IWAIT
            drv_com_ivt = np.full((n_zones, n_zones), np.nan, dtype=np.float64)
            drv_com_ivt[0, 3] = 18.0
            drv_com_iwait = np.full((n_zones, n_zones), np.nan, dtype=np.float64)
            drv_com_iwait[0, 3] = -0.5  # negative artifact to be sanitized to 0.0
            data_grp.create_dataset(f"DRV_COM_WLK_TOTIVT__{tp}", data=drv_com_ivt)
            data_grp.create_dataset(f"DRV_COM_WLK_IWAIT__{tp}", data=drv_com_iwait)

            walk_dist_with_nan = walk_dist.copy()
            walk_dist_with_nan[0, 3] = np.nan  # unrouted long walk
            del data_grp[f"WALK_DIST__{tp}"]
            data_grp.create_dataset(f"WALK_DIST__{tp}", data=walk_dist_with_nan)

    return path


def test_convert_beam_full_skims_to_activitysim_omx(tmp_path: Path) -> None:
    """Verify scaling, HOV copy, EXP disable, RH multipliers, WLK_TRN_WLK, NaN sanitization, and diagonal imputation."""
    raw_omx = _create_synthetic_raw_beam_omx(tmp_path / "raw_skims.omx")
    out_omx = tmp_path / "activitysim_skims.omx"

    cfg = FullSkimConversionConfig(
        transit_scale_factor=100.0,
        disable_express_bus=True,
        copy_hov_from_sov=True,
        synthesize_ridehail_from_sovtoll=True,
        rh_pooled_time_multiplier=1.15,
        rh_pooled_dist_multiplier=1.15,
        rh_solo_wait_minutes=5.0,
        rh_pooled_wait_minutes=7.0,
    )
    report = convert_beam_full_skims_to_activitysim_omx(raw_omx, out_omx, config=cfg)

    assert out_omx.is_file()
    assert is_omx_already_converted(out_omx)
    assert not is_omx_already_converted(raw_omx)
    assert report.zone_count == 4
    assert report.intrazonal_imputed_matrix_count > 0

    with h5py.File(out_omx, "r") as f:
        data = f["data"]
        lookup = f["lookup"]

        # 0. Zero NaNs, zero Infs, zero negative values across ALL matrices
        for mat_name in data.keys():
            arr = data[mat_name][:]
            assert not np.any(np.isnan(arr)), f"NaN found in {mat_name}"
            assert not np.any(np.isinf(arr)), f"Inf found in {mat_name}"
            assert not np.any(arr < 0.0), f"Negative value found in {mat_name}"

        # 1. Dual zone lookups + zone_number
        assert "TAZ" in lookup
        assert "zone_id" in lookup
        assert "zone_number" in lookup
        np.testing.assert_array_equal(lookup["TAZ"][:], [1, 2, 3, 4])
        np.testing.assert_array_equal(lookup["zone_id"][:], [1, 2, 3, 4])
        np.testing.assert_array_equal(lookup["zone_number"][:], [1, 2, 3, 4])

        # Logical matrices SHA-256 stamped, dynamic timestamp omitted from HDF5 attrs
        assert "logical_matrices_sha256" in f.attrs
        assert f.attrs["logical_matrices_sha256"].startswith("sha256:logical_matrices:")
        assert "pilates_converted_at" not in f.attrs

        # 2. Intrazonal diagonal imputation & WALK_DIST NaN backfill
        dist_2d = data["DIST"][:]
        assert np.all(np.diag(dist_2d) > 0.0)
        # Row 0 has off-diagonals [1.0, 2.0, 3.0] -> mean(3 nearest) = 2.0 -> * 0.5 = 1.0
        assert pytest.approx(dist_2d[0, 0], rel=1e-5) == 1.0
        assert "DISTBIKE" in data
        assert "DISTWALK" in data
        # Cell (0, 3) in WALK_DIST was NaN -> backfilled from SOV_DIST (3.0)
        assert pytest.approx(data["DISTWALK"][0, 3], rel=1e-5) == 3.0
        assert np.all(np.diag(data["SOV_TIME__AM"][:]) > 0.0)

        # 3. Transit 100x scaling vs unscaled FAR/BOARDS
        # Raw WLK_LOC_WLK_TOTIVT__AM[0, 1] was 20.0 -> scaled 2000.0
        assert pytest.approx(data["WLK_LOC_WLK_TOTIVT__AM"][0, 1]) == 2000.0
        # WAIT = IWAIT (500) + XWAIT (300) = 800
        assert pytest.approx(data["WLK_LOC_WLK_WAIT__AM"][0, 1]) == 800.0
        # FAR remains 250.0 cents (unscaled)
        assert pytest.approx(data["WLK_LOC_WLK_FAR__AM"][0, 1]) == 250.0
        assert pytest.approx(data["WLK_LOC_WLK_BOARDS__AM"][0, 1]) == 1.0
        # DRV_HVY_WLK_DDIST scaled by 100x (1.5 -> 150.0)
        assert pytest.approx(data["DRV_HVY_WLK_DDIST__AM"][0, 1]) == 150.0

        # 4. Express Bus disabled (all zero across WLK_EXP_WLK, DRV_EXP_WLK, WLK_EXP_DRV)
        assert np.all(data["WLK_EXP_WLK_TOTIVT__AM"][:] == 0.0)
        assert np.all(data["DRV_EXP_WLK_TOTIVT__AM"][:] == 0.0)
        assert np.all(data["WLK_EXP_DRV_TOTIVT__AM"][:] == 0.0)

        # 5. General walk-transit WLK_TRN_WLK picks min positive TOTIVT
        # Cell (0, 1): only LOC is available (2000.0, FAR 250.0)
        assert pytest.approx(data["WLK_TRN_WLK_IVT__AM"][0, 1]) == 2000.0
        assert pytest.approx(data["WLK_TRN_WLK_FAR__AM"][0, 1]) == 250.0
        # Cell (0, 2): HVY has TOTIVT 1200.0 < LOC 2000.0 -> picks HVY (FAR 450.0)
        assert pytest.approx(data["WLK_TRN_WLK_IVT__AM"][0, 2]) == 1200.0
        assert pytest.approx(data["WLK_TRN_WLK_FAR__AM"][0, 2]) == 450.0

        # 6. HOV2 / HOV3 copied from SOV; HOV2TOLL / HOV3TOLL copied from SOVTOLL
        np.testing.assert_allclose(data["HOV2_TIME__AM"][:], data["SOV_TIME__AM"][:])
        np.testing.assert_allclose(data["HOV3_DIST__PM"][:], data["SOV_DIST__PM"][:])
        np.testing.assert_allclose(
            data["HOV2TOLL_TIME__AM"][:], data["SOVTOLL_TIME__AM"][:]
        )
        np.testing.assert_allclose(
            data["HOV3TOLL_BTOLL__AM"][:], data["SOVTOLL_BTOLL__AM"][:]
        )

        # 7. Ridehail (RH_SOLO = 1.0x SOVTOLL, RH_POOLED & RH_SHARED = 1.15x SOVTOLL)
        np.testing.assert_allclose(
            data["RH_SOLO_TOTIVT__AM"][:], data["SOVTOLL_TIME__AM"][:]
        )
        np.testing.assert_allclose(
            data["RH_SOLO_TIME__AM"][:], data["SOVTOLL_TIME__AM"][:]
        )
        np.testing.assert_allclose(
            data["RH_POOLED_TOTIVT__AM"][:],
            data["SOVTOLL_TIME__AM"][:] * np.float32(1.15),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            data["RH_SHARED_TIME__AM"][:],
            data["SOVTOLL_TIME__AM"][:] * np.float32(1.15),
            rtol=1e-5,
        )
        assert pytest.approx(data["RH_SOLO_WAIT__AM"][0, 1]) == 5.0
        assert pytest.approx(data["RH_POOLED_WAIT__AM"][0, 1]) == 7.0
        assert np.all(data["RH_SOLO_REJECTIONPROB__AM"][:] == 0.0)

    # 8. Idempotency guard refuses to double-scale an already-converted OMX file
    with pytest.raises(ValueError, match="already has"):
        convert_beam_full_skims_to_activitysim_omx(
            out_omx, tmp_path / "double_converted.omx", config=cfg
        )


def test_run_tracked_full_skim_conversion_records_consist_run_and_sidecar(
    tmp_path: Path,
) -> None:
    """Verify standalone Consist-tracked conversion logs artifacts and writes .provenance.json."""
    raw_omx = _create_synthetic_raw_beam_omx(tmp_path / "raw_skims.omx")
    out_omx = tmp_path / "converted" / "skims.omx"
    db_path = tmp_path / ".consist" / "provenance.duckdb"

    cfg = FullSkimConversionConfig()
    report = run_tracked_full_skim_conversion(
        raw_omx,
        out_omx,
        config=cfg,
        region="sfbay",
        year=2019,
        iteration=0,
        tracker_db_path=db_path,
        run_dir=tmp_path / "consist_runs",
    )

    assert out_omx.is_file()
    assert report.output_matrix_count > 100
    sidecar = read_skim_provenance_sidecar(skim_provenance_sidecar_path(out_omx))
    assert sidecar is not None
    assert sidecar["producer"]["step"] == "beam_full_skim_omx_converter"
    assert sidecar["producer"]["region"] == "sfbay"
    assert sidecar["producer"]["year"] == 2019
    assert sidecar["identity"].startswith("sha256:file:")
    assert sidecar["skim_metadata"]["logical_matrices_sha256"].startswith(
        "sha256:logical_matrices:"
    )
    assert "pilates_converted_at" in sidecar["skim_metadata"]


def test_native_beam_full_skim_converts_final_skims_omx(tmp_path: Path) -> None:
    """Verify _native_beam_full_skim keeps BEAM_FULL_SKIMS raw and converts FINAL_SKIMS_OMX."""
    from pilates.beam.launch_config import BeamLaunchConfig
    from pilates.workflows.artifact_keys import BEAM_FULL_SKIMS, FINAL_SKIMS_OMX
    from pilates.workflows.steps.beam import _native_beam_full_skim

    raw_omx = _create_synthetic_raw_beam_omx(
        tmp_path / "beam_output" / "sfbay" / "year-2019-iteration-0" / "skims.omx"
    )
    final_omx_target = tmp_path / "beam" / "input" / "sfbay" / "skims.omx"

    launch_root = tmp_path / "launch"
    launch_root.mkdir()
    primary_conf = launch_root / "sfbay.conf"
    primary_conf.write_text("# beam conf\n", encoding="utf-8")
    launch_config = BeamLaunchConfig(root=launch_root, primary_config=primary_conf)

    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")
    state = SimpleNamespace(
        run_info_path=str(tmp_path / "run_info.json"),
        year=2019,
        iteration=0,
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            full_skim=SimpleNamespace(convert_to_activitysim_omx=True)
        ),
    )
    consist_ctx = SimpleNamespace(
        run_metadata={},
        declared_outputs={
            BEAM_FULL_SKIMS: raw_omx,
            FINAL_SKIMS_OMX: final_omx_target,
        },
        tracker=None,
        current_run=SimpleNamespace(id="test_full_skim_run"),
    )
    fake_ref = SimpleNamespace(staged_path=tmp_path / "init.linkstats.parquet")

    fn = getattr(_native_beam_full_skim, "__wrapped__", _native_beam_full_skim)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "pilates.workflows.steps.beam.validate_staged_linkstats_reference",
                return_value=fake_ref,
            )
        )
        stack.enter_context(
            patch(
                "pilates.workflows.steps.beam.has_staged_linkstats_admission_policy",
                return_value=False,
            )
        )
        stack.enter_context(
            patch("pilates.workflows.steps.beam.validate_r5_execution_reference")
        )
        mock_runner_cls = stack.enter_context(
            patch("pilates.beam.runner.BeamFullSkimRunner")
        )
        stack.enter_context(
            patch("pilates.workflows.steps.beam._validate_native_outputs")
        )
        stack.enter_context(
            patch(
                "pilates.workflows.steps.beam._beam_full_skim_native_output_paths",
                return_value={
                    BEAM_FULL_SKIMS: raw_omx,
                    FINAL_SKIMS_OMX: final_omx_target,
                },
            )
        )
        mock_runner_cls.return_value.run.return_value = SimpleNamespace(
            full_skims=raw_omx,
            final_skims_omx=raw_omx,
        )
        fn(
            settings=settings,
            state=state,
            workspace=workspace,
            beam_launch_config=launch_config,
            _consist_ctx=consist_ctx,
        )

    assert raw_omx.is_file()
    assert final_omx_target.is_file()
    assert not is_omx_already_converted(raw_omx)
    assert is_omx_already_converted(final_omx_target)
    assert "full_skim_conversion" in consist_ctx.run_metadata

    raw_sidecar = read_skim_provenance_sidecar(skim_provenance_sidecar_path(raw_omx))
    final_sidecar = read_skim_provenance_sidecar(
        skim_provenance_sidecar_path(final_omx_target)
    )
    assert raw_sidecar is not None
    assert final_sidecar is not None
    assert raw_sidecar["producer"]["step"] == "beam_full_skim"
    assert final_sidecar["producer"]["step"] == "beam_full_skim"
    assert raw_sidecar["identity"] != final_sidecar["identity"]
