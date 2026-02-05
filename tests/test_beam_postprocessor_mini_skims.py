from pathlib import Path
import shutil

import numpy as np
import xarray as xr
import pandas as pd

from pilates.beam import postprocessor as pp
from pilates.beam.postprocessor import copy_skims_for_unobserved_modes


def _copy_mini_skims(tmp_path: Path) -> Path:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "skims"
    source = fixtures_dir / "mini_skims.zarr"
    target = tmp_path / "mini_skims.zarr"
    shutil.copytree(source, target)
    return target


def _copy_expected_skims(tmp_path: Path) -> Path:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "skims"
    source = fixtures_dir / "mini_skims_expected.zarr"
    target = tmp_path / "mini_skims_expected.zarr"
    shutil.copytree(source, target)
    return target


def _write_zeroed_main_from_mini(tmp_path: Path, mini_path: Path) -> Path:
    main_path = tmp_path / "main_skims.zarr"
    src = xr.open_zarr(mini_path, consolidated=True)
    try:
        zero_vars = {}
        for name, da in src.data_vars.items():
            zero_vars[name] = (da.dims, np.zeros(da.shape, dtype=da.dtype))
        ds = xr.Dataset(zero_vars, coords=src.coords, attrs=src.attrs)
        for name in ds.variables:
            ds[name].encoding = {}
        ds.to_zarr(main_path, mode="w", consolidated=True, zarr_format=2)
    finally:
        src.close()
    return main_path


def test_mini_skims_fixture_shape(tmp_path) -> None:
    zarr_path = _copy_mini_skims(tmp_path)
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        assert ds.sizes["otaz"] == 5
        assert ds.sizes["dtaz"] == 5
        assert ds.sizes["time_period"] == 5
        assert "SOV_TRIPS" in ds.data_vars
        assert ds["SOV_TRIPS"].shape == (5, 5, 5)
        assert "DISTWALK" in ds.data_vars
        assert ds["DISTWALK"].shape == (5, 5)
    finally:
        ds.close()


def test_copy_skims_for_unobserved_modes_overwrites(tmp_path) -> None:
    zarr_path = _copy_mini_skims(tmp_path)
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        # Force a difference so we can verify overwrite behavior.
        ds["HOV2_TIME"].data[:] = 0.0
        before = ds["HOV2_TIME"].values.copy()
        copy_skims_for_unobserved_modes({"SOV": ["HOV2"]}, ds)
        after = ds["HOV2_TIME"].values
        assert not np.array_equal(before, after)
        assert np.array_equal(after, ds["SOV_TIME"].values)
    finally:
        ds.close()


def test_merge_mini_skims_matches_expected(tmp_path, monkeypatch) -> None:
    mini_path = _copy_mini_skims(tmp_path)
    expected_path = _copy_expected_skims(tmp_path)
    main_path = _write_zeroed_main_from_mini(tmp_path, mini_path)

    monkeypatch.setattr(
        pp, "verify_skim_zone_order", lambda settings, skim_file_path, workspace: list(range(5))
    )
    monkeypatch.setattr(
        pp.zone_utils, "load_canonical_zones", lambda settings, workspace: pd.DataFrame(index=range(5))
    )
    monkeypatch.setattr(pp, "ensure_0_based_and_flag_zarr_skims", lambda *args, **kwargs: None)

    def _get_setting(settings, key, default=None):
        if key == "shared.skims.periods":
            return ["EA", "AM", "MD", "PM", "EV"]
        if key == "consolidate_tnc_fleets":
            return True
        if key == "beam.skim_previous_weight":
            return 0.9
        if key == "beam.ridehail_path_map":
            return default
        return default

    monkeypatch.setattr(pp, "get_setting", _get_setting)

    pp._merge_beam_skims_to_zarr(
        all_skims_path=str(main_path),
        iteration_skims_path=str(mini_path),
        beam_output_dir="",
        settings=None,
        workspace=None,
        override=str(mini_path),
    )

    expected = xr.open_zarr(expected_path, consolidated=True)
    actual = xr.open_zarr(main_path, consolidated=True)
    try:
        assert set(expected.data_vars) == set(actual.data_vars)
        for name in expected.data_vars:
            np.testing.assert_allclose(
                actual[name].values,
                expected[name].values,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Mismatch for {name}",
            )
    finally:
        expected.close()
        actual.close()
