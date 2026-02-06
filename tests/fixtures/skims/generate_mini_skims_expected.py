#!/usr/bin/env python3
"""
Regenerate the golden output fixture for mini skims.

Usage:
  /Users/zaneedell/miniforge3/envs/PILATES/bin/python \
    tests/fixtures/skims/generate_mini_skims_expected.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pilates.beam import postprocessor as pp


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures" / "skims"
    mini_path = fixtures_dir / "mini_skims.zarr"
    expected_path = fixtures_dir / "mini_skims_expected.zarr"

    if not mini_path.exists():
        raise FileNotFoundError(f"Missing mini skims fixture: {mini_path}")

    work_dir = Path("/tmp/pilates_mini_skims_build")
    main_path = work_dir / "main_skims.zarr"
    partial_path = work_dir / "partial_skims.zarr"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(mini_path, partial_path)

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

    # Patch dependencies to keep deterministic + avoid filesystem requirements.
    pp.verify_skim_zone_order = lambda settings, skim_file_path, workspace: list(range(5))
    pp.zone_utils.load_canonical_zones = (
        lambda settings, workspace: pd.DataFrame(index=range(5))
    )
    pp.ensure_0_based_and_flag_zarr_skims = lambda *args, **kwargs: None

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

    pp.get_setting = _get_setting

    pp._merge_beam_skims_to_zarr(
        all_skims_path=str(main_path),
        iteration_skims_path=str(partial_path),
        beam_output_dir="",
        settings=None,
        workspace=None,
        override=str(partial_path),
    )

    if expected_path.exists():
        shutil.rmtree(expected_path)
    shutil.copytree(main_path, expected_path)
    print(f"Wrote expected fixture to {expected_path}")


if __name__ == "__main__":
    # Avoid matplotlib cache warnings in some environments.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    main()
