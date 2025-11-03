#!/usr/bin/env python3
"""
Create minimal zarr fixture for testing.

This creates a tiny zarr store with the same structure as PILATES skims
but with minimal data (3 zones, 1 time period, 2 variables).
"""

import numpy as np
import xarray as xr
import zarr
from pathlib import Path


def create_minimal_zarr():
    """Create minimal zarr fixture."""
    fixtures_dir = Path(__file__).parent
    zarr_path = fixtures_dir / "minimal_skims.zarr"

    # Minimal dimensions
    n_zones = 3
    time_periods = ["AM"]

    # Coordinates
    coords = {
        "otaz": np.array([1, 2, 3]),
        "dtaz": np.array([1, 2, 3]),
        "time_period": time_periods,
    }

    # Create minimal skim data (3x3x1 for each variable)
    # Shape should be (otaz, dtaz, time_period) = (3, 3, 1)
    sov_time = np.array([
        [[10.5], [12.0], [14.5]],  # zone 1 to zones 1,2,3
        [[12.0], [15.2], [18.0]],  # zone 2 to zones 1,2,3
        [[14.5], [18.0], [20.1]],  # zone 3 to zones 1,2,3
    ])

    hov2_time = np.array([
        [[11.0], [13.0], [15.0]],
        [[13.0], [16.0], [19.0]],
        [[15.0], [19.0], [21.0]],
    ])

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "SOV_TIME": (["otaz", "dtaz", "time_period"], sov_time),
            "HOV2_TIME": (["otaz", "dtaz", "time_period"], hov2_time),
        },
        coords=coords,
    )

    # Save as zarr (force v2 format for PILATES compatibility)
    print(f"Creating minimal zarr at {zarr_path}")

    # Remove existing if present
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    # Use zarr v2 explicitly
    import os
    os.environ['ZARR_V3_EXPERIMENTAL_API'] = '0'

    # Save with xarray (defaults to zarr v2 when v3 is not enabled)
    ds.to_zarr(zarr_path, mode="w")

    # Verify
    ds_loaded = xr.open_zarr(zarr_path)
    print(f"\nCreated zarr with:")
    print(f"  Variables: {list(ds_loaded.data_vars)}")
    print(f"  Coordinates: {list(ds_loaded.coords)}")
    print(f"  Shape: {ds_loaded['SOV_TIME'].shape}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
    print(f"  Size: {total_size / 1024:.1f} KB")

    print(f"\n✅ Minimal zarr fixture created successfully!")


if __name__ == "__main__":
    create_minimal_zarr()
