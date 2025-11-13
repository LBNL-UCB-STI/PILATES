#!/usr/bin/env python
"""
Clear corrupted zone_geoms cache from H5 datastore.

This script removes the cached zone geometries that were stored with the
integer conversion bug. The next run will re-download and cache correctly.

Usage:
    python clear_zone_cache.py <path_to_h5_file>

Example:
    python clear_zone_cache.py /global/scratch/users/zaneedell/pilates-output/new-asim-fast-beam-bigger-pool-20251113-153418/urbansim/data/custom_mpo_53199100_model_data.h5
"""

import sys
import os
import pandas as pd


def clear_zone_cache(h5_path: str):
    """Remove zone_geoms from H5 file."""

    if not os.path.exists(h5_path):
        print(f"Error: H5 file not found: {h5_path}")
        return False

    print(f"Opening {h5_path}")

    try:
        # Open in append mode to allow modifications
        store = pd.HDFStore(h5_path, mode='a')

        # Find all zone_geoms keys
        zone_keys = [k for k in store.keys() if 'zone_geoms' in k]

        if not zone_keys:
            print("No zone_geoms tables found in H5 file")
            store.close()
            return True

        print(f"\nFound {len(zone_keys)} zone_geoms table(s):")
        for key in zone_keys:
            print(f"  {key}")

        # Check if they're corrupted
        for key in zone_keys:
            zones = store[key]

            # Get sample ID
            if zones.index.name and zones.index.name != 'index':
                sample_id = str(zones.index[0]) if len(zones) > 0 else ""
            elif 'GEOID' in zones.columns:
                sample_id = str(zones['GEOID'].iloc[0]) if len(zones) > 0 else ""
            else:
                sample_id = ""

            print(f"\n{key}:")
            print(f"  Sample zone ID: '{sample_id}'")
            print(f"  Shape: {zones.shape}")

            # Determine if corrupted (short ID when we expect GEOID)
            zone_type = key.replace('/','').replace('_zone_geoms', '')
            is_census = zone_type in ['block', 'block_group', 'tract']
            is_corrupted = is_census and len(sample_id) < 12

            if is_corrupted:
                print(f"  ❌ CORRUPTED (expected 12+ digit GEOID, got '{sample_id}')")
                print(f"  Deleting {key}...")
                del store[key]
                print(f"  ✓ Deleted")
            else:
                print(f"  ✓ Appears valid")

        store.close()
        print("\n✓ Done")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    h5_path = sys.argv[1]
    success = clear_zone_cache(h5_path)
    sys.exit(0 if success else 1)
