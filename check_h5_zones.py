#!/usr/bin/env python
"""Check what's stored in the H5 zone_geoms table."""

import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python check_h5_zones.py <path_to_h5_file>")
    sys.exit(1)

h5_path = sys.argv[1]

print(f"Opening {h5_path}")
store = pd.HDFStore(h5_path, mode='r')

print("\nKeys in H5 file:")
for key in store.keys():
    print(f"  {key}")

# Look for zone_geoms
zone_keys = [k for k in store.keys() if 'zone_geoms' in k]

if zone_keys:
    for zone_key in zone_keys:
        print(f"\n{zone_key}:")
        zones = store[zone_key]
        print(f"  Shape: {zones.shape}")
        print(f"  Index name: {zones.index.name}")
        print(f"  Index dtype: {zones.index.dtype}")
        print(f"  First 10 index values: {zones.index[:10].tolist()}")

        if 'GEOID' in zones.columns:
            print(f"  GEOID column dtype: {zones['GEOID'].dtype}")
            print(f"  First 10 GEOID values: {zones['GEOID'].head(10).tolist()}")
else:
    print("\nNo zone_geoms tables found")

store.close()
