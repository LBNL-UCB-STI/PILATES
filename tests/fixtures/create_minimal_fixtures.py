#!/usr/bin/env python3
"""
Create minimal fixture data for stub-based provenance testing.

This script generates tiny but realistic data files that mimic the structure
of actual PILATES model outputs, allowing fast integration tests.
"""

import os
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent


def create_minimal_urbansim_h5():
    """Create minimal UrbanSim H5 file with realistic structure."""
    print("📦 Creating minimal UrbanSim H5 file...")

    h5_path = FIXTURES_DIR / "minimal_urbansim_2017.h5"

    # Create minimal but realistic data
    households_data = pd.DataFrame({
        'household_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'building_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'persons': [2, 1, 3, 2, 4, 1, 2, 3, 2, 1],
        'income': [50000, 75000, 60000, 45000, 80000, 55000, 70000, 65000, 50000, 60000],
        'cars': [1, 2, 1, 0, 2, 1, 2, 1, 1, 0],
        'workers': [1, 1, 2, 1, 2, 1, 2, 2, 1, 0],
        'children': [1, 0, 2, 1, 2, 0, 1, 1, 0, 0],
        'age_of_head': [35, 45, 40, 28, 42, 55, 38, 45, 32, 65],
        'block_id': ['block1', 'block2', 'block3', 'block1', 'block2',
                     'block3', 'block1', 'block2', 'block3', 'block1'],
    })
    households_data.set_index('household_id', inplace=True)

    persons_data = pd.DataFrame({
        'person_id': list(range(1, 21)),
        'household_id': [1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9],
        'age': [35, 8, 45, 40, 12, 8, 28, 5, 42, 38, 10, 6, 55, 38, 35, 45, 15, 10, 32, 30],
        'worker': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        'student': [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        'sex': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2],
        'race_id': [1, 1, 2, 3, 3, 3, 1, 1, 2, 2, 2, 2, 1, 3, 3, 1, 1, 1, 2, 2],
    })
    persons_data.set_index('person_id', inplace=True)

    jobs_data = pd.DataFrame({
        'job_id': list(range(1, 16)),
        'building_id': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 201, 202, 203, 204, 205],
        'sector_id': ['retail', 'office', 'industrial', 'retail', 'office'] * 3,
        'home_based_status': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    jobs_data.set_index('job_id', inplace=True)

    blocks_data = pd.DataFrame({
        'block_id': ['block1', 'block2', 'block3'],
        'zone_id': [1, 2, 3],
        'county_id': ['001', '001', '001'],
    })
    blocks_data.set_index('block_id', inplace=True)

    buildings_data = pd.DataFrame({
        'building_id': list(range(101, 111)) + list(range(201, 211)),
        'parcel_id': list(range(1001, 1011)) + list(range(1001, 1011)),
        'building_type_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6],
        'residential_units': [1, 1, 1, 4, 4, 4, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'non_residential_sqft': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5000, 5000, 5000, 10000, 10000, 10000, 8000, 8000, 8000, 8000],
    })
    buildings_data.set_index('building_id', inplace=True)

    parcels_data = pd.DataFrame({
        'parcel_id': list(range(1001, 1011)),
        'zone_id': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
        'parcel_size': [5000, 6000, 5500, 7000, 4500, 6500, 5000, 5500, 6000, 5000],
    })
    parcels_data.set_index('parcel_id', inplace=True)

    # Write to H5 file
    with pd.HDFStore(h5_path, mode='w') as store:
        # Store at root level (base year format)
        store.put('/households', households_data, format='table')
        store.put('/persons', persons_data, format='table')
        store.put('/jobs', jobs_data, format='table')
        store.put('/blocks', blocks_data, format='table')
        store.put('/buildings', buildings_data, format='table')
        store.put('/parcels', parcels_data, format='table')

        # Also store in year-specific format (forecast year format)
        store.put('/2017/households', households_data, format='table')
        store.put('/2017/persons', persons_data, format='table')
        store.put('/2017/jobs', jobs_data, format='table')
        store.put('/2017/blocks', blocks_data, format='table')
        store.put('/2017/buildings', buildings_data, format='table')
        store.put('/2017/parcels', parcels_data, format='table')

    print(f"   ✅ Created {h5_path} ({os.path.getsize(h5_path) / 1024:.1f} KB)")
    return h5_path


def create_minimal_activitysim_outputs():
    """Create minimal ActivitySim output files (parquet format)."""
    print("📦 Creating minimal ActivitySim outputs...")

    output_dir = FIXTURES_DIR / "minimal_activitysim_outputs" / "final_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create households output
    households_dir = output_dir / "households"
    households_dir.mkdir(exist_ok=True)
    households_data = pd.DataFrame({
        'household_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'TAZ': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'income': [50000, 75000, 60000, 45000, 80000, 55000, 70000, 65000, 50000, 60000],
        'persons': [2, 1, 3, 2, 4, 1, 2, 3, 2, 1],
        'cars': [1, 2, 1, 0, 2, 1, 2, 1, 1, 0],
        'workers': [1, 1, 2, 1, 2, 1, 2, 2, 1, 0],
    })
    households_data.to_parquet(households_dir / "final.parquet", index=False)

    # Create persons output
    persons_dir = output_dir / "persons"
    persons_dir.mkdir(exist_ok=True)
    persons_data = pd.DataFrame({
        'person_id': list(range(1, 21)),
        'household_id': [1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9],
        'age': [35, 8, 45, 40, 12, 8, 28, 5, 42, 38, 10, 6, 55, 38, 35, 45, 15, 10, 32, 30],
        'ptype': [1, 4, 1, 1, 4, 4, 1, 4, 1, 1, 4, 4, 1, 1, 1, 1, 4, 4, 1, 2],
        'workplace_taz': [2, -1, 3, 1, -1, -1, 2, -1, 3, 2, -1, -1, 1, 3, -1, 2, -1, -1, 1, -1],
        'school_taz': [-1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, 2, 2, -1, -1],
    })
    persons_data.to_parquet(persons_dir / "final.parquet", index=False)

    # Create beam_plans output (critical for BEAM)
    plans_dir = output_dir / "beam_plans"
    plans_dir.mkdir(exist_ok=True)
    plans_data = pd.DataFrame({
        'person_id': list(range(1, 21)),
        'plan': ['<plan>...</plan>'] * 20,  # Simplified XML plan strings
        'trip_mode_choice_logsum': [10.5] * 20,
    })
    plans_data.to_parquet(plans_dir / "final.parquet", index=False)

    # Create land_use output
    landuse_dir = output_dir / "land_use"
    landuse_dir.mkdir(exist_ok=True)
    landuse_data = pd.DataFrame({
        'TAZ': [1, 2, 3],
        'TOTPOP': [1000, 1500, 800],
        'TOTHH': [400, 600, 300],
        'TOTEMP': [500, 800, 200],
        'RETEMPN': [200, 350, 80],
        'FPSEMPN': [100, 150, 50],
        'HEREMPN': [50, 80, 20],
        'OTHEMPN': [150, 220, 50],
    })
    landuse_data.to_parquet(landuse_dir / "final.parquet", index=False)

    print(f"   ✅ Created ActivitySim outputs in {output_dir}")
    return output_dir


def create_minimal_beam_skims():
    """Create minimal BEAM skim matrix (OMX format)."""
    print("📦 Creating minimal BEAM skims...")

    try:
        import openmatrix as omx
    except ImportError:
        print("   ⚠️  openmatrix not installed, skipping OMX creation")
        return None

    skims_path = FIXTURES_DIR / "minimal_beam_skims.omx"

    # Create 5x5 zone matrix
    n_zones = 5

    with omx.open_file(str(skims_path), 'w') as omx_file:
        # Create some basic skim matrices
        # SOV (Single Occupancy Vehicle) time matrix
        sov_time = np.random.uniform(10, 60, (n_zones, n_zones))
        np.fill_diagonal(sov_time, 0)  # Zero intra-zonal time
        omx_file['SOV_TIME__AM'] = sov_time

        # SOV distance matrix
        sov_dist = np.random.uniform(1, 30, (n_zones, n_zones))
        np.fill_diagonal(sov_dist, 0)
        omx_file['SOV_DIST__AM'] = sov_dist

        # HOV2 (carpoo) time matrix
        hov2_time = sov_time * 0.9  # Slightly faster than SOV
        omx_file['HOV2_TIME__AM'] = hov2_time

        # Add zone mapping
        omx_file.create_mapping('zone_number', list(range(1, n_zones + 1)))

    print(f"   ✅ Created {skims_path} ({os.path.getsize(skims_path) / 1024:.1f} KB)")
    return skims_path


def main():
    """Create all minimal fixtures."""
    print("🔧 Creating minimal fixture data for stub-based testing...")
    print("=" * 60)

    create_minimal_urbansim_h5()
    create_minimal_activitysim_outputs()
    create_minimal_beam_skims()

    print("=" * 60)
    print("✅ All minimal fixtures created successfully!")
    print(f"📁 Fixtures location: {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
