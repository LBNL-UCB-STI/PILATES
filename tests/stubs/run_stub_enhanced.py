#!/usr/bin/env python3
"""
Enhanced stub runner that creates realistic output file structures.

This stub uses pre-generated minimal fixtures to create outputs that match
the expected structure of real PILATES models, enabling full provenance testing.
"""

import argparse
import os
import time
import shutil
from pathlib import Path


def get_fixtures_dir():
    """Get the fixtures directory path."""
    # Assume we're in tests/stubs/run_stub_enhanced.py
    tests_dir = Path(__file__).parent.parent
    return tests_dir / "fixtures"


def create_urbansim_outputs(output_dir, year):
    """Create UrbanSim outputs using minimal fixture."""
    print(f"Creating UrbanSim outputs for year {year}...")

    fixtures_dir = get_fixtures_dir()
    source_h5 = fixtures_dir / "minimal_urbansim_2017.h5"

    if not source_h5.exists():
        print(f"⚠️  Fixture not found: {source_h5}")
        print("   Run: python tests/fixtures/create_minimal_fixtures.py")
        # Create dummy output instead
        output_path = os.path.join(output_dir, f"model_data_{year}.h5")
        with open(output_path, "w") as f:
            f.write("Dummy UrbanSim Output")
        return

    # Copy fixture to output location
    output_path = os.path.join(output_dir, f"model_data_{year}.h5")
    shutil.copy2(source_h5, output_path)
    print(f"   ✅ Created {output_path}")


def create_activitysim_outputs(output_dir):
    """Create ActivitySim outputs using minimal fixtures."""
    print("Creating ActivitySim outputs...")

    fixtures_dir = get_fixtures_dir()
    source_outputs = fixtures_dir / "minimal_activitysim_outputs" / "final_pipeline"

    if not source_outputs.exists():
        print(f"⚠️  Fixture not found: {source_outputs}")
        print("   Run: python tests/fixtures/create_minimal_fixtures.py")
        # Create minimal dummy structure
        plans_dir = os.path.join(output_dir, "final_pipeline", "beam_plans")
        os.makedirs(plans_dir, exist_ok=True)
        with open(os.path.join(plans_dir, "final.parquet"), "w") as f:
            f.write("Dummy ActivitySim Plans")
        return

    # Copy entire final_pipeline directory
    dest_pipeline = os.path.join(output_dir, "final_pipeline")
    if os.path.exists(dest_pipeline):
        shutil.rmtree(dest_pipeline)
    shutil.copytree(source_outputs, dest_pipeline)
    print("   ✅ Created ActivitySim final_pipeline outputs")


def create_beam_outputs(output_dir, region, year, iteration):
    """Create BEAM outputs."""
    print(f"Creating BEAM outputs for year {year}, iteration {iteration}...")

    # Create BEAM output directory structure
    beam_output_path = os.path.join(
        output_dir, region, f"year-{year}-iteration-{iteration}"
    )
    os.makedirs(beam_output_path, exist_ok=True)

    # Create dummy skims files that BEAM would produce
    # BEAM creates individual CSV files for skim matrices
    skim_files = [
        "0.SOV_TIME_General_Purpose_Lanes__AM.csv.gz",
        "1.SOV_DIST_General_Purpose_Lanes__AM.csv.gz",
        "2.HOV2_TIME_General_Purpose_Lanes__AM.csv.gz",
    ]

    for skim_file in skim_files:
        skim_path = os.path.join(beam_output_path, skim_file)
        # Create a minimal CSV file (gzipped would be ideal but text works for testing)
        with open(skim_path, "w") as f:
            f.write("origin,destination,value\n1,2,15.5\n2,3,20.0\n")

    print(f"   ✅ Created BEAM output files in {beam_output_path}")


def create_atlas_outputs(output_dir, year):
    """Create ATLAS outputs."""
    print(f"Creating ATLAS outputs for year {year}...")

    # ATLAS creates vehicle CSV files
    vehicles_file = os.path.join(output_dir, f"vehicles_{year}.csv")
    with open(vehicles_file, "w") as f:
        f.write("vehicle_id,household_id,bodytype,modelyear,pred_power\n")
        f.write("1,1,Car,2015,ICE\n")
        f.write("2,2,Car,2018,ICE\n")
        f.write("3,3,SUV,2020,BEV\n")

    householdv_file = os.path.join(output_dir, f"householdv_{year}.csv")
    with open(householdv_file, "w") as f:
        f.write("household_id,nvehicles\n")
        f.write("1,1\n")
        f.write("2,2\n")
        f.write("3,1\n")

    print("   ✅ Created ATLAS vehicle files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, help="Name of the model being stubbed"
    )
    parser.add_argument(
        "--cwd", required=True, help="Current working directory for the model"
    )
    parser.add_argument(
        "--config_name", required=True, help="Name of the configuration file"
    )
    parser.add_argument("--year", type=int, default=2017, help="Simulation year")
    parser.add_argument(
        "--iteration", type=int, default=0, help="Iteration number (for BEAM)"
    )
    parser.add_argument("--region", default="sfbay", help="Region name (for BEAM)")
    args = parser.parse_args()

    print(f"🔧 Running enhanced stub for model: {args.model_name}")
    print(f"   Working directory: {args.cwd}")
    print(f"   Config name: {args.config_name}")
    print(f"   Year: {args.year}")

    # Simulate some work
    time.sleep(0.5)

    # Create realistic output files based on model_name
    if args.model_name == "urbansim":
        create_urbansim_outputs(args.cwd, args.year)

    elif args.model_name == "atlas":
        create_atlas_outputs(args.cwd, args.year)

    elif args.model_name == "activitysim":
        create_activitysim_outputs(args.cwd)

    elif args.model_name == "beam":
        create_beam_outputs(args.cwd, args.region, args.year, args.iteration)

    else:
        print(f"⚠️  Unknown model_name for stub: {args.model_name}")
        print("   Creating generic dummy output...")
        output_path = os.path.join(args.cwd, f"{args.model_name}_output.txt")
        with open(output_path, "w") as f:
            f.write(f"Dummy {args.model_name} Output")

    print(f"✅ Finished enhanced stub for model: {args.model_name}")


if __name__ == "__main__":
    main()
