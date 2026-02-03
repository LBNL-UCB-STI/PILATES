import geopandas as gpd
import pandas as pd
import os
import sys

# Add pilates to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pilates.utils.geog import geoid_to_zone_map
from pilates.config.models import load_config


def verify_zone_consistency(config_path="settings.yaml"):
    """
    Verifies that the zone order in BEAM's shapefile matches the
    canonical zone order derived from the UrbanSim datastore via geoid_to_zone_map.
    """
    print("--- Starting Zone Order Verification ---")

    # 1. Load Pilates Config
    try:
        settings = load_config(config_path)
        print(f"Successfully loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error: Could not load or parse config file at '{config_path}'.")
        print(f"Details: {e}")
        return

    # 2. Get Canonical Order from Pilates (via geoid_to_zone_map)
    try:
        print("Generating canonical zone order from datastore...")
        # Note: geoid_to_zone_map returns a dict, we need the sorted keys
        mapping = geoid_to_zone_map(settings)
        # The mapping's values are 1-based strings. We sort by them to get the canonical order.
        pilates_order_df = pd.DataFrame.from_dict(
            mapping, orient="index", columns=["zone_id"]
        )
        pilates_order_df["zone_id"] = pd.to_numeric(pilates_order_df["zone_id"])
        pilates_order_df = pilates_order_df.sort_values("zone_id")
        pilates_canonical_order = pilates_order_df.index.tolist()
        print(
            f"Successfully generated canonical order for {len(pilates_canonical_order)} zones."
        )
        print(f"First 5 canonical GEOIDs: {pilates_canonical_order[:5]}")
    except Exception as e:
        print("Error: Could not generate canonical zone order from geoid_to_zone_map.")
        print("This likely means the HDF5 datastore is missing or inaccessible.")
        print(f"Details: {e}")
        return

    # 3. Get Zone Order from BEAM Shapefile
    try:
        region = settings.run.region
        beam_input_folder = settings.beam.local_input_folder
        shapefile_relative_path = settings.beam.skims_shapefile
        shapefile_path = os.path.join(
            beam_input_folder, region, shapefile_relative_path
        )

        print(f"Loading BEAM shapefile from: {shapefile_path}")
        if not os.path.exists(shapefile_path):
            print(f"Error: BEAM shapefile not found at '{shapefile_path}'.")
            print("Please ensure the path is correct and the file exists.")
            return

        shapefile_gdf = gpd.read_file(shapefile_path)

        # Use the geoid column from the config that BEAM uses
        geoid_col = settings.beam.skim_zone_geoid_col
        if geoid_col not in shapefile_gdf.columns:
            print(f"Error: Column '{geoid_col}' not found in the shapefile.")
            print(f"Available columns are: {shapefile_gdf.columns.tolist()}")
            return

        beam_shapefile_order = shapefile_gdf[geoid_col].tolist()
        print(
            f"Successfully loaded {len(beam_shapefile_order)} zones from BEAM shapefile."
        )
        print(f"First 5 shapefile GEOIDs: {beam_shapefile_order[:5]}")

    except Exception as e:
        print("Error: Could not load or parse BEAM shapefile.")
        print(f"Details: {e}")
        return

    # 4. Compare the two orders
    print("\n--- Comparison Results ---")

    if len(pilates_canonical_order) != len(beam_shapefile_order):
        print("FAIL: Mismatch in number of zones!")
        print(f"  - Pilates Canonical Order: {len(pilates_canonical_order)} zones")
        print(f"  - BEAM Shapefile Order: {len(beam_shapefile_order)} zones")
    else:
        print("OK: Number of zones is consistent.")

    if pilates_canonical_order == beam_shapefile_order:
        print("OK: Zone orders are identical. Consistency is confirmed.")
    else:
        print("FAIL: Zone orders are NOT identical!")
        print("This will lead to scrambled skim data.")

        # Find the first point of divergence
        for i, (pilates_zone, beam_zone) in enumerate(
            zip(pilates_canonical_order, beam_shapefile_order)
        ):
            if pilates_zone != beam_zone:
                print(f"\nFirst mismatch found at index {i}:")
                print(f"  - Pilates Canonical Order expects: {pilates_zone}")
                print(f"  - BEAM Shapefile has:            {beam_zone}")
                break

        print(
            "\nRecommendation: The BEAM shapefile needs to be sorted to match the canonical order."
        )
        print("The canonical order is derived from the UrbanSim HDF5 datastore.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="settings-new-asim-seattle.yaml",
        help="Path to the hierarchical config file to use for verification.",
    )
    args = parser.parse_args()

    verify_zone_consistency(config_path=args.config)
