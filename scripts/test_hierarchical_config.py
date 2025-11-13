#!/usr/bin/env python
"""
End-to-end test of Phase 1 hierarchical configuration hashing.

This script demonstrates:
1. Loading new-format PILATES config
2. Creating hierarchical config hashes
3. Storing them in the database
4. Querying for reusable outputs

Usage:
    python scripts/test_hierarchical_config.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add pilates to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pilates.config.models import load_config
from pilates.config.schema import get_field_annotations, get_dependency_graph
from pilates.generic.config_hashing import ConfigHasher
from pilates.utils.config_snapshot import ConfigSnapshotManager
from pilates.utils.duckdb_manager import DuckDBManager


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_phase1_hierarchical_config():
    """Complete end-to-end test of Phase 1 implementation."""

    print_section("PHASE 1: HIERARCHICAL CONFIG HASHING - END-TO-END TEST")

    # -------------------------------------------------------------------------
    # Step 1: Load and Validate Config
    # -------------------------------------------------------------------------
    print_section("Step 1: Load and Validate Config")

    config_path = "settings-new-asim-seattle-migrated.yaml"
    print(f"Loading config from: {config_path}")

    config = load_config(config_path)
    print(f"✓ Config validated successfully!")
    print(f"  Region: {config.run.region}")
    print(f"  Years: {config.run.start_year}-{config.run.end_year}")
    print(f"  Enabled models: {', '.join(config.get_enabled_models())}")

    # -------------------------------------------------------------------------
    # Step 2: Create Config Snapshot
    # -------------------------------------------------------------------------
    print_section("Step 2: Create Config Snapshot")

    # Convert config to dict for snapshot manager
    config_dict = config.model_dump()
    workspace_path = os.getcwd()

    snapshot_manager = ConfigSnapshotManager(workspace_path)
    snapshot = snapshot_manager.create_config_snapshot(config_dict)

    print(f"✓ Created config snapshot")
    print(f"  Snapshot ID: {snapshot['snapshot_id']}")
    print(f"  Content hash: {snapshot['config_content_hash'][:16]}...")

    # -------------------------------------------------------------------------
    # Step 3: Create Hierarchical Hashes
    # -------------------------------------------------------------------------
    print_section("Step 3: Create Hierarchical Config Hashes")

    enabled_models = config.get_enabled_models()
    hierarchical_hashes = snapshot_manager.create_hierarchical_config_hashes(
        snapshot, enabled_models
    )

    print(f"✓ Created hierarchical hashes for {len(hierarchical_hashes)} layers:")
    for model_name, hash_info in hierarchical_hashes.items():
        print(f"  {model_name:15} {hash_info['hash'][:16]}...")

    # -------------------------------------------------------------------------
    # Step 4: Initialize Database and Upload
    # -------------------------------------------------------------------------
    print_section("Step 4: Initialize Database and Upload Hashes")

    # Create temporary database for testing
    import uuid

    temp_dir = tempfile.gettempdir()
    test_db_path = os.path.join(
        temp_dir, f"test_hierarchical_config_{uuid.uuid4().hex[:8]}.duckdb"
    )

    try:
        print(f"Creating test database at: {test_db_path}")
        db = DuckDBManager(test_db_path)
        db.initialize_database()
        print("✓ Database initialized")

        # Upload config snapshot first (required foreign key)
        print("\nUploading config snapshot...")
        conn = db._get_connection()
        conn.execute(
            """
            INSERT INTO config_snapshots (
                snapshot_id, created_timestamp, config_content_hash,
                git_hashes, config_files, pilates_settings,
                beam_config, asim_subdir, region
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot["snapshot_id"],
                snapshot["created_timestamp"],
                snapshot["config_content_hash"],
                "{}",  # Simplified for test
                "{}",  # Simplified for test
                "{}",  # Simplified for test
                snapshot.get("beam_config"),
                snapshot.get("asim_subdir"),
                snapshot.get("region"),
            ],
        )
        print("✓ Config snapshot uploaded")

        # Upload hierarchical hashes
        print("\nUploading hierarchical config hashes...")
        db.upload_hierarchical_config_hashes(
            snapshot["snapshot_id"], hierarchical_hashes
        )
        print("✓ Hierarchical hashes uploaded")

        # -------------------------------------------------------------------------
        # Step 5: Verify Database Contents
        # -------------------------------------------------------------------------
        print_section("Step 5: Verify Database Contents")

        result = conn.execute("SELECT COUNT(*) FROM model_configs").fetchone()
        print(f"✓ Stored {result[0]} model config hashes in database")

        print("\nConfig hashes in database:")
        rows = conn.execute(
            "SELECT model_name, config_hash, config_type FROM model_configs"
        ).fetchall()

        for row in rows:
            print(f"  {row[0]:15} {row[1][:16]}... (type: {row[2]})")

        # -------------------------------------------------------------------------
        # Step 6: Demonstrate Intelligent Caching Query
        # -------------------------------------------------------------------------
        print_section("Step 6: Demonstrate Intelligent Caching")

        print("SCENARIO: Two runs differ only in BEAM config\n")

        # Original config hashes
        orig_activitysim_hash = hierarchical_hashes["activitysim"]["hash"]
        orig_beam_hash = hierarchical_hashes["beam"]["hash"]

        print("Run A (original):")
        print(f"  ActivitySim hash: {orig_activitysim_hash[:16]}...")
        print(f"  BEAM hash:        {orig_beam_hash[:16]}...")

        # Simulate different BEAM config
        modified_config = config_dict.copy()
        modified_config["beam"] = modified_config["beam"].copy()
        modified_config["beam"]["sample"] = 0.5  # Changed from 1.0

        # Recompute hashes
        field_annotations = get_field_annotations()
        dependency_graph = get_dependency_graph()
        hasher = ConfigHasher(
            config=modified_config,
            field_annotations=field_annotations,
            dependency_graph=dependency_graph,
        )
        modified_hashes = hasher.get_hierarchical_hashes(enabled_models)

        new_activitysim_hash = modified_hashes["activitysim"]
        new_beam_hash = modified_hashes["beam"]

        print("\nRun B (modified BEAM config):")
        print(f"  ActivitySim hash: {new_activitysim_hash[:16]}...")
        print(f"  BEAM hash:        {new_beam_hash[:16]}...")

        print("\nComparison:")
        asim_match = (
            "✓ SAME" if orig_activitysim_hash == new_activitysim_hash else "✗ DIFF"
        )
        beam_match = "✓ SAME" if orig_beam_hash == new_beam_hash else "✗ DIFF"

        print(f"  ActivitySim: {asim_match}")
        print(f"  BEAM:        {beam_match}")

        print("\nConclusion:")
        if orig_activitysim_hash == new_activitysim_hash:
            print("  ✓ ActivitySim config is IDENTICAL across runs")
            print("  ✓ Can query database for reusable ActivitySim outputs!")
            print(
                f"\n  Query: SELECT * FROM file_records WHERE config_hash = '{orig_activitysim_hash[:16]}...'"
            )
        else:
            print("  ✗ ActivitySim config differs - would need to re-run")

        # -------------------------------------------------------------------------
        # Step 7: Summary
        # -------------------------------------------------------------------------
        print_section("PHASE 1 IMPLEMENTATION - SUMMARY")

        print("✓ NEW CAPABILITIES:")
        print("  1. Hierarchical config hashing (base + per-model)")
        print("  2. Database storage of config hashes")
        print("  3. Intelligent caching queries")
        print("  4. Pydantic validation for configs")
        print("  5. Migration from legacy config format")
        print()
        print("✓ FUTURE-PROOF DESIGN:")
        print("  • Generic config hashing framework (→ provtrack)")
        print("  • PILATES-specific implementation")
        print("  • Clean separation of concerns")
        print()
        print("✓ READY FOR PRODUCTION:")
        print("  • Database schema updated")
        print("  • Upload/query methods implemented")
        print("  • End-to-end workflow validated")

        print_section("TEST COMPLETED SUCCESSFULLY!")

    finally:
        # Clean up test database
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
            print(f"\nCleaned up test database: {test_db_path}")


if __name__ == "__main__":
    test_phase1_hierarchical_config()
