#!/usr/bin/env python3
"""
Workflow utilities for H5 datastore management with PILATES database.

This module provides high-level workflows for the two main use cases:
1. Extract ActivitySim inputs from H5 and upload to database (bypass preprocessing)
2. Reconstruct UrbanSim H5 datastore from database (enable UrbanSim runs)
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.h5_to_database import H5ActivitySimExtractor
from pilates.utils.database_to_h5 import DatabaseToH5Reconstructor
from pilates.utils.database_upload import create_database_manager
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def workflow_extract_activitysim_inputs(
    h5_path: str,
    settings: Dict[str, Any],
    year: Optional[int] = None,
    save_csv: bool = False,
    output_dir: Optional[str] = None,
) -> bool:
    """
    Workflow 1: Extract ActivitySim inputs from H5 and upload to database.

    This workflow enables bypassing the complex ActivitySimPreprocessor when
    UrbanSim is turned off by pre-extracting and storing ActivitySim inputs.

    Args:
        h5_path: Path to UrbanSim H5 file
        settings: PILATES settings dictionary
        year: Optional year to extract
        save_csv: Whether to save CSV files as backup
        output_dir: Directory for CSV files

    Returns:
        bool: True if successful
    """
    logger.info("=== WORKFLOW 1: Extract ActivitySim Inputs ===")
    logger.info(f"Source H5: {h5_path}")
    logger.info(
        f"Target: Database ({get_setting(settings, 'shared.database.path', 'not configured')})"
    )

    try:
        # Initialize extractor
        extractor = H5ActivitySimExtractor(h5_path, settings)

        # Extract data
        logger.info("Step 1: Extracting ActivitySim input tables from H5 file...")
        extracted_data = extractor.extract_activitysim_inputs(
            year=year, save_csv=save_csv, output_dir=output_dir
        )

        if not extracted_data:
            logger.error("No ActivitySim input data found in H5 file")
            return False

        logger.info(f"Extracted {len(extracted_data)} tables:")
        for table_name, df in extracted_data.items():
            logger.info(f"  {table_name}: {len(df)} records")

        # Upload to database
        logger.info("Step 2: Uploading extracted data to database...")
        success = extractor.upload_to_database()

        if success:
            logger.info("✅ WORKFLOW 1 COMPLETED SUCCESSFULLY")
            logger.info("ActivitySim inputs are now available in database")
            logger.info(f"Run ID: {extractor.run_id}")

            # Show how to use this data
            logger.info("\n📋 Next Steps:")
            logger.info("1. Configure UrbanSim to be turned OFF in your settings")
            logger.info(
                "2. Modify ActivitySimPreprocessor to query database instead of H5 file"
            )
            logger.info("3. Use the following OpenLineage IDs to retrieve data:")

            # Log OpenLineage IDs for reference
            for table_name in extracted_data.keys():
                for record in extractor.file_records.values():
                    if record.metadata.get("table_name") == table_name:
                        logger.info(f"   {table_name}: {record.openlineage_id}")
                        break
        else:
            logger.error("❌ WORKFLOW 1 FAILED: Database upload unsuccessful")

        return success

    except Exception as e:
        logger.error(f"❌ WORKFLOW 1 FAILED: {e}")
        return False


def workflow_reconstruct_urbansim_h5(
    settings: Dict[str, Any],
    output_h5_path: str,
    config_hash: Optional[str] = None,
    year: Optional[int] = None,
    include_synthetic: bool = False,
) -> bool:
    """
    Workflow 2: Reconstruct UrbanSim H5 datastore from database.

    This workflow enables UrbanSim runs by reconstructing the required H5 input
    file from database records of previous runs.

    Args:
        settings: PILATES settings dictionary
        output_h5_path: Path where H5 file will be created
        config_hash: Optional configuration hash to match
        year: Optional year to reconstruct for
        include_synthetic: Whether to generate synthetic data for missing tables

    Returns:
        bool: True if successful
    """
    logger.info("=== WORKFLOW 2: Reconstruct UrbanSim H5 ===")
    logger.info(
        f"Source: Database ({get_setting(settings, 'shared.database.path', 'not configured')})"
    )
    logger.info(f"Target H5: {output_h5_path}")

    try:
        # Initialize reconstructor
        reconstructor = DatabaseToH5Reconstructor(settings, output_h5_path)

        # Analyze available data first
        logger.info("Step 1: Analyzing available data in database...")
        reconstructor.db_manager = create_database_manager(settings)

        if not reconstructor.db_manager:
            logger.error("Failed to create database manager")
            return False

        with reconstructor.db_manager:
            activitysim_data = reconstructor._query_available_activitysim_data()
            urbansim_data = reconstructor._query_urbansim_output_data(config_hash, year)

            logger.info(
                f"Found data for {len(activitysim_data)} ActivitySim input tables"
            )
            logger.info(f"Found data for {len(urbansim_data)} UrbanSim output tables")

            all_available = set(activitysim_data.keys()) | set(urbansim_data.keys())
            all_expected = set(
                reconstructor.CORE_URBANSIM_TABLES + reconstructor.ACTIVITYSIM_TABLES
            )
            missing = all_expected - all_available

            if missing:
                logger.warning(f"Missing expected tables: {missing}")
                if not include_synthetic:
                    logger.warning(
                        "Consider using --include-synthetic to generate placeholder data"
                    )

        # Reconstruct H5 file
        logger.info("Step 2: Reconstructing H5 file...")
        success = reconstructor.reconstruct_h5_file(
            config_hash=config_hash, year=year, include_synthetic=include_synthetic
        )

        if success:
            logger.info("✅ WORKFLOW 2 COMPLETED SUCCESSFULLY")
            logger.info(f"H5 file created: {output_h5_path}")

            if reconstructor.missing_tables:
                logger.warning(
                    "⚠️  IMPORTANT: Some tables are missing or contain synthetic data"
                )
                logger.warning(
                    "Review the H5 file before using in production UrbanSim runs"
                )
                logger.warning(f"Missing tables: {reconstructor.missing_tables}")

            # Show how to use this H5 file
            logger.info("\n📋 Next Steps:")
            logger.info("1. Configure UrbanSim to be turned ON in your settings")
            logger.info(
                "2. Update UrbanSim data path to point to reconstructed H5 file"
            )
            logger.info("3. Run PILATES with UrbanSim enabled")
            logger.info("4. Monitor for any missing data issues during the run")
        else:
            logger.error("❌ WORKFLOW 2 FAILED: H5 reconstruction unsuccessful")

        return success

    except Exception as e:
        logger.error(f"❌ WORKFLOW 2 FAILED: {e}")
        return False


def workflow_full_cycle_demo(
    base_h5_path: str,
    settings: Dict[str, Any],
    temp_h5_path: str,
    year: Optional[int] = None,
) -> bool:
    """
    Demo workflow: Complete cycle from H5 -> Database -> H5.

    This demonstrates both workflows in sequence for testing/validation.

    Args:
        base_h5_path: Original H5 file to extract from
        settings: PILATES settings dictionary
        temp_h5_path: Path for reconstructed H5 file
        year: Optional year to work with

    Returns:
        bool: True if both workflows successful
    """
    logger.info("=== DEMO: Full H5 <-> Database Cycle ===")

    # Workflow 1: Extract to database
    success1 = workflow_extract_activitysim_inputs(
        base_h5_path,
        settings,
        year=year,
        save_csv=True,
        output_dir="/tmp/extracted_csv",
    )

    if not success1:
        logger.error("Demo failed at extraction step")
        return False

    # Workflow 2: Reconstruct from database
    success2 = workflow_reconstruct_urbansim_h5(
        settings, temp_h5_path, year=year, include_synthetic=True
    )

    if not success2:
        logger.error("Demo failed at reconstruction step")
        return False

    logger.info("✅ DEMO COMPLETED SUCCESSFULLY")
    logger.info("Both H5 -> Database and Database -> H5 workflows work")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="H5 datastore workflow utilities for PILATES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflows:

1. EXTRACT: H5 -> Database (enables bypassing UrbanSim preprocessing)
   python h5_workflows.py extract --h5-file urbansim_data.h5 --settings settings.yaml

2. RECONSTRUCT: Database -> H5 (enables UrbanSim runs from database)
   python h5_workflows.py reconstruct --output urbansim_data.h5 --settings settings.yaml

3. DEMO: Full cycle test
   python h5_workflows.py demo --h5-file original.h5 --output reconstructed.h5 --settings settings.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="workflow", help="Workflow to run")

    # Extract workflow
    extract_parser = subparsers.add_parser(
        "extract", help="Extract ActivitySim inputs from H5 to database"
    )
    extract_parser.add_argument(
        "--h5-file", required=True, help="Path to UrbanSim H5 file"
    )
    extract_parser.add_argument(
        "--settings", required=True, help="Path to PILATES settings YAML file"
    )
    extract_parser.add_argument("--year", type=int, help="Year to extract")
    extract_parser.add_argument(
        "--save-csv", action="store_true", help="Save extracted data as CSV backup"
    )
    extract_parser.add_argument("--output-dir", help="Directory for CSV files")

    # Reconstruct workflow
    reconstruct_parser = subparsers.add_parser(
        "reconstruct", help="Reconstruct UrbanSim H5 from database"
    )
    reconstruct_parser.add_argument(
        "--output", required=True, help="Path for output H5 file"
    )
    reconstruct_parser.add_argument(
        "--settings", required=True, help="Path to PILATES settings YAML file"
    )
    reconstruct_parser.add_argument("--config-hash", help="Configuration hash to match")
    reconstruct_parser.add_argument("--year", type=int, help="Year to reconstruct for")
    reconstruct_parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Generate synthetic data for missing tables",
    )

    # Demo workflow
    demo_parser = subparsers.add_parser("demo", help="Run full cycle demo")
    demo_parser.add_argument(
        "--h5-file", required=True, help="Path to original H5 file"
    )
    demo_parser.add_argument(
        "--output", required=True, help="Path for reconstructed H5 file"
    )
    demo_parser.add_argument(
        "--settings", required=True, help="Path to PILATES settings YAML file"
    )
    demo_parser.add_argument("--year", type=int, help="Year to work with")

    # Common arguments
    for subparser in [extract_parser, reconstruct_parser, demo_parser]:
        subparser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )

    args = parser.parse_args()

    if not args.workflow:
        parser.print_help()
        return 1

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load settings
    try:
        with open(args.settings, "r") as f:
            settings = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return 1

    try:
        if args.workflow == "extract":
            success = workflow_extract_activitysim_inputs(
                args.h5_file,
                settings,
                year=args.year,
                save_csv=args.save_csv,
                output_dir=args.output_dir,
            )

        elif args.workflow == "reconstruct":
            success = workflow_reconstruct_urbansim_h5(
                settings,
                args.output,
                config_hash=args.config_hash,
                year=args.year,
                include_synthetic=args.include_synthetic,
            )

        elif args.workflow == "demo":
            success = workflow_full_cycle_demo(
                args.h5_file, settings, args.output, year=args.year
            )
        else:
            logger.error(f"Unknown workflow: {args.workflow}")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Workflow cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
