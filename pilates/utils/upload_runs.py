#!/usr/bin/env python3
"""
Command-line utility for uploading PILATES run data to database.

This script can upload individual runs or batch upload multiple runs
to the configured database backend.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.database_upload import (
    upload_run_directory_to_database,
    upload_run_info_to_database,
    batch_upload_runs_to_database,
)
from pilates.config.models import load_config, PilatesConfig

logger = logging.getLogger(__name__)


def load_settings(settings_path: str) -> PilatesConfig:
    """Load PILATES settings from YAML file using Pydantic validation."""
    try:
        return load_config(settings_path)
    except Exception as e:
        logger.error(f"Failed to load and validate settings from {settings_path}: {e}")
        raise  # Re-raise the exception after logging


def find_run_directories(search_path: str) -> list:
    """Find all run directories containing run_info.json files."""
    pattern = os.path.join(search_path, "**/run_info.json")
    run_info_files = glob.glob(pattern, recursive=True)
    return [os.path.dirname(f) for f in run_info_files]


def main():
    parser = argparse.ArgumentParser(
        description="Upload PILATES run data to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single run
  python upload_runs.py --run-dir /path/to/run --settings settings.yaml
  
  # Upload from run_info.json file
  python upload_runs.py --run-info /path/to/run_info.json --settings settings.yaml
  
  # Batch upload all runs in tmp directory
  python upload_runs.py --batch-dir /path/to/tmp --settings settings.yaml
  
  # Upload all runs matching pattern
  python upload_runs.py --batch-pattern "/tmp/pilates-run-*" --settings settings.yaml
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--run-dir", help="Path to run directory containing run_info.json"
    )
    input_group.add_argument("--run-info", help="Path to run_info.json file")
    input_group.add_argument(
        "--batch-dir", help="Directory to search for run directories (recursive)"
    )
    input_group.add_argument(
        "--batch-pattern", help="Glob pattern to match run directories"
    )

    # Required settings
    parser.add_argument(
        "--settings",
        required=True,
        help="Path to PILATES settings YAML file with database configuration",
    )

    # Optional arguments
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load settings
    settings = load_settings(args.settings)
    if not settings:
        logger.error("Failed to load settings")
        return 1

    # Check database configuration
    db_config = settings.shared.database
    if not db_config.get("enabled", False):
        logger.warning("Database upload is not enabled in settings")
        if not args.dry_run:
            return 1

    try:
        if args.run_dir:
            # Upload single run directory
            if args.dry_run:
                run_info_path = os.path.join(args.run_dir, "run_info.json")
                if os.path.exists(run_info_path):
                    print(f"Would upload: {args.run_dir}")
                else:
                    print(f"No run_info.json found in: {args.run_dir}")
                return 0

            success = upload_run_directory_to_database(args.run_dir, settings)
            return 0 if success else 1

        elif args.run_info:
            # Upload single run_info.json file
            if args.dry_run:
                if os.path.exists(args.run_info):
                    print(f"Would upload: {args.run_info}")
                else:
                    print(f"File not found: {args.run_info}")
                return 0

            success = upload_run_info_to_database(args.run_info, settings)
            return 0 if success else 1

        elif args.batch_dir:
            # Batch upload from directory
            run_dirs = find_run_directories(args.batch_dir)

            if args.dry_run:
                print(f"Found {len(run_dirs)} run directories in {args.batch_dir}:")
                for run_dir in sorted(run_dirs):
                    print(f"  {run_dir}")
                return 0

            if not run_dirs:
                logger.warning(f"No run directories found in {args.batch_dir}")
                return 1

            results = batch_upload_runs_to_database(run_dirs, settings)
            successful = sum(1 for success in results.values() if success)
            total = len(results)

            logger.info(f"Batch upload results: {successful}/{total} successful")
            return 0 if successful == total else 1

        elif args.batch_pattern:
            # Batch upload from glob pattern
            run_dirs = glob.glob(args.batch_pattern)
            # Filter to only directories that contain run_info.json
            run_dirs = [
                d
                for d in run_dirs
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "run_info.json"))
            ]

            if args.dry_run:
                print(f"Found {len(run_dirs)} run directories matching pattern:")
                for run_dir in sorted(run_dirs):
                    print(f"  {run_dir}")
                return 0

            if not run_dirs:
                logger.warning(
                    f"No run directories found matching pattern: {args.batch_pattern}"
                )
                return 1

            results = batch_upload_runs_to_database(run_dirs, settings)
            successful = sum(1 for success in results.values() if success)
            total = len(results)

            logger.info(f"Batch upload results: {successful}/{total} successful")
            return 0 if successful == total else 1

    except KeyboardInterrupt:
        logger.info("Upload cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Upload failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
