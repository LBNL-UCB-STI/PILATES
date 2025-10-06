#!/usr/bin/env python3
"""
Export PILATES database data dictionary in various formats.

This utility script allows you to export the complete database schema
documentation including table descriptions, column types, and relationships.

Usage:
    python pilates/utils/export_data_dictionary.py \\
        --database /path/to/database.duckdb \\
        --output docs/database_schema.md \\
        --format markdown

    # Export all formats at once
    python pilates/utils/export_data_dictionary.py \\
        --database /path/to/database.duckdb \\
        --output-dir docs/database/ \\
        --all-formats
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilates.utils.duckdb_manager import DuckDBManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export PILATES database data dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export as Markdown
  python pilates/utils/export_data_dictionary.py \\
      --database pilates.duckdb \\
      --output schema.md

  # Export as HTML for sharing with non-technical users
  python pilates/utils/export_data_dictionary.py \\
      --database pilates.duckdb \\
      --output schema.html \\
      --format html

  # Export all formats
  python pilates/utils/export_data_dictionary.py \\
      --database pilates.duckdb \\
      --output-dir docs/ \\
      --all-formats
        """,
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Path to DuckDB database file",
    )

    parser.add_argument(
        "--output",
        help="Output file path (required unless --all-formats is used)",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for --all-formats mode",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json", "csv", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--all-formats",
        action="store_true",
        help="Export in all formats to output-dir",
    )

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip row counts and statistics (faster)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all_formats and not args.output:
        parser.error("--output is required unless --all-formats is specified")

    if args.all_formats and not args.output_dir:
        parser.error("--output-dir is required with --all-formats")

    if not os.path.exists(args.database):
        logger.error(f"Database file not found: {args.database}")
        sys.exit(1)

    # Create database manager
    try:
        db_manager = DuckDBManager(args.database)
        logger.info(f"Connected to database: {args.database}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    include_stats = not args.no_stats

    try:
        if args.all_formats:
            # Export all formats
            os.makedirs(args.output_dir, exist_ok=True)

            formats = {
                "markdown": "database_schema.md",
                "json": "database_schema.json",
                "csv": "database_schema.csv",
                "html": "database_schema.html",
            }

            for fmt, filename in formats.items():
                output_path = os.path.join(args.output_dir, filename)
                logger.info(f"Exporting {fmt} dictionary to {output_path}...")

                success = db_manager.export_data_dictionary(
                    output_path, format=fmt, include_stats=include_stats
                )

                if success:
                    logger.info(f"✓ Successfully exported {fmt} to {output_path}")
                else:
                    logger.error(f"✗ Failed to export {fmt}")

            logger.info(f"\nAll formats exported to {args.output_dir}/")

        else:
            # Export single format
            logger.info(
                f"Exporting {args.format} dictionary to {args.output}..."
            )

            success = db_manager.export_data_dictionary(
                args.output, format=args.format, include_stats=include_stats
            )

            if success:
                logger.info(f"✓ Successfully exported to {args.output}")
                logger.info(f"\nYou can now share {args.output} with your team!")
            else:
                logger.error(f"✗ Failed to export dictionary")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Error during export: {e}")
        sys.exit(1)

    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
