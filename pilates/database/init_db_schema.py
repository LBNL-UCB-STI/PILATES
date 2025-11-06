import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from pilates.*
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.duckdb_manager import DuckDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a DuckDB database with the PILATES schema."
    )
    parser.add_argument(
        "database_path", help="Path to the DuckDB database file to initialize."
    )
    args = parser.parse_args()

    logging.info(f"Attempting to initialize database at: {args.database_path}")

    try:
        # Instantiating DuckDBManager automatically calls initialize_database
        db_manager = DuckDBManager(args.database_path)

        if db_manager.initialize_database():
            logging.info(
                f"Successfully initialized database schema at {args.database_path}"
            )
        else:
            logging.error(
                f"Failed to initialize database schema at {args.database_path}"
            )

    except Exception as e:
        logging.error(f"An error occurred during database initialization: {e}")


if __name__ == "__main__":
    main()
