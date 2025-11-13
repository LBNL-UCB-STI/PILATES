#!/usr/bin/env python3
"""Utility script to initialize a fresh DuckDB database for PILATES.

This script discovers all *.sql files in the `pilates/database/schema/`
directory and executes them in alphabetical order, creating the complete
schema required by the application (including the `config_snapshots`
table with the `config_hash` column).

Usage:
    python -m pilates.utils.init_duckdb --db-path /path/to/your.duckdb

If the database file does not exist it will be created; if it exists it
will be overwritten after user confirmation.
"""

import argparse
import os
import sys
import duckdb
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_schema_files(schema_dir: str):
    """Return a sorted list of .sql files in *schema_dir*.
    """
    if not os.path.isdir(schema_dir):
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
    files = [f for f in os.listdir(schema_dir) if f.endswith('.sql')]
    return sorted(files)

def execute_sql_file(conn: duckdb.DuckDBPyConnection, sql_path: str):
    """Read and execute the SQL statements from *sql_path*.
    """
    logger.info(f"Executing {os.path.basename(sql_path)}")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    conn.execute(sql)

def initialize_database(db_path: str, schema_dir: str):
    """Create or replace the DuckDB database at *db_path* using the schema files.
    """
    # If the DB already exists, ask for confirmation before overwriting.
    if os.path.exists(db_path):
        resp = input(f"Database file '{db_path}' already exists. Overwrite? [y/N]: ")
        if resp.lower() != "y":
            logger.info("Aborting initialization.")
            sys.exit(0)
        else:
            os.remove(db_path)
            logger.info("Existing database removed.")

    conn = duckdb.connect(database=db_path)
    try:
        for sql_file in get_schema_files(schema_dir):
            sql_path = os.path.join(schema_dir, sql_file)
            execute_sql_file(conn, sql_path)
        logger.info("Database initialized successfully at %s", db_path)
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Initialize a fresh PILATES DuckDB database.")
    parser.add_argument("--db-path", required=True, help="Path to the DuckDB file to create.")
    args = parser.parse_args()

    # Determine the absolute path to the schema directory regardless of where the script is run.
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    pilates_root = os.path.abspath(os.path.join(utils_dir, os.pardir))
    schema_dir = os.path.join(pilates_root, "database", "schema")

    initialize_database(args.db_path, schema_dir)

if __name__ == "__main__":
    main()
