
import argparse
import json
import logging
import os
import re
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from pilates.*
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.database.schema_generator import _normalize_table_name # Import the normalization function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_approved_tables(schema_dir: str) -> set:
    """
    Scans the approved schema directory and its 'generated' subdirectory
    for .sql files and extracts table names, normalizing them.
    """
    approved_tables = set()
    
    # List of directories to scan
    dirs_to_scan = [schema_dir]
    generated_dir = os.path.join(schema_dir, 'generated')
    if os.path.isdir(generated_dir):
        dirs_to_scan.append(generated_dir)

    for current_dir in dirs_to_scan:
        if not os.path.isdir(current_dir):
            logging.warning(f"Schema directory not found: {current_dir}")
            continue

        for filename in os.listdir(current_dir):
            if filename.endswith(".sql"):
                try:
                    with open(os.path.join(current_dir, filename), 'r') as f:
                        content = f.read()
                        found = re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", content, re.IGNORECASE)
                        for table in found:
                            # Extract year from filename for normalization
                            match = re.search(r'(\d{4})', filename)
                            year = int(match.group(1)) if match else None
                            normalized_table_name = _normalize_table_name(table, year)
                            approved_tables.add(normalized_table_name.lower())
                except IOError as e:
                    logging.error(f"Could not read schema file {filename} in {current_dir}: {e}")
    
    logging.info(f"Found {len(approved_tables)} approved table definitions across scanned directories.")
    return approved_tables

def sanitize_name(name: str) -> str:
    """Sanitizes a name to be a valid SQL identifier."""
    name = name.strip().strip('/')
    name = re.sub(r'[\s./-]', '_', name)
    if re.match(r'^[0-9]', name):
        name = '_' + name
    return name.lower()

def main():
    """Main function to drive the selective data upload process."""
    parser = argparse.ArgumentParser(description="Selectively upload table data from a PILATES run to the database.")
    parser.add_argument("run_info_path", help="Path to the run_info.json file.")
    parser.add_argument("database_path", help="Path to the DuckDB database file.")
    parser.add_argument("--table", action='append', help="Specify a table to upload. Can be used multiple times. If not provided, all approved tables found in the run will be uploaded.")
    args = parser.parse_args()

    if not os.path.exists(args.run_info_path):
        logging.error(f"Input file not found: {args.run_info_path}")
        return

    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    approved_schema_dir = os.path.join(script_dir, 'schema')
    run_dir = os.path.dirname(args.run_info_path)

    # --- Logic ---
    approved_tables = get_approved_tables(approved_schema_dir)
    if not approved_tables:
        logging.warning("No approved schemas found. Nothing to upload.")
        return

    # If user specifies tables, filter the approved list
    target_tables = set(args.table) if args.table else approved_tables
    upload_list = approved_tables.intersection(target_tables)

    if not upload_list:
        logging.warning("No tables matched the criteria for upload.")
        return

    logging.info(f"Attempting to upload data for tables: {sorted(list(upload_list))}")

    # Load run_info.json
    try:
        with open(args.run_info_path, 'r') as f:
            run_info = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to read or parse {args.run_info_path}: {e}")
        return

    # Connect to the database
    db_manager = DuckDBManager(args.database_path)

    # Find records to upload
    records_to_upload = []
    for record in run_info.get('file_records', {}).values():
        if not record.get('schema') or not record.get('short_name'):
            continue
        
        # If year is not explicitly in the record, try to extract it from the short_name
        if record.get('year') is None:
            match = re.search(r'(\d{4})', record['short_name'])
            if match:
                record['year'] = int(match.group(1))

        # Normalize table name for comparison and storage
        normalized_table_name = _normalize_table_name(record['short_name'], record.get('year'))
        
        if normalized_table_name in upload_list:
            records_to_upload.append(record)

    if not records_to_upload:
        logging.warning("Found no data in the run_info file for the specified tables.")
        return

    # Process each record
    for record in records_to_upload:
        # Normalize table name for logging and data storage
        normalized_table_name = _normalize_table_name(record['short_name'], record.get('year'))
        logging.info(f"Processing table '{normalized_table_name}' from source record...")

        try:
            df = None
            data_file_path = None

            # --- Path Resolution Logic ---
            path_from_record = record['file_path']
            is_h5_table = record.get('h5_file_unique_id')
            if is_h5_table:
                parent_h5_id = record['h5_file_unique_id']
                path_from_record = run_info['file_records'][parent_h5_id]['file_path']

            if os.path.isabs(path_from_record):
                if os.path.exists(path_from_record):
                    data_file_path = path_from_record
            else:
                # 1. Try path relative to the run_info.json directory
                candidate_path = os.path.join(run_dir, path_from_record)
                if os.path.exists(candidate_path):
                    data_file_path = candidate_path
                else:
                    # 2. Try path relative to the parent of the run_info.json directory
                    parent_run_dir = os.path.dirname(run_dir)
                    candidate_path_from_parent = os.path.join(parent_run_dir, path_from_record)
                    if os.path.exists(candidate_path_from_parent):
                        data_file_path = candidate_path_from_parent
            
            if not data_file_path:
                raise FileNotFoundError(f"Could not find data file. Record path: {path_from_record}")

            # --- Data Reading Logic ---
            if is_h5_table:
                logging.info(f"Inspecting keys in H5 file: {data_file_path}")
                with pd.HDFStore(data_file_path, mode='r') as store:
                    h5_keys = store.keys()
                    logging.info(f"Available keys: {h5_keys}")
                    if record['table_name'] not in h5_keys:
                        raise KeyError(f"Key '{record['table_name']}' not found in H5 file. Please check the file content.")
                df = pd.read_hdf(data_file_path, key=record['table_name'])
            elif data_file_path.endswith(('.parquet')):
                df = pd.read_parquet(data_file_path)
            else:
                logging.warning(f"Unsupported file type for table {normalized_table_name}: {data_file_path}")
                continue

            # Rename 'year' column to 'data_year' for specific tables to match schema
            if normalized_table_name in ['householdv', 'atlas_vehicles_input', 'new_vehicle_annual_medians', 
                                        'new_vehicle_representative_vehicle', 'new_vehicles', 'vehicles', 
                                        'atlas_vehicles2_output']:
                if 'year' in df.columns:
                    df.rename(columns={'year': 'data_year'}, inplace=True)

            # Add metadata to the dataframe before upload
            df['run_id'] = run_info['run_id']
            df['file_record_id'] = record['unique_id']
            df['year'] = record.get('year')
            # Note: Iteration is on the model_run, not the file_record. This is a simplification.
            df['iteration'] = None 

            # Upload the data
            success = db_manager.store_generic_table(normalized_table_name, df)
            if success:
                logging.info(f"Successfully uploaded {len(df)} rows to table '{normalized_table_name}'.")
            else:
                logging.error(f"Failed to upload data for table '{normalized_table_name}'.")

        except Exception as e:
            logging.error(f"An error occurred while processing table {normalized_table_name}: {e}")

    logging.info("Selective upload process complete.")

if __name__ == "__main__":
    main()
