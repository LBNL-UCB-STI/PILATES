
import json
import os
import re
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_definitions(schema_dir: str) -> set:
    """
    Scans a directory for .sql files and extracts all defined table and view names,
    normalizing them to account for year-specific naming conventions.
    """
    defined_names = set()
    logging.info(f"Scanning for existing schemas in: {schema_dir}")
    if not os.path.isdir(schema_dir):
        logging.warning(f"Schema directory not found: {schema_dir}")
        return defined_names

    files_in_dir = os.listdir(schema_dir)
    logging.info(f"Files found in schema directory: {files_in_dir}")

    for filename in files_in_dir:
        if filename.endswith(".sql"):
            file_path = os.path.join(schema_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Regex to find 'CREATE TABLE [IF NOT EXISTS]' and 'CREATE [OR REPLACE] VIEW'
                    # It captures the name of the table/view.
                    found_tables = re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", content, re.IGNORECASE)
                    found_views = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)", content, re.IGNORECASE)
                    
                    for table in found_tables:
                        # Attempt to extract year from filename for normalization
                        match = re.search(r'(\d{4})', filename)
                        year = int(match.group(1)) if match else None
                        normalized_table_name = _normalize_table_name(table, year)
                        defined_names.add(normalized_table_name.lower())
                    for view in found_views:
                        # Views might also have year, but for now, just sanitize
                        defined_names.add(sanitize_name(view).lower())
            except IOError as e:
                logging.error(f"Could not read schema file {filename}: {e}")
    
    logging.info(f"Found {len(defined_names)} existing table/view definitions.")
    return defined_names

def map_json_type_to_sql(json_type: str) -> str:
    """
    Maps a data type from the run_info.json schema to a DuckDB SQL type.

    Args:
        json_type: The type string from the JSON schema (e.g., 'int64', 'float64', 'object').

    Returns:
        The corresponding SQL data type as a string.
    """
    json_type = json_type.lower()
    if 'int' in json_type:
        return 'BIGINT'
    elif 'float' in json_type:
        return 'DOUBLE'
    elif 'bool' in json_type:
        return 'BOOLEAN'
    elif 'date' in json_type:
        return 'TIMESTAMP'
    else:
        return 'VARCHAR'

def sanitize_name(name: str, is_column=False) -> str:
    """
    Sanitizes a name to be a valid SQL identifier.

    Args:
        name: The string to sanitize.
        is_column: If True, applies stricter rules for column names.

    Returns:
        A sanitized string safe for use as a SQL table or column name.
    """
    # Remove leading/trailing whitespace and slashes
    name = name.strip().strip('/')
    # Replace problematic characters with underscores
    # For column names, be more aggressive
    if is_column:
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    else:
        name = re.sub(r'[\s./-]', '_', name)

    # Ensure it doesn't start with a number
    if re.match(r'^[0-9]', name):
        name = '_' + name
        
    return name.lower()

def _normalize_table_name(short_name: str, year: Optional[int]) -> str:
    normalized_name = short_name.lower()
    
    # First, try to remove the year provided in the record, if any
    if year:
        year_str = str(year)
        pattern = r"([_-]?" + re.escape(year_str) + r")(?![0-9])"
        normalized_name = re.sub(pattern, "", normalized_name).strip('_-')
            
    # Specific patterns for year removal
    # Pattern: model_data_YYYY_YYYY_NAME -> model_data_NAME
    normalized_name = re.sub(r'model_data_(\d{4})_(\d{4})_(.*)', r'model_data_\3', normalized_name)
    # Pattern: usim_output_YYYY_YYYY_NAME -> usim_output_NAME
    normalized_name = re.sub(r'usim_output_(\d{4})_(\d{4})_(.*)', r'usim_output_\3', normalized_name)
    # Pattern: usim_input_archive_YYYY_NAME -> usim_input_archive_NAME
    normalized_name = re.sub(r'usim_input_archive_(\d{4})_(.*)', r'usim_input_archive_\2', normalized_name)
    # Pattern: NAME_YYYY -> NAME (general case for year at the end)
    normalized_name = re.sub(r'([_-]?\d{4})(?![0-9])', '', normalized_name).strip('_-')
    
        # Further sanitize the normalized name    normalized_name = re.sub(r'[^a-zA-Z0-9_]', '_', normalized_name)
    # Remove any double underscores that might result from year removal
    normalized_name = re.sub(r'_{2,}', '_', normalized_name)
    normalized_name = normalized_name.strip('_')

    return normalized_name

def generate_sql_for_record(record: dict, table_name: str) -> str:
    """
    Generates a CREATE TABLE and COMMENT statements for a given file record.

    Args:
        record: The file record dictionary from run_info.json.
        table_name: The sanitized name to use for the SQL table.

    Returns:
        A string containing the full SQL definition for the new table.
    """
    if not record.get('schema'):
        return ""

    # Heuristic to find a potential semantic primary key
    suggested_pk = None
    for col in record['schema']:
        col_name = col.get('name', '').lower()
        if col_name.endswith('_id') or col_name == 'id':
            suggested_pk = sanitize_name(col_name, is_column=True)
            break

    sql_lines = [
        f"-- Auto-generated schema for table: {table_name}",
        f"-- Source: {record.get('short_name', 'N/A')}",
        f"-- Description: {record.get('description', 'N/A')}",
        "-- IMPORTANT: This is a placeholder. Review and adjust data types, add primary keys, and add indexes as needed.",
    ]
    if suggested_pk:
        sql_lines.append(f"-- Suggested Primary Key: {suggested_pk}")
    sql_lines.append("")

    sql_lines.append(f"CREATE SEQUENCE IF NOT EXISTS {table_name}_id_seq START 1;")
    sql_lines.append("")

    sql_lines.extend([
        f"CREATE TABLE IF NOT EXISTS {table_name} (",
        "    -- Add an auto-incrementing primary key for uniqueness",
        f"    id BIGINT PRIMARY KEY DEFAULT nextval('{table_name}_id_seq'),",
        "",
        "    -- Foreign keys and metadata to link to the main run, file, and context",
        "    run_id VARCHAR,",
        "    file_record_id VARCHAR,",
        "    year INTEGER,",
        "    iteration INTEGER,",
    ])

    # Add columns from schema
    for col in record['schema']:
        col_name = sanitize_name(col['name'], is_column=True)
        sql_type = map_json_type_to_sql(col['type'])
        sql_lines.append(f"    {col_name} {sql_type},")

    sql_lines.extend([
        "    FOREIGN KEY (run_id) REFERENCES runs(run_id),",
        "    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)",
        ");",
        "",
        f"COMMENT ON TABLE {table_name} IS '{record.get('description', 'Table auto-generated from run data.').replace("'", "''")}';",

        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_run_id ON {table_name}(run_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_year_iter ON {table_name}(year, iteration);",
        ""
    ])

    # Add placeholder comments for each column
    for col in record['schema']:
        col_name = sanitize_name(col['name'], is_column=True)
        sql_lines.append(f"COMMENT ON COLUMN {table_name}.{col_name} IS 'TODO: Add description.';")

    return "\n".join(sql_lines)

def main():
    """
    Main function to drive the schema generation process.
    """
    parser = argparse.ArgumentParser(description="Generate placeholder SQL schema files from a PILATES run_info.json file.")
    parser.add_argument("run_info_path", help="Path to the run_info.json file to analyze.")
    args = parser.parse_args()

    if not os.path.exists(args.run_info_path):
        logging.error(f"Input file not found: {args.run_info_path}")
        return

    # Define directories based on the location of this script
    # __file__ is pilates/database/schema_generator.py
    script_location = os.path.dirname(os.path.abspath(__file__))
    # approved_schema_dir is pilates/database/schema/
    approved_schema_dir = os.path.join(script_location, 'schema')
    # generated_dir is pilates/database/schema/generated/
    generated_dir = os.path.join(approved_schema_dir, 'generated')
    
    # Ensure generated directory exists
    os.makedirs(generated_dir, exist_ok=True)

    # 1. Get existing table definitions
    existing_tables = get_existing_definitions(approved_schema_dir)

    # 2. Load the run_info.json file
    try:
        with open(args.run_info_path, 'r') as f:
            run_info = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to read or parse {args.run_info_path}: {e}")
        return

    # 3. Iterate through records and generate schemas
    new_files_generated = 0
    file_records = run_info.get('file_records', {})
    
    for record in file_records.values():
        # We are interested in records that are tables (have a schema and a short_name)
        if not record.get('schema') or not record.get('short_name'):
            continue
            
        # If year is not explicitly in the record, try to extract it from the short_name
        if record.get('year') is None:
            match = re.search(r'(\d{4})', record['short_name'])
            if match:
                record['year'] = int(match.group(1))

        # Normalize table name to handle year-specific tables
        normalized_short_name = _normalize_table_name(record['short_name'], record.get('year'))
        table_name = sanitize_name(normalized_short_name)

        if table_name in existing_tables:
            logging.info(f"Schema for table '{table_name}' already exists. Skipping.")
            continue

        logging.info(f"New table found: '{record['short_name']}'. Generating placeholder schema...")

        # 4. Generate SQL
        sql_content = generate_sql_for_record(record, table_name)
        if not sql_content:
            continue

        # 5. Write to placeholder file
        output_filename = os.path.join(generated_dir, f"{table_name}.sql")
        try:
            with open(output_filename, 'w') as f:
                f.write(sql_content)
            logging.info(f"Successfully generated placeholder file: {output_filename}")
            new_files_generated += 1
        except IOError as e:
            logging.error(f"Failed to write schema file for {table_name}: {e}")

    logging.info(f"Schema generation complete. Generated {new_files_generated} new placeholder files.")

if __name__ == "__main__":
    main()
