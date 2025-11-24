import os
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
# The schema directory is a sibling to the 'scripts' directory
SCHEMA_DIR = SCRIPT_DIR.parent / "schema"
GENERATED_DIR = SCHEMA_DIR
# We start with just the activitysim outputs as per user feedback
FILES_TO_PROCESS = ["06_asim_outputs.sql"]

# Template for the unified view
VIEW_TEMPLATE = r"""-- View: {table_name}
CREATE OR REPLACE VIEW {table_name} AS
SELECT
{column_list}
FROM uploaded_{table_name}
UNION ALL
SELECT
{parquet_column_list}
FROM
    file_records fr,
    read_parquet(fr.file_path) p
WHERE
    fr.logical_table_name = '{table_name}' AND fr.storage_location = 'external';

COMMENT ON VIEW {table_name} IS 'Unified view for {table_name} data, combining uploaded and external sources.';
"""


def parse_create_table(sql_content: str):
    """
    Parses a 'CREATE TABLE' statement to extract table name and column definitions.
    """
    create_table_pattern = re.compile(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\((.*?)\);",
        re.DOTALL | re.IGNORECASE
    )

    tables = {}
    for match in create_table_pattern.finditer(sql_content):
        table_name = match.group(1)
        body = match.group(2)
        
        # Remove SQL comments
        body = re.sub(r'--.*$', '', body, flags=re.MULTILINE)
        
        columns = []
        # Split into potential column definitions or constraints
        # Split by comma that is NOT inside parentheses (for constraint definitions)
        parts = re.split(r',\s*(?![^()]*\))', body)
        
        for part in parts:
            part = part.strip()
            if not part: continue
            
            # Check if this part is a constraint definition
            if re.match(r'(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CONSTRAINT)', part, re.IGNORECASE):
                continue
            
            # Extract column name - it's the first word. Can be quoted.
            col_match = re.match(r'^("?\w+"?)\s+', part)
            if col_match:
                column_name = col_match.group(1) # Keep quotes for now
                columns.append(column_name)

        # Filter out duplicates and ensure proper order
        unique_ordered_columns = []
        seen = set()
        for col in columns:
            if col not in seen:
                unique_ordered_columns.append(col)
                seen.add(col)

        tables[table_name] = unique_ordered_columns

    return tables


def generate_sql(file_path: Path):
    """
    Generates the SQL for renamed tables from a given schema file.
    The concept of a static, unified view is deprecated in favor of 
    programmatic query generation.
    """
    logging.info(f"Processing schema file: {file_path.name}")
    content = file_path.read_text()

    # Find all CREATE TABLE statements in the file
    tables_and_columns = parse_create_table(content)

    renamed_tables_sql = content

    for table_name, columns in tables_and_columns.items():
        # 1. Rename the original table to "uploaded_<name>"
        renamed_tables_sql = renamed_tables_sql.replace(
            f"CREATE TABLE IF NOT EXISTS {table_name}",
            f"CREATE TABLE IF NOT EXISTS uploaded_{table_name}"
        )
        
        # 2. Update foreign key references within this file
        for t_name in tables_and_columns.keys():
            renamed_tables_sql = renamed_tables_sql.replace(
                f"REFERENCES {t_name}",
                f"REFERENCES uploaded_{t_name}"
            )
        
        # 3. Rename indexes
        renamed_tables_sql = renamed_tables_sql.replace(
            f"ON {table_name}",
            f"ON uploaded_{table_name}"
        )
        renamed_tables_sql = renamed_tables_sql.replace(
            f"idx_{table_name}_",
            f"idx_uploaded_{table_name}_"
        )
        renamed_tables_sql = renamed_tables_sql.replace(
            f"COMMENT ON TABLE {table_name}",
            f"COMMENT ON TABLE uploaded_{table_name}"
        )

    return renamed_tables_sql


def main():
    """
    Main function to drive the SQL generation process.
    """
    logging.info("Starting SQL table rename script.")
    GENERATED_DIR.mkdir(exist_ok=True)
    
    all_renamed_sql = []
    
    for file_name in FILES_TO_PROCESS:
        file_path = SCHEMA_DIR / file_name
        if not file_path.exists():
            logging.warning(f"Schema file not found: {file_path}, skipping.")
            continue

        renamed_sql = generate_sql(file_path)
        all_renamed_sql.append(renamed_sql)

    # Write the renamed table definitions to a single file
    renamed_path = GENERATED_DIR / "06_asim_outputs_uploaded.sql"
    renamed_path.write_text("---\n".join(all_renamed_sql))
    logging.info(f"Generated renamed schema: {renamed_path}")

    logging.info("SQL rename script complete.")


if __name__ == "__main__":
    main()
