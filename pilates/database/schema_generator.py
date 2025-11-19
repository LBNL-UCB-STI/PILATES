import json
import os
import re
import argparse
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants for Integer Downcasting
SMALLINT_MIN, SMALLINT_MAX = -32768, 32767
INTEGER_MIN, INTEGER_MAX = -2147483648, 2147483647

CONVERT_ENUMS = False


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

    for filename in files_in_dir:
        if filename.endswith(".sql"):
            file_path = os.path.join(schema_dir, filename)
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    found_tables = re.findall(
                        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
                        content,
                        re.IGNORECASE,
                    )
                    found_views = re.findall(
                        r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)",
                        content,
                        re.IGNORECASE,
                    )

                    for table in found_tables:
                        match = re.search(r"(\d{4})", filename)
                        year = int(match.group(1)) if match else None
                        normalized_table_name = _normalize_table_name(table, year)
                        defined_names.add(normalized_table_name.lower())
                    for view in found_views:
                        defined_names.add(sanitize_name(view).lower())
            except IOError as e:
                logging.error(f"Could not read schema file {filename}: {e}")

    logging.info(f"Found {len(defined_names)} existing table/view definitions.")
    return defined_names


def sanitize_name(name: str, is_column=False) -> str:
    """
    Sanitizes a name to be a valid SQL identifier.
    """
    name = name.strip().strip("/")
    if is_column:
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    else:
        name = re.sub(r"[\s./-]", "_", name)

    if re.match(r"^[0-9]", name):
        name = "_" + name

    return name.lower()


def _normalize_table_name(short_name: str, year: Optional[int]) -> str:
    """
    Normalizes table names by removing year/iteration suffixes.
    """
    normalized_name = short_name.lower()

    if year:
        year_str = str(year)
        pattern = r"([_-]?" + re.escape(year_str) + r")(?![0-9])"
        normalized_name = re.sub(pattern, "", normalized_name).strip("_-")

    normalized_name = re.sub(
        r"model_data_(\d{4})_(\d{4})_(.*)", r"model_data_\3", normalized_name
    )
    normalized_name = re.sub(
        r"usim_output_(\d{4})_(\d{4})_(.*)", r"usim_output_\3", normalized_name
    )
    normalized_name = re.sub(
        r"usim_input_archive_(\d{4})_(.*)", r"usim_input_archive_\2", normalized_name
    )
    normalized_name = re.sub(r"([_-]?\d{4})(?![0-9])", "", normalized_name).strip("_-")
    normalized_name = re.sub(r"_{2,}", "_", normalized_name)

    return normalized_name.strip("_")


def determine_duckdb_type(
    col_data: Dict[str, Any], table_name: str, enum_definitions: List[str]
) -> str:
    """
    Analyzes column metadata to determine the optimal DuckDB SQL type.
    Populates enum_definitions list if an ENUM type is created.
    """
    raw_type = str(col_data.get("type", "varchar")).lower()
    col_name = sanitize_name(col_data.get("name", "unknown"))

    # 1. Handle Enums
    if col_data.get("is_enum") and col_data.get("enum_values"):
        if CONVERT_ENUMS:
            # Create a unique name for the enum type
            enum_type_name = f"{table_name}_{col_name}_enum"

            # Clean values: escape single quotes
            clean_values = [str(v).replace("'", "''") for v in col_data["enum_values"]]
            values_str = ", ".join([f"'{v}'" for v in clean_values])

            enum_sql = f"CREATE TYPE {enum_type_name} AS ENUM ({values_str});"
            enum_definitions.append(enum_sql)
            return enum_type_name
        else:
            return "VARCHAR"

    # 2. Handle Integers (Downcasting Logic)
    # Check if it is explicitly an int, or a float that looks like an int (common in pandas)
    is_int = "int" in raw_type
    is_int_like = col_data.get("is_integer_like", False)

    if is_int or is_int_like:
        min_val = col_data.get("min")
        max_val = col_data.get("max")

        # If we don't have stats, default to BIGINT to be safe
        if min_val is None or max_val is None:
            return "BIGINT"

        # Apply ranges
        if min_val >= SMALLINT_MIN and max_val <= SMALLINT_MAX:
            return "SMALLINT"
        elif min_val >= INTEGER_MIN and max_val <= INTEGER_MAX:
            return "INTEGER"
        else:
            return "BIGINT"

    # 3. Handle Floats
    if "float" in raw_type:
        return "DOUBLE"  # DuckDB prefers DOUBLE

    # 4. Handle Booleans
    if "bool" in raw_type:
        return "BOOLEAN"

    # 5. Handle Dates
    if "date" in raw_type or "time" in raw_type:
        return "TIMESTAMP"

    # 6. Fallback
    return "VARCHAR"


def generate_sql_for_record(record: dict, table_name: str) -> str:
    """
    Generates SQL definition including ENUM types and CREATE TABLE.
    Groups columns by type for readability.
    """
    if not record.get("schema"):
        return ""

    enum_definitions = []
    columns_by_category = {
        "keys": [],
        "integers": [],
        "doubles": [],
        "enums": [],
        "others": [],
    }

    # 1. Process columns and categorize them
    for col in record["schema"]:
        col_name = sanitize_name(col["name"], is_column=True)
        sql_type = determine_duckdb_type(col, table_name, enum_definitions)

        col_def = f"    {col_name} {sql_type},"

        # Categorize for sorting
        if col_name.endswith("_id") or col_name == "id":
            columns_by_category["keys"].append(col_def)
        elif "INT" in sql_type:
            columns_by_category["integers"].append(col_def)
        elif "DOUBLE" in sql_type:
            columns_by_category["doubles"].append(col_def)
        elif (
            "ENUM" in sql_type
        ):  # The type name will contain 'enum' based on our naming convention
            columns_by_category["enums"].append(col_def)
        else:
            columns_by_category["others"].append(col_def)

    # 2. Build SQL
    sql_lines = [
        f"-- Auto-generated schema for table: {table_name}",
        f"-- Source: {record.get('short_name', 'N/A')}",
        f"-- Description: {record.get('description', 'N/A')}",
        "",
    ]

    # Add ENUM definitions first
    if enum_definitions:
        sql_lines.append("-- Enum Definitions")
        sql_lines.extend(enum_definitions)
        sql_lines.append("")

    sql_lines.append(f"CREATE SEQUENCE IF NOT EXISTS {table_name}_id_seq START 1;")
    sql_lines.append("")

    sql_lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")

    # 3. Add Columns in specific order
    sql_lines.append("    -- Primary & Foreign Keys")
    sql_lines.append(
        f"    id BIGINT PRIMARY KEY DEFAULT nextval('{table_name}_id_seq'),"
    )
    sql_lines.append("    run_id VARCHAR,")
    sql_lines.append("    file_record_id VARCHAR,")
    sql_lines.append("    year INTEGER,")
    sql_lines.append("    iteration INTEGER,")
    sql_lines.append("    sub_iteration INTEGER,")

    # Add analyzed columns sorted by category
    if columns_by_category["keys"]:
        sql_lines.extend(sorted(columns_by_category["keys"]))

    if columns_by_category["integers"]:
        sql_lines.append("")
        sql_lines.append("    -- Integer attributes (optimized)")
        sql_lines.extend(sorted(columns_by_category["integers"]))

    if columns_by_category["enums"]:
        sql_lines.append("")
        sql_lines.append("    -- Categorical Enums")
        sql_lines.extend(sorted(columns_by_category["enums"]))

    if columns_by_category["doubles"]:
        sql_lines.append("")
        sql_lines.append("    -- Continuous variables")
        sql_lines.extend(sorted(columns_by_category["doubles"]))

    if columns_by_category["others"]:
        sql_lines.append("")
        sql_lines.append("    -- Strings and other attributes")
        sql_lines.extend(sorted(columns_by_category["others"]))

    # 4. Add standard Constraints
    sql_lines.append("")
    sql_lines.append("    -- Constraints")
    sql_lines.append("    FOREIGN KEY (run_id) REFERENCES runs(run_id),")
    sql_lines.append(
        "    FOREIGN KEY (file_record_id) REFERENCES file_records(unique_id)"
    )
    sql_lines.append(");")
    sql_lines.append("")

    description = (
        record.get("description", "Table auto-generated from run data.") or ""
    ).replace("'", "''")
    sql_lines.append(f"COMMENT ON TABLE {table_name} IS '{description}';")

    # 5. Add Indexes (REVISED FOR DUCKDB)

    # We KEEP indexes on high-cardinality identifiers (run_id, unique IDs)
    # because they are often used for joins or strict lookups.
    sql_lines.extend(
        [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_run_id ON {table_name}(run_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_year_iter ON {table_name}(year, iteration);",
        ]
    )

    # If the table has a 'person_id' or 'household_id', we create an index
    # because users might query these randomly, and the data might be sorted by Zone.
    entity_ids = ["person_id", "household_id", "tour_id", "trip_id"]

    col_names = [c["name"].lower() for c in record["schema"]]

    for eid in entity_ids:
        if eid in col_names:
            sql_lines.append(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{eid} ON {table_name}({eid});"
            )

    # C. Spatial Sorting Note (Keep this from previous step)
    sort_candidates = []
    if "zone_id" in col_names:
        sort_candidates.append("zone_id")
    if "origin" in col_names:
        sort_candidates.append("origin")

    if sort_candidates:
        cols_str = ", ".join(sort_candidates)
        sql_lines.append("")
        sql_lines.append(
            f"-- PERFORMANCE NOTE: Data should be physically sorted by (run_id, {cols_str})"
        )

    return "\n".join(sql_lines)


def main():
    """
    Main function to drive the schema generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate optimized DuckDB SQL schema files from a PILATES run_info.json file."
    )
    parser.add_argument(
        "run_info_path", help="Path to the run_info.json file to analyze."
    )
    args = parser.parse_args()

    if not os.path.exists(args.run_info_path):
        logging.error(f"Input file not found: {args.run_info_path}")
        return

    script_location = os.path.dirname(os.path.abspath(__file__))
    approved_schema_dir = os.path.join(script_location, "schema")
    generated_dir = os.path.join(approved_schema_dir, "generated")

    os.makedirs(generated_dir, exist_ok=True)

    existing_tables = get_existing_definitions(approved_schema_dir)

    try:
        with open(args.run_info_path, "r") as f:
            run_info = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to read or parse {args.run_info_path}: {e}")
        return

    new_files_generated = 0
    file_records = run_info.get("file_records", {})

    for record in file_records.values():
        if not record.get("schema") or not record.get("short_name"):
            continue

        if record.get("year") is None:
            match = re.search(r"(\d{4})", record["short_name"])
            if match:
                record["year"] = int(match.group(1))

        record["iteration"] = None
        record["sub_iteration"] = None

        sub_iter_match = re.search(r"_sub(\d+)$", record["short_name"])
        if sub_iter_match:
            record["sub_iteration"] = int(sub_iter_match.group(1))
            record["short_name"] = re.sub(r"_sub\d+$", "", record["short_name"])

        iter_match = re.search(r"_(\d+)$", record["short_name"])
        if iter_match:
            record["iteration"] = int(iter_match.group(1))
            record["short_name"] = re.sub(r"_\d+$", "", record["short_name"])

        normalized_short_name = _normalize_table_name(
            record["short_name"], record.get("year")
        )
        table_name = sanitize_name(normalized_short_name)

        if table_name in existing_tables:
            logging.info(f"Schema for table '{table_name}' already exists. Skipping.")
            continue

        logging.info(
            f"New table found: '{record['short_name']}' -> '{table_name}'. Generating optimized schema..."
        )

        sql_content = generate_sql_for_record(record, table_name)
        if not sql_content:
            continue

        output_filename = os.path.join(generated_dir, f"{table_name}.sql")
        try:
            with open(output_filename, "w") as f:
                f.write(sql_content)
            logging.info(f"Successfully generated file: {output_filename}")
            new_files_generated += 1
        except IOError as e:
            logging.error(f"Failed to write schema file for {table_name}: {e}")

    logging.info(
        f"Schema generation complete. Generated {new_files_generated} new files."
    )


if __name__ == "__main__":
    main()
