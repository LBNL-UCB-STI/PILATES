import logging
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def analyze_series_stats(series: pd.Series, distinct_threshold: int = 63) -> Dict[str, Any]:
    """
    Analyzes a pandas Series to extract metadata for SQL optimization.

    Args:
        series: The data column to analyze.
        distinct_threshold: Max distinct values to consider for ENUM generation.

    Returns:
        Dict containing type, min, max, is_enum, and enum_values.
    """
    stats = {
        "name": str(series.name),
        "type": str(series.dtype),
        "count": int(series.count()),  # Non-null count
        "nullable": bool(series.isnull().any())
    }

    # 1. Handle Numeric Data (Integers and Floats)
    if pd.api.types.is_numeric_dtype(series):
        # Drop NaNs for stat calculation
        clean_series = series.dropna()
        if not clean_series.empty:
            # specific numpy casting to ensure JSON serializable
            min_val = clean_series.min()
            max_val = clean_series.max()
            stats["min"] = min_val.item() if hasattr(min_val, 'item') else min_val
            stats["max"] = max_val.item() if hasattr(max_val, 'item') else max_val

            # Check if it's a disguised integer (float column with no decimals)
            # This helps decide if we can use INTEGER instead of DOUBLE
            if pd.api.types.is_float_dtype(series):
                is_integer_like = np.all(np.mod(clean_series, 1) == 0)
                stats["is_integer_like"] = bool(is_integer_like)

    # 2. Handle Categorical / String Data (Potential ENUMs)
    # We check object types, string types, or categorical types
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype,
                                                                                                  pd.CategoricalDtype):
        try:
            # Calculate cardinality (number of unique values)
            unique_count = series.nunique()
            stats["unique_count"] = unique_count

            # Heuristic: If unique count is low, it's a good ENUM candidate
            if 0 < unique_count <= distinct_threshold:
                stats["is_enum"] = True
                # Get the actual values for the ENUM definition, sorted
                # dropna=True is default in unique(), but being explicit
                unique_vals = series.unique()
                # Filter out NaNs and convert to string
                valid_vals = [str(x) for x in unique_vals if pd.notna(x)]
                stats["enum_values"] = sorted(valid_vals)
            else:
                stats["is_enum"] = False
        except Exception as e:
            logger.debug(f"Could not analyze enum stats for {series.name}: {e}")

    return stats

def _get_schema_from_h5(file_path: str, sample_rows: int = 1000) -> List[Dict[str, str]]:
    """
    Extracts a flattened schema from an HDF5 file using pandas.HDFStore
    to correctly identify tables and column names.
    """
    flat_schema = []
    try:
        with pd.HDFStore(file_path, mode="r") as store:
            for table_name in store.keys():
                # Select a sample to infer stats
                # Note: 'stop' argument in select limits rows read
                try:
                    df_sample = store.select(table_name, stop=sample_rows)

                    for col_name in df_sample.columns:
                        col_stats = analyze_series_stats(df_sample[col_name])
                        # Prepend table name to field name for flattening
                        col_stats["name"] = f"{table_name}:{col_name}".replace("/", "")
                        col_stats["h5_table"] = table_name
                        flat_schema.append(col_stats)
                except Exception as e:
                    # Fallback for tables that might be weird formats (fixed, etc)
                    logger.warning(f"Could not sample H5 table '{table_name}' with pandas.select, trying direct pytables read: {e}")

                    try:
                        node = store.get_node(table_name)

                        # Check if it's a PyTables Table object which can be sampled
                        if hasattr(node, 'read') and hasattr(node, 'colnames') and len(node.colnames) > 0:
                            # Read a sample of rows directly from the PyTables node.
                            # This is efficient and works for 'fixed' format tables.
                            sample_data = node.read(stop=sample_rows)
                            df_sample = pd.DataFrame(sample_data)

                            # Now that we have a sample DataFrame, analyze it
                            for col_name in df_sample.columns:
                                col_stats = analyze_series_stats(df_sample[col_name])
                                col_stats["name"] = f"{table_name}:{col_name}".replace("/", "")
                                col_stats["h5_table"] = table_name
                                flat_schema.append(col_stats)
                        else:
                            # If it's not a readable table, just get columns as a last resort
                            logger.warning(f"Node '{table_name}' is not a readable PyTables Table, extracting column names only.")
                            cols = []
                            if hasattr(node, "colnames"):
                                cols = node.colnames
                            elif hasattr(node, "description") and hasattr(node.description, "_v_names"):
                                cols = node.description._v_names

                            for n in cols:
                                flat_schema.append({"name": f"{table_name}:{n}", "type": "unknown", "h5_table": table_name})

                    except Exception as e_fallback:
                        logger.warning(f"Fallback schema inference for H5 table '{table_name}' failed: {e_fallback}")

    except Exception as e:
        logger.warning(f"Could not read HDF5 schema from {file_path}: {e}")
    return flat_schema


def _get_sparse_schema_for_wide_csv(file_path: str) -> List[Dict[str, str]]:
    """Reads a wide CSV, converts to long format, and returns the sparse schema."""
    try:
        df = pd.read_csv(file_path, index_col=0)
        df_long = df.stack().reset_index()
        df_long.columns = ["taz", "tract", "proportion"]
        df_long = df_long[df_long["proportion"] > 0]

        schema_info = []
        for col_name, col_type in df_long.dtypes.items():
            schema_info.append({"name": col_name, "type": str(col_type)})
        return schema_info
    except Exception as e:
        logger.warning(f"Could not generate sparse schema for {file_path}: {e}")
        return []


def get_schema_from_parquet(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads Parquet metadata directly. This is highly optimized and does NOT
    need to read the actual rows to get Min/Max stats, as Parquet
    stores these in the footer.
    """
    schema_info = []
    try:
        pq_file = pq.ParquetFile(file_path)

        # FIX: Convert physical ParquetSchema to logical ArrowSchema
        # This gives us access to .field() and rich types (Int8 vs Int64)
        schema = pq_file.schema.to_arrow_schema()
        metadata = pq_file.metadata

        # Iterate over columns using the Arrow schema
        for i in range(len(schema.names)):
            name = schema.names[i]
            field = schema.field(name)  # Now this works!

            col_stats = {
                "name": name,
                "type": str(field.type),
                "nullable": field.nullable
            }

            # Parquet stores stats in RowGroups. We iterate them to find global min/max.
            min_val = None
            max_val = None
            has_stats = False

            for rg in range(metadata.num_row_groups):
                rg_meta = metadata.row_group(rg)
                col_meta = rg_meta.column(i)
                if col_meta.is_stats_set:
                    has_stats = True
                    rg_min = col_meta.statistics.min
                    rg_max = col_meta.statistics.max

                    # Update global min
                    if min_val is None or (rg_min is not None and rg_min < min_val):
                        min_val = rg_min
                    # Update global max
                    if max_val is None or (rg_max is not None and rg_max > max_val):
                        max_val = rg_max

            if has_stats:
                # Parquet stats might be bytes, ensure serializability
                if isinstance(min_val, (bytes, bytearray)):
                    # Decode if text, otherwise leave for now (or skip)
                    try:
                        col_stats["min"] = min_val.decode('utf-8')
                        col_stats["max"] = max_val.decode('utf-8')
                    except:
                        pass  # Binary data
                else:
                    col_stats["min"] = min_val
                    col_stats["max"] = max_val

            schema_info.append(col_stats)

    except Exception as e:
        # This log message was visible in your test output, which confirmed the location
        logger.warning(f"Error reading Parquet metadata for {file_path}: {e}")

    return schema_info


def get_schema_from_file(file_path: str, sample_rows: int = 1000) -> List[Dict[str, str]]:
    """
    Infers the schema from a .csv or .parquet file.

    Args:
        file_path (str): The path to the file.

    Returns:
        A list of dictionaries, where each dictionary represents a field
        in the schema (e.g., [{'name': 'col1', 'type': 'int64'}]).
    """
    if "taz_to_tract" in os.path.basename(file_path):
        logger.info(f"Generating sparse schema for wide file: {file_path}")
        return _get_sparse_schema_for_wide_csv(file_path)

    schema_info = []
    try:
        if file_path.endswith(".csv") | file_path.endswith(".csv.gz"):
            # low_memory=False ensures pandas guesses types accurately for the chunk
            df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
            for col_name in df.columns:
                schema_info.append(analyze_series_stats(df[col_name]))
        elif file_path.endswith(".parquet"):
            # Use the optimized metadata reader
            schema_info = get_schema_from_parquet(file_path)
        elif file_path.endswith((".h5", ".hdf5")):
            return _get_schema_from_h5(file_path)
    except Exception as e:
        logger.warning(f"Could not automatically infer schema for {file_path}: {e}")
    return schema_info
