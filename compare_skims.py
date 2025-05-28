import sys
import os
import argparse
import logging

import openmatrix as omx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # We don't specify handlers here, so default console output is prevented.
    # We will explicitly add a StreamHandler and a FileHandler.
)

logger = logging.getLogger(__name__) # Get the logger for this module

# Get the root logger
root_logger = logging.getLogger()
# Clear any default handlers that might have been added by basicConfig or previous calls
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Define the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a console handler (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Set level for console output
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler) # Add handler to the root logger

# Create a file handler
log_file_path = "compare_skims.log" # Define the name of the log file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO) # Set level for file output
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler) # Add handler to the root logger

logger.info(f"Logging output to console and file: {log_file_path}")

def parse_matrix_name(matrix_name):
    """Parses a matrix name into path, measure, and time period."""
    # Expected format: PATH_MEASURE__TIMEPERIOD or MEASURE (for non-OD skims like DIST)
    parts = matrix_name.rsplit('__', 1)
    if len(parts) == 2:
        measure_part, time_period = parts
        # Try to split measure_part into path and measure
        measure_parts = measure_part.rsplit('_', 1)
        if len(measure_parts) == 2:
            path, measure = measure_parts
        else:
            # Handle cases like 'DIST__AM' where there's no explicit path
            path = None
            measure = measure_part # This line was correct
    else:
        # Handle cases with no time period like 'DIST'
        measure_part = matrix_name
        time_period = None
        measure_parts = measure_part.rsplit('_', 1)
        if len(measure_parts) == 2:
             # Could be SOV_DIST or SOVTOLL_TOLL
             path, measure = measure_parts
        else:
            # Could be just DIST, or a non-standard name
            path = None
            measure = measure_part # FIX: Changed 'matrix_part' to 'measure_part'

    return path, measure, time_period

def calculate_matrix_statistics(data, matrix_name, file_path):
    """
    Calculates statistics for a given NumPy array matrix.

    Parameters
    ----------
    data : np.ndarray
        The matrix data.
    matrix_name : str
        The name of the matrix.
    file_path : str
        The path to the OMX file (for logging).

    Returns
    -------
    dict
        A dictionary containing calculated statistics.
    """
    if data is None:
        return {"exists": False}

    total_elements = data.size
    is_nan = np.isnan(data)
    nan_count = np.sum(is_nan)
    nan_percent = (nan_count / total_elements) * 100 if total_elements > 0 else 0

    # Consider zeros as potentially "missing" or invalid for some skims
    is_zero = (data == 0)
    zero_count = np.sum(is_zero)
    zero_percent = (zero_count / total_elements) * 100 if total_elements > 0 else 0

    # Valid data: not NaN and not zero
    is_valid = ~(is_nan | is_zero)
    valid_count = np.sum(is_valid)
    valid_percent = (valid_count / total_elements) * 100 if total_elements > 0 else 0

    stats = {
        "exists": True,
        "shape": data.shape,
        "total_elements": total_elements,
        "nan_count": nan_count,
        "nan_percent": nan_percent,
        "zero_count": zero_count,
        "zero_percent": zero_percent,
        "valid_count": valid_count,
        "valid_percent": valid_percent,
        "descriptive_stats": {}
    }

    if valid_count > 0:
        valid_data = data[is_valid]
        stats["descriptive_stats"] = {
            "mean": np.mean(valid_data),
            "median": np.median(valid_data),
            "min": np.min(valid_data),
            "max": np.max(valid_data),
            "std": np.std(valid_data)
        }

    return stats

def compare_skims(path1, path2):
    """
    Compares two OMX skim files matrix by matrix.

    Parameters
    ----------
    path1 : str
        Path to the first OMX file.
    path2 : str
        Path to the second OMX file.
    """
    if not os.path.exists(path1):
        logger.error(f"File not found: {path1}")
        return
    if not os.path.exists(path2):
        logger.error(f"File not found: {path2}")
        return

    logger.info(f"Comparing skim files:")
    logger.info(f"  File 1: {path1}")
    logger.info(f"  File 2: {path2}")
    logger.info("-" * 80)

    skims1 = None
    skims2 = None

    try:
        skims1 = omx.open_file(path1, 'r')
        skims2 = omx.open_file(path2, 'r')

        tables1 = set(skims1.list_matrices())
        tables2 = set(skims2.list_matrices())

        all_tables = sorted(list(tables1.union(tables2)))

        missing_in_2 = sorted(list(tables1 - tables2))
        missing_in_1 = sorted(list(tables2 - tables1))

        if missing_in_2:
            logger.warning(f"Matrices present in '{os.path.basename(path1)}' but not in '{os.path.basename(path2)}': {missing_in_2}")
        if missing_in_1:
            logger.warning(f"Matrices present in '{os.path.basename(path2)}' but not in '{os.path.basename(path1)}': {missing_in_1}")

        comparison_summary = []

        for table_name in all_tables:
            logger.info(f"\n--- Comparing matrix: {table_name} ---")

            path, measure, period = parse_matrix_name(table_name)

            # --- Basic Statistics (Keep existing logic) ---
            data1 = None
            data2 = None
            stats1 = {"exists": False}
            stats2 = {"exists": False}

            # Load data if exists
            if table_name in tables1:
                try:
                    data1 = np.array(skims1[table_name])
                    stats1 = calculate_matrix_statistics(data1, table_name, path1)
                except Exception as e:
                    logger.error(f"Error reading matrix {table_name} from {path1}: {e}")
                    stats1 = {"exists": True, "error": str(e)}  # Mark as exists but error happened

            if table_name in tables2:
                try:
                    data2 = np.array(skims2[table_name])
                    stats2 = calculate_matrix_statistics(data2, table_name, path2)
                except Exception as e:
                    logger.error(f"Error reading matrix {table_name} from {path2}: {e}")
                    stats2 = {"exists": True, "error": str(e)}  # Mark as exists but error happened

            # Report basic statistics for each file (Keep existing logic)
            logger.info(f"  File 1 ({os.path.basename(path1)}):")
            # ... (existing reporting for stats1) ...
            if stats1["exists"]:
                if "error" in stats1:
                    logger.info(f"    Error: {stats1['error']}")
                else:
                    logger.info(f"    Shape: {stats1['shape']}")
                    logger.info(f"    Total Elements: {stats1['total_elements']}")
                    logger.info(f"    NaNs: {stats1['nan_count']} ({stats1['nan_percent']:.2f}%)")
                    logger.info(f"    Zeros: {stats1['zero_count']} ({stats1['zero_percent']:.2f}%)")
                    logger.info(f"    Valid: {stats1['valid_count']} ({stats1['valid_percent']:.2f}%)")
                    if stats1['valid_count'] > 0:
                        desc_stats = stats1["descriptive_stats"]
                        logger.info(
                            f"    Valid Data Stats: Mean={desc_stats['mean']:.4f}, Median={desc_stats['median']:.4f}, Min={desc_stats['min']:.4f}, Max={desc_stats['max']:.4f}, Std Dev={desc_stats['std']:.4f}")
            else:
                logger.info("    Matrix not found.")

            logger.info(f"  File 2 ({os.path.basename(path2)}):")
            # ... (existing reporting for stats2) ...
            if stats2["exists"]:
                if "error" in stats2:
                    logger.info(f"    Error: {stats2['error']}")
                else:
                    logger.info(f"    Shape: {stats2['shape']}")
                    logger.info(f"    Total Elements: {stats2['total_elements']}")
                    logger.info(f"    NaNs: {stats2['nan_count']} ({stats2['nan_percent']:.2f}%)")
                    logger.info(f"    Zeros: {stats2['zero_count']} ({stats2['zero_percent']:.2f}%)")
                    logger.info(f"    Valid: {stats2['valid_count']} ({stats2['valid_percent']:.2f}%)")
                    if stats2['valid_count'] > 0:
                        desc_stats = stats2["descriptive_stats"]
                        logger.info(
                            f"    Valid Data Stats: Mean={desc_stats['mean']:.4f}, Median={desc_stats['median']:.4f}, Min={desc_stats['min']:.4f}, Max={desc_stats['max']:.4f}, Std Dev={desc_stats['std']:.4f}")
            else:
                logger.info("    Matrix not found.")

            # --- Feasibility and Segmented Average Analysis ---
            # Only perform this for measures other than TRIPS/FAILURES and if a path and period were parsed
            if path and period and measure not in ["TRIPS", "FAILURES"]:
                # Determine potential feasibility matrices
                feasibility_measures = ["TOTIVT", "IVT", "TIME", "DIST"]  # Prioritize time/dist measures
                feasibility_matrix_name = None

                for fm in feasibility_measures:
                    potential_name = f"{path}_{fm}__{period}"
                    if potential_name in all_tables:
                        feasibility_matrix_name = potential_name
                        break  # Found a feasibility matrix

                if feasibility_matrix_name:
                    logger.info(f"  Using '{feasibility_matrix_name}' to determine feasible ODs.")

                    feasibility_data1 = None
                    feasibility_data2 = None
                    mask1 = None
                    mask2 = None

                    # Load feasibility data
                    if feasibility_matrix_name in tables1:
                        try:
                            feasibility_data1 = np.array(skims1[feasibility_matrix_name])
                        except Exception as e:
                            logger.error(
                                f"Error reading feasibility matrix {feasibility_matrix_name} from {path1}: {e}")

                    if feasibility_matrix_name in tables2:
                        try:
                            feasibility_data2 = np.array(skims2[feasibility_matrix_name])
                        except Exception as e:
                            logger.error(
                                f"Error reading feasibility matrix {feasibility_matrix_name} from {path2}: {e}")

                    # Create feasibility masks (where value > 0)
                    shape_match = True
                    if feasibility_data1 is not None and feasibility_data2 is not None:
                        if feasibility_data1.shape != feasibility_data2.shape:
                            logger.warning(
                                f"  Shape mismatch for feasibility matrix {feasibility_matrix_name}: {feasibility_data1.shape} vs {feasibility_data2.shape}. Skipping feasibility analysis.")
                            shape_match = False
                        else:
                            # Ensure data is not NaN before checking > 0
                            mask1 = (~np.isnan(feasibility_data1)) & (feasibility_data1 > 0)
                            mask2 = (~np.isnan(feasibility_data2)) & (feasibility_data2 > 0)
                    elif feasibility_data1 is not None:
                        logger.warning(
                            f"  Feasibility matrix {feasibility_matrix_name} only found in {os.path.basename(path1)}. Cannot compare feasibility counts or segmented averages.")
                        shape_match = False  # Cannot do full comparison
                    elif feasibility_data2 is not None:
                        logger.warning(
                            f"  Feasibility matrix {feasibility_matrix_name} only found in {os.path.basename(path2)}. Cannot compare feasibility counts or segmented averages.")
                        shape_match = False  # Cannot do full comparison
                    else:
                        logger.debug(f"  Feasibility matrix {feasibility_matrix_name} not found in either file.")
                        shape_match = False  # Cannot do full comparison

                    if mask1 is not None and mask2 is not None and shape_match:
                        feasible_count1 = np.sum(mask1)
                        feasible_count2 = np.sum(mask2)
                        feasible_diff = feasible_count2 - feasible_count1

                        logger.info(f"  Feasible ODs (where {feasibility_matrix_name} > 0):")
                        logger.info(f"    File 1: {feasible_count1}")
                        logger.info(f"    File 2: {feasible_count2}")
                        logger.info(f"    Difference (File 2 - File 1): {feasible_diff}")

                        if feasible_count1 > 0 or feasible_count2 > 0:
                            # Define OD categories based on feasibility
                            mask_both_feasible = mask1 & mask2
                            mask_file1_only = mask1 & ~mask2
                            mask_file2_only = ~mask1 & mask2

                            both_count = np.sum(mask_both_feasible)
                            f1_only_count = np.sum(mask_file1_only)
                            f2_only_count = np.sum(mask_file2_only)

                            logger.info(f"  OD Category Counts (based on {feasibility_matrix_name} > 0):")
                            logger.info(f"    Feasible in both: {both_count}")
                            logger.info(f"    Feasible in File 1 only: {f1_only_count}")
                            logger.info(f"    Feasible in File 2 only: {f2_only_count}")

                            # Calculate and report segmented averages for the CURRENT matrix (table_name)
                            if stats1["exists"] and stats2[
                                "exists"] and "error" not in stats1 and "error" not in stats2:
                                if stats1["shape"] == stats2["shape"]:
                                    logger.info(f"  Segmented Averages for {table_name}:")

                                    def safe_mean(arr, mask):
                                        # Calculate mean only if mask is not empty and filtered data is not all NaN
                                        filtered_data = arr[mask]
                                        if filtered_data.size > 0 and not np.all(np.isnan(filtered_data)):
                                            return np.nanmean(
                                                filtered_data)  # Use nanmean to ignore NaNs in selected slice
                                        return np.nan  # Return NaN if no data or all NaN

                                    # Both feasible
                                    mean1_both = safe_mean(data1, mask_both_feasible)
                                    mean2_both = safe_mean(data2, mask_both_feasible)
                                    logger.info(
                                        f"    Feasible in both ({both_count} ODs): File 1 Avg={mean1_both:.4f}, File 2 Avg={mean2_both:.4f}")

                                    # File 1 only feasible
                                    mean1_f1only = safe_mean(data1, mask_file1_only)
                                    mean2_f1only = safe_mean(data2,
                                                             mask_file1_only)  # Value in file 2 should ideally be 0 or NaN if not feasible
                                    logger.info(
                                        f"    Feasible in File 1 only ({f1_only_count} ODs): File 1 Avg={mean1_f1only:.4f}, File 2 Avg={mean2_f1only:.4f}")

                                    # File 2 only feasible
                                    mean1_f2only = safe_mean(data1,
                                                             mask_file2_only)  # Value in file 1 should ideally be 0 or NaN if not feasible
                                    mean2_f2only = safe_mean(data2, mask_file2_only)
                                    logger.info(
                                        f"    Feasible in File 2 only ({f2_only_count} ODs): File 1 Avg={mean1_f2only:.4f}, File 2 Avg={mean2_f2only:.4f}")

                                else:
                                    logger.warning(
                                        f"  Shape mismatch for {table_name} ({stats1['shape']} vs {stats2['shape']}). Cannot calculate segmented averages.")
                            else:
                                logger.debug(
                                    f"  Skipping segmented average calculation for {table_name}: matrix not found or error occurred in one file.")
                    # Else: Feasibility matrix existed but had errors or shape mismatch, detailed analysis skipped.
                else:
                    logger.debug(
                        f"  No common feasibility matrix found for {path}_{measure}__{period}. Skipping segmented analysis.")
            # Else: Basic statistics already reported.

            # --- Overall Comparison (Keep existing logic, perhaps refine warnings based on detailed analysis) ---
            # This section can remain largely the same, providing high-level comparison
            # based on basic stats (NaN, Zero, Valid counts, overall mean/median/etc. of Valid data).
            # If a detailed segmented analysis was performed, the warnings here about NaN/Zero/Valid counts
            # might be less critical than the findings in the segmented analysis, but they are still useful.
            if stats1["exists"] and stats2["exists"] and "error" not in stats1 and "error" not in stats2:
                if stats1["shape"] != stats2["shape"]:
                    pass  # Shape mismatch already reported above
                elif stats1["total_elements"] == 0:
                    pass  # Empty matrices already reported above
                else:
                    # Compare missing data percentages (NaNs, Zeros)
                    nan_diff = abs(stats1['nan_percent'] - stats2['nan_percent'])
                    zero_diff = abs(stats1['zero_percent'] - stats2['zero_percent'])

                    if nan_diff > 1.0:  # Threshold for warning
                        logger.warning(
                            f"  Overall Significant difference in NaN percentage: {stats1['nan_percent']:.2f}% vs {stats2['nan_percent']:.2f}% (Diff: {nan_diff:.2f}%)")
                        comparison_summary.append(f"{table_name}: Overall NaN Diff={nan_diff:.2f}%")

                    if zero_diff > 1.0:  # Threshold for warning
                        logger.warning(
                            f"  Overall Significant difference in Zero percentage: {stats1['zero_percent']:.2f}% vs {stats2['zero_percent']:.2f}% (Diff: {zero_diff:.2f}%)")
                        comparison_summary.append(f"{table_name}: Overall Zero Diff={zero_diff:.2f}%")

                    # Compare descriptive statistics of *all* valid data (non-NaN, non-zero)
                    if stats1['valid_count'] > 0 and stats2['valid_count'] > 0:
                        desc1 = stats1["descriptive_stats"]
                        desc2 = stats2["descriptive_stats"]
                        comp_notes = []

                        for stat_name in ["mean", "median", "min", "max", "std"]:
                            val1 = desc1[stat_name]
                            val2 = desc2[stat_name]

                            if abs(val1) < 1e-9 and abs(val2) < 1e-9:
                                ratio = 1.0
                            elif abs(val1) < 1e-9:
                                ratio = 0.0
                            elif abs(val2) < 1e-9:
                                ratio = float('inf')
                            else:
                                ratio = val2 / val1

                            # Check for significant ratio difference (e.g., outside 0.5x to 2x)
                            # or large absolute difference for small values
                            significant_ratio_diff = ratio < 0.5 or ratio > 2.0
                            significant_abs_diff = abs(val1 - val2) > 0.01 and (
                                        abs(val1) > 1e-9 or abs(val2) > 1e-9)  # Reduced threshold for absolute diff

                            if significant_ratio_diff or significant_abs_diff:
                                warning_msg = f"  Overall Significant difference in {stat_name}: {val1:.4f} vs {val2:.4f}"
                                if ratio != 0.0 and ratio != float('inf'):
                                    warning_msg += f" (Ratio: {ratio:.2f})"
                                logger.warning(warning_msg)
                                comp_notes.append(f"Overall {stat_name} Ratio={ratio:.2f}")

                        if comp_notes:
                            comparison_summary.append(f"{table_name}: {', '.join(comp_notes)}")

                    elif stats1['valid_count'] == 0 and stats2['valid_count'] > 0:
                        logger.warning(
                            f"  Overall Valid data exists in File 2 but not in File 1 ({stats1['valid_count']} vs {stats2['valid_count']}).")
                        comparison_summary.append(f"{table_name}: Overall NO VALID DATA IN FILE 1")
                    elif stats1['valid_count'] > 0 and stats2['valid_count'] == 0:
                        logger.warning(
                            f"  Overall Valid data exists in File 1 but not in File 2 ({stats1['valid_count']} vs {stats2['valid_count']}).")
                        comparison_summary.append(f"{table_name}: Overall NO VALID DATA IN FILE 2")
                    elif stats1['valid_count'] == 0 and stats2['valid_count'] == 0:
                        pass  # Both have no valid data, already noted.

        logger.info("\n" + "=" * 80)
        logger.info("Comparison Summary:")
        logger.info("-" * 80)
        if comparison_summary:
             for note in comparison_summary:
                  logger.info(note)
        else:
             logger.info("No significant differences found.")
        logger.info("=" * 80)


    except Exception as e:
        logger.critical(f"An error occurred during skim comparison: {e}")
    finally:
        if skims1:
            try:
                skims1.close()
            except Exception as e:
                logger.error(f"Error closing {path1}: {e}")
        if skims2:
            try:
                skims2.close()
            except Exception as e:
                logger.error(f"Error closing {path2}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare statistics of matrices in two OMX skim files.")
    parser.add_argument(
        "path1",
        help="Path to the first OMX skim file.")
    parser.add_argument(
        "path2",
        help="Path to the second OMX skim file.")

    args = parser.parse_args()

    compare_skims(args.path1, args.path2)