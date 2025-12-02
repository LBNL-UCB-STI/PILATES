import json
import argparse
import shutil
import re
from pathlib import Path


def add_missing_iterations(run_info_path_str: str, no_backup: bool):
    """
    Parses short_names in a run_info.json file to add missing 'iteration'
    and 'sub_iteration' keys to file records.
    """
    run_info_path = Path(run_info_path_str)

    if not run_info_path.exists():
        print(f"Error: File not found at {run_info_path}")
        return

    if not no_backup:
        backup_path = run_info_path.with_suffix(run_info_path.suffix + ".bak")
        print(f"Creating backup at {backup_path}")
        shutil.copy(run_info_path, backup_path)

    try:
        with open(run_info_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return

    if "file_records" not in data:
        print("Error: 'file_records' key not found in JSON data.")
        return

    updated_count = 0
    # Regex to find an iteration number at the end of a short_name, e.g., "name_2018_1" -> "1"
    iter_pattern = re.compile(r"_(\d+)$")

    for record in data["file_records"].values():
        # Check if iteration is missing or null
        if record.get("iteration") is None:
            short_name = record.get("short_name", "")
            match = iter_pattern.search(short_name)

            if match:
                iteration_num = int(match.group(1))
                record["iteration"] = iteration_num

                # Also ensure sub_iteration exists
                if record.get("sub_iteration") is None:
                    record["sub_iteration"] = 0

                updated_count += 1
                print(
                    f"Patched record '{short_name}': set iteration to {iteration_num}"
                )

    if updated_count > 0:
        try:
            with open(run_info_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nSuccessfully patched {updated_count} records in {run_info_path}")
        except IOError as e:
            print(f"Error writing to file: {e}")
    else:
        print("No records needed patching for missing iteration numbers.")


def main():
    """Main function to parse arguments and run the patcher."""
    parser = argparse.ArgumentParser(
        description="Adds missing 'iteration' fields to records in a run_info.json file by parsing the 'short_name'.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "run_info_file", help="Path to the run_info.json file to patch."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="If set, a backup of the original file will not be created.",
    )

    args = parser.parse_args()
    add_missing_iterations(args.run_info_file, args.no_backup)


if __name__ == "__main__":
    main()
