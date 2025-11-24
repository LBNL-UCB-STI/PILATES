
import json
import argparse
import shutil
from pathlib import Path

def patch_run_info(run_info_path_str: str, file_type: str, no_backup: bool):
    """
    Finds all 'trips_asim_out' records in a run_info.json file, sorts them
    chronologically, and updates their file_path and iteration number.
    """
    run_info_path = Path(run_info_path_str)

    if not run_info_path.exists():
        print(f"Error: File not found at {run_info_path}")
        return

    # Create a backup unless disabled
    if not no_backup:
        backup_path = run_info_path.with_suffix(run_info_path.suffix + '.bak')
        print(f"Creating backup at {backup_path}")
        shutil.copy(run_info_path, backup_path)

    # Read the JSON data
    try:
        with open(run_info_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return

    if 'file_records' not in data:
        print("Error: 'file_records' key not found in JSON data.")
        return

    # Find and collect all 'trips_asim_out' records
    trips_records = []
    for record in data['file_records'].values():
        if record.get("short_name") == file_type:
            trips_records.append(record)

    if not trips_records:
        print(f"No '{file_type}' records found to update.")
        return

    print(f"Found {len(trips_records)} '{file_type}' records to update.")

    # Sort records by their creation timestamp to ensure correct order
    trips_records.sort(key=lambda r: r.get('created_at', ''))

    # Iterate through the sorted records and apply the patches
    for i, record in enumerate(trips_records):
        year = record.get('year')
        if year is None:
            print(f"Warning: Record {record.get('unique_id')} is missing a 'year' and will be skipped.")
            continue

        print(f"Patching record for iteration {i} (Year: {year})")

        # 1. Construct the new, correct file path
        short_name = file_type.replace("_asim_out", "")
        new_path = f"activitysim/output/year-{year}-iteration-{i}/{short_name}.parquet"
        record['file_path'] = new_path

        # 2. Update the iteration number
        record['iteration'] = i

        # Note: sub_iteration is left as-is, as requested.

    # Write the modified data back to the file
    try:
        with open(run_info_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSuccessfully patched {run_info_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")


def main():
    """Main function to parse arguments and run the patcher."""
    parser = argparse.ArgumentParser(
        description="Patches the file_path and iteration for 'trips_asim_out' records in a run_info.json file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run_info_file", help="Path to the run_info.json file to patch.")
    parser.add_argument("file_type", default="trips_asim_out", help="File type to patch.")
    parser.add_argument("--no-backup", action="store_true", help="If set, a backup of the original file will not be created.")
    
    args = parser.parse_args()
    patch_run_info(args.run_info_file, args.file_type, args.no_backup)


if __name__ == "__main__":
    main()
