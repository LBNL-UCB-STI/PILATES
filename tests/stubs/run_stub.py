import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, help="Name of the model being stubbed"
    )
    parser.add_argument(
        "--cwd", required=True, help="Current working directory for the model"
    )
    parser.add_argument(
        "--config_name", required=True, help="Name of the configuration file"
    )
    args = parser.parse_args()

    print(f"Running stub for model: {args.model_name}")
    print(f"Working directory: {args.cwd}")
    print(f"Config name: {args.config_name}")

    # Simulate some work
    time.sleep(1)

    # Create dummy output files based on model_name
    if args.model_name == "urbansim":
        output_path = os.path.join(args.cwd, "urbansim_output.h5")
        with open(output_path, "w") as f:
            f.write("Dummy UrbanSim Output")
        print(f"Created dummy UrbanSim output file: {output_path}")
    elif args.model_name == "atlas":
        output_path = os.path.join(args.cwd, "atlas_output.csv")
        with open(output_path, "w") as f:
            f.write("Dummy Atlas Output")
        print(f"Created dummy Atlas output file: {output_path}")
    elif args.model_name == "activitysim":
        output_path = os.path.join(args.cwd, "activitysim_output.parquet")
        with open(output_path, "w") as f:
            f.write("Dummy ActivitySim Output")
        print(f"Created dummy ActivitySim output file: {output_path}")
    elif args.model_name == "beam":
        output_path = os.path.join(args.cwd, "beam_output.csv.gz")
        with open(output_path, "w") as f:
            f.write("Dummy BEAM Output")
        print(f"Created dummy BEAM output file: {output_path}")
    elif args.model_name == "activitysim":
        for output_type in ["beam_plans", "person", "trips", "households", "tours"]:
            output_dir = os.path.join(args.cwd, "output", "final_pipeline", output_type)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "final.parquet")
            with open(output_path, "w") as f:
                f.write("Dummy ActivitySim Output")
            print(f"Created dummy ActivitySim output file: {output_path}")
    else:
        print(f"Unknown model_name for stub: {args.model_name}")

    print(f"Finished stub for model: {args.model_name}")


if __name__ == "__main__":
    main()
