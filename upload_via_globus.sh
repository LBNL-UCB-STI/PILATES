#!/bin/sh

if  [ $# -eq 3 ]
    then
        file=$1
        region=$2
        output_dir=$3

        echo "Reading run state from file $file"

        folder_name=$(awk  -F ':' '/folder_name/ {print $2=$2;}' $file | xargs)
        path=$(awk  -F ':' '/path/ {print $2=$2;}' $file | xargs)

        echo "Looking for output files at $path/$folder_name"

        echo "Uploading results to GCloud in Region: $region via Globus";
        echo "Output directory: $output_dir";

        # Globus Tutorial Collection IDs
        FROM="5791d5ee-c85a-4753-91f0-502a80d050d7:$path/$folder_name"
        TO="54047297-0b17-4dd9-ba50-ba1dc2063468:beam-core-outputs"

        find "$FROM/activitysim/output/" -type f \( -name "households" -o -name "land_use" -o -name "persons" -o -name "plans" -o -name "tours" -o -name "trips" \) -exec sh -c 'echo "Renaming: $1 -> $1.parquet"; mv "$1" "$1.parquet"' _ {} \;

        globus mkdir "$TO/$output_dir"
        globus mkdir "$TO/$output_dir/beam"
        globus mkdir "$TO/$output_dir/activitysim"
        globus mkdir "$TO/$output_dir/activitysim/data"
        globus transfer "$FROM/beam/beam_output/$region/" "$TO/$output_dir/beam/" -s size --recursive --label "BEAM Outputs" --exclude "*xml*"
        globus transfer "$FROM/activitysim/data/" "$TO/$output_dir/activitysim/data/" -s size --recursive --label "ASim Inputs"
        for dir in "$output_dir/activitysim/output/"year*; do
          if [ -d "$dir" ]; then
            # Example command: list files in the directory
            echo "Processing $dir"
            # Place your command here, e.g.:
             globus transfer "$FROM/$dir" "$TO/$output_dir/activitysim/$(basename "$dir")/" -s size --recursive --label "ASim Outputs"
          fi
        done
else
    echo "Please provide a region (e.g. 'austin' or 'sfbay') and S3 directory name"
fi        

