#!/bin/sh

if  [ $# -eq 2 ]
    then
        echo "Uploading results to GCloud in Region: $1 via Globus";
        echo "Output directory: $2";

        # Globus Tutorial Collection IDs
        FROM="5791d5ee-c85a-4753-91f0-502a80d050d7:/global/scratch/users/zaneedell/sources/PILATES"
        TO="54047297-0b17-4dd9-ba50-ba1dc2063468:beam-core-outputs"

        globus mkdir "$TO/$2"
        globus transfer "$FROM/pilates/beam/beam_output/$1/" "$TO/$2/beam/" --recursive --label "BEAM Outputs" --exclude "*xml*" --include "year-*" --exclude "*"
        globus transfer "$FROM/pilates/activitysim/output/" "$TO/$2/activitysim/" --recursive --label "ASim Outputs" --include "final*" --include "year*" --exclude "*"
        globus transfer "$FROM/pilates/activitysim/data/" "$TO/$2/activitysim/data/" --recursive --label "ASim Inputs"
else
    echo "Please provide a region (e.g. 'austin' or 'sfbay') and S3 directory name"
fi        

