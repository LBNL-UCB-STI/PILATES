#!/bin/sh

if  [ $# -eq 2 ]
    then
        echo "Uploading results to Google Cloud in Region: $1";
        echo "GCloud directory: $2";

        gcloud alpha storage cp -r -n pilates/beam/beam_output/$1/year-* gs://cruise-outputs/$2/beam/
        gcloud alpha storage cp -r -n pilates/activitysim/output/final* gs://cruise-outputs/$2/activitysim/
        gcloud alpha storage cp -r -n pilates/activitysim/output/year* gs://cruise-outputs/$2/activitysim/
        gcloud alpha storage cp -r -n pilates/activitysim/output/pipeline.h5 gs://cruise-outputs/$2/activitysim/pipeline.h5
        gcloud alpha storage cp -r -n pilates/activitysim/data/ gs://cruise-outputs/$2/activitysim/data/
        gcloud alpha storage cp -r -n pilates/postprocessing/output/ gs://cruise-outputs/$2/inexus/
        gcloud alpha storage cp -r -n pilates/postprocessing/MEP/ gs://cruise-outputs/$2/MEP/
        gcloud alpha storage cp -n log_pilates_$1.out gs://cruise-outputs/$2/log_pilates_$1.out
        rclone copy pilates/postprocessing/MEP/ s3://cruise-outputs/pilates-outputs/$2/MEP/

else
    echo "Please provide a region (e.g. 'austin' or 'sfbay') and GCloud directory name"
fi        
