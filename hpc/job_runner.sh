#!/bin/bash

RANDOM_PART="$(tr -dc A-Z0-9 </dev/urandom | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

PARTITION="es1"
QOS="es_normal"
MEMORY_LIMIT="80"
ACCOUNT="pc_beamcore"
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"
EXPECTED_EXECUTION_DURATION="0-01:00:00"

mkdir -p "/global/scratch/users/$USER/pilates_logs"

sbatch --partition="$PARTITION" \
    --exclusive \
    --mem="${MEMORY_LIMIT}G" \
    --qos="$QOS" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="$JOB_LOG_FILE_PATH" \
    --time="$EXPECTED_EXECUTION_DURATION" \
    job.sh

echo $JOB_LOG_FILE_PATH
