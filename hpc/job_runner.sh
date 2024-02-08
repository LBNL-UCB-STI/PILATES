#!/bin/bash

RANDOM_PART="$(tr -dc A-Z0-9 </dev/urandom | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

PARTITION="lr7"
QOS="lr_normal"
NUM_CPUS=50
MEMORY_LIMIT="240"
ACCOUNT="pc_beamcore"
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"
EXPECTED_EXECUTION_DURATION="2-00:00:00"

mkdir -p "/global/scratch/users/$USER/pilates_logs"

sbatch --partition="$PARTITION" \
    --exclusive \
    --cpus-per-task="$NUM_CPUS" \
    --mem="${MEMORY_LIMIT}G" \
    --qos="$QOS" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="$JOB_LOG_FILE_PATH" \
    --time="$EXPECTED_EXECUTION_DURATION" \
    job.sh

echo $JOB_LOG_FILE_PATH
