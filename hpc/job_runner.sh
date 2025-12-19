#!/bin/bash

RANDOM_PART="$(tr -dc A-Z0-9 </dev/urandom | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

settings_file="settings.yaml"
stage_file="current_stage_${JOB_NAME}.yaml"

# Default values
PARTITION="lr7"
QOS="lr_normal"
NUM_CPUS=56
MEMORY_LIMIT="500"
# PARTITION="lr8"
# QOS="lr8_normal"
# NUM_CPUS=128
# MEMORY_LIMIT="700"

while getopts :c:s:p: name
do
    case $name in
    c)    cflag=1
          settings_file="$OPTARG";;
    s)    sflag=1
          stage_file="$OPTARG";;
    p)    pflag=1
          partition_arg="$OPTARG";;
    ?)   printf "Usage: %s: [-c settings/config file loc] [-s stage file loc] [-p partition] args\n" $0
          exit 2;;
    esac
done

# Set values based on partition argument
if [ "$partition_arg" = "lr8" ]; then
    PARTITION="lr8"
    QOS="lr8_normal"
    NUM_CPUS=128
    MEMORY_LIMIT="700"
elif [ "$partition_arg" = "lr7" ]; then
    PARTITION="lr7"
    QOS="lr_normal"
    NUM_CPUS=56
    MEMORY_LIMIT="500"
fi

ACCOUNT="pc_envisionai"
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"
EXPECTED_EXECUTION_DURATION="3-00:00:00"

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
    job.sh "$settings_file" "$stage_file"

echo $JOB_LOG_FILE_PATH