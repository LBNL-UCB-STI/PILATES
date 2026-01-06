#!/bin/bash

# Change to repo root directory if we're in the hpc/ subdirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "hpc" ]]; then
    cd "$SCRIPT_DIR/.." || exit 1
fi

RANDOM_PART="$(tr -dc A-Z0-9 </dev/urandom | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

template_file="settings_template.yaml" # Template file with variables
settings_file="settings.yaml" # This file will be generated
stage_file="current_stage_${JOB_NAME}.yaml" # Current stage file

# Default values
PARTITION="lr7"
QOS="lr_normal"
NUM_CPUS=56
MEMORY_LIMIT="500"
BEAM_MEMORY="400g"
# PARTITION="lr8"
# QOS="lr8_normal"
# NUM_CPUS=128
# MEMORY_LIMIT="700"
# BEAM_MEMORY="600g"

while getopts :c:s:p: name
do
    case $name in
    c)    cflag=1
          template_file="$OPTARG";; # Command line argument for settings template file
    s)    sflag=1
          stage_file="$OPTARG";; # Command line argument for stage file
    p)    pflag=1
          partition_arg="$OPTARG";; # Partition argument
    ?)   printf "Usage: %s: [-c settings template file loc] [-s stage file loc] [-p partition] args\n" $0
          exit 2;;
    esac
done

# Set values based on partition argument
if [ "$partition_arg" = "lr8" ]; then
    PARTITION="lr8"
    QOS="lr8_normal"
    NUM_CPUS=128
    MEMORY_LIMIT="700" # In GB
    BEAM_MEMORY="600g"
elif [ "$partition_arg" = "lr7" ]; then
    PARTITION="lr7"
    QOS="lr_normal"
    NUM_CPUS=56
    MEMORY_LIMIT="500" # In GB
    BEAM_MEMORY="400g"
fi

# Export the BEAM_MEMORY environment variable
export BEAM_MEMORY

# Debug: Print current directory and template file path
echo "Current directory: $(pwd)"
echo "Template file: $template_file"
echo "Settings file: $settings_file"

# Check if template file exists
if [ ! -f "$template_file" ]; then
    echo "Error: Template file '$template_file' not found"
    exit 1
fi

# Create the final settings file from the template
envsubst < "$template_file" > "$settings_file"

ACCOUNT="pc_envisionai"
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"
EXPECTED_EXECUTION_DURATION="3-00:00:00"

# Create logs directory if it does not exist
mkdir -p "/global/scratch/users/$USER/pilates_logs"

# Submit the job
sbatch --partition="$PARTITION" \
    --exclusive \
    --cpus-per-task="$NUM_CPUS" \
    --mem="${MEMORY_LIMIT}G" \
    --qos="$QOS" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="$JOB_LOG_FILE_PATH" \
    --time="$EXPECTED_EXECUTION_DURATION" \
    "$SCRIPT_DIR/job.sh" "$settings_file" "$stage_file"

# Print the job log file path
echo $JOB_LOG_FILE_PATH