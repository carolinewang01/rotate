#!/bin/bash

# Algorithm to run
algo="open_ended_minimax"
label="baselines-v0"

# Create log directory if it doesn't exist
mkdir -p results/logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/logs/${algo}/${label}/experiment_${timestamp}.log"

# Available algorithms (commented out for reference)
# algorithms=(
#     "open_ended_minimax"
#     "open_ended_paired"
#     "paired_ued"
#     "open_ended_fcp"
# )

# Tasks to run
tasks=(
    "overcooked/asymm_advantages"
    # "overcooked/coord_ring"
    # "overcooked/counter_circuit"
    # "overcooked/cramped_room"
    # "overcooked/forced_coord"
)

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] ${message}" | tee -a "${log_file}"
}

# Initialize counters
success_count=0
failure_count=0

# Run experiments for each task
for task in "${tasks[@]}"; do
    log "Starting task: ${algo}/${task}"
    
    if python open_ended_training/run.py algorithm="${algo}/${task}" task="${task}" label="${label}" 2>> "${log_file}"; then
        log "✅ Successfully completed task: ${algo}/${task}"
        ((success_count++))
    else
        log "❌ Failed to complete task: ${algo}/${task}"
        ((failure_count++))
    fi
done

# Print final summary
log "Experiment Summary:"
log "Total tasks attempted: ${#tasks[@]}"
log "Successful tasks: ${success_count}"
log "Failed tasks: ${failure_count}"

if [ ${failure_count} -gt 0 ]; then
    log "Warning: Some tasks failed. Check the log file for details: ${log_file}"
    exit 1
else
    log "All tasks completed successfully!"
    exit 0
fi

