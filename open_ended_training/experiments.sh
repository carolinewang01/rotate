#!/bin/bash

# Algorithm to run
algo="rotate"
partner_algo="rotate_without_pop" # choices: rotate_without_pop, rotate_with_mixed_play
conf_obj_type="sreg-xp_ret-sp_ret-sxp" # rotate results in paper use sreg-xp_ret-sp_ret-sxp
label="paper-v0:rotate"
num_seeds=3

# Create log directory if it doesn't exist
mkdir -p results/oe_logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/oe_logs/${algo}/${label}/experiment_${timestamp}.log"

# Available algorithms (commented out for reference)
# algorithms=(
#     "rotate"
#     "open_ended_minimax"
#     "paired"
# )

# Tasks to run
tasks=(
    "overcooked/asymm_advantages"
    "overcooked/coord_ring"
    "overcooked/counter_circuit"
    "overcooked/cramped_room"
    "overcooked/forced_coord"
    "lbf"
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
    
    if python open_ended_training/run.py algorithm="${algo}/${task}" task="${task}" label="${label}" \
        algorithm.NUM_SEEDS="${num_seeds}" \
        2>> "${log_file}"; then
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

