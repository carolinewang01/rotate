#!/bin/bash

# Algorithm to run
algo="ppo_ego"
config_name="ppo_ego"
label="ego-vs-population"
num_seeds=2

# Declare an associative array for partner paths
declare -A partner_paths
partner_paths=(
    ["overcooked/counter_circuit"]="results/overcooked-v1/counter_circuit/oe_persistent_paired/method-explore:uniform/2025-04-29_23-21-19/saved_train_run"
    ["overcooked/cramped_room"]="results/overcooked-v1/cramped_room/oe_persistent_paired/method-explore:uniform/2025-04-30_00-47-07/saved_train_run"
    ["overcooked/forced_coord"]="results/overcooked-v1/forced_coord/oe_persistent_paired/method-explore:uniform/2025-04-30_01-34-09/saved_train_run"
    ["lbf"]="results/lbf/oe_persistent_paired/method-explore:uniform/2025-04-30_03-14-07/saved_train_run"
)

# Create log directory if it doesn't exist
mkdir -p results/ego_agent_training_logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/ego_agent_training_logs/${algo}/${label}/experiment_${timestamp}.log"

# Tasks to run
tasks=(
    # "overcooked/asymm_advantages"
    # "overcooked/coord_ring"
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
    
    # Get the partner path for the current task
    partner_path=${partner_paths[$task]}

    if [ -z "$partner_path" ]; then
        log "❌ Error: No partner path found for task: ${task}"
        ((failure_count++))
        continue # Skip to the next task
    fi
    
    log "Using partner path: ${partner_path}"

    if python ego_agent_training/run.py -cn "${config_name}" task="${task}" \
        label="${label}" algorithm.NUM_EGO_TRAIN_SEEDS="${num_seeds}" \
        partner_agent.path="${partner_path}" 2>> "${log_file}"; then
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

