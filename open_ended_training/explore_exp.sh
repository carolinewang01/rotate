#!/bin/bash

# Algorithm to run
algo="oe_paired_reset"
label="method-explore:br-ego:popsize-3"
# sampling_strategy="plr"
partner_pop_size=3
num_seeds=1

# DEBUG COMMAND
# python open_ended_training/run.py algorithm=open_ended_lagrange/lbf task=lbf algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=8e4 algorithm.TIMESTEPS_PER_ITER_EGO=8e4 label=debug algorithm.NUM_SEEDS=1

# Create log directory if it doesn't exist
mkdir -p results/oe_logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/oe_logs/${algo}/${label}/experiment_${timestamp}.log"

# Available algorithms (commented out for reference)
# algorithms=(
#     "open_ended_lagrange"
#     "open_ended_minimax"
#     "open_ended_paired"
#     "oe_paired_reset"
#     "oe_persistent_paired"
#     "paired_ued"
#     "open_ended_fcp"
# )

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
    
    if python open_ended_training/run.py algorithm="${algo}/${task}" \
        task="${task}" label="${label}" algorithm.NUM_SEEDS="${num_seeds}" \
        algorithm.PARTNER_POP_SIZE="${partner_pop_size}" \
        # algorithm.SAMPLING_STRATEGY="${sampling_strategy}" \
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

