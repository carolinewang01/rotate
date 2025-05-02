#!/bin/bash

# Algorithm to run
algo="open_ended_lagrange"
label="method-explore:scale-all-loss"
partner_pop_size=1
num_seeds=1
log_train_out=false
log_eval_out=false

# DEBUG COMMAND
# CUDA_VISIBLE_DEVICES=1 python open_ended_training/run.py algorithm=open_ended_paired/lbf task=lbf algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=5e4 algorithm.TIMESTEPS_PER_ITER_EGO=5e4 label=d
# ebug logger.mode=offline algorithm.NUM_SEEDS=1

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
        logger.log_train_out="${log_train_out}" \
        logger.log_eval_out="${log_eval_out}" \
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