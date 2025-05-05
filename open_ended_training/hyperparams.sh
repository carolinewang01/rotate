#!/bin/bash

# Algorithm to run
algo="open_ended_lagrange"
experiment_name="gae-obj:normfix"
partner_pop_size=1
num_seeds=1
log_train_out=false
log_eval_out=false

# Define hyperparameters for each task
# Format: key=task_name, value="[start1,end1] [start2,end2] ..."
declare -A task_hyperparams
# task_hyperparams["lbf"]="[-0.2,0] [-0.2,-0.1]"
task_hyperparams["overcooked/cramped_room"]="[-250,-150]" #  [-200,-150]
# task_hyperparams["overcooked/counter_circuit"]="[-250,-50] [-150,-60]"
task_hyperparams["overcooked/forced_coord"]="[-25,0]" # [-10,0]


# Create log directory if it doesn't exist
log_dir_base="results/oe_logs/${algo}/${experiment_name}"
mkdir -p "${log_dir_base}"

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir_base}/experiment_${timestamp}.log"

# Tasks to run
tasks=(
    # "overcooked/asymm_advantages"
    # "overcooked/coord_ring"
    "lbf"
    "overcooked/counter_circuit"
    "overcooked/cramped_room"
    "overcooked/forced_coord"
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
    if [[ -v task_hyperparams["$task"] ]]; then
        log "Starting task: ${algo}/${task} with specific hyperparameters"
        hyperparam_pairs=${task_hyperparams["$task"]}
        for pair in $hyperparam_pairs; do
            # Extract start and end values from the pair string like "[-0.2,0]"
            current_lrs=$(echo "$pair" | sed 's/\[//; s/\].*//; s/,.*//')
            current_lre=$(echo "$pair" | sed 's/.*,//; s/\]//')

            label="hyperparam:${experiment_name}:lrs-${current_lrs}:lre-${current_lre}"
            task_log_dir="${log_dir_base}/${label}"
            mkdir -p "${task_log_dir}"
            task_log_file="${task_log_dir}/run_${timestamp}.log"

            log "--- Running with LRS: ${current_lrs}, LRE: ${current_lre}, Label: ${label} ---"

            if python open_ended_training/run.py algorithm="${algo}/${task}" \
                task="${task}" label="${label}" algorithm.NUM_SEEDS="${num_seeds}" \
                algorithm.PARTNER_POP_SIZE="${partner_pop_size}" \
                algorithm.LOWER_REGRET_THRESHOLD_START="${current_lrs}" \
                algorithm.LOWER_REGRET_THRESHOLD_END="${current_lre}" \
                logger.log_train_out="${log_train_out}" \
                logger.log_eval_out="${log_eval_out}" \
                >> "${task_log_file}" 2>&1; then
                log "✅ Successfully completed task: ${algo}/${task} with LRS=${current_lrs}, LRE=${current_lre}"
                ((success_count++))
            else
                log "❌ Failed to complete task: ${algo}/${task} with LRS=${current_lrs}, LRE=${current_lre}. Check log: ${task_log_file}"
                ((failure_count++))
            fi
        done
    else
        log "⚠️ No specific hyperparameters defined for task: ${algo}/${task}. Skipping."
    fi
done

# Print final summary
log "Experiment Summary:"
log "Total hyperparameter combinations attempted: $((success_count + failure_count))"
log "Successful tasks: ${success_count}"
log "Failed tasks: ${failure_count}"

if [ ${failure_count} -gt 0 ]; then
    log "Warning: Some tasks failed. Check the log file for details: ${log_file}"
    exit 1
else
    log "All tasks completed successfully!"
    exit 0
fi

