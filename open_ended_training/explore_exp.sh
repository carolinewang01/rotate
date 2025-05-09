#!/bin/bash

# Algorithm to run
algo="oe_paired_resets" # oe_paired_resets
label="method-explore:bengret-s5-new"
partner_pop_size=1
num_checkpoints=3
num_seeds=1
log_train_out=false
log_eval_out=false
conf_obj_type="sreg-xp_sreg-sp_ret-sxp" # choices: sreg-xp_ret-sp_ret-sxp, sreg-xp_sreg-sp_ret-sxp, sreg-xp_sreg-sp_-ret-xsp_ret-sxp
ego_teammate="all" # choices: [final, all]
# pretrain_ppo=false

# DEBUG COMMAND
# CUDA_VISIBLE_DEVICES=1 python open_ended_training/run.py algorithm=oe_paired_resets/lbf task=lbf algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=5e4 algorithm.TIMESTEPS_PER_ITER_EGO=5e4 label=debug logger.mode=offline algorithm.NUM_SEEDS=1

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
#     "oe_paired_resets"
#     "oe_persistent"
#     "paired_ued"
#     "open_ended_fcp"
# )

# Tasks to run
tasks=(
    # "lbf"
    # "overcooked/cramped_room"
    # "overcooked/counter_circuit"
    "overcooked/forced_coord"
    "overcooked/asymm_advantages"
    # "overcooked/coord_ring"
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
        algorithm.CONF_OBJ_TYPE="${conf_obj_type}" \
        algorithm.NUM_CHECKPOINTS="${num_checkpoints}" \
        algorithm.EGO_TEAMMATE="${ego_teammate}" \
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