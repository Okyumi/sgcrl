#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=valid_dcc_ablation
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/valid_dcc_ablation_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/valid_dcc_ablation_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-6%7

# Fourteen configurations, packed two per underutilized L40S:
#   0-7:  Task 5/8, plain DCC vs terminal-success BC, 4M steps
#   8-13: continual Tasks 0-9, plain / one-step Advantage / terminal BC,
#         4M steps per task by default
#
# Submit:
#   sbatch DRAFT_valid_dcc_ablation.sh
# Full 8M-per-task version:
#   CONTINUAL_STEPS_PER_TASK=8000000 sbatch DRAFT_valid_dcc_ablation.sh
#
# The 10-task cells may exceed one 48-hour allocation. Re-submit the same
# command: completed task-boundary checkpoints are detected automatically.
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_valid_dcc_ablation.py"
RUNS_PER_GPU=2
FIRST_CONFIG=$(( ${SLURM_ARRAY_TASK_ID:-0} * RUNS_PER_GPU ))

cd "$REPO_DIR"
export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"
module purge 2>/dev/null || true
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

# Two online-RL processes share the otherwise underused GPU. Bound host-side
# math/thread pools so simulator work does not oversubscribe the 16 CPU cores.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python tests/test_valid_dcc_ablation.py
TOTAL_CONFIGS="$(python "$CONFIG_SCRIPT" --total)"
CELL_LOG_ROOT="${REPO_DIR}/logs/valid_dcc_ablation_v1/cells"
mkdir -p "$CELL_LOG_ROOT"

run_cell() {
  local config_index="$1"
  eval "$(python "$CONFIG_SCRIPT" --setting "$config_index")"

  local action_effect_flag="--noaction_effect_enabled"
  if [ "$ACTION_EFFECT_ENABLED" = "true" ]; then
    action_effect_flag="--action_effect_enabled"
  fi

  local log_root="${REPO_DIR}/logs/valid_dcc_ablation_v1/${SUITE}/runs/${NAME}"
  local checkpoint_root="${REPO_DIR}/logs/valid_dcc_ablation_v1/${SUITE}/checkpoints/${NAME}"
  mkdir -p "$log_root" "$checkpoint_root"

  echo "[$config_index/$TOTAL_CONFIGS] $NAME"
  echo "  W&B: $WANDB_GROUP"
  echo "  budget: $STEPS_PER_TASK steps/task x $NUM_TASKS task(s)"
  echo "  SuccessBC: weight=$SUCCESS_BC_WEIGHT labels=$SUCCESS_BC_LABEL_MODE"

  python -u run_continual_contrastive.py \
    --alg=contrastive_cpc \
    --actor_mode="$ACTOR_MODE" \
    --critic_mode="$CRITIC_MODE" \
    --seed="$SEED" \
    --single_task="$SINGLE_TASK" \
    --num_tasks="$NUM_TASKS" \
    --steps_per_task="$STEPS_PER_TASK" \
    --base_steps="$BASE_STEPS" \
    --start_task=0 \
    --eval_every="$EVAL_EVERY" \
    --eval_episodes="$EVAL_EPISODES" \
    --use_wandb \
    --wandb_project=continual_gcrl_paper \
    --wandb_group="$WANDB_GROUP" \
    --add_uid \
    --log_dir="$log_root" \
    --checkpoint_dir="$checkpoint_root" \
    --nouse_task_id \
    --nointra_eval_previous_tasks \
    --adapt_heads_only \
    --noencoder_from_base \
    --nouse_20_tasks \
    --use_residual \
    --network_width="$NETWORK_WIDTH" \
    --critic_depth="$CRITIC_DEPTH" \
    --actor_depth="$ACTOR_DEPTH" \
    --goal_conditioning_mode="$GOAL_CONDITIONING_MODE" \
    --sawyer_success_mode="$SAWYER_SUCCESS_MODE" \
    --profile_runtime \
    --noactor_auto_reset \
    --dyn_aux_weight="$DYN_AUX_WEIGHT" \
    --phi_task_width="$PHI_TASK_WIDTH" \
    --phi_task_depth="$PHI_TASK_DEPTH" \
    --combine_mode="$COMBINE_MODE" \
    --goal_encoder_mode="$GOAL_ENCODER_MODE" \
    --in_trajectory_negative_repeats="$IN_TRAJECTORY_NEGATIVE_REPEATS" \
    "$action_effect_flag" \
    --action_effect_loss_weight="$ACTION_EFFECT_LOSS_WEIGHT" \
    --action_effect_actor_weight="$ACTION_EFFECT_ACTOR_WEIGHT" \
    --action_effect_actor_mode="$ACTION_EFFECT_ACTOR_MODE" \
    --action_effect_target_mode="$ACTION_EFFECT_TARGET_MODE" \
    --success_bc_weight="$SUCCESS_BC_WEIGHT" \
    --success_bc_label_mode="$SUCCESS_BC_LABEL_MODE" \
    --success_buffer_capacity="$SUCCESS_BUFFER_CAPACITY" \
    --success_bc_batch_size="$SUCCESS_BC_BATCH_SIZE" \
    --counterfactual_rank_interval_steps=0 \
    --counterfactual_oracle_interval_steps=0 \
    --action_landscape_diagnostic_interval_steps=0 \
    --shortcut_diagnostic_interval=0 \
    --nolog_rl_metrics \
    --nolog_pool_cosine \
    --nolog_mixture_norm \
    --nolog_probe_data \
    --post_task_eval_scope=current
}

pids=()
indices=()
for slot in 0 1; do
  config_index=$(( FIRST_CONFIG + slot ))
  if [ "$config_index" -ge "$TOTAL_CONFIGS" ]; then
    continue
  fi
  cell_out="${CELL_LOG_ROOT}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${config_index}.out"
  cell_err="${CELL_LOG_ROOT}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${config_index}.err"
  run_cell "$config_index" >"$cell_out" 2>"$cell_err" &
  pids+=("$!")
  indices+=("$config_index")
  echo "Launched config $config_index; logs: $cell_out and $cell_err"
done

exit_status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Config ${indices[$i]} failed." >&2
    exit_status=1
  else
    echo "Config ${indices[$i]} finished."
  fi
done
exit "$exit_status"
