#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_alg_ablation
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_algorithm_ablation_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_algorithm_ablation_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-11%6

# Corrected-wrapper pilot, two seeds per task:
#   0-3:  original one-step Advantage-DCC
#   4-7:  raw H=25 Advantage-DCC without BC
#   8-11: plain DCC + terminal-success BC, without an Advantage head
#
# Submit with:
#   sbatch DRAFT_task58_algorithm_ablation.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_task58_algorithm_ablation.py"
CONFIG_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LOG_DIR="${REPO_DIR}/logs/task58_algorithm_ablation_v1/runs"
CHECKPOINT_ROOT="${REPO_DIR}/logs/task58_algorithm_ablation_v1/checkpoints"

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
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

python tests/test_task58_algorithm_ablation.py
eval "$(python "$CONFIG_SCRIPT" --setting "$CONFIG_INDEX")"

if [ "$ACTION_EFFECT_ENABLED" = "true" ]; then
  ACTION_EFFECT_FLAG="--action_effect_enabled"
else
  ACTION_EFFECT_FLAG="--noaction_effect_enabled"
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_ROOT/$NAME"

echo "Config       : $CONFIG_INDEX / $(python "$CONFIG_SCRIPT" --total)"
echo "Variant      : $NAME"
echo "Environment  : $SINGLE_TASK"
echo "Seed         : $SEED"
echo "W&B group    : $WANDB_GROUP"
echo "BC labels    : $SUCCESS_BC_LABEL_MODE"

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
  --log_dir="$LOG_DIR" \
  --checkpoint_dir="$CHECKPOINT_ROOT/$NAME" \
  --nouse_task_id \
  --nointra_eval_previous_tasks \
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
  --in_trajectory_negative_repeats="$IN_TRAJECTORY_NEGATIVE_REPEATS" \
  "$ACTION_EFFECT_FLAG" \
  --action_effect_loss_weight="$ACTION_EFFECT_LOSS_WEIGHT" \
  --action_effect_actor_weight="$ACTION_EFFECT_ACTOR_WEIGHT" \
  --action_effect_actor_mode="$ACTION_EFFECT_ACTOR_MODE" \
  --action_effect_target_mode="$ACTION_EFFECT_TARGET_MODE" \
  --outcome_horizon="$OUTCOME_HORIZON" \
  --outcome_success_threshold="$OUTCOME_SUCCESS_THRESHOLD" \
  --outcome_progress_loss_weight="$OUTCOME_PROGRESS_LOSS_WEIGHT" \
  --outcome_success_loss_weight="$OUTCOME_SUCCESS_LOSS_WEIGHT" \
  --outcome_success_actor_weight="$OUTCOME_SUCCESS_ACTOR_WEIGHT" \
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
