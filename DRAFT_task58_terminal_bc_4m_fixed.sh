#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_bc4m_fix
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_terminal_bc_4m_fixed_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_terminal_bc_4m_fixed_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-1%2

# Four corrected-wrapper Terminal-Success-BC cells, packed two per L40S:
#   array 0: Task 5, seeds 5 and 6
#   array 1: Task 8, seeds 5 and 6
# Submit: sbatch DRAFT_task58_terminal_bc_4m_fixed.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_valid_dcc_ablation.py"
GROUP="TASK58-CORRECTED-4M-TERMINAL-SUCCESS-BC-L0.1-FIXED-DIST-V1"
FIRST_CONFIG=$(( 4 + ${SLURM_ARRAY_TASK_ID:-0} * 2 ))

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

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python tests/test_valid_dcc_ablation.py
CELL_LOG_ROOT="${REPO_DIR}/logs/task58_terminal_bc_4m_fixed_v1/cells"
mkdir -p "$CELL_LOG_ROOT"

run_cell() {
  local config_index="$1"
  eval "$(python "$CONFIG_SCRIPT" --setting "$config_index")"
  if [ "$VARIANT" != "terminal_success_bc_l0p1" ]; then
    echo "Refusing to launch non-BC config $config_index: $VARIANT" >&2
    return 2
  fi

  NAME="${NAME}_fixed_dist_v1"
  local log_root="${REPO_DIR}/logs/task58_terminal_bc_4m_fixed_v1/runs/${NAME}"
  local checkpoint_root="${REPO_DIR}/logs/task58_terminal_bc_4m_fixed_v1/checkpoints/${NAME}"
  mkdir -p "$log_root" "$checkpoint_root"

  echo "[$config_index] $NAME"
  echo "  W&B: $GROUP"
  echo "  budget: $STEPS_PER_TASK steps"
  echo "  SuccessBC: weight=$SUCCESS_BC_WEIGHT labels=$SUCCESS_BC_LABEL_MODE"

  python -u run_continual_contrastive.py \
    --alg=contrastive_cpc \
    --actor_mode="$ACTOR_MODE" \
    --critic_mode="$CRITIC_MODE" \
    --seed="$SEED" \
    --single_task="$SINGLE_TASK" \
    --num_tasks=1 \
    --steps_per_task=4000000 \
    --base_steps=4000000 \
    --start_task=0 \
    --eval_every=100000 \
    --eval_episodes=10 \
    --use_wandb \
    --wandb_project=continual_gcrl_paper \
    --wandb_group="$GROUP" \
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
    --noaction_effect_enabled \
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
