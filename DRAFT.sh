#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=continual_crl
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-3
#
# ==========================================================================
# Continual Goal-Conditioned Contrastive RL – Torch Batch SLURM Launcher
#
# Runs MULTIPLE experiments per GPU using a job array. Each array task
# launches TASKS_PER_GPU experiments in parallel on the same GPU.
#
# Experiment configurations are defined in experiment_configs.py.
# Use "python experiment_configs.py --list" to see all configurations.
# Use "python experiment_configs.py --total" to see the total count.
#
# To calculate the array range:
#   total = $(python experiment_configs.py --total)
#   max_array_id = ceil(total / TASKS_PER_GPU) - 1
#
#   For 45 configs with TASKS_PER_GPU=2: ceil(45/2) - 1 = 22
#   So: --array=0-22
#
# Usage:
#   sbatch draft_4_torch.sh                     # run all configs
#   sbatch --array=0-4 draft_4_torch.sh         # run first chunk only
#
# Torch HPC note:
#   This script uses scratch Miniconda + set_up/torch_hpc_env.sh
#   (no cuda module load).
# ==========================================================================

set -euo pipefail

# ---- number of parallel tasks per GPU ------------------------------------
TASKS_PER_GPU=3

# Optional contiguous config window. Defaults preserve the historical
# behaviour (start at config 0 and run through experiment_configs.py --total).
# Torch-specific wrappers can select a subset without duplicating this file.
CONFIG_INDEX_OFFSET="${CONFIG_INDEX_OFFSET:-0}"
CONFIG_LIMIT="${CONFIG_LIMIT:-0}"

# ---- shared defaults ------------------------------------------------------
ALG="${ALG:-contrastive_cpc}"
NUM_TASKS="${NUM_TASKS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-8000000}"
BASE_STEPS="${BASE_STEPS:-8000000}"
K_MAX="${K_MAX:-10}"
START_TASK="${START_TASK:-0}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
WANDB_GROUP="${WANDB_GROUP:-C2: decomposed single-cell sanity}"
ADD_UID="${ADD_UID:-true}"
USE_TASK_ID="${USE_TASK_ID:-false}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
INTRA_EVAL_PREVIOUS="${INTRA_EVAL_PREVIOUS:-false}"
LOG_RL_METRICS="${LOG_RL_METRICS:-true}"
K_SAMPLE_K="${K_SAMPLE_K:-0}"
ADAPT_HEADS_ONLY="${ADAPT_HEADS_ONLY:-true}"
ENCODER_FROM_BASE="${ENCODER_FROM_BASE:-false}"
USE_20_TASKS="${USE_20_TASKS:-false}"

# Scaling architecture
USE_RESIDUAL="${USE_RESIDUAL:-true}"
NETWORK_WIDTH="${NETWORK_WIDTH:-256}"
CRITIC_DEPTH="${CRITIC_DEPTH:-4}"
ACTOR_DEPTH="${ACTOR_DEPTH:-4}"
ENERGY_FN="${ENERGY_FN:-inner_product}"
LOGSUMEXP_PENALTY="${LOGSUMEXP_PENALTY:-0.01}"
SINGLE_TASK="${SINGLE_TASK:-}"
ACTOR_AUTO_RESET="${ACTOR_AUTO_RESET:-false}"
ACTOR_RESET_DORMANT_THRESHOLD="${ACTOR_RESET_DORMANT_THRESHOLD:-0.1}"
ACTOR_RESET_WARMUP="${ACTOR_RESET_WARMUP:-200000}"
ACTOR_RESET_MAX="${ACTOR_RESET_MAX:-3}"

# Decomposed-critic + diagnostic flags (defaults preserve prior behaviour).
# Per-cell overrides come from experiment_configs.py via the eval line
# below; cells that don't set these get the dataclass / flag defaults.
DYN_AUX_WEIGHT="${DYN_AUX_WEIGHT:-1.0}"
DYN_AUX_AFTER_TASK0="${DYN_AUX_AFTER_TASK0:--1.0}"
PHI_TASK_WIDTH="${PHI_TASK_WIDTH:-256}"
PHI_TASK_DEPTH="${PHI_TASK_DEPTH:-4}"
COMBINE_MODE="${COMBINE_MODE:-add}"
GOAL_ENCODER_MODE="${GOAL_ENCODER_MODE:-shared}"
IN_TRAJECTORY_NEGATIVE_REPEATS="${IN_TRAJECTORY_NEGATIVE_REPEATS:-1}"
BELLMAN_LOSS_WEIGHT="${BELLMAN_LOSS_WEIGHT:-1.0}"
BELLMAN_RESIDUAL_L2_WEIGHT="${BELLMAN_RESIDUAL_L2_WEIGHT:-0.0001}"
BELLMAN_DISCOUNT="${BELLMAN_DISCOUNT:-0.99}"
BELLMAN_TAU="${BELLMAN_TAU:-0.005}"
BELLMAN_HIDDEN_DIM="${BELLMAN_HIDDEN_DIM:-256}"
HER_REWARD_THRESHOLD="${HER_REWARD_THRESHOLD:-0.05}"
STEP_PENALTY_REWARD="${STEP_PENALTY_REWARD:-true}"
DCC_SAC_Q_LOSS_WEIGHT="${DCC_SAC_Q_LOSS_WEIGHT:-1.0}"
DCC_SAC_Q_LEARNING_RATE="${DCC_SAC_Q_LEARNING_RATE:-0.0003}"
DCC_SAC_DISCOUNT="${DCC_SAC_DISCOUNT:-0.99}"
DCC_SAC_TAU="${DCC_SAC_TAU:-0.005}"
DCC_SAC_Q_HIDDEN_DIM="${DCC_SAC_Q_HIDDEN_DIM:-1024}"
DCC_SAC_BETA_MAX="${DCC_SAC_BETA_MAX:-0.1}"
DCC_SAC_Q_WARMUP_UPDATES="${DCC_SAC_Q_WARMUP_UPDATES:-10000}"
DCC_SAC_Q_RAMP_UPDATES="${DCC_SAC_Q_RAMP_UPDATES:-25000}"
DCC_SAC_TD_ERROR_THRESHOLD="${DCC_SAC_TD_ERROR_THRESHOLD:-0.5}"
DCC_SAC_TWIN_DISAGREEMENT_THRESHOLD="${DCC_SAC_TWIN_DISAGREEMENT_THRESHOLD:-0.1}"
DCC_SAC_EMA_DECAY="${DCC_SAC_EMA_DECAY:-0.99}"
DCC_SAC_CANDIDATE_ACTIONS="${DCC_SAC_CANDIDATE_ACTIONS:-8}"
DCC_SAC_NORMALIZATION_EPS="${DCC_SAC_NORMALIZATION_EPS:-0.001}"
DCC_SAC_CORRECTION_CLIP="${DCC_SAC_CORRECTION_CLIP:-5.0}"
ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-1.0}"
ACTION_CONTRAST_TEMPERATURE="${ACTION_CONTRAST_TEMPERATURE:-1.0}"
ACTION_CONTRAST_BATCH_SIZE="${ACTION_CONTRAST_BATCH_SIZE:-32}"
SHORTCUT_DIAGNOSTIC_INTERVAL="${SHORTCUT_DIAGNOSTIC_INTERVAL:-0}"
SHORTCUT_DIAGNOSTIC_BATCH_SIZE="${SHORTCUT_DIAGNOSTIC_BATCH_SIZE:-32}"
SHORTCUT_CANDIDATE_ACTIONS="${SHORTCUT_CANDIDATE_ACTIONS:-16}"
POST_TASK_EVAL_SCOPE="${POST_TASK_EVAL_SCOPE:-all_seen}"
LOG_POOL_COSINE="${LOG_POOL_COSINE:-true}"
LOG_MIXTURE_NORM="${LOG_MIXTURE_NORM:-false}"
LOG_PROBE_DATA="${LOG_PROBE_DATA:-false}"

# Directories
LOG_DIR="${LOG_DIR:-/scratch/yd2247/sgcrl/logs/continual}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/sgcrl/logs/continual_checkpoints}"
REPO_DIR="/scratch/yd2247/sgcrl"

# ---- torch HPC environment setup (match submit_continual_torch.sh) -------
export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"

module purge 2>/dev/null || true

# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl

# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

# Keep the same per-process memory control used by draft_4 parallel runs.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ---- helper: build flags for a single experiment --------------------------
build_flags() {
  # Arguments: ACTOR_MODE CRITIC_MODE SEED
  local _ACTOR_MODE="$1"
  local _CRITIC_MODE="$2"
  local _SEED="$3"

  local _FLAGS="--seed=$_SEED"
  _FLAGS="$_FLAGS --alg=$ALG"
  _FLAGS="$_FLAGS --num_tasks=$NUM_TASKS"
  _FLAGS="$_FLAGS --steps_per_task=$STEPS_PER_TASK"
  _FLAGS="$_FLAGS --base_steps=$BASE_STEPS"
  _FLAGS="$_FLAGS --k_max=$K_MAX"
  _FLAGS="$_FLAGS --start_task=$START_TASK"
  _FLAGS="$_FLAGS --eval_every=$EVAL_EVERY"
  _FLAGS="$_FLAGS --log_dir=$LOG_DIR"
  _FLAGS="$_FLAGS --checkpoint_dir=$CHECKPOINT_DIR"

  if [ "$USE_WANDB" = "true" ]; then
    _FLAGS="$_FLAGS --use_wandb"
  fi
  _FLAGS="$_FLAGS --wandb_project=$WANDB_PROJECT"
  _FLAGS="$_FLAGS --wandb_group=$WANDB_GROUP"
  if [ "$ADD_UID" = "true" ]; then
    _FLAGS="$_FLAGS --add_uid"
  fi

  _FLAGS="$_FLAGS --critic_mode=$_CRITIC_MODE"
  if [ "$USE_TASK_ID" = "true" ]; then
    _FLAGS="$_FLAGS --use_task_id"
  else
    _FLAGS="$_FLAGS --nouse_task_id"
  fi

  _FLAGS="$_FLAGS --eval_episodes=$EVAL_EPISODES"
  if [ "$INTRA_EVAL_PREVIOUS" = "true" ]; then
    _FLAGS="$_FLAGS --intra_eval_previous_tasks"
  else
    _FLAGS="$_FLAGS --nointra_eval_previous_tasks"
  fi
  if [ "$LOG_RL_METRICS" = "true" ]; then
    _FLAGS="$_FLAGS --log_rl_metrics"
  else
    _FLAGS="$_FLAGS --nolog_rl_metrics"
  fi
  _FLAGS="$_FLAGS --k_sample_k=$K_SAMPLE_K"

  if [ "$ADAPT_HEADS_ONLY" = "true" ]; then
    _FLAGS="$_FLAGS --adapt_heads_only"
  else
    _FLAGS="$_FLAGS --noadapt_heads_only"
  fi
  if [ "$ENCODER_FROM_BASE" = "true" ]; then
    _FLAGS="$_FLAGS --encoder_from_base"
  else
    _FLAGS="$_FLAGS --noencoder_from_base"
  fi
  if [ "$USE_20_TASKS" = "true" ]; then
    _FLAGS="$_FLAGS --use_20_tasks"
  else
    _FLAGS="$_FLAGS --nouse_20_tasks"
  fi
  _FLAGS="$_FLAGS --actor_mode=$_ACTOR_MODE"

  if [ "$USE_RESIDUAL" = "true" ]; then
    _FLAGS="$_FLAGS --use_residual"
  else
    _FLAGS="$_FLAGS --nouse_residual"
  fi
  _FLAGS="$_FLAGS --network_width=$NETWORK_WIDTH"
  _FLAGS="$_FLAGS --critic_depth=$CRITIC_DEPTH"
  _FLAGS="$_FLAGS --actor_depth=$ACTOR_DEPTH"
  _FLAGS="$_FLAGS --energy_fn=$ENERGY_FN"
  _FLAGS="$_FLAGS --logsumexp_penalty=$LOGSUMEXP_PENALTY"
  if [ -n "$SINGLE_TASK" ]; then
    _FLAGS="$_FLAGS --single_task=$SINGLE_TASK"
  fi
  if [ "$ACTOR_AUTO_RESET" = "true" ]; then
    _FLAGS="$_FLAGS --actor_auto_reset"
  else
    _FLAGS="$_FLAGS --noactor_auto_reset"
  fi
  _FLAGS="$_FLAGS --actor_reset_dormant_threshold=$ACTOR_RESET_DORMANT_THRESHOLD"
  _FLAGS="$_FLAGS --actor_reset_warmup=$ACTOR_RESET_WARMUP"
  _FLAGS="$_FLAGS --actor_reset_max=$ACTOR_RESET_MAX"

  # Decomposed-critic + diagnostic flags. Read directly from the
  # surrounding shell environment so per-cell overrides from
  # experiment_configs.py take effect.
  _FLAGS="$_FLAGS --dyn_aux_weight=$DYN_AUX_WEIGHT"
  _FLAGS="$_FLAGS --dyn_aux_after_task0=$DYN_AUX_AFTER_TASK0"
  _FLAGS="$_FLAGS --phi_task_width=$PHI_TASK_WIDTH"
  _FLAGS="$_FLAGS --phi_task_depth=$PHI_TASK_DEPTH"
  _FLAGS="$_FLAGS --combine_mode=$COMBINE_MODE"
  _FLAGS="$_FLAGS --goal_encoder_mode=$GOAL_ENCODER_MODE"
  _FLAGS="$_FLAGS --in_trajectory_negative_repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS"
  _FLAGS="$_FLAGS --bellman_loss_weight=$BELLMAN_LOSS_WEIGHT"
  _FLAGS="$_FLAGS --bellman_residual_l2_weight=$BELLMAN_RESIDUAL_L2_WEIGHT"
  _FLAGS="$_FLAGS --bellman_discount=$BELLMAN_DISCOUNT"
  _FLAGS="$_FLAGS --bellman_tau=$BELLMAN_TAU"
  _FLAGS="$_FLAGS --bellman_hidden_dim=$BELLMAN_HIDDEN_DIM"
  _FLAGS="$_FLAGS --her_reward_threshold=$HER_REWARD_THRESHOLD"
_FLAGS="$_FLAGS --dcc_sac_q_loss_weight=$DCC_SAC_Q_LOSS_WEIGHT"
_FLAGS="$_FLAGS --dcc_sac_q_learning_rate=$DCC_SAC_Q_LEARNING_RATE"
_FLAGS="$_FLAGS --dcc_sac_discount=$DCC_SAC_DISCOUNT"
_FLAGS="$_FLAGS --dcc_sac_tau=$DCC_SAC_TAU"
_FLAGS="$_FLAGS --dcc_sac_q_hidden_dim=$DCC_SAC_Q_HIDDEN_DIM"
_FLAGS="$_FLAGS --dcc_sac_beta_max=$DCC_SAC_BETA_MAX"
_FLAGS="$_FLAGS --dcc_sac_q_warmup_updates=$DCC_SAC_Q_WARMUP_UPDATES"
_FLAGS="$_FLAGS --dcc_sac_q_ramp_updates=$DCC_SAC_Q_RAMP_UPDATES"
_FLAGS="$_FLAGS --dcc_sac_td_error_threshold=$DCC_SAC_TD_ERROR_THRESHOLD"
_FLAGS="$_FLAGS --dcc_sac_twin_disagreement_threshold=$DCC_SAC_TWIN_DISAGREEMENT_THRESHOLD"
_FLAGS="$_FLAGS --dcc_sac_ema_decay=$DCC_SAC_EMA_DECAY"
_FLAGS="$_FLAGS --dcc_sac_candidate_actions=$DCC_SAC_CANDIDATE_ACTIONS"
_FLAGS="$_FLAGS --dcc_sac_normalization_eps=$DCC_SAC_NORMALIZATION_EPS"
_FLAGS="$_FLAGS --dcc_sac_correction_clip=$DCC_SAC_CORRECTION_CLIP"
_FLAGS="$_FLAGS --action_contrast_weight=$ACTION_CONTRAST_WEIGHT"
_FLAGS="$_FLAGS --action_contrast_temperature=$ACTION_CONTRAST_TEMPERATURE"
_FLAGS="$_FLAGS --action_contrast_batch_size=$ACTION_CONTRAST_BATCH_SIZE"
_FLAGS="$_FLAGS --shortcut_diagnostic_interval=$SHORTCUT_DIAGNOSTIC_INTERVAL"
_FLAGS="$_FLAGS --shortcut_diagnostic_batch_size=$SHORTCUT_DIAGNOSTIC_BATCH_SIZE"
_FLAGS="$_FLAGS --shortcut_candidate_actions=$SHORTCUT_CANDIDATE_ACTIONS"
_FLAGS="$_FLAGS --post_task_eval_scope=$POST_TASK_EVAL_SCOPE"
  if [ "$STEP_PENALTY_REWARD" = "true" ]; then
    _FLAGS="$_FLAGS --step_penalty_reward"
  else
    _FLAGS="$_FLAGS --nostep_penalty_reward"
  fi
  if [ "$LOG_POOL_COSINE" = "true" ]; then
    _FLAGS="$_FLAGS --log_pool_cosine"
  else
    _FLAGS="$_FLAGS --nolog_pool_cosine"
  fi
  if [ "$LOG_MIXTURE_NORM" = "true" ]; then
    _FLAGS="$_FLAGS --log_mixture_norm"
  else
    _FLAGS="$_FLAGS --nolog_mixture_norm"
  fi
  if [ "$LOG_PROBE_DATA" = "true" ]; then
    _FLAGS="$_FLAGS --log_probe_data"
  else
    _FLAGS="$_FLAGS --nolog_probe_data"
  fi

  echo "$_FLAGS"
}

cd "$REPO_DIR"
TOTAL_CONFIGS=$(python experiment_configs.py --total)
CONFIG_END="$TOTAL_CONFIGS"
if [ "$CONFIG_LIMIT" -gt 0 ]; then
  CONFIG_END=$(( CONFIG_INDEX_OFFSET + CONFIG_LIMIT ))
  if [ "$CONFIG_END" -gt "$TOTAL_CONFIGS" ]; then
    CONFIG_END="$TOTAL_CONFIGS"
  fi
fi

echo "============================================================"
echo "Continual Goal-Conditioned Contrastive RL — Torch Batch"
echo "============================================================"
echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda prefix       : ${CONDA_PREFIX:-N/A}"
echo "Tasks per GPU      : $TASKS_PER_GPU"
echo "Total configs      : $TOTAL_CONFIGS"
echo "Selected configs   : [$CONFIG_INDEX_OFFSET, $CONFIG_END)"
echo "JAX mem fraction   : $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "============================================================"

PIDS=()

for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( CONFIG_INDEX_OFFSET
      + TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))

  if [ "$CONFIG_IDX" -ge "$CONFIG_END" ]; then
    echo "[slot $i] Config index $CONFIG_IDX >= selected end $CONFIG_END - skipping."
    continue
  fi

  eval "$(python experiment_configs.py --setting "$CONFIG_IDX")"

  FLAGS=$(build_flags "$ACTOR_MODE" "$CRITIC_MODE" "$SEED")

  EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${CONFIG_IDX}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[slot $i] Config #${CONFIG_IDX}: actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
  echo "[slot $i] Log: ${EXP_LOG_PREFIX}.{out,err}"
  echo "[slot $i] Running: python -u run_continual_contrastive.py $FLAGS"
  echo "------------------------------------------------------------"

  (
    echo "============================================================"
    echo "Continual Goal-Conditioned Contrastive RL (Torch HPC)"
    echo "============================================================"
    echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
    echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Config Index       : $CONFIG_IDX"
    echo "Node               : $(hostname)"
    echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Conda prefix       : ${CONDA_PREFIX:-N/A}"
    echo "------------------------------------------------------------"
    echo "Seed            : $SEED"
    echo "Algorithm       : $ALG"
    echo "Num tasks       : $NUM_TASKS"
    echo "Steps per task  : $STEPS_PER_TASK"
    echo "Base steps      : $BASE_STEPS"
    echo "K_max           : $K_MAX"
    echo "Start task      : $START_TASK"
    echo "Eval every      : $EVAL_EVERY"
    echo "W&B             : $USE_WANDB"
    echo "W&B project     : $WANDB_PROJECT"
    echo "W&B group       : $WANDB_GROUP"
    echo "Critic mode     : $CRITIC_MODE"
    echo "Actor mode      : $ACTOR_MODE"
    echo "Use task ID     : $USE_TASK_ID"
    echo "Eval episodes   : $EVAL_EPISODES"
    echo "Intra-eval prev : $INTRA_EVAL_PREVIOUS"
    echo "RL metrics      : $LOG_RL_METRICS"
    echo "K-sample K      : $K_SAMPLE_K"
    echo "Heads only      : $ADAPT_HEADS_ONLY"
    echo "Encoder base    : $ENCODER_FROM_BASE"
    echo "20-task         : $USE_20_TASKS"
    echo "Use residual    : $USE_RESIDUAL"
    echo "Network width   : $NETWORK_WIDTH"
    echo "Critic depth    : $CRITIC_DEPTH"
    echo "Actor depth     : $ACTOR_DEPTH"
    echo "Energy fn       : $ENERGY_FN"
    echo "LSE penalty     : $LOGSUMEXP_PENALTY"
    echo "Single task     : ${SINGLE_TASK:-none}"
    echo "Actor auto-reset: $ACTOR_AUTO_RESET (threshold=$ACTOR_RESET_DORMANT_THRESHOLD, warmup=$ACTOR_RESET_WARMUP, max=$ACTOR_RESET_MAX)"
    echo "Decomp critic   : dyn_aux_weight=$DYN_AUX_WEIGHT (after_task0=$DYN_AUX_AFTER_TASK0) phi_task=${PHI_TASK_WIDTH}x${PHI_TASK_DEPTH}"
    echo "In-traj negs    : repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS"
    echo "Diagnostics     : log_pool_cosine=$LOG_POOL_COSINE log_mixture_norm=$LOG_MIXTURE_NORM log_probe_data=$LOG_PROBE_DATA"
    echo "Log dir         : $LOG_DIR"
    echo "Checkpoint dir  : $CHECKPOINT_DIR"
    echo "============================================================"
    echo ""
    echo "Running: python -u run_continual_contrastive.py $FLAGS"
    echo ""

    python -u run_continual_contrastive.py $FLAGS

    echo ""
    echo "============================================================"
    echo "Run complete. Checkpoints saved to: $CHECKPOINT_DIR"
    echo "============================================================"
  ) > "${EXP_LOG_PREFIX}.out" 2> "${EXP_LOG_PREFIX}.err" &

  PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} experiment(s). PIDs: ${PIDS[*]:-none}"
echo "Waiting for all to finish..."

wait

echo ""
echo "============================================================"
echo "All experiments on this GPU complete."
echo "============================================================"
