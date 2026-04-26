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
#SBATCH --array=0-2
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

# ---- shared defaults ------------------------------------------------------
ALG="${ALG:-contrastive_cpc}"
NUM_TASKS="${NUM_TASKS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-8000000}"
BASE_STEPS="${BASE_STEPS:-8000000}"
K_MAX="${K_MAX:-10}"
START_TASK="${START_TASK:-0}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
USE_WANDB="${USE_WANDB:-true}"
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

  echo "$_FLAGS"
}

cd "$REPO_DIR"
TOTAL_CONFIGS=$(python experiment_configs.py --total)

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
echo "JAX mem fraction   : $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "============================================================"

PIDS=()

for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))

  if [ "$CONFIG_IDX" -ge "$TOTAL_CONFIGS" ]; then
    echo "[slot $i] Config index $CONFIG_IDX >= $TOTAL_CONFIGS - skipping."
    continue
  fi

  eval "$(python experiment_configs.py --setting "$CONFIG_IDX")"

  FLAGS=$(build_flags "$ACTOR_MODE" "$CRITIC_MODE" "$SEED")

  EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${CONFIG_IDX}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[slot $i] Config #${CONFIG_IDX}: actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
  echo "[slot $i] Log: ${EXP_LOG_PREFIX}.{out,err}"
  echo "[slot $i] Running: python run_continual_contrastive.py $FLAGS"
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
    echo "Log dir         : $LOG_DIR"
    echo "Checkpoint dir  : $CHECKPOINT_DIR"
    echo "============================================================"
    echo ""
    echo "Running: python run_continual_contrastive.py $FLAGS"
    echo ""

    python run_continual_contrastive.py $FLAGS

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
