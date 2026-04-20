#!/bin/bash
#SBATCH --job-name=continual_crl
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%j.err
#SBATCH --mail-user=yd2247@nyu.edu

# ==========================================================================
# Continual Goal-Conditioned Contrastive RL – SLURM launcher
#
# Usage (submit the full 10-task run):
#   sbatch draft_3.sh
#
# Usage (override any flag via environment variables):
#   SEED=0 NUM_TASKS=10 sbatch draft_3.sh
#   START_TASK=5 sbatch draft_3.sh          # resume from task 5
#   NUM_TASKS=2 STEPS=10000 sbatch draft_3.sh  # quick debug run
#
# All tuneable parameters are in the "EXPERIMENT PARAMETERS" block below.
# ==========================================================================

set -euo pipefail

# ---- tuneable parameters (override via env vars when submitting) ----------
SEED="${SEED:-42}"
ALG="${ALG:-contrastive_cpc}"
NUM_TASKS="${NUM_TASKS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-8000000}"
BASE_STEPS="${BASE_STEPS:-8000000}"
K_MAX="${K_MAX:-10}"
START_TASK="${START_TASK:-0}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
USE_WANDB="${USE_WANDB:-true}"
ADD_UID="${ADD_UID:-true}"
CRITIC_MODE="${CRITIC_MODE:-persistent}"
USE_TASK_ID="${USE_TASK_ID:-false}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
INTRA_EVAL_PREVIOUS="${INTRA_EVAL_PREVIOUS:-false}"
LOG_RL_METRICS="${LOG_RL_METRICS:-true}"
K_SAMPLE_K="${K_SAMPLE_K:-0}"
ADAPT_HEADS_ONLY="${ADAPT_HEADS_ONLY:-true}"
ENCODER_FROM_BASE="${ENCODER_FROM_BASE:-false}"
USE_20_TASKS="${USE_20_TASKS:-false}"
ACTOR_MODE="${ACTOR_MODE:-cka}"

# Scaling architecture
USE_RESIDUAL="${USE_RESIDUAL:-true}"
NETWORK_WIDTH="${NETWORK_WIDTH:-256}"
CRITIC_DEPTH="${CRITIC_DEPTH:-4}"
ACTOR_DEPTH="${ACTOR_DEPTH:-4}"
ENERGY_FN="${ENERGY_FN:-inner_product}"
LOGSUMEXP_PENALTY="${LOGSUMEXP_PENALTY:-0.01}"
SINGLE_TASK="${SINGLE_TASK:-}"
ACTOR_RESET_INTERVAL="${ACTOR_RESET_INTERVAL:-0}"

# Directories (all on scratch to avoid home quota issues)
LOG_DIR="${LOG_DIR:-/scratch/yd2247/sgcrl/logs/continual}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/sgcrl/logs/continual_checkpoints}"
REPO_DIR="/scratch/yd2247/sgcrl"

# ---- environment setup (same as draft.sh) ---------------------------------
module purge
module load cuda/11.8.0

# Headless MuJoCo rendering
export MUJOCO_GL=egl

# Protobuf 4.x compat
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Ignore ~/.local packages (avoid protobuf / TF conflicts)
export PYTHONNOUSERSITE=1

# Reduce log noise
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_VLOG_LEVEL=3

# Force unbuffered Python output so logs appear immediately in SLURM .out
export PYTHONUNBUFFERED=1

# Scratch-based caches
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Conda
export MKL_INTERFACE_LAYER=LP64,GNU
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl

export PATH="${CONDA_PREFIX}/bin:$PATH"

# Library paths
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
[ -d "${MUJOCO_DIR}/bin" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
[ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
for _d in /usr/lib/nvidia /usr/lib64/nvidia; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$_d" && break
done
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null) || true
[ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDNN_LIB}"

# ---- create output dirs ---------------------------------------------------
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ---- print run info -------------------------------------------------------
echo "============================================================"
echo "Continual Goal-Conditioned Contrastive RL"
echo "============================================================"
echo "SLURM Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node           : $(hostname)"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "------------------------------------------------------------"
echo "Seed           : $SEED"
echo "Algorithm      : $ALG"
echo "Num tasks      : $NUM_TASKS"
echo "Steps per task : $STEPS_PER_TASK"
echo "Base steps     : $BASE_STEPS"
echo "K_max          : $K_MAX"
echo "Start task     : $START_TASK"
echo "Eval every     : $EVAL_EVERY"
echo "W&B            : $USE_WANDB"
echo "Critic mode    : $CRITIC_MODE"
echo "Use task ID    : $USE_TASK_ID"
echo "Eval episodes  : $EVAL_EPISODES"
echo "Intra-eval prev: $INTRA_EVAL_PREVIOUS"
echo "RL metrics     : $LOG_RL_METRICS"
echo "K-sample K     : $K_SAMPLE_K"
echo "Heads only     : $ADAPT_HEADS_ONLY"
echo "Encoder base   : $ENCODER_FROM_BASE"
echo "20-task        : $USE_20_TASKS"
echo "Actor mode     : $ACTOR_MODE"
echo "Use residual   : $USE_RESIDUAL"
echo "Network width  : $NETWORK_WIDTH"
echo "Critic depth   : $CRITIC_DEPTH"
echo "Actor depth    : $ACTOR_DEPTH"
echo "Energy fn      : $ENERGY_FN"
echo "LSE penalty    : $LOGSUMEXP_PENALTY"
echo "Single task    : ${SINGLE_TASK:-none}"
echo "Actor reset    : $ACTOR_RESET_INTERVAL"
echo "Log dir        : $LOG_DIR"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "============================================================"

# ---- build flag string ----------------------------------------------------
FLAGS="--seed=$SEED"
FLAGS="$FLAGS --alg=$ALG"
FLAGS="$FLAGS --num_tasks=$NUM_TASKS"
FLAGS="$FLAGS --steps_per_task=$STEPS_PER_TASK"
FLAGS="$FLAGS --base_steps=$BASE_STEPS"
FLAGS="$FLAGS --k_max=$K_MAX"
FLAGS="$FLAGS --start_task=$START_TASK"
FLAGS="$FLAGS --eval_every=$EVAL_EVERY"
FLAGS="$FLAGS --log_dir=$LOG_DIR"
FLAGS="$FLAGS --checkpoint_dir=$CHECKPOINT_DIR"

if [ "$USE_WANDB" = "true" ]; then
  FLAGS="$FLAGS --use_wandb"
fi
if [ "$ADD_UID" = "true" ]; then
  FLAGS="$FLAGS --add_uid"
fi

FLAGS="$FLAGS --critic_mode=$CRITIC_MODE"
if [ "$USE_TASK_ID" = "true" ]; then
  FLAGS="$FLAGS --use_task_id"
else
  FLAGS="$FLAGS --nouse_task_id"
fi

FLAGS="$FLAGS --eval_episodes=$EVAL_EPISODES"
if [ "$INTRA_EVAL_PREVIOUS" = "true" ]; then
  FLAGS="$FLAGS --intra_eval_previous_tasks"
else
  FLAGS="$FLAGS --nointra_eval_previous_tasks"
fi
if [ "$LOG_RL_METRICS" = "true" ]; then
  FLAGS="$FLAGS --log_rl_metrics"
else
  FLAGS="$FLAGS --nolog_rl_metrics"
fi
FLAGS="$FLAGS --k_sample_k=$K_SAMPLE_K"

if [ "$ADAPT_HEADS_ONLY" = "true" ]; then
  FLAGS="$FLAGS --adapt_heads_only"
else
  FLAGS="$FLAGS --noadapt_heads_only"
fi
if [ "$ENCODER_FROM_BASE" = "true" ]; then
  FLAGS="$FLAGS --encoder_from_base"
else
  FLAGS="$FLAGS --noencoder_from_base"
fi
if [ "$USE_20_TASKS" = "true" ]; then
  FLAGS="$FLAGS --use_20_tasks"
else
  FLAGS="$FLAGS --nouse_20_tasks"
fi
FLAGS="$FLAGS --actor_mode=$ACTOR_MODE"

# Scaling architecture flags
if [ "$USE_RESIDUAL" = "true" ]; then
  FLAGS="$FLAGS --use_residual"
else
  FLAGS="$FLAGS --nouse_residual"
fi
FLAGS="$FLAGS --network_width=$NETWORK_WIDTH"
FLAGS="$FLAGS --critic_depth=$CRITIC_DEPTH"
FLAGS="$FLAGS --actor_depth=$ACTOR_DEPTH"
FLAGS="$FLAGS --energy_fn=$ENERGY_FN"
FLAGS="$FLAGS --logsumexp_penalty=$LOGSUMEXP_PENALTY"
if [ -n "$SINGLE_TASK" ]; then
  FLAGS="$FLAGS --single_task=$SINGLE_TASK"
fi
FLAGS="$FLAGS --actor_reset_interval=$ACTOR_RESET_INTERVAL"

# ---- run -------------------------------------------------------------------
cd "$REPO_DIR"

echo ""
echo "Running: python run_continual_contrastive.py $FLAGS"
echo ""

python run_continual_contrastive.py $FLAGS

echo ""
echo "============================================================"
echo "Run complete. Checkpoints saved to: $CHECKPOINT_DIR"
echo "============================================================"
