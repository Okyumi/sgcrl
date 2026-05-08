#!/bin/bash
#SBATCH --job-name=continual_crl
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-22

# ==========================================================================
# Continual Goal-Conditioned Contrastive RL – Batch SLURM Launcher
#
# Runs MULTIPLE experiments per GPU using a job array. Each array task
# launches TASKS_PER_GPU=2 experiments in parallel on the same GPU.
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
#   sbatch draft_4.sh                   # run all 45 configs (23 GPUs)
#   sbatch --array=0-4 draft_4.sh       # run first 10 configs only
#
# JAX Memory:
#   Each process preallocates 45% of GPU memory (total 90%, 10% headroom).
#   This is set via XLA_PYTHON_CLIENT_MEM_FRACTION=0.45.
# ==========================================================================

set -euo pipefail

# ---- number of parallel tasks per GPU ------------------------------------
TASKS_PER_GPU=2

# ---- shared defaults (same as draft_3.sh) ---------------------------------
# These are the defaults for ALL experiments in the batch.  Per-experiment
# overrides (actor_mode, critic_mode, seed) come from experiment_configs.py.
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
NEG_BANK_MODE="${NEG_BANK_MODE:-off}"
NEG_BANK_PER_TASK_CAPACITY="${NEG_BANK_PER_TASK_CAPACITY:-10000}"
NEG_BANK_N_PER_STEP="${NEG_BANK_N_PER_STEP:-256}"
NEG_BANK_CANDIDATE_POOL="${NEG_BANK_CANDIDATE_POOL:-1024}"
NEG_BANK_WEIGHT="${NEG_BANK_WEIGHT:-0.3}"
NEG_BANK_MAX_TASKS="${NEG_BANK_MAX_TASKS:-20}"

# Decomposed-critic + diagnostic flags (defaults preserve prior behaviour).
# Per-cell overrides come from experiment_configs.py via the eval line
# below; cells that don't set these get the dataclass / flag defaults.
DYN_AUX_WEIGHT="${DYN_AUX_WEIGHT:-1.0}"
PHI_TASK_WIDTH="${PHI_TASK_WIDTH:-256}"
PHI_TASK_DEPTH="${PHI_TASK_DEPTH:-2}"
LOG_POOL_COSINE="${LOG_POOL_COSINE:-true}"
LOG_MIXTURE_NORM="${LOG_MIXTURE_NORM:-false}"
LOG_PROBE_DATA="${LOG_PROBE_DATA:-false}"

# Directories (all on scratch to avoid home quota issues)
LOG_DIR="${LOG_DIR:-/scratch/yd2247/sgcrl/logs/continual}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/sgcrl/logs/continual_checkpoints}"
REPO_DIR="/scratch/yd2247/sgcrl"

# ---- environment setup (identical to draft_3.sh — do not modify) ----------
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

# ---- JAX GPU memory allocation for multi-process sharing ------------------
# By default JAX preallocates 75% of GPU memory, which prevents a second
# process from using the same GPU.  With TASKS_PER_GPU=2 we give each
# process 45% (total 90%, leaving 10% headroom for CUDA context etc.).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# ---- create output dirs ---------------------------------------------------
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

  # Scaling architecture flags
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
  _FLAGS="$_FLAGS --neg_bank_mode=$NEG_BANK_MODE"
  _FLAGS="$_FLAGS --neg_bank_per_task_capacity=$NEG_BANK_PER_TASK_CAPACITY"
  _FLAGS="$_FLAGS --neg_bank_n_per_step=$NEG_BANK_N_PER_STEP"
  _FLAGS="$_FLAGS --neg_bank_candidate_pool=$NEG_BANK_CANDIDATE_POOL"
  _FLAGS="$_FLAGS --neg_bank_weight=$NEG_BANK_WEIGHT"
  _FLAGS="$_FLAGS --neg_bank_max_tasks=$NEG_BANK_MAX_TASKS"

  # Decomposed-critic + diagnostic flags. Read directly from the
  # surrounding shell environment so per-cell overrides from
  # experiment_configs.py take effect.
  _FLAGS="$_FLAGS --dyn_aux_weight=$DYN_AUX_WEIGHT"
  _FLAGS="$_FLAGS --phi_task_width=$PHI_TASK_WIDTH"
  _FLAGS="$_FLAGS --phi_task_depth=$PHI_TASK_DEPTH"
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

# ---- get total number of configurations -----------------------------------
cd "$REPO_DIR"
TOTAL_CONFIGS=$(python experiment_configs.py --total)

# ---- launch TASKS_PER_GPU experiments on this GPU -------------------------
echo "============================================================"
echo "Continual Goal-Conditioned Contrastive RL — Batch Launcher"
echo "============================================================"
echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Tasks per GPU      : $TASKS_PER_GPU"
echo "Total configs      : $TOTAL_CONFIGS"
echo "JAX mem fraction   : $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "============================================================"

PIDS=()

for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))

  # Skip if this index exceeds the total number of configurations
  if [ "$CONFIG_IDX" -ge "$TOTAL_CONFIGS" ]; then
    echo "[slot $i] Config index $CONFIG_IDX >= $TOTAL_CONFIGS — skipping."
    continue
  fi

  # Read per-experiment overrides from experiment_configs.py
  eval "$(python experiment_configs.py --setting "$CONFIG_IDX")"
  # This sets ACTOR_MODE, CRITIC_MODE, SEED (and any extras)

  # Build the full flag string
  FLAGS=$(build_flags "$ACTOR_MODE" "$CRITIC_MODE" "$SEED")

  # Per-experiment log files: preserve all .out/.err information
  # Format: {SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}_{CONFIG_IDX}.{out,err}
  EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${CONFIG_IDX}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[slot $i] Config #${CONFIG_IDX}: actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
  echo "[slot $i] Log: ${EXP_LOG_PREFIX}.{out,err}"
  echo "[slot $i] Running: python run_continual_contrastive.py $FLAGS"
  echo "------------------------------------------------------------"

  # Launch in background.
  # Each experiment gets its own .out and .err file with the FULL output
  # (run info, training progress, metrics — identical to draft_3.sh).
  (
    echo "============================================================"
    echo "Continual Goal-Conditioned Contrastive RL"
    echo "============================================================"
    echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
    echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Config Index       : $CONFIG_IDX"
    echo "Node               : $(hostname)"
    echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
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
    echo "Actor mode     : $ACTOR_MODE"
    echo "Use task ID    : $USE_TASK_ID"
    echo "Eval episodes  : $EVAL_EPISODES"
    echo "Intra-eval prev: $INTRA_EVAL_PREVIOUS"
    echo "RL metrics     : $LOG_RL_METRICS"
    echo "K-sample K     : $K_SAMPLE_K"
    echo "Heads only     : $ADAPT_HEADS_ONLY"
    echo "Encoder base   : $ENCODER_FROM_BASE"
    echo "20-task        : $USE_20_TASKS"
    echo "Use residual   : $USE_RESIDUAL"
    echo "Network width  : $NETWORK_WIDTH"
    echo "Critic depth   : $CRITIC_DEPTH"
    echo "Actor depth    : $ACTOR_DEPTH"
    echo "Energy fn      : $ENERGY_FN"
    echo "LSE penalty    : $LOGSUMEXP_PENALTY"
    echo "Single task    : ${SINGLE_TASK:-none}"
    echo "Actor auto-reset: $ACTOR_AUTO_RESET (threshold=$ACTOR_RESET_DORMANT_THRESHOLD, warmup=$ACTOR_RESET_WARMUP, max=$ACTOR_RESET_MAX)"
    echo "Neg bank       : mode=$NEG_BANK_MODE (M=$NEG_BANK_N_PER_STEP, pool=$NEG_BANK_CANDIDATE_POOL, weight=$NEG_BANK_WEIGHT)"
    echo "Decomp critic  : dyn_aux_weight=$DYN_AUX_WEIGHT phi_task=${PHI_TASK_WIDTH}x${PHI_TASK_DEPTH}"
    echo "Diagnostics    : log_pool_cosine=$LOG_POOL_COSINE log_mixture_norm=$LOG_MIXTURE_NORM log_probe_data=$LOG_PROBE_DATA"
    echo "Log dir        : $LOG_DIR"
    echo "Checkpoint dir : $CHECKPOINT_DIR"
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
echo "Launched ${#PIDS[@]} experiment(s). PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

# Wait for all background processes
wait

echo ""
echo "============================================================"
echo "All experiments on this GPU complete."
echo "============================================================"
