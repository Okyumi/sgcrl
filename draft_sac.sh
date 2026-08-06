#!/bin/bash
#SBATCH --job-name=continual_sac
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual_sac/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual_sac/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# ==========================================================================
# Continual Goal-Conditioned SAC + HER - SLURM launcher
#
# SAC counterpart of draft_3.sh / draft_4.sh. Runs TASKS_PER_GPU experiments
# in parallel per array task, reading per-run overrides from
# experiment_configs_sac.py.
#
# Sizing the array:
#   total=$(python experiment_configs_sac.py --total)   # 6 by default
#   max_array_id = ceil(total / TASKS_PER_GPU) - 1      # ceil(6/2)-1 = 2
#
# Usage:
#   sbatch draft_sac.sh                       # the whole grid (6 runs, 3 GPUs)
#   sbatch --array=0-0 draft_sac.sh           # first 2 runs only
#
#   # Single run, no config file: override anything via env vars.
#   ACTOR_MODE=reset CRITIC_MODE=reset SEED=1 SINGLE_RUN=true sbatch draft_sac.sh
#   ACTOR_MODE=persistent CRITIC_MODE=persistent SEED=1 SINGLE_RUN=true \
#       sbatch draft_sac.sh
#
#   # Smoke test (2 tasks x 10k steps, offline W&B), runs in minutes.
#   SMOKE=true SINGLE_RUN=true sbatch draft_sac.sh
#
# W&B: pass the project/entity/group via env vars (WANDB_PROJECT_NAME,
# WANDB_ENTITY_NAME, WANDB_GROUP_NAME). Credentials are NOT stored here --
# run `wandb login` once on the login node, or set WANDB_MODE=offline and
# `wandb sync` the run directories later.
#
# JAX memory: each process preallocates XLA_PYTHON_CLIENT_MEM_FRACTION of the
# GPU, so TASKS_PER_GPU * fraction must stay below 1.
# ==========================================================================

set -euo pipefail

# ---- parallelism ----------------------------------------------------------
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
# When true, ignore experiment_configs_sac.py and run exactly one experiment
# from the env vars below.
SINGLE_RUN="${SINGLE_RUN:-false}"
# When true, shrink the run to a smoke test and force W&B offline.
SMOKE="${SMOKE:-false}"

# ---- shared defaults (override via env vars when submitting) --------------
SEED="${SEED:-42}"
ALG="${ALG:-sac_her}"
NUM_TASKS="${NUM_TASKS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-8000000}"
BASE_STEPS="${BASE_STEPS:-8000000}"
K_MAX="${K_MAX:-10}"
START_TASK="${START_TASK:-0}"
AUTO_RESUME="${AUTO_RESUME:-true}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
INTRA_EVAL_PREVIOUS="${INTRA_EVAL_PREVIOUS:-false}"
POST_TASK_EVAL_SCOPE="${POST_TASK_EVAL_SCOPE:-all_seen}"
K_SAMPLE_K="${K_SAMPLE_K:-0}"
LOG_RL_METRICS="${LOG_RL_METRICS:-true}"
ADD_UID="${ADD_UID:-true}"
ACTOR_MODE="${ACTOR_MODE:-cka}"
CRITIC_MODE="${CRITIC_MODE:-persistent}"
USE_TASK_ID="${USE_TASK_ID:-false}"
ADAPT_HEADS_ONLY="${ADAPT_HEADS_ONLY:-true}"
ENCODER_FROM_BASE="${ENCODER_FROM_BASE:-false}"
USE_20_TASKS="${USE_20_TASKS:-false}"
SINGLE_TASK="${SINGLE_TASK:-}"
TASK_SEQUENCE="${TASK_SEQUENCE:-}"

# SAC-specific: sparse HER reward.
HER_REWARD_THRESHOLD="${HER_REWARD_THRESHOLD:-0.05}"
STEP_PENALTY_REWARD="${STEP_PENALTY_REWARD:-true}"

# Architecture (256 wide matches the collaborator's tuned setting).
USE_RESIDUAL="${USE_RESIDUAL:-true}"
NETWORK_WIDTH="${NETWORK_WIDTH:-256}"
CRITIC_DEPTH="${CRITIC_DEPTH:-4}"
ACTOR_DEPTH="${ACTOR_DEPTH:-4}"

# Actor auto-reset (task 0 only).
ACTOR_AUTO_RESET="${ACTOR_AUTO_RESET:-false}"
ACTOR_RESET_DORMANT_THRESHOLD="${ACTOR_RESET_DORMANT_THRESHOLD:-0.1}"
ACTOR_RESET_WARMUP="${ACTOR_RESET_WARMUP:-200000}"
ACTOR_RESET_MAX="${ACTOR_RESET_MAX:-3}"

# W&B. No API key here: `wandb login` once, or use WANDB_MODE_FLAG=offline.
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-continual_sac}"
WANDB_ENTITY_NAME="${WANDB_ENTITY_NAME:-}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-sac_baseline}"
WANDB_MODE_FLAG="${WANDB_MODE_FLAG:-online}"

# Directories (scratch, to stay off the home quota).
LOG_DIR="${LOG_DIR:-/scratch/yd2247/sgcrl/logs/continual_sac}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/sgcrl/logs/continual_sac_checkpoints}"
REPO_DIR="${REPO_DIR:-/scratch/yd2247/sgcrl}"

# ---- smoke-test overrides -------------------------------------------------
if [ "$SMOKE" = "true" ]; then
  NUM_TASKS=2
  STEPS_PER_TASK=10000
  BASE_STEPS=10000
  K_MAX=2
  EVAL_EVERY=5000
  EVAL_EPISODES=2
  WANDB_MODE_FLAG=offline
  WANDB_GROUP_NAME="${WANDB_GROUP_NAME}_smoke"
  echo "[smoke] 2 tasks x 10k steps, W&B offline."
fi

# ---- environment setup (identical to draft_3.sh) --------------------------
module purge
module load cuda/11.8.0

export MUJOCO_GL=egl                                  # headless rendering
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # protobuf 4.x compat
export PYTHONNOUSERSITE=1                             # ignore ~/.local
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_VLOG_LEVEL=3
export PYTHONUNBUFFERED=1

export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

export MKL_INTERFACE_LAYER=LP64,GNU
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
export PATH="${CONDA_PREFIX}/bin:$PATH"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
[ -d "${MUJOCO_DIR}/bin" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
[ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
for _d in /usr/lib/nvidia /usr/lib64/nvidia; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$_d" && break
done
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null) || true
[ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDNN_LIB}"

# Split the GPU between the co-located processes.
if [ "$SINGLE_RUN" = "true" ]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
cd "$REPO_DIR"

# ---- flag builder --------------------------------------------------------
# Emits the flag string for the *current* values of the variables above, so
# the per-config `eval` below can change ACTOR_MODE / SEED / ... and re-call it.
build_flags() {
  local f="--seed=$SEED"
  f="$f --alg=$ALG"
  f="$f --num_tasks=$NUM_TASKS"
  f="$f --steps_per_task=$STEPS_PER_TASK"
  f="$f --base_steps=$BASE_STEPS"
  f="$f --k_max=$K_MAX"
  f="$f --start_task=$START_TASK"
  f="$f --eval_every=$EVAL_EVERY"
  f="$f --eval_episodes=$EVAL_EPISODES"
  f="$f --post_task_eval_scope=$POST_TASK_EVAL_SCOPE"
  f="$f --k_sample_k=$K_SAMPLE_K"
  f="$f --log_dir=$LOG_DIR"
  f="$f --checkpoint_dir=$CHECKPOINT_DIR"
  f="$f --actor_mode=$ACTOR_MODE"
  f="$f --critic_mode=$CRITIC_MODE"
  f="$f --her_reward_threshold=$HER_REWARD_THRESHOLD"
  f="$f --network_width=$NETWORK_WIDTH"
  f="$f --critic_depth=$CRITIC_DEPTH"
  f="$f --actor_depth=$ACTOR_DEPTH"
  f="$f --actor_reset_dormant_threshold=$ACTOR_RESET_DORMANT_THRESHOLD"
  f="$f --actor_reset_warmup=$ACTOR_RESET_WARMUP"
  f="$f --actor_reset_max=$ACTOR_RESET_MAX"
  f="$f --wandb_project=$WANDB_PROJECT_NAME"
  f="$f --wandb_group=$WANDB_GROUP_NAME"
  f="$f --wandb_mode=$WANDB_MODE_FLAG"
  [ -n "$WANDB_ENTITY_NAME" ] && f="$f --wandb_entity=$WANDB_ENTITY_NAME"
  [ -n "$SINGLE_TASK" ] && f="$f --single_task=$SINGLE_TASK"
  [ -n "$TASK_SEQUENCE" ] && f="$f --task_sequence=$TASK_SEQUENCE"

  # Boolean flags: absl accepts --flag / --noflag.
  local name value
  for pair in \
      "use_wandb:$USE_WANDB" \
      "add_uid:$ADD_UID" \
      "auto_resume:$AUTO_RESUME" \
      "use_task_id:$USE_TASK_ID" \
      "adapt_heads_only:$ADAPT_HEADS_ONLY" \
      "encoder_from_base:$ENCODER_FROM_BASE" \
      "use_20_tasks:$USE_20_TASKS" \
      "intra_eval_previous_tasks:$INTRA_EVAL_PREVIOUS" \
      "log_rl_metrics:$LOG_RL_METRICS" \
      "use_residual:$USE_RESIDUAL" \
      "step_penalty_reward:$STEP_PENALTY_REWARD" \
      "actor_auto_reset:$ACTOR_AUTO_RESET"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [ "$value" = "true" ]; then
      f="$f --$name"
    else
      f="$f --no$name"
    fi
  done

  echo "$f"
}

print_header() {
  echo "============================================================"
  echo "Continual Goal-Conditioned SAC + HER"
  echo "============================================================"
  echo "SLURM job        : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
  echo "Node / GPU       : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
  echo "Seed             : $SEED"
  echo "Actor / Critic   : $ACTOR_MODE / $CRITIC_MODE"
  echo "Tasks            : $NUM_TASKS (base=$BASE_STEPS, per-task=$STEPS_PER_TASK)"
  echo "Reward           : step_penalty=$STEP_PENALTY_REWARD tau=$HER_REWARD_THRESHOLD"
  echo "Network          : residual=$USE_RESIDUAL width=$NETWORK_WIDTH"
  echo "                   critic_depth=$CRITIC_DEPTH actor_depth=$ACTOR_DEPTH"
  echo "Resume           : start_task=$START_TASK auto_resume=$AUTO_RESUME"
  echo "W&B              : $USE_WANDB project=$WANDB_PROJECT_NAME"
  echo "                   group=$WANDB_GROUP_NAME mode=$WANDB_MODE_FLAG"
  echo "Log dir          : $LOG_DIR"
  echo "Checkpoint dir   : $CHECKPOINT_DIR"
  echo "JAX mem fraction : $XLA_PYTHON_CLIENT_MEM_FRACTION"
  echo "============================================================"
}

# ---- single run ----------------------------------------------------------
if [ "$SINGLE_RUN" = "true" ]; then
  print_header
  FLAGS=$(build_flags)
  echo ""
  echo "Running: python run_continual_sac.py $FLAGS"
  echo ""
  python run_continual_sac.py $FLAGS
  echo ""
  echo "Run complete. Checkpoints in: $CHECKPOINT_DIR"
  exit 0
fi

# ---- batch: TASKS_PER_GPU runs per array task ----------------------------
TOTAL_CONFIGS=$(python experiment_configs_sac.py --total)
echo "Total configs: $TOTAL_CONFIGS (TASKS_PER_GPU=$TASKS_PER_GPU)"

PIDS=()
for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))
  if [ "$CONFIG_IDX" -ge "$TOTAL_CONFIGS" ]; then
    echo "Config index $CONFIG_IDX >= $TOTAL_CONFIGS; nothing more to run."
    break
  fi

  (
    # Per-run overrides (ACTOR_MODE / CRITIC_MODE / SEED / ...). Runs in a
    # subshell so one config cannot leak into the next.
    eval "$(python experiment_configs_sac.py --setting "$CONFIG_IDX")"

    EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${CONFIG_IDX}"
    echo "[config $CONFIG_IDX] actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
    print_header > "${EXP_LOG_PREFIX}.out"
    python run_continual_sac.py $(build_flags) \
        >> "${EXP_LOG_PREFIX}.out" 2> "${EXP_LOG_PREFIX}.err"
    echo "[config $CONFIG_IDX] done -> ${EXP_LOG_PREFIX}.out"
  ) &
  PIDS+=($!)
done

wait "${PIDS[@]}"

echo ""
echo "============================================================"
echo "Array task ${SLURM_ARRAY_TASK_ID:-0} complete."
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "============================================================"
