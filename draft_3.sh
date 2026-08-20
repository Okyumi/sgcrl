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
WANDB_GROUP="${WANDB_GROUP:-}"
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

# Decomposed-critic + diagnostic flags. All default to the dataclass
# defaults in contrastive/continual_config.py + the absl flag defaults
# in run_continual_contrastive.py, so omitting these from the env vars
# preserves bit-identical behaviour vs prior runs.
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
# Previous-replay negative bank (off | vanilla | hard_weighted)
NEG_BANK_MODE="${NEG_BANK_MODE:-off}"
NEG_BANK_PER_TASK_CAPACITY="${NEG_BANK_PER_TASK_CAPACITY:-10000}"
NEG_BANK_N_PER_STEP="${NEG_BANK_N_PER_STEP:-256}"
NEG_BANK_CANDIDATE_POOL="${NEG_BANK_CANDIDATE_POOL:-1024}"
NEG_BANK_WEIGHT="${NEG_BANK_WEIGHT:-0.3}"
NEG_BANK_MAX_TASKS="${NEG_BANK_MAX_TASKS:-20}"

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
echo "In-traj negs   : repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS"
echo "Actor auto-reset: $ACTOR_AUTO_RESET (threshold=$ACTOR_RESET_DORMANT_THRESHOLD, warmup=$ACTOR_RESET_WARMUP, max=$ACTOR_RESET_MAX)"
echo "Neg bank       : mode=$NEG_BANK_MODE (M=$NEG_BANK_N_PER_STEP, pool=$NEG_BANK_CANDIDATE_POOL, weight=$NEG_BANK_WEIGHT, cap/task=$NEG_BANK_PER_TASK_CAPACITY)"
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
if [ -n "$WANDB_GROUP" ]; then
  FLAGS="$FLAGS --wandb_group=$WANDB_GROUP"
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
if [ "$ACTOR_AUTO_RESET" = "true" ]; then
  FLAGS="$FLAGS --actor_auto_reset"
else
  FLAGS="$FLAGS --noactor_auto_reset"
fi
FLAGS="$FLAGS --actor_reset_dormant_threshold=$ACTOR_RESET_DORMANT_THRESHOLD"
FLAGS="$FLAGS --actor_reset_warmup=$ACTOR_RESET_WARMUP"
FLAGS="$FLAGS --actor_reset_max=$ACTOR_RESET_MAX"
FLAGS="$FLAGS --neg_bank_mode=$NEG_BANK_MODE"
FLAGS="$FLAGS --neg_bank_per_task_capacity=$NEG_BANK_PER_TASK_CAPACITY"
FLAGS="$FLAGS --neg_bank_n_per_step=$NEG_BANK_N_PER_STEP"
FLAGS="$FLAGS --neg_bank_candidate_pool=$NEG_BANK_CANDIDATE_POOL"
FLAGS="$FLAGS --neg_bank_weight=$NEG_BANK_WEIGHT"
FLAGS="$FLAGS --neg_bank_max_tasks=$NEG_BANK_MAX_TASKS"

# Decomposed-critic + diagnostic flags
FLAGS="$FLAGS --dyn_aux_weight=$DYN_AUX_WEIGHT"
FLAGS="$FLAGS --dyn_aux_after_task0=$DYN_AUX_AFTER_TASK0"
FLAGS="$FLAGS --phi_task_width=$PHI_TASK_WIDTH"
FLAGS="$FLAGS --phi_task_depth=$PHI_TASK_DEPTH"
FLAGS="$FLAGS --combine_mode=$COMBINE_MODE"
FLAGS="$FLAGS --goal_encoder_mode=$GOAL_ENCODER_MODE"
FLAGS="$FLAGS --in_trajectory_negative_repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS"
FLAGS="$FLAGS --bellman_loss_weight=$BELLMAN_LOSS_WEIGHT"
FLAGS="$FLAGS --bellman_residual_l2_weight=$BELLMAN_RESIDUAL_L2_WEIGHT"
FLAGS="$FLAGS --bellman_discount=$BELLMAN_DISCOUNT"
FLAGS="$FLAGS --bellman_tau=$BELLMAN_TAU"
FLAGS="$FLAGS --bellman_hidden_dim=$BELLMAN_HIDDEN_DIM"
FLAGS="$FLAGS --her_reward_threshold=$HER_REWARD_THRESHOLD"
FLAGS="$FLAGS --dcc_sac_q_loss_weight=$DCC_SAC_Q_LOSS_WEIGHT"
FLAGS="$FLAGS --dcc_sac_q_learning_rate=$DCC_SAC_Q_LEARNING_RATE"
FLAGS="$FLAGS --dcc_sac_discount=$DCC_SAC_DISCOUNT"
FLAGS="$FLAGS --dcc_sac_tau=$DCC_SAC_TAU"
FLAGS="$FLAGS --dcc_sac_q_hidden_dim=$DCC_SAC_Q_HIDDEN_DIM"
FLAGS="$FLAGS --dcc_sac_beta_max=$DCC_SAC_BETA_MAX"
FLAGS="$FLAGS --dcc_sac_q_warmup_updates=$DCC_SAC_Q_WARMUP_UPDATES"
FLAGS="$FLAGS --dcc_sac_q_ramp_updates=$DCC_SAC_Q_RAMP_UPDATES"
FLAGS="$FLAGS --dcc_sac_td_error_threshold=$DCC_SAC_TD_ERROR_THRESHOLD"
FLAGS="$FLAGS --dcc_sac_twin_disagreement_threshold=$DCC_SAC_TWIN_DISAGREEMENT_THRESHOLD"
FLAGS="$FLAGS --dcc_sac_ema_decay=$DCC_SAC_EMA_DECAY"
FLAGS="$FLAGS --dcc_sac_candidate_actions=$DCC_SAC_CANDIDATE_ACTIONS"
FLAGS="$FLAGS --dcc_sac_normalization_eps=$DCC_SAC_NORMALIZATION_EPS"
FLAGS="$FLAGS --dcc_sac_correction_clip=$DCC_SAC_CORRECTION_CLIP"
FLAGS="$FLAGS --action_contrast_weight=$ACTION_CONTRAST_WEIGHT"
FLAGS="$FLAGS --action_contrast_temperature=$ACTION_CONTRAST_TEMPERATURE"
FLAGS="$FLAGS --action_contrast_batch_size=$ACTION_CONTRAST_BATCH_SIZE"
FLAGS="$FLAGS --shortcut_diagnostic_interval=$SHORTCUT_DIAGNOSTIC_INTERVAL"
FLAGS="$FLAGS --shortcut_diagnostic_batch_size=$SHORTCUT_DIAGNOSTIC_BATCH_SIZE"
FLAGS="$FLAGS --shortcut_candidate_actions=$SHORTCUT_CANDIDATE_ACTIONS"
FLAGS="$FLAGS --post_task_eval_scope=$POST_TASK_EVAL_SCOPE"
if [ "$STEP_PENALTY_REWARD" = "true" ]; then
  FLAGS="$FLAGS --step_penalty_reward"
else
  FLAGS="$FLAGS --nostep_penalty_reward"
fi
if [ "$LOG_POOL_COSINE" = "true" ]; then
  FLAGS="$FLAGS --log_pool_cosine"
else
  FLAGS="$FLAGS --nolog_pool_cosine"
fi
if [ "$LOG_MIXTURE_NORM" = "true" ]; then
  FLAGS="$FLAGS --log_mixture_norm"
else
  FLAGS="$FLAGS --nolog_mixture_norm"
fi
if [ "$LOG_PROBE_DATA" = "true" ]; then
  FLAGS="$FLAGS --log_probe_data"
else
  FLAGS="$FLAGS --nolog_probe_data"
fi

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
