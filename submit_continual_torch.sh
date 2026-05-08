#!/bin/bash
#SBATCH --job-name=continual_crl
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#
# Continual run on NYU Torch: scratch Miniconda env, conda/pip CUDA (no cuda module).
#
#   sbatch submit_continual_torch.sh
#   NUM_TASKS=2 STEPS_PER_TASK=10000 sbatch submit_continual_torch.sh
#
# Override Miniconda root if needed:
#   MINICONDA_ROOT=/scratch/yd2247/miniconda3 sbatch submit_continual_torch.sh

set -euo pipefail

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
USE_TASK_ID="${USE_TASK_ID:-true}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
K_SAMPLE_K="${K_SAMPLE_K:-0}"
ADAPT_HEADS_ONLY="${ADAPT_HEADS_ONLY:-true}"
ENCODER_FROM_BASE="${ENCODER_FROM_BASE:-false}"
USE_20_TASKS="${USE_20_TASKS:-false}"
ACTOR_MODE="${ACTOR_MODE:-cka}"

USE_RESIDUAL="${USE_RESIDUAL:-true}"
NETWORK_WIDTH="${NETWORK_WIDTH:-256}"
CRITIC_DEPTH="${CRITIC_DEPTH:-4}"
ACTOR_DEPTH="${ACTOR_DEPTH:-4}"
ENERGY_FN="${ENERGY_FN:-inner_product}"
LOGSUMEXP_PENALTY="${LOGSUMEXP_PENALTY:-0.01}"
SINGLE_TASK="${SINGLE_TASK:-}"

# Decomposed-critic + diagnostic flags (defaults preserve prior behaviour).
DYN_AUX_WEIGHT="${DYN_AUX_WEIGHT:-1.0}"
PHI_TASK_WIDTH="${PHI_TASK_WIDTH:-256}"
PHI_TASK_DEPTH="${PHI_TASK_DEPTH:-4}"
LOG_POOL_COSINE="${LOG_POOL_COSINE:-true}"
LOG_MIXTURE_NORM="${LOG_MIXTURE_NORM:-false}"
LOG_PROBE_DATA="${LOG_PROBE_DATA:-false}"

LOG_DIR="${LOG_DIR:-/scratch/yd2247/sgcrl/logs/continual}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/sgcrl/logs/continual_checkpoints}"
REPO_DIR="/scratch/yd2247/sgcrl"

export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"

module purge 2>/dev/null || true

# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl

# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "============================================================"
echo "Continual Goal-Conditioned Contrastive RL (Torch / conda CUDA)"
echo "============================================================"
echo "SLURM Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node           : $(hostname)"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda prefix   : $CONDA_PREFIX"
echo "Seed           : $SEED"
echo "Algorithm      : $ALG"
echo "Num tasks      : $NUM_TASKS"
echo "Steps / task   : $STEPS_PER_TASK"
echo "============================================================"

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

# Decomposed-critic + diagnostic flags
FLAGS="$FLAGS --dyn_aux_weight=$DYN_AUX_WEIGHT"
FLAGS="$FLAGS --phi_task_width=$PHI_TASK_WIDTH"
FLAGS="$FLAGS --phi_task_depth=$PHI_TASK_DEPTH"
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

cd "$REPO_DIR"
echo "Running: python run_continual_contrastive.py $FLAGS"
python run_continual_contrastive.py $FLAGS

echo "Done. Checkpoints: $CHECKPOINT_DIR"
