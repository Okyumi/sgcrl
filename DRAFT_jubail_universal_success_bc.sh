#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_univ_bc
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/universal_success_bc/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/universal_success_bc/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3

# One-algorithm Success-BC search: help 5/8 without hurting 4/7.
#   0  Task 4 terminal_episode, 1M from paper BC task_3
#   1  Task 7 terminal_episode, 1M from paper BC task_6
#   2  Task 5 current_sparse_reward, 1M from scratch
#   3  Task 8 current_sparse_reward, 1M from scratch
#
#   sbatch DRAFT_jubail_universal_success_bc.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_universal_success_bc.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=4
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LOG_DIR="/scratch/yd2247/sgcrl/logs/universal_success_bc/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/universal_success_bc/checkpoints"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-PAPER-DCC-UNIVERSAL-SUCCESS-BC}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

python tests/test_universal_success_bc.py

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG_INDEX_OFFSET="$CONFIG_IDX"
export CONFIG_LIMIT=1
SLURM_ARRAY_TASK_ID=0 bash "$REPO_DIR/DRAFT_jubail.sh"

echo "Universal Success-BC cell ${CONFIG_IDX} finished."
