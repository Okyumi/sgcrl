#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_cur_bc
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/current_sparse_bc_task47/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/current_sparse_bc_task47/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1

# Per-step Success-BC probe on Task 4 and Task 7, resumed from paper
# Success-BC task_3 / task_6 (seed 6).
#   sbatch DRAFT_jubail_current_sparse_bc_task47.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_current_sparse_bc_task47.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=2
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LOG_DIR="/scratch/yd2247/sgcrl/logs/current_sparse_bc_task47/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/current_sparse_bc_task47/checkpoints"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-PAPER-DCC-CURRENT-SPARSE-BC-TASK47}"
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

python tests/test_current_sparse_bc_task47.py

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG_INDEX_OFFSET="$CONFIG_IDX"
export CONFIG_LIMIT=1
SLURM_ARRAY_TASK_ID=0 bash "$REPO_DIR/DRAFT_jubail.sh"

echo "Current-sparse Success-BC cell ${CONFIG_IDX} finished."
