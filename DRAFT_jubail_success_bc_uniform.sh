#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_bc_uni
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/success_bc_uniform/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/success_bc_uniform/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-49

# Uniform Success-BC ideas after Task 7 warmup (job 18064660):
#   0–9    Task 7 recency
#   10–19  Task 5 warmup
#   20–29  Task 5 recency
#   30–39  Task 8 warmup
#   40–49  Task 8 recency
#   sbatch DRAFT_jubail_success_bc_uniform.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_success_bc_uniform.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=50
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LOG_DIR="/scratch/yd2247/sgcrl/logs/success_bc_uniform/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/success_bc_uniform/checkpoints"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-PAPER-DCC-SUCCESS-BC-UNIFORM}"
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

python tests/test_success_bc_uniform.py

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG_INDEX_OFFSET="$CONFIG_IDX"
export CONFIG_LIMIT=1
SLURM_ARRAY_TASK_ID=0 bash "$REPO_DIR/DRAFT_jubail.sh"

echo "Success-BC uniform cell ${CONFIG_IDX} finished."
