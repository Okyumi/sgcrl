#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=native_success_8m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/native_success/promote_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/native_success/promote_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Guarded 8M/task paper revalidation. It is intentionally unavailable until
# the V4 wrapper controls and 100k/task ten-task smoke matrix pass.
# Re-submit the same command after a wall-time exit; task-boundary checkpoints
# are mode-separated and auto-resume without colliding with legacy runs.
#
# Launch after promotion review:
#   NATIVE_SUCCESS_WRAPPER_PROMOTED=true \
#     sbatch DRAFT_native_success_wrapper_promotion.sh
set -euo pipefail

if [ "${NATIVE_SUCCESS_WRAPPER_PROMOTED:-false}" != "true" ]; then
  echo "V4 controls and the ten-task smoke must pass before promotion." >&2
  exit 2
fi

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"
export NATIVE_SUCCESS_WRAPPER_PROMOTED
export CONFIG_SCRIPT="experiment_configs_native_success_wrapper_promotion.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=9
export TASKS_PER_GPU=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export LOG_DIR="/scratch/yd2247/sgcrl/logs/native_success/v2_promotion"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/native_success_checkpoints/v2_promotion"

python tests/test_sawyer_native_success_wrapper.py
python tests/test_native_success_wrapper_configs.py
exec bash "$REPO_DIR/DRAFT.sh"
