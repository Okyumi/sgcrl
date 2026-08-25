#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=native_success_smoke
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/native_success/smoke_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/native_success/smoke_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Nine paired 100k/task runs: DCC, reset/reset, persistent/persistent × seeds
# 5/6/7. Three CPU-actor/JAX-learner processes share each underused GPU.
#
# Prerequisite:
#   sbatch DRAFT_goal_wrapper_positive_controls.sh
#   python scripts/evaluate_goal_wrapper_positive_controls.py \
#     logs/goal_validity/positive_controls_v3_seed{5,6,7}.json \
#     --strict-promotion
#
# Launch:
#   sbatch DRAFT_native_success_wrapper_smoke.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_native_success_wrapper.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=9
export TASKS_PER_GPU=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export LOG_DIR="/scratch/yd2247/sgcrl/logs/native_success/v1_smoke"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/native_success_checkpoints/v1_smoke"

python tests/test_sawyer_native_success_wrapper.py
python tests/test_native_success_wrapper_configs.py
python tests/test_goal_wrapper_positive_controls.py
exec bash "$REPO_DIR/DRAFT.sh"
