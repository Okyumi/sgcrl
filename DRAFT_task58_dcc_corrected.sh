#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_dcc_goal_v2
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_dcc_goal_v2_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_dcc_goal_v2_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Six clean single-task DCC baselines: Task 5/8 x seeds 5/6/7. Two learners
# share each L40S. All simulator-based diagnostic probes are disabled; the
# Task-5/8 stage metrics reuse ordinary deterministic evaluation trajectories.
#
#   sbatch DRAFT_task58_dcc_corrected.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_dcc_corrected.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=6
export TASKS_PER_GPU=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_dcc_corrected_goal_v2/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_dcc_corrected_goal_v2/checkpoints"

python tests/test_task58_dcc_corrected.py
python tests/test_task58_reachable_success_goals.py
exec bash "$REPO_DIR/DRAFT.sh"
