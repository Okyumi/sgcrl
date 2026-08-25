#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_axis_dcc
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_axis/slurm_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_axis/slurm_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-1

# Four small cells: Task 5/8 x seeds 5/6. Two ordinary DCC learners share
# each L40S. The only experimental change is the direct sparse-reward axis.
#
# Launch:
#   sbatch DRAFT_task58_axis_reward.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_axis_reward.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=4
export TASKS_PER_GPU=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_axis/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_axis/checkpoints"

python tests/test_task58_axis_reward.py
exec bash "$REPO_DIR/DRAFT.sh"
