#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=goal_semantics_8m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/goal_validity/promote_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/goal_validity/promote_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Gate V2: full-horizon promotion, guarded by the V1 W&B evaluator.
#   python scripts/evaluate_goal_semantics.py
#   GOAL_VALIDITY_PROMOTED=true sbatch DRAFT_goal_semantics_promotion.sh
set -euo pipefail

if [ "${GOAL_VALIDITY_PROMOTED:-false}" != "true" ]; then
  echo "Run scripts/evaluate_goal_semantics.py first; promotion is blocked." >&2
  exit 2
fi

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"
export GOAL_VALIDITY_PROMOTED
export CONFIG_SCRIPT="experiment_configs_goal_semantics_promotion.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=6
export TASKS_PER_GPU=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/goal_validity/v2_8m"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/goal_validity_checkpoints/v2_8m"

python tests/test_goal_semantics.py
exec bash "$REPO_DIR/DRAFT.sh"

