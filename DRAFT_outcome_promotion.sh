#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=ocsdcc_promote
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/outcome_promotion/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/outcome_promotion/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-5

# Only use after one 1M stage passes the promotion gates:
#   OUTCOME_WINNER_STAGE=2 sbatch DRAFT_outcome_promotion.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

OUTCOME_WINNER_STAGE="${OUTCOME_WINNER_STAGE:-0}"
if [[ ! "$OUTCOME_WINNER_STAGE" =~ ^[123]$ ]]; then
  echo "Set OUTCOME_WINNER_STAGE=1, 2, or 3 after gate review." >&2
  exit 2
fi
export OUTCOME_WINNER_STAGE
export CONFIG_SCRIPT="experiment_configs_outcome_promotion.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=6
export TASKS_PER_GPU=1
export LOG_DIR="/scratch/yd2247/sgcrl/logs/outcome_promotion/stage${OUTCOME_WINNER_STAGE}"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/outcome_promotion_checkpoints/stage${OUTCOME_WINNER_STAGE}"
export ACTION_LANDSCAPE_SELF_TEST=true
export OUTCOME_FALSIFICATION_SELF_TEST=true

exec bash "$REPO_DIR/DRAFT.sh"
