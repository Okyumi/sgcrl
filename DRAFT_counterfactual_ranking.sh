#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=cfrdcc_1m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/counterfactual_ranking/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/counterfactual_ranking/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-3

# Four matched 1M-step cells: Tasks 5/8 x seeds 5/6.
# Submit with: sbatch DRAFT_counterfactual_ranking.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_counterfactual_ranking.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=4
export TASKS_PER_GPU=1
export LOG_DIR="/scratch/yd2247/sgcrl/logs/counterfactual_ranking"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/counterfactual_ranking_checkpoints"
export ACTION_LANDSCAPE_SELF_TEST=true
export COUNTERFACTUAL_RANK_SELF_TEST=true

exec bash "$REPO_DIR/DRAFT.sh"
