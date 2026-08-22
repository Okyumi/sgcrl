#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=ocsdcc_1m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/outcome_falsification/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/outcome_falsification/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-3

# Submit one stage at a time:
#   FALSIFICATION_STAGE=1 sbatch DRAFT_outcome_falsification.sh
# Promote to stage 2/3 only after the documented gates are evaluated.
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

FALSIFICATION_STAGE="${FALSIFICATION_STAGE:-1}"
if [[ ! "$FALSIFICATION_STAGE" =~ ^[123]$ ]]; then
  echo "FALSIFICATION_STAGE must be 1, 2, or 3" >&2
  exit 2
fi

export CONFIG_SCRIPT="experiment_configs_outcome_falsification.py"
export CONFIG_INDEX_OFFSET=$(( (FALSIFICATION_STAGE - 1) * 4 ))
export CONFIG_LIMIT=4
export TASKS_PER_GPU=1
export LOG_DIR="/scratch/yd2247/sgcrl/logs/outcome_falsification/stage${FALSIFICATION_STAGE}"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/outcome_falsification_checkpoints/stage${FALSIFICATION_STAGE}"
export ACTION_LANDSCAPE_SELF_TEST=true
export OUTCOME_FALSIFICATION_SELF_TEST=true

exec bash "$REPO_DIR/DRAFT.sh"
