#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_landscape
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-1

# Torch-HPC wrapper for the six causal DCC action-landscape experiments.
# DRAFT.sh launches three configurations per GPU:
#   array 0 -> configs 12, 13, 14 (Task 5; seeds 5, 6, 7)
#   array 1 -> configs 15, 16, 17 (Task 8; seeds 5, 6, 7)

set -euo pipefail

REPO_DIR=/scratch/yd2247/sgcrl
export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"

module purge 2>/dev/null || true
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

cd "$REPO_DIR"

# Do not spend a 48-hour allocation on a simulator whose wrapper state cannot
# be restored exactly.  This checks both target environments before training.
python -m contrastive.action_ranking_diagnostics \
  --self-test-env=sawyer_handle_press_side \
  --self-test-env=sawyer_window_close \
  --seed=5

export CONFIG_INDEX_OFFSET=12
export CONFIG_LIMIT=6
# Keep diagnostic checkpoints separate from prior plain-DCC single-task runs;
# otherwise auto-resume could incorrectly decide these jobs are complete.
export CHECKPOINT_DIR=/scratch/yd2247/sgcrl/logs/action_landscape_checkpoints

exec bash "$REPO_DIR/DRAFT.sh"
