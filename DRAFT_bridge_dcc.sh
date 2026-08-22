#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=bridge_dcc
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/bridge_dcc/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/bridge_dcc/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-5

# Matched task-5/task-8 pilots (indices 18--35):
#   IWR only, forward action-effect only, and Bridge-DCC (both), seeds 5/6/7.
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

# Counterfactual probes are only meaningful if MuJoCo state restoration is
# exact on the installed Torch environment. Abort the whole array otherwise.
python -m contrastive.action_ranking_diagnostics \
  --self-test-env=sawyer_handle_press_side \
  --self-test-env=sawyer_window_close \
  --seed=5

export CONFIG_INDEX_OFFSET=18
export CONFIG_LIMIT=18
export LOG_DIR="/scratch/yd2247/sgcrl/logs/bridge_dcc"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/bridge_dcc_checkpoints"

exec bash "$REPO_DIR/DRAFT.sh"
