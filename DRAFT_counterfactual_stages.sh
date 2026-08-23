#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=cfr_staged
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/counterfactual_stages_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/counterfactual_stages_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-3

# Torch staged counterfactual experiments.
#
#   COUNTERFACTUAL_STAGE=A sbatch DRAFT_counterfactual_stages.sh
#   COUNTERFACTUAL_STAGE=B sbatch DRAFT_counterfactual_stages.sh
#   COUNTERFACTUAL_STAGE=C sbatch DRAFT_counterfactual_stages.sh
#   COUNTERFACTUAL_STAGE=D COUNTERFACTUAL_REACH_MODE=scripted_contact \
#     sbatch DRAFT_counterfactual_stages.sh
#   COUNTERFACTUAL_STAGE=D COUNTERFACTUAL_REACH_MODE=policy \
#     sbatch DRAFT_counterfactual_stages.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export COUNTERFACTUAL_STAGE="${COUNTERFACTUAL_STAGE:-A}"
export COUNTERFACTUAL_REACH_MODE="${COUNTERFACTUAL_REACH_MODE:-scripted_contact}"
case "$COUNTERFACTUAL_STAGE" in
  A|B|C|D) ;;
  *) echo "COUNTERFACTUAL_STAGE must be A, B, C, or D" >&2; exit 2 ;;
esac
case "$COUNTERFACTUAL_REACH_MODE" in
  policy|scripted_contact) ;;
  *) echo "COUNTERFACTUAL_REACH_MODE must be policy or scripted_contact" >&2; exit 2 ;;
esac

export CONFIG_SCRIPT="experiment_configs_counterfactual_stages.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=4
# Simulator counterfactuals are CPU-heavy; one job per GPU avoids the
# oversubscription and synchronized crashes seen in the previous batch.
export TASKS_PER_GPU=1
export LOG_DIR="/scratch/yd2247/sgcrl/logs/counterfactual_stages/${COUNTERFACTUAL_STAGE}_${COUNTERFACTUAL_REACH_MODE}"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/counterfactual_stages_checkpoints/${COUNTERFACTUAL_STAGE}_${COUNTERFACTUAL_REACH_MODE}"
export ACTION_LANDSCAPE_SELF_TEST=true
export COUNTERFACTUAL_RANK_SELF_TEST=true
export COUNTERFACTUAL_STAGES_SELF_TEST=true

exec bash "$REPO_DIR/DRAFT.sh"
