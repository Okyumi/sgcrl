#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=goal_wrapper_audit
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/goal_validity/wrapper_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/goal_validity/wrapper_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Gate V0: no GPU and no training.  Every seed must pass before V1.
#   sbatch DRAFT_goal_wrapper_audit.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"
module purge 2>/dev/null || true
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

SEEDS=(5 6 7)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTPUT="logs/goal_validity/wrapper_audit_seed${SEED}.json"
mkdir -p logs/goal_validity

python -u scripts/validate_sawyer_goal_wrapper.py \
  --seed "$SEED" \
  --episodes 50 \
  --max-steps 150 \
  --expert-success-min 0.80 \
  --output "$OUTPUT" \
  --wandb-project continual_gcrl_paper \
  --strict

