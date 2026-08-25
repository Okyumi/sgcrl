#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=goal_semantics_1m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/goal_validity/train_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/goal_validity/train_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-3

# Gate V1: matched 1M full-state vs success-mechanism DCC cells.
# Run only after all three V0 array jobs pass:
#   sbatch DRAFT_goal_semantics.sh
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
export MUJOCO_GL=egl

export CONFIG_SCRIPT="experiment_configs_goal_semantics.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=8
# Two ordinary learner jobs multiplex the GPU while their single MuJoCo actor
# loops are on CPU. No serial counterfactual probes run in these cells.
export TASKS_PER_GPU=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/goal_validity/v1_1m"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/goal_validity_checkpoints/v1_1m"

python tests/test_goal_semantics.py
exec bash "$REPO_DIR/DRAFT.sh"

