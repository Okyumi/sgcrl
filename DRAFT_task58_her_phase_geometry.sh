#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_her_phase
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_her_phase_geometry/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_her_phase_geometry/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-11

# Cross-task HER future-phase geometry probe:
#   0-2  push_progress
#   3-5  handle_contact
#   6-8  faucet_contact
#   9-11 peg_contact
#
#   sbatch DRAFT_task58_her_phase_geometry.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_her_phase_geometry.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=12
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_her_phase_geometry/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_her_phase_geometry/checkpoints"

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

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

python tests/test_her_future_phase.py
python tests/test_task58_her_phase_geometry.py
exec bash "$REPO_DIR/DRAFT.sh"
