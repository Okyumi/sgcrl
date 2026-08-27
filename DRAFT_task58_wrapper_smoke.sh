#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_wrapper_smoke
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/wrapper_smoke/task58_%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/wrapper_smoke/task58_%j.err
#SBATCH --mail-user=yd2247@nyu.edu

# Evaluation-only simulator gate for the original Task-5/Task-8 wrappers.
# No learner, replay server, checkpoint, or W&B run is created.
#
#   sbatch DRAFT_task58_wrapper_smoke.sh
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
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p logs/wrapper_smoke
python tests/test_task58_corrected_wrapper_smoke.py
python tests/test_task58_axis_reward.py
python -u scripts/smoke_test_task58_corrected_wrapper.py \
  --seeds 5 6 7 \
  --episodes 5 \
  --training-horizon 150 \
  --max-steps 200 \
  --expert-success-min 0.80 \
  --output logs/wrapper_smoke/task58_corrected_wrapper.json
