#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=goal_wrapper_controls
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/goal_validity/controls_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/goal_validity/controls_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Paired positive controls for seeds 5, 6, and 7. A GPU is required so
# mujoco_py uses EGL instead of compiling the OSMesa CPU shim
# (GL/osmesa.h is not present on Torch CPU nodes). No training.
#   sbatch DRAFT_goal_wrapper_positive_controls.sh
#
# After all three finish:
#   python scripts/evaluate_goal_wrapper_positive_controls.py \
#     logs/goal_validity/positive_controls_v4_seed{5,6,7}.json
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

SETTING="${SLURM_ARRAY_TASK_ID:-0}"
eval "$(python experiment_configs_goal_wrapper_positive_controls.py \
  --setting "$SETTING")"
mkdir -p logs/goal_validity

# This explicitly runs inside contrastive_rl so NumPy, MetaWorld, and MuJoCo
# are available; do not invoke the module with Torch's system Python.
python tests/test_goal_wrapper_positive_controls.py
python -u scripts/audit_sawyer_goal_positive_controls.py \
  --seed "$SEED" \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --expert-success-min "$EXPERT_SUCCESS_MIN" \
  --fixed-success-max "$FIXED_SUCCESS_MAX" \
  --trajectory-tolerance "$TRAJECTORY_TOLERANCE" \
  --target-tolerance "$TARGET_TOLERANCE" \
  --output "$OUTPUT" \
  --wandb-project continual_gcrl_paper \
  --wandb-group "$WANDB_GROUP"
