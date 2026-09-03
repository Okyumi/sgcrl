#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_bc_video
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_success_bc_video_v1/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_success_bc_video_v1/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-5

# Task-5 corrected-wrapper retry: plain DCC vs success-buffer BC + videos.
#
#   0-2  dcc_baseline      seeds 5/6/7  (decomposed critic, no BC)
#   3-5  success_bc_combined seeds 5/6/7 (advantage_decomposed + raw_horizon
#        retention, success_bc_weight=0.1, combined actor — not effect_only)
#
# W&B group: TASK58-SUCCESS-BC-VIDEO-4M
# Videos: evaluator/rollout_video every 100k env steps on W&B Media tab.
#
#   sbatch DRAFT_task58_success_bc_video.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_success_bc_video.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=6
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_success_bc_video_v1/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_success_bc_video_v1/checkpoints"

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

python tests/test_task58_success_bc_video.py
python tests/test_task58_reachable_success_goals.py
exec bash "$REPO_DIR/DRAFT.sh"
