#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_ret_v2
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_retention_diagnostics_v2/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_retention_diagnostics_v2/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-14

# Task-5 retention diagnostics v2 (checkpoint-identity fix):
#
#   0-2   her_uniform
#   3-5   her_final_state
#   6-8   her_success_oversample
#   9-11  freeze_critic_0p3
#  12-14  discounted_entropy_off
#
# Separate log/checkpoint roots from v1 so auto-resume cannot collide with
# the incomplete v1 tree. W&B: TASK58-RETENTION-DIAGNOSTICS-1M-V2
#
#   sbatch DRAFT_task58_retention_diagnostics.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_retention_diagnostics.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=15
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_retention_diagnostics_v2/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_retention_diagnostics_v2/checkpoints"

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

python tests/test_task58_retention_diagnostics.py
python tests/test_critic_phase_probe.py
exec bash "$REPO_DIR/DRAFT.sh"
