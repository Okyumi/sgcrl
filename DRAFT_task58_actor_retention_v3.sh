#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_actor_v3
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_actor_retention_v3/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_actor_retention_v3/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-14

# Task-5 actor retention v3:
#
#   0-2   dcc_control            (extended ψ probe; legacy HER actor)
#   3-5   actor_goal_task        (actor trained on env desired goal)
#   6-8   actor_goal_mix         (50/50 HER + task goal)
#   9-11  actor_success_score    (critic score on success buffer; no BC)
#  12-14  success_bc_terminal    (positive control; terminal Success-BC)
#
# W&B: TASK58-ACTOR-RETENTION-1M-V3
#
#   sbatch DRAFT_task58_actor_retention_v3.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_actor_retention_v3.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=15
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_actor_retention_v3/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_actor_retention_v3/checkpoints"

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

python tests/test_task58_actor_retention_v3.py
python tests/test_critic_phase_probe.py
exec bash "$REPO_DIR/DRAFT.sh"
