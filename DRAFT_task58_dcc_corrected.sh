#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_dcc_fullnet8m
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_dcc_fullnet_8m_v1/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_dcc_fullnet_8m_v1/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-11

# Twelve independent single-task DCC runs at full network capacity:
#   Task 5 (handle press) and Task 8 (window close)
#   x dynamics auxiliary on/off (dyn_aux_weight 1.0 / 0.0)
#   x seeds 5/6/7
# Each run trains 8M env steps on one L40S. Eval every 50k steps logs
# evaluator/success_rate, evaluator/task58/* stage metrics, and learner RL
# metrics (log_rl_metrics=true).
#
#   sbatch DRAFT_task58_dcc_corrected.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_task58_dcc_corrected.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=12
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LOG_DIR="/scratch/yd2247/sgcrl/logs/task58_dcc_fullnet_8m_v1/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/task58_dcc_fullnet_8m_v1/checkpoints"

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

python tests/test_task58_dcc_corrected.py
python tests/test_task58_reachable_success_goals.py
exec bash "$REPO_DIR/DRAFT.sh"
