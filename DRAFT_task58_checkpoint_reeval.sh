#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=task58_ckpt_eval
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_checkpoint_reeval_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_checkpoint_reeval_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-2

# Evaluation only: Task 5/8 x seeds 5/6/7. Each array job evaluates two
# existing DCC checkpoints; it never creates a learner or replay buffer.
#
#   sbatch DRAFT_task58_checkpoint_reeval.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/yd2247/sgcrl/logs/continual_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/yd2247/sgcrl/logs/task58_checkpoint_reeval/results}"
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
python tests/test_task58_checkpoint_reevaluation.py

run_setting() {
  local setting="$1"
  eval "$(python experiment_configs_task58_checkpoint_reeval.py \
    --setting "$setting")"
  local checkpoint="$CHECKPOINT_ROOT/$CHECKPOINT_RELATIVE"
  local output="$OUTPUT_ROOT/$OUTPUT_RELATIVE"
  python -u scripts/reevaluate_task58_dcc_checkpoints.py \
    --checkpoint "$checkpoint" \
    --env-name "$ENV_NAME" \
    --task-id "$TASK_ID" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --output "$output"
}

base_setting=$(( ${SLURM_ARRAY_TASK_ID:-0} * 2 ))
run_setting "$base_setting" &
pid_a=$!
run_setting "$((base_setting + 1))" &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
