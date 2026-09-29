#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_gate10
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4

# Paper-seed fill: seeds 10–14 from scratch (configs 5–9 of the 10-seed grid).
# Same checkpoint root and W&B group as seeds 5–9.
#
#   sbatch DRAFT_jubail_full_critic_gate_10seed_fill.sh
set -euo pipefail

CHAIN_INDEX="${FULL_GATE_CHAIN_INDEX:-0}"
MAX_CHAIN="${FULL_GATE_MAX_CHAIN:-8}"
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_full_critic_gate_5seed.py"
export CONFIG_INDEX_OFFSET=5
export CONFIG_LIMIT=5
export TASKS_PER_GPU="${TASKS_PER_GPU:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export LOG_DIR="/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/checkpoints"
SLURM_LOG_DIR="/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/runs"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-PAPER-DCC-FULL-CRITICGATE-5SEED}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export START_TASK=0
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

mkdir -p "$SLURM_LOG_DIR" "$LOG_DIR" "$CHECKPOINT_DIR"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

if [ "$CHAIN_INDEX" -eq 0 ]; then
  python tests/test_full_critic_gate_5seed.py
fi

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if python scripts/full_critic_gate_5seed_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} already complete. Skipping."
  exit 0
fi

NEXT_JOB=""
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$CHAIN_INDEX" -lt "$MAX_CHAIN" ]; then
  NEXT_CHAIN=$((CHAIN_INDEX + 1))
  NEXT_JOB="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --array="${ARRAY_TASK_ID}" \
      --export=ALL,FULL_GATE_CHAIN_INDEX="${NEXT_CHAIN}",FULL_GATE_MAX_CHAIN="${MAX_CHAIN}" \
      "$REPO_DIR/DRAFT_jubail_full_critic_gate_10seed_fill.sh" || true)"
  if [ -n "$NEXT_JOB" ]; then
    echo "Scheduled continuation ${NEXT_JOB} (chain ${NEXT_CHAIN}/${MAX_CHAIN})."
  else
    echo "WARNING: failed to schedule a 48h continuation job." >&2
  fi
fi

echo "Full critic-gate 10-seed fill chain index: ${CHAIN_INDEX}/${MAX_CHAIN}"
bash "$REPO_DIR/DRAFT_jubail.sh"

if [ -n "$NEXT_JOB" ] && python scripts/full_critic_gate_5seed_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} finished the curriculum; cancelling ${NEXT_JOB}."
  scancel "$NEXT_JOB" || true
fi
