#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=paper9base
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/paper_9baseline/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/paper_9baseline/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-22

# 90 matched runs: nine actor/critic reset-persistent-CKA combinations
# on seeds 5..14, native_info wrapper, representation metrics on.
# Four CPU-bound learners share each L40S. Re-submit is automatic:
# each array task chains a dependent follow-up before training, and
# run_continual_contrastive.py resumes from the latest task_*.pkl.
#
# Launch:
#   sbatch DRAFT_paper_9baseline_10seed.sh
# Resume only unfinished array tasks:
#   ARRAY=$(python scripts/paper_9baseline_status.py --incomplete-array-ids)
#   sbatch --array="$ARRAY" DRAFT_paper_9baseline_10seed.sh
set -euo pipefail

CHAIN_INDEX="${PAPER_BASELINE_CHAIN_INDEX:-0}"
MAX_CHAIN="${PAPER_BASELINE_MAX_CHAIN:-8}"
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_paper_9baseline_10seed.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=90
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.22}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export LOG_DIR="/scratch/yd2247/sgcrl/logs/paper_9baseline/10seed/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/paper_9baseline_checkpoints/10seed"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" \
    /scratch/yd2247/sgcrl/logs/paper_9baseline

python tests/test_paper_9baseline_10seed.py

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if python scripts/paper_9baseline_status.py \
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
      --export=ALL,PAPER_BASELINE_CHAIN_INDEX="${NEXT_CHAIN}",PAPER_BASELINE_MAX_CHAIN="${MAX_CHAIN}" \
      "$REPO_DIR/DRAFT_paper_9baseline_10seed.sh" || true)"
  if [ -n "$NEXT_JOB" ]; then
    echo "Scheduled continuation ${NEXT_JOB} (chain ${NEXT_CHAIN}/${MAX_CHAIN})."
  else
    echo "WARNING: failed to schedule a 48h continuation job." >&2
  fi
fi

echo "Paper 9-baseline chain index: ${CHAIN_INDEX}/${MAX_CHAIN}"
bash "$REPO_DIR/DRAFT.sh"

if [ -n "$NEXT_JOB" ] && python scripts/paper_9baseline_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} finished the curriculum; cancelling ${NEXT_JOB}."
  scancel "$NEXT_JOB" || true
fi
