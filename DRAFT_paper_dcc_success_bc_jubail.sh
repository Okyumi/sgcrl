#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_bc_jubail
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-9

# 20 matched runs on Jubail: DCC without dynamics / Success-BC vs DCC with
# Success-BC, seeds 5..14, native_info wrapper, representation metrics on.
# Two CPU-bound learners share each A100. Re-submit is automatic: each
# array task chains a dependent follow-up before training, and
# run_continual_contrastive.py resumes from the latest task_*.pkl.
#
# Launch:
#   sbatch DRAFT_paper_dcc_success_bc_jubail.sh
# Resume only unfinished array tasks:
#   ARRAY=$(python scripts/paper_dcc_success_bc_jubail_status.py --incomplete-array-ids)
#   sbatch --array="$ARRAY" DRAFT_paper_dcc_success_bc_jubail.sh
set -euo pipefail

CHAIN_INDEX="${PAPER_DCC_BC_CHAIN_INDEX:-0}"
MAX_CHAIN="${PAPER_DCC_BC_MAX_CHAIN:-8}"
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_paper_dcc_success_bc_jubail.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=20
export TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export LOG_DIR="/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail/10seed/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed"
SLURM_LOG_DIR="/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export START_TASK=0
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

# SLURM opens --output/--error before this script runs, so the log
# directory must already exist at submit time. Recreate it here too so
# continuation hops cannot lose the path.
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
  python tests/test_paper_dcc_success_bc_jubail.py
fi

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if python scripts/paper_dcc_success_bc_jubail_status.py \
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
      --export=ALL,PAPER_DCC_BC_CHAIN_INDEX="${NEXT_CHAIN}",PAPER_DCC_BC_MAX_CHAIN="${MAX_CHAIN}" \
      "$REPO_DIR/DRAFT_paper_dcc_success_bc_jubail.sh" || true)"
  if [ -n "$NEXT_JOB" ]; then
    echo "Scheduled continuation ${NEXT_JOB} (chain ${NEXT_CHAIN}/${MAX_CHAIN})."
  else
    echo "WARNING: failed to schedule a 48h continuation job." >&2
  fi
fi

echo "Paper DCC Success-BC Jubail chain index: ${CHAIN_INDEX}/${MAX_CHAIN}"
bash "$REPO_DIR/DRAFT_jubail.sh"

if [ -n "$NEXT_JOB" ] && python scripts/paper_dcc_success_bc_jubail_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} finished the curriculum; cancelling ${NEXT_JOB}."
  scancel "$NEXT_JOB" || true
fi
