#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t5_advice
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2

# Jubail 1-seed Task-5 diagnostic (action advice vs 20% mass inject).
#   0  handle_measure
#   1  push_measure
#   2  handle_inject_20pct
#
#   sbatch DRAFT_jubail_task5_action_advice.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_jubail_task5_action_advice.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=3
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LOG_DIR="/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/checkpoints"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-TASK58-JUBAIL-ACTION-ADVICE-1SEED}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

python tests/test_jubail_task5_action_advice.py
python -c "import experiment_configs_jubail_task5_action_advice as c; assert len(c.build_configs())==3"

bash "$REPO_DIR/DRAFT_jubail.sh"

CONFIG_IDX=$(( CONFIG_INDEX_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))
eval "$(python "$CONFIG_SCRIPT" --setting "$CONFIG_IDX")"

if [ "${RUN_ACTION_ADVICE_PROBE:-false}" = "true" ]; then
  echo "Running same-state action-advice probe for ${VARIANT:-unknown}"
  python scripts/run_action_advice_from_checkpoints.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --env-name "$SINGLE_TASK" \
    --seed "$SEED" \
    --targets "${ACTION_ADVICE_TARGETS:-latest}" \
    --output-dir "$LOG_DIR" \
    --episodes 40
fi

echo "Jubail Task-5 diagnostic cell ${CONFIG_IDX} finished."
