#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_aimass
#SBATCH --partition=nvidia
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_action_informative_mass/runs/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_action_informative_mass/runs/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL

# Train Task-8 window-close if the matched DCC checkpoint is missing, then
# extract critic action-sensitivity proxies for handle / window / push and
# render the paper figure.
#
#   sbatch DRAFT_jubail_action_informative_mass.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

CKPT_ROOT="/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/checkpoints"
OUT_DIR="/scratch/yd2247/sgcrl/logs/jubail_action_informative_mass/runs"
mkdir -p "$OUT_DIR" "$CKPT_ROOT"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

python tests/test_action_informative_mass.py

WINDOW_CKPT_DIR="${CKPT_ROOT}/actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_sawyer_window_close/seed_6"
if ! ls "${WINDOW_CKPT_DIR}"/task_0_step_*.pkl >/dev/null 2>&1; then
  echo "Training matched window-close seed-6 1M DCC run"
  export CONFIG_SCRIPT="experiment_configs_jubail_window_measure.py"
  export CONFIG_INDEX_OFFSET=0
  export CONFIG_LIMIT=1
  export TASKS_PER_GPU=1
  export LOG_DIR="$OUT_DIR"
  export CHECKPOINT_DIR="$CKPT_ROOT"
  export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
  export WANDB_GROUP="${WANDB_GROUP:-TASK58-JUBAIL-ACTION-MASS-WINDOW}"
  export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
  export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
    export WANDB_API_KEY
  fi
  bash "$REPO_DIR/DRAFT_jubail.sh"
else
  echo "Window-close checkpoints already present; skipping train."
fi

python scripts/run_action_informative_mass.py \
  --checkpoint-dir "$CKPT_ROOT" \
  --seed 6 \
  --output-dir "$OUT_DIR" \
  --episodes 40 \
  --envs sawyer_handle_press_side,sawyer_window_close,sawyer_push

echo "Action-informative-mass job finished."
ls -l "$OUT_DIR"/action_mass_*.json \
  /scratch/yd2247/sgcrl/results/img/paper/fig_action_informative_mass.pdf
