#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t5_feat
#SBATCH --partition=nvidia
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task5_feature_shortcut/runs/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task5_feature_shortcut/runs/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL

# Offline critic-feature probe on the seed-6 Jubail checkpoints.
# Does not retrain. Isolates handle-z vs object-xy vs action.
#
#   sbatch DRAFT_jubail_task5_feature_shortcut.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

CKPT_ROOT="/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/checkpoints"
OUT_DIR="/scratch/yd2247/sgcrl/logs/jubail_task5_feature_shortcut/runs"
mkdir -p "$OUT_DIR"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

python tests/test_critic_feature_shortcut.py

python scripts/run_feature_shortcut_from_checkpoints.py \
  --checkpoint-dir "$CKPT_ROOT" \
  --env-name sawyer_handle_press_side \
  --seed 6 \
  --targets 100000,latest \
  --output-dir "$OUT_DIR" \
  --episodes 40

python scripts/run_feature_shortcut_from_checkpoints.py \
  --checkpoint-dir "$CKPT_ROOT" \
  --env-name sawyer_push \
  --seed 6 \
  --targets 250000,latest \
  --output-dir "$OUT_DIR" \
  --episodes 40

echo "Feature-shortcut probe finished."
ls -l "$OUT_DIR"/feature_shortcut_*.json
