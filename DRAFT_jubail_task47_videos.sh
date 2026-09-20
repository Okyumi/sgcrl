#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t47_vid
#SBATCH --partition=nvidia
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task47_videos/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task47_videos/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3

# Paper Task-4 / Task-7 final-policy GIFs (plain DCC vs Success-BC).
#   0  stick_pull plain
#   1  stick_pull Success-BC
#   2  shelf_place plain
#   3  shelf_place Success-BC
#
#   sbatch DRAFT_jubail_task47_videos.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export VIDEO_DIR="/scratch/yd2247/sgcrl/logs/jubail_task47_videos/gifs"
export LOG_DIR="/scratch/yd2247/sgcrl/logs/jubail_task47_videos/runs"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

python tests/test_jubail_task47_videos.py

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-0}"
eval "$(python experiment_configs_jubail_task47_videos.py --setting "$CONFIG_IDX")"

GIF_OUT="${VIDEO_DIR}/${VARIANT}"
mkdir -p "$GIF_OUT"

python scripts/record_checkpoint_rollout_gifs.py \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --env-name "$SINGLE_TASK" \
  --seed "$SEED" \
  --label "$LABEL" \
  --rollouts "$ROLLOUTS" \
  --hunt-episodes "$HUNT_EPISODES" \
  --network-width "$NETWORK_WIDTH" \
  --critic-depth "$CRITIC_DEPTH" \
  --actor-depth "$ACTOR_DEPTH" \
  --sawyer-success-mode "$SAWYER_SUCCESS_MODE" \
  --output-dir "$GIF_OUT"

echo "Jubail Task-4/7 video cell ${CONFIG_IDX} (${VARIANT}) finished."
