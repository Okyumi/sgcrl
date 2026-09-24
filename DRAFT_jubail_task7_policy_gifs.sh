#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t7_gifs
#SBATCH --partition=nvidia
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task7_policy_gifs/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task7_policy_gifs/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2

# Task-7 shelf_place GIFs, seed 6.
#   0  plain DCC final + first-success hunt
#   1  episode-wide Success-BC final + first-success hunt
#   2  terminal-episode Success-BC: 10 paced mid-ckpts over 8M
#      plus first-success hunt at the first 10% eval (~1.5M)
#
#   sbatch DRAFT_jubail_task7_policy_gifs.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export VIDEO_DIR="/scratch/yd2247/sgcrl/logs/jubail_task7_policy_gifs/gifs"
export LOG_DIR="/scratch/yd2247/sgcrl/logs/jubail_task7_policy_gifs/runs"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

python tests/test_jubail_task7_policy_gifs.py

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-0}"
eval "$(python experiment_configs_jubail_task7_policy_gifs.py --setting "$CONFIG_IDX")"

GIF_OUT="${VIDEO_DIR}/${VARIANT}"
mkdir -p "$GIF_OUT"

if [ "$JOB_MODE" = "checkpoint_file" ]; then
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
elif [ "$JOB_MODE" = "mid_ckpts" ]; then
  python scripts/record_checkpoint_rollout_gifs.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --env-name "$SINGLE_TASK" \
    --seed "$SEED" \
    --label "$LABEL" \
    --even "$EVEN" \
    --horizon "$HORIZON" \
    --task-id "$TASK_ID" \
    --first-success-step "$FIRST_SUCCESS_STEP" \
    --hunt-episodes "$HUNT_EPISODES" \
    --network-width "$NETWORK_WIDTH" \
    --critic-depth "$CRITIC_DEPTH" \
    --actor-depth "$ACTOR_DEPTH" \
    --sawyer-success-mode "$SAWYER_SUCCESS_MODE" \
    --output-dir "$GIF_OUT"
else
  echo "Unknown JOB_MODE=${JOB_MODE}" >&2
  exit 1
fi

echo "Task-7 policy GIF cell ${CONFIG_IDX} (${VARIANT}) finished."
