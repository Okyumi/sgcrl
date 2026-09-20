#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t8_vid
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task_videos/runs/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task_videos/runs/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=3-4

# Task-8 visual rollouts (GIFs), same protocol as Task 5.
#   3  DCC w/o BC, window-close: 10 paced GIFs + first-success hunt at ~250k
#      from existing seed-6 mid-ckpts (job 17954891)
#   4  DCC + Success-BC window-close: train 1M, save paced + first-success
#      GIFs during eval, then re-export from mid-ckpts
#
#   sbatch DRAFT_jubail_task8_videos.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export CONFIG_SCRIPT="experiment_configs_jubail_task_videos.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=5
export TASKS_PER_GPU=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LOG_DIR="/scratch/yd2247/sgcrl/logs/jubail_task_videos/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/jubail_task_videos/checkpoints"
export VIDEO_DIR="/scratch/yd2247/sgcrl/logs/jubail_task_videos/gifs"
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_GROUP="${WANDB_GROUP:-TASK58-JUBAIL-TASK-VIDEOS}"
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

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$VIDEO_DIR"

python tests/test_eval_video.py
python tests/test_jubail_task_videos.py
python -c "import experiment_configs_jubail_task_videos as c; assert len(c.build_configs())==5"

CONFIG_IDX="${SLURM_ARRAY_TASK_ID:-3}"
eval "$(python "$CONFIG_SCRIPT" --setting "$CONFIG_IDX")"

GIF_OUT="${VIDEO_DIR}/${VARIANT}"
mkdir -p "$GIF_OUT"

record_gifs() {
  local ckpt_dir="$1"
  python scripts/record_checkpoint_rollout_gifs.py \
    --checkpoint-dir "$ckpt_dir" \
    --env-name "$SINGLE_TASK" \
    --seed "$SEED" \
    --label "$LABEL" \
    --even 10 \
    --first-success-step "$FIRST_SUCCESS_STEP" \
    --output-dir "$GIF_OUT"
}

if [ "$JOB_MODE" = "offline_gifs" ]; then
  echo "Recording offline GIFs for ${VARIANT} from ${SOURCE_CHECKPOINT_DIR}"
  record_gifs "$SOURCE_CHECKPOINT_DIR"
elif [ "$JOB_MODE" = "train" ]; then
  echo "Training ${VARIANT} with disk GIFs, then re-exporting mid-ckpts"
  export CONFIG_INDEX_OFFSET="$CONFIG_IDX"
  export CONFIG_LIMIT=1
  SLURM_ARRAY_TASK_ID=0 bash "$REPO_DIR/DRAFT_jubail.sh"
  record_gifs "$CHECKPOINT_DIR"
else
  echo "Unknown JOB_MODE=${JOB_MODE}" >&2
  exit 1
fi

echo "Jubail Task-8 video cell ${CONFIG_IDX} (${VARIANT}) finished."
