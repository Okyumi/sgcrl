#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t7_dsucc_gifs
#SBATCH --partition=nvidia
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task7_success_buffer_gifs/runs/%A.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task7_success_buffer_gifs/runs/%A.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL

# Render every contiguous D_succ run in the Task-7 success-buffer
# snapshots (terminal, warmup, first-success-window).
#
#   sbatch DRAFT_jubail_task7_success_buffer_gifs.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

export VIDEO_DIR="/scratch/yd2247/sgcrl/logs/jubail_task7_success_buffer_gifs/gifs"
export LOG_DIR="/scratch/yd2247/sgcrl/logs/jubail_task7_success_buffer_gifs/runs"

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

python tests/test_success_buffer_gifs.py

python scripts/record_success_buffer_gifs.py \
  --output-dir "$VIDEO_DIR" \
  --webp \
  --episode-len 150 \
  --max-frames 150 \
  --max-segments 0
