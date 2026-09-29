#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jubail_t56_strip
#SBATCH --partition=nvidia
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/jubail_task56_episode_strips/runs/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/jubail_task56_episode_strips/runs/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL

# Record Task-5 / Task-6 appendix filmstrip from existing seed-6 DCC
# checkpoints. No training.
#
#   sbatch DRAFT_jubail_task56_episode_strips.sh
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

mkdir -p logs/jubail_task56_episode_strips/runs
mkdir -p results/data/task56_episode_strips
mkdir -p results/img/paper

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

python tests/test_task56_episode_strips.py
python scripts/record_task56_episode_strips.py
