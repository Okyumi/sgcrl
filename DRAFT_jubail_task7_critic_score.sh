#!/bin/bash
#SBATCH --job-name=t7_score
#SBATCH --partition=nvidia
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task7_critic_score_gate/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task7_critic_score_gate/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL

set -euo pipefail
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"
mkdir -p /scratch/yd2247/sgcrl/logs/task7_critic_score_gate

module purge 2>/dev/null || true
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/jubail_hpc_env.sh"
export MUJOCO_GL=egl

python scripts/score_task7_success_path_critic.py --episodes 30 \
  --output /scratch/yd2247/sgcrl/logs/task7_critic_score_gate/scores.json
