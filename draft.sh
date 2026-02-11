#!/bin/bash
#SBATCH --job-name=sgcrl_test
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=preempt
#SBATCH --output=/scratch/yd2247/sgcrl/logs/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/%j.err
#SBATCH --mail-user=yd2247@nyu.edu

set -euo pipefail

# Headless MuJoCo (no display on compute node)
export MUJOCO_GL=egl

# Use scratch for caches/tmp (avoid home quota)
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Logs directory for this job
mkdir -p /scratch/yd2247/sgcrl/logs

# Modules (match setup_env.sh / setup_gpu.sh)
module purge
module load miniconda/3-4.11.0
module load cuda/11.8.0

# Conda and env
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
export PATH="${CONDA_PREFIX}/bin:$PATH"

# Library path: conda + MuJoCo (required by README)
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
fi

# Run from repo root
cd /scratch/yd2247/sgcrl

# Short test run (default env sawyer_bin, alg contrastive_cpc)
python lp_contrastive.py \
  --lp_launch_type=local_mt \
  --env=sawyer_bin \
  --alg=contrastive_cpc \
  --num_steps=50_000 \
  --seed=42
