#!/bin/bash
#SBATCH --job-name=sgcrl_test
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/sgcrl/logs/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/%j.err
#SBATCH --mail-user=yd2247@nyu.edu

set -euo pipefail

module purge
module load cuda/11.8.0
# module load ffmpeg/4.2.2
# module load mesa/20.2.1

# Use EGL for headless MuJoCo rendering
export MUJOCO_GL=egl

# Protobuf 4.x is incompatible with TensorFlow's generated _pb2.py; use pure-Python impl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Use only conda env packages; ignore ~/.local (avoids protobuf / TF conflict from user installs)
export PYTHONNOUSERSITE=1

# Reduce log noise (optional: comment out if you want verbose TF/JAX logs)
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_VLOG_LEVEL=3
# Do NOT set JAX_PLATFORMS=cuda: actors/evaluators use device='cpu'; with cuda-only JAX that raises "Unknown backend cpu".
# export JAX_PLATFORMS=cuda


# Use scratch for caches/tmp (avoid home quota)
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Initialize conda (conda environment provides its own Python, no need for python/3.10 module)
export MKL_INTERFACE_LAYER=LP64,GNU
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate contrastive_rl

# Ensure conda environment's Python is first in PATH
export PATH="${CONDA_PREFIX}/bin:$PATH"

# Required so C extensions (e.g. courier) can find libpython3.9.so
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# MuJoCo and CUDA (if present)
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
[ -d "${MUJOCO_DIR}/bin" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
[ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
# mujoco_py needs NVIDIA driver libs (e.g. libnvidia-glcore) on GPU nodes
for _d in /usr/lib/nvidia /usr/lib64/nvidia; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$_d" && break
done
# TensorFlow GPU: cuDNN from conda env (e.g. nvidia-cudnn-cu11)
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null) || true
[ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDNN_LIB}"

# Run from repo root
cd /scratch/yd2247/sgcrl

# Short debug run (default env sawyer_bin, alg contrastive_cpc)
# python lp_contrastive.py \
#   --lp_launch_type=local_mt \
#   --env=sawyer_push_back \
#   --alg=contrastive_cpc \
#   --num_steps=50_000 \
#   --seed=42 \
#   --use_wandb

# Replication run (paper settings: 8M steps, fixed goals, unique log dir per run).
# For 8M steps consider increasing SBATCH --time (e.g. 24:00:00 or 48:00:00) and --mem (e.g. 32GB).
python lp_contrastive.py \
  --lp_launch_type=local_mt \
  --env=sawyer_window_close \
  --alg=contrastive_cpc \
  --num_steps=8_000_000 \
  --seed=13 \
  --add_uid \
  --use_wandb \
  --log_dir_path=logs/