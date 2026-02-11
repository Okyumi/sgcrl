#!/bin/bash
###############################################################################
# setup_gpu.sh - GPU-specific setup for SGCRL on NYUAD HPC
#
# IMPORTANT: This script must be run from a GPU node (e.g., inside an
# interactive GPU session or a SLURM GPU job).
#
# Usage:
#   # First, get a GPU interactive session:
#   srun --gres=gpu:1 --pty bash
#   # Then run:
#   bash setup_gpu.sh
###############################################################################

set -e

ENV_NAME="contrastive_rl"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"

# Redirect caches to /scratch to avoid home quota issues
export PIP_CACHE_DIR="/scratch/$(whoami)/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/$(whoami)/.cache/conda/pkgs"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" 2>/dev/null || true

echo "============================================="
echo "GPU Setup for SGCRL"
echo "============================================="

# ======================== STEP 1: Load Modules ==============================
echo ""
echo "Step 1: Loading modules..."

ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
echo "Loading: $ANACONDA_MODULE"
module load "$ANACONDA_MODULE"

# NYUAD HPC has cuda/11.8.0 (no separate cuDNN module)
echo "Loading: cuda/11.8.0"
module load cuda/11.8.0

echo "Modules loaded."

# ======================== STEP 2: Activate Conda Env ========================
echo ""
echo "Step 2: Activating conda environment..."

eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Active environment: $CONDA_PREFIX"

# ======================== STEP 3: Install cuDNN via conda ===================
echo ""
echo "Step 3: Installing cuDNN via conda (not available as HPC module)..."

conda install -c conda-forge cudnn=8.2 -y

# ======================== STEP 4: Set Library Paths =========================
echo ""
echo "Step 4: Setting library paths..."

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add CUDA lib from the loaded module
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
    echo "Added CUDA_HOME lib: $CUDA_HOME/lib64"
fi

# Fallback: check common locations
for cuda_lib_dir in /usr/local/cuda-11.8/lib64 /usr/local/cuda-11/lib64 /usr/local/cuda/lib64; do
    if [ -d "$cuda_lib_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${cuda_lib_dir}"
        echo "Added CUDA lib: $cuda_lib_dir"
        break
    fi
done

for nvidia_dir in /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu/nvidia; do
    if [ -d "$nvidia_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_dir}"
        echo "Added NVIDIA lib: $nvidia_dir"
        break
    fi
done

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ======================== STEP 5: Install GPU JAX ===========================
echo ""
echo "Step 5: Installing GPU-compatible JAX and optax..."

pip install optax==0.1.7

# Use CUDA 11 + cuDNN 8.2 build (matches cuda/11.8.0 + conda cudnn=8.2)
pip install --upgrade "jax==0.4.7" "jaxlib==0.4.7+cuda11.cudnn82" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "============================================="
echo "Step 6: Verifying GPU access from JAX..."
echo "============================================="

python -c "
import jax
print('JAX version:', jax.__version__)
devices = jax.devices()
print('Available devices:', devices)
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print('GPU devices found:', len(gpu_devices))
    for d in gpu_devices:
        print('  -', d)
    print('GPU setup successful!')
else:
    print('WARNING: No GPU devices found. Check CUDA/cuDNN installation.')
"

echo ""
echo "============================================================"
echo "  GPU SETUP COMPLETE"
echo "============================================================"
echo ""
echo "You can now run experiments with GPU acceleration."
echo "Use the SLURM scripts to submit jobs:"
echo "  sbatch run_experiment.slurm"
echo "  bash run_all_experiments.sh"
echo ""
