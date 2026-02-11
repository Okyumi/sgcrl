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

echo "============================================="
echo "GPU Setup for SGCRL"
echo "============================================="

# ======================== STEP 1: Load Modules ==============================
echo ""
echo "Step 1: Loading modules (anaconda, CUDA, cuDNN)..."

# Auto-detect the correct anaconda module name
ANACONDA_MODULE=""
for mod in anaconda3 anaconda anaconda/2023.07 anaconda/3 miniconda/3-4.11.0 miniconda-nobashrc/3-4.11.0; do
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
        ANACONDA_MODULE="$mod"
        break
    fi
done
if [ -z "$ANACONDA_MODULE" ]; then
    echo "ERROR: Could not find an anaconda module."
    module avail 2>&1 | grep -i conda || true
    exit 1
fi
echo "Loading: $ANACONDA_MODULE"
module load "$ANACONDA_MODULE"

module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

echo "Modules loaded."

# ======================== STEP 2: Activate Conda Env ========================
echo ""
echo "Step 2: Activating conda environment..."

eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Active environment: $CONDA_PREFIX"

# ======================== STEP 3: Set Library Paths =========================
echo ""
echo "Step 3: Setting library paths..."

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add CUDA library paths
# These paths may vary on NYUAD HPC; adjust if needed
if [ -d "/usr/local/cuda-11.3/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.3/lib64"
elif [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
fi

if [ -d "/usr/lib/nvidia" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ======================== STEP 4: Install GPU JAX ===========================
echo ""
echo "Step 4: Installing GPU-compatible JAX and optax..."

# Install optax version compatible with this JAX
pip install optax==0.1.7

# Install JAX with CUDA 11 + cuDNN 8.2 support
pip install --upgrade "jax==0.4.7" "jaxlib==0.4.7+cuda11.cudnn82" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "============================================="
echo "Step 5: Verifying GPU access from JAX..."
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
